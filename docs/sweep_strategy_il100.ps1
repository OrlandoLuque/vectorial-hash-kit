# Does LCA still beat Legacy at the recommended item_limit=100?
#
# The original strategy sweep ran at il ∈ {3,10,30} where LCA's advantage
# shrank as trees got shallower (near parity at il=30). At il=100 the tree
# is shallower still — this checks whether the "LCA is the default" decision
# survives at the config we now recommend.
#
# Settings: pop=10000, il ∈ {30, 100, 200}, full sim, --respawn 0.05,
# 3 seeds. Strategy legacy/lca on the no-neighbors binary (clean, no
# bookkeeping); lca-ropes on the neighbors binary.

param(
    [string]$CsvPath = "docs/sweep_strategy_il100.csv",
    [string]$BinNbrs   = "target/release/critters_headless_nbrs.exe",
    [string]$BinNoNbrs = "target/release/critters_headless_nonbrs.exe"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture
if (-not (Test-Path $BinNbrs))   { throw "Missing: $BinNbrs" }
if (-not (Test-Path $BinNoNbrs)) { throw "Missing: $BinNoNbrs" }

$pop          = 10000
$itemLimits   = @(30, 100, 200)
$seeds        = @(42, 7, 123)
$warmup       = 120
$frames       = 120
$respawn      = 0.05
$perKind  = [int]($pop / 3)
$drifters = $pop - 2 * $perKind
$hunters  = $perKind
$pulsars  = $perKind

# (strategy, feature) combos.
$variants = @(
    @{ strat='legacy';    feature='off' }
    @{ strat='lca';       feature='off' }
    @{ strat='lca-ropes'; feature='on'  }
)

$csvHeader = 'timestamp,pop,item_limit,seed,feature,strategy,alive,leaves,arena_nodes,wall_s,mv_mean_us,mv_p50_us,mv_p95_us,vis_avg_us,atk_avg_us,ins_rm_us'
if (-not (Test-Path $CsvPath)) { $csvHeader | Out-File -FilePath $CsvPath -Encoding utf8 }
$done = New-Object 'System.Collections.Generic.HashSet[string]'
Import-Csv $CsvPath | ForEach-Object {
    [void]$done.Add("$($_.pop)|$($_.item_limit)|$($_.strategy)|$($_.seed)")
}

function Get-NumberAfter([string]$lines, [string]$label) {
    $escLabel = [regex]::Escape($label)
    $rx = "binary\s+$escLabel.*?(\d[\d\.]*)\s+(\d[\d\.]*)\s+(\d[\d\.]*)\s*$"
    foreach ($line in ($lines -split "`r?`n")) { if ($line -match $rx) { return @([double]$Matches[1], [double]$Matches[2], [double]$Matches[3]) } }
    return $null
}
function Get-IntAfter([string]$lines, [string]$pattern) {
    foreach ($line in ($lines -split "`r?`n")) { if ($line -match $pattern) { return [int64]$Matches[1] } }
    return $null
}

$totalCells = $itemLimits.Count * $variants.Count * $seeds.Count
$cellIdx = 0
$swStart = Get-Date

foreach ($il in $itemLimits) {
    foreach ($v in $variants) {
        $bin = if ($v.feature -eq 'on') { $BinNbrs } else { $BinNoNbrs }
        foreach ($seed in $seeds) {
            $cellIdx++
            $key = "$pop|$il|$($v.strat)|$seed"
            if ($done.Contains($key)) { Write-Host "[$cellIdx/$totalCells] SKIP: $key"; continue }
            Write-Host "[$cellIdx/$totalCells] il=$il strat=$($v.strat) feature=$($v.feature) seed=$seed ..."
            $cellSw = Get-Date
            $stdout = & $bin --mode binary --frames $frames --warmup $warmup `
                --drifters $drifters --hunters $hunters --pulsars $pulsars `
                --split $il --merge $il --seed $seed --update-strategy $v.strat `
                --respawn $respawn 2>&1 | Out-String
            $cellSecs = ((Get-Date) - $cellSw).TotalSeconds
            $mv  = Get-NumberAfter $stdout 'move+update (us)'
            $vis = Get-NumberAfter $stdout 'vision cull avg (us)'
            $atk = Get-NumberAfter $stdout 'attack cull avg (us)'
            $rm  = Get-NumberAfter $stdout 'insert+remove (us)'
            $alive  = Get-IntAfter $stdout 'alive\s+(\d+)'
            $leaves = Get-IntAfter $stdout 'binary:\s+(\d+)\s+leaves'
            $arena  = Get-IntAfter $stdout '(\d+)\s+arena nodes'
            if ($null -eq $mv) { Write-Host "  WARN parse failed"; Write-Host $stdout; continue }
            $ts  = (Get-Date).ToString('o')
            $inv = [System.Globalization.CultureInfo]::InvariantCulture
            function F([object]$x) { ([double]$x).ToString('F4', $inv) }
            $row = @(
                $ts, $pop, $il, $seed, $v.feature, $v.strat,
                $alive, $leaves, $arena, ([double]$cellSecs).ToString('F2', $inv),
                (F $mv[0]), (F $mv[1]), (F $mv[2]),
                (F ($(if ($vis) { $vis[0] } else { 0 }))),
                (F ($(if ($atk) { $atk[0] } else { 0 }))),
                (F ($(if ($rm)  { $rm[0]  } else { 0 })))
            ) -join ','
            Add-Content -Path $CsvPath -Value $row
            [void]$done.Add($key)
            Write-Host ("  -> mv={0:N0} vis={1:N1} | leaves={2} | {3:N1}s" -f $mv[0], $(if ($vis) { $vis[0] } else { 0 }), $leaves, $cellSecs)
        }
    }
}
$elapsed = ((Get-Date) - $swStart).TotalMinutes
Write-Host ("`nDone. Total wall: {0:N1} min. Rows in {1}" -f $elapsed, $CsvPath)
