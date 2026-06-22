# cull_walk break-even bench: does any walk strategy on the dominant
# vision_prey path beat descent on the full critters workload?
#
# Settings: pop=10000, il=100, merge=100, lca, full sim, --respawn 0.05,
# 3 seeds. Cull strategies: descent (baseline), walk-samet, walk-probe,
# walk-ropes. WalkRopes only meaningful with the neighbors feature on
# (otherwise it falls back to walk-samet inside Sims).
#
# Both binaries:
#   - critters_headless_nonbrs.exe (no feature) — used for descent /
#     walk-samet / walk-probe runs; legacy/lca update strategy without
#     bookkeeping cost
#   - critters_headless_nbrs.exe   (with feature) — used for walk-ropes,
#     and also descent for the "feature ON but no walk" comparison

param(
    [string]$CsvPath = "docs/sweep_cull_walk.csv",
    [string]$BinNbrs   = "target/release/critters_headless_nbrs.exe",
    [string]$BinNoNbrs = "target/release/critters_headless_nonbrs.exe"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path $BinNbrs))   { throw "Missing: $BinNbrs" }
if (-not (Test-Path $BinNoNbrs)) { throw "Missing: $BinNoNbrs" }

# The cells we want to compare. For each, the binary + the cull strategy.
$cells = @(
    @{ bin='nonbrs'; cull='descent';    feature='off' }
    @{ bin='nonbrs'; cull='walk-samet'; feature='off' }
    @{ bin='nonbrs'; cull='walk-probe'; feature='off' }
    @{ bin='nbrs';   cull='descent';    feature='on'  }
    @{ bin='nbrs';   cull='walk-ropes'; feature='on'  }
)
$seeds = @(42, 7, 123)

$pop      = 10000
$il       = 100
$warmup   = 120
$frames   = 120
$respawn  = 0.05
$perKind  = [int]($pop / 3)
$drifters = $pop - 2 * $perKind
$hunters  = $perKind
$pulsars  = $perKind

$csvHeader = 'timestamp,pop,item_limit,seed,feature,cull,alive,leaves,arena_nodes,wall_s,mv_mean_us,mv_p50_us,mv_p95_us,vis_avg_us,atk_avg_us,ins_rm_us'
if (-not (Test-Path $CsvPath)) {
    $csvHeader | Out-File -FilePath $CsvPath -Encoding utf8
}
$done = New-Object 'System.Collections.Generic.HashSet[string]'
Import-Csv $CsvPath | ForEach-Object {
    [void]$done.Add("$($_.pop)|$($_.feature)|$($_.cull)|$($_.seed)")
}

function Get-NumberAfter([string]$lines, [string]$label) {
    $escLabel = [regex]::Escape($label)
    $rx = "binary\s+$escLabel.*?(\d[\d\.]*)\s+(\d[\d\.]*)\s+(\d[\d\.]*)\s*$"
    foreach ($line in ($lines -split "`r?`n")) {
        if ($line -match $rx) {
            return @([double]$Matches[1], [double]$Matches[2], [double]$Matches[3])
        }
    }
    return $null
}
function Get-IntAfter([string]$lines, [string]$pattern) {
    foreach ($line in ($lines -split "`r?`n")) {
        if ($line -match $pattern) { return [int64]$Matches[1] }
    }
    return $null
}

$totalCells = $cells.Count * $seeds.Count
$cellIdx = 0
$swStart = Get-Date

foreach ($c in $cells) {
    $bin = if ($c.bin -eq 'nbrs') { $BinNbrs } else { $BinNoNbrs }
    foreach ($seed in $seeds) {
        $cellIdx++
        $key = "$pop|$($c.feature)|$($c.cull)|$seed"
        if ($done.Contains($key)) {
            Write-Host "[$cellIdx/$totalCells] SKIP: $key"
            continue
        }
        Write-Host "[$cellIdx/$totalCells] feature=$($c.feature) cull=$($c.cull) seed=$seed ..."
        $cellSw = Get-Date
        $stdout = & $bin `
            --mode binary `
            --frames $frames `
            --warmup $warmup `
            --drifters $drifters `
            --hunters $hunters `
            --pulsars $pulsars `
            --split $il `
            --merge $il `
            --seed $seed `
            --update-strategy lca `
            --respawn $respawn `
            --cull-strategy $c.cull 2>&1 | Out-String
        $cellSecs = ((Get-Date) - $cellSw).TotalSeconds

        $mv  = Get-NumberAfter $stdout 'move+update (us)'
        $vis = Get-NumberAfter $stdout 'vision cull avg (us)'
        $atk = Get-NumberAfter $stdout 'attack cull avg (us)'
        $rm  = Get-NumberAfter $stdout 'insert+remove (us)'
        $alive  = Get-IntAfter $stdout 'alive\s+(\d+)'
        $leaves = Get-IntAfter $stdout 'binary:\s+(\d+)\s+leaves'
        $arena  = Get-IntAfter $stdout '(\d+)\s+arena nodes'

        if ($null -eq $mv) {
            Write-Host "  WARN parse failed"
            Write-Host $stdout
            continue
        }

        $ts  = (Get-Date).ToString('o')
        $inv = [System.Globalization.CultureInfo]::InvariantCulture
        function F([object]$x) { ([double]$x).ToString('F4', $inv) }
        $row = @(
            $ts, $pop, $il, $seed, $c.feature, $c.cull,
            $alive, $leaves, $arena, ([double]$cellSecs).ToString('F2', $inv),
            (F $mv[0]), (F $mv[1]), (F $mv[2]),
            (F ($(if ($vis) { $vis[0] } else { 0 }))),
            (F ($(if ($atk) { $atk[0] } else { 0 }))),
            (F ($(if ($rm)  { $rm[0]  } else { 0 })))
        ) -join ','
        Add-Content -Path $CsvPath -Value $row
        [void]$done.Add($key)
        Write-Host ("  -> mv={0:N0} vis={1:N1} atk={2:N1} | alive={3} leaves={4} | {5:N1}s" -f `
            $mv[0], $(if ($vis) { $vis[0] } else { 0 }),
            $(if ($atk) { $atk[0] } else { 0 }), $alive, $leaves, $cellSecs)
    }
}

$elapsed = ((Get-Date) - $swStart).TotalMinutes
Write-Host ("`nDone. Total wall: {0:N1} min. Rows in {1}" -f $elapsed, $CsvPath)
