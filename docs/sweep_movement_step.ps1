# Movement-step sensitivity: bigger dt = critters move further per frame =
# more cross-leaf jumps per update, stressing the LCA relocation path.
# Uses --no-attack so the workload is pure movement (no kill/respawn churn),
# isolating the update cost. pop=10000, il=100, lca, neighbors off, 3 seeds.
#
# Note: dt also scales sim-time progression, but with --no-attack there is
# no firing/cooldown/respawn, so the only effect is movement magnitude.

param(
    [string]$CsvPath = "docs/sweep_movement_step.csv",
    [string]$BinPath = "target/release/critters_headless_nonbrs.exe"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture
if (-not (Test-Path $BinPath)) { throw "Missing: $BinPath" }

$pop          = 10000
$il           = 100
$dts          = @(0.0083, 0.0167, 0.0333, 0.0667)  # ~120/60/30/15 fps step
$seeds        = @(42, 7, 123)
$warmup       = 120
$frames       = 120
$perKind  = [int]($pop / 3)
$drifters = $pop - 2 * $perKind
$hunters  = $perKind
$pulsars  = $perKind

$csvHeader = 'timestamp,pop,item_limit,dt,seed,warmup,frames,alive,leaves,arena_nodes,wall_s,mv_mean_us,mv_p50_us,mv_p95_us,vis_avg_us,atk_avg_us,ins_rm_us'
if (-not (Test-Path $CsvPath)) { $csvHeader | Out-File -FilePath $CsvPath -Encoding utf8 }
$done = New-Object 'System.Collections.Generic.HashSet[string]'
Import-Csv $CsvPath | ForEach-Object { [void]$done.Add("$($_.pop)|$($_.dt)|$($_.seed)") }

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

$totalCells = $dts.Count * $seeds.Count
$cellIdx = 0
$swStart = Get-Date

foreach ($dt in $dts) {
    foreach ($seed in $seeds) {
        $cellIdx++
        $key = "$pop|$dt|$seed"
        if ($done.Contains($key)) { Write-Host "[$cellIdx/$totalCells] SKIP: $key"; continue }
        Write-Host "[$cellIdx/$totalCells] dt=$dt seed=$seed ..."
        $cellSw = Get-Date
        $stdout = & $BinPath --mode binary --frames $frames --warmup $warmup `
            --drifters $drifters --hunters $hunters --pulsars $pulsars `
            --split $il --merge $il --seed $seed --update-strategy lca `
            --dt $dt --no-attack 2>&1 | Out-String
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
            $ts, $pop, $il, $dt, $seed, $warmup, $frames,
            $alive, $leaves, $arena, ([double]$cellSecs).ToString('F2', $inv),
            (F $mv[0]), (F $mv[1]), (F $mv[2]),
            (F ($(if ($vis) { $vis[0] } else { 0 }))),
            (F ($(if ($atk) { $atk[0] } else { 0 }))),
            (F ($(if ($rm)  { $rm[0]  } else { 0 })))
        ) -join ','
        Add-Content -Path $CsvPath -Value $row
        [void]$done.Add($key)
        Write-Host ("  -> mv={0:N0} p95={1:N0} | leaves={2} | {3:N1}s" -f $mv[0], $mv[2], $leaves, $cellSecs)
    }
}
$elapsed = ((Get-Date) - $swStart).TotalMinutes
Write-Host ("`nDone. Total wall: {0:N1} min. Rows in {1}" -f $elapsed, $CsvPath)
