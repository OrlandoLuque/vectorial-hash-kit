# Fine sweep: item_limit x merge_limit at pop=10000 under the FULL sim
# workload (attacks on, so cull/insert/remove are also measured), strategy
# fixed to `lca`, neighbors feature off (production default per
# UPDATE_STRATEGIES.md).
#
# Answers: at what item_limit does the cull cost start to outweigh the
# update gains? Does merge_limit < item_limit (hysteresis) matter at
# sustained pop?
#
# Run from the repo root:
#   pwsh -File docs/sweep_item_merge.ps1
#
# Resumable: existing rows in the CSV are skipped on rerun.

param(
    [string]$CsvPath = "docs/sweep_item_merge.csv",
    [string]$BinPath = "target/release/critters_headless_nonbrs.exe"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path $BinPath)) { throw "Missing binary: $BinPath (build it first)" }

# ---------- matrix ----------
# For each item_limit, three merge_limit values: 1, ~half, and =item_limit.
# Constraints: merge_limit <= item_limit, merge_limit >= 1.
$cells = @()
$itemLimits = @(3, 6, 10, 15, 20, 30, 50)
foreach ($il in $itemLimits) {
    $merges = @(1, [Math]::Max(1, [int]($il / 2)), $il) | Sort-Object -Unique
    foreach ($m in $merges) {
        foreach ($seed in @(42, 7, 123)) {
            $cells += [PSCustomObject]@{ il = $il; merge = $m; seed = $seed }
        }
    }
}
$totalCells = $cells.Count
$pop      = 10000
$warmup   = 120
$frames   = 120
$respawn  = 0.05

# ---------- CSV bootstrap ----------
$csvHeader = 'timestamp,pop,item_limit,merge_limit,strategy,neighbors,seed,warmup,frames,alive,leaves,arena_nodes,wall_s,mv_mean_us,mv_p50_us,mv_p95_us,vis_avg_us,atk_avg_us,ins_rm_us'
if (-not (Test-Path $CsvPath)) {
    $csvHeader | Out-File -FilePath $CsvPath -Encoding utf8
}
$done = New-Object 'System.Collections.Generic.HashSet[string]'
Import-Csv $CsvPath | ForEach-Object {
    [void]$done.Add("$($_.pop)|$($_.item_limit)|$($_.merge_limit)|$($_.seed)")
}

# ---------- helpers ----------
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

# ---------- sweep ----------
Write-Host "Sweep plan: $totalCells cells (pop=$pop, full sim, lca, no neighbors). CSV: $CsvPath"
$cellIdx = 0
$swStart = Get-Date

$perKind  = [int]($pop / 3)
$drifters = $pop - 2 * $perKind
$hunters  = $perKind
$pulsars  = $perKind

foreach ($cell in $cells) {
    $cellIdx++
    $key = "$pop|$($cell.il)|$($cell.merge)|$($cell.seed)"
    if ($done.Contains($key)) {
        Write-Host "[$cellIdx/$totalCells] SKIP (already in CSV): $key"
        continue
    }
    Write-Host "[$cellIdx/$totalCells] pop=$pop il=$($cell.il) merge=$($cell.merge) seed=$($cell.seed) ..."
    $cellSw = Get-Date
    $stdout = & $BinPath `
        --mode binary `
        --frames $frames `
        --warmup $warmup `
        --drifters $drifters `
        --hunters $hunters `
        --pulsars $pulsars `
        --split $cell.il `
        --merge $cell.merge `
        --seed $cell.seed `
        --update-strategy lca `
        --respawn $respawn 2>&1 | Out-String
    $cellSecs = ((Get-Date) - $cellSw).TotalSeconds

    $mv  = Get-NumberAfter $stdout 'move+update (us)'
    $vis = Get-NumberAfter $stdout 'vision cull avg (us)'
    $atk = Get-NumberAfter $stdout 'attack cull avg (us)'
    $rm  = Get-NumberAfter $stdout 'insert+remove (us)'
    $alive  = Get-IntAfter $stdout 'alive\s+(\d+)'
    $leaves = Get-IntAfter $stdout 'binary:\s+(\d+)\s+leaves'
    $arena  = Get-IntAfter $stdout '(\d+)\s+arena nodes'

    if ($null -eq $mv) {
        Write-Host "  WARN: failed to parse move+update from:"
        Write-Host $stdout
        continue
    }

    $ts  = (Get-Date).ToString('o')
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    function F([object]$x) { ([double]$x).ToString('F4', $inv) }
    $row = @(
        $ts, $pop, $cell.il, $cell.merge, 'lca', 'off', $cell.seed, $warmup, $frames,
        $alive, $leaves, $arena, ([double]$cellSecs).ToString('F2', $inv),
        (F $mv[0]), (F $mv[1]), (F $mv[2]),
        (F ($(if ($vis) { $vis[0] } else { 0 }))),
        (F ($(if ($atk) { $atk[0] } else { 0 }))),
        (F ($(if ($rm)  { $rm[0]  } else { 0 })))
    ) -join ','
    Add-Content -Path $CsvPath -Value $row
    [void]$done.Add($key)
    Write-Host ("  -> mv={0:N0} vis={1:N1} atk={2:N1} rm={3:N1} | alive={4} leaves={5} | {6:N1}s" -f `
        $mv[0], $(if ($vis) { $vis[0] } else { 0 }),
        $(if ($atk) { $atk[0] } else { 0 }),
        $(if ($rm)  { $rm[0]  } else { 0 }),
        $alive, $leaves, $cellSecs)
}

$elapsed = ((Get-Date) - $swStart).TotalMinutes
Write-Host ("`nDone. Total wall: {0:N1} min. Rows in {1}" -f $elapsed, $CsvPath)
