# Full-sim head-to-head: Tree (float) vs IntegerTree, at the same workload
# settings, sweeping item_limit. Pop=10000, strategy=lca, neighbors=off,
# 3 seeds, merge_limit = item_limit (chosen default).
#
# Runs each cell twice (once per mode) and writes a single row capturing
# both for direct comparison.
#
# Run from the repo root:
#   pwsh -File docs/sweep_tree_vs_itree.ps1

param(
    [string]$CsvPath = "docs/sweep_tree_vs_itree.csv",
    [string]$BinPath = "target/release/critters_headless.exe"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path $BinPath)) { throw "Missing binary: $BinPath (build first)" }

$itemLimits = @(30, 50, 100, 200, 500)
$seeds      = @(42, 7, 123)
$pop      = 10000
$warmup   = 120
$frames   = 120
$respawn  = 0.05

# CSV bootstrap
$csvHeader = 'timestamp,pop,item_limit,seed,mode,alive,leaves,arena_nodes,wall_s,mv_mean_us,mv_p50_us,mv_p95_us,vis_avg_us,atk_avg_us,ins_rm_us'
if (-not (Test-Path $CsvPath)) {
    $csvHeader | Out-File -FilePath $CsvPath -Encoding utf8
}
$done = New-Object 'System.Collections.Generic.HashSet[string]'
Import-Csv $CsvPath | ForEach-Object {
    [void]$done.Add("$($_.pop)|$($_.item_limit)|$($_.mode)|$($_.seed)")
}

function Get-NumberAfter([string]$lines, [string]$label, [string]$structKey) {
    $escLabel = [regex]::Escape($label)
    $rx = "$structKey\s+$escLabel.*?(\d[\d\.]*)\s+(\d[\d\.]*)\s+(\d[\d\.]*)\s*$"
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

$totalCells = $itemLimits.Count * $seeds.Count * 2
$cellIdx = 0
$swStart = Get-Date
$perKind  = [int]($pop / 3)
$drifters = $pop - 2 * $perKind
$hunters  = $perKind
$pulsars  = $perKind

foreach ($il in $itemLimits) {
    foreach ($seed in $seeds) {
        foreach ($mode in @('binary', 'int_binary')) {
            $cellIdx++
            $modeKey = if ($mode -eq 'binary') { 'tree' } else { 'itree' }
            $key = "$pop|$il|$modeKey|$seed"
            if ($done.Contains($key)) {
                Write-Host "[$cellIdx/$totalCells] SKIP (already in CSV): $key"
                continue
            }
            Write-Host "[$cellIdx/$totalCells] mode=$mode il=$il seed=$seed ..."
            $cellSw = Get-Date
            $stdout = & $BinPath `
                --mode $mode `
                --frames $frames `
                --warmup $warmup `
                --drifters $drifters `
                --hunters $hunters `
                --pulsars $pulsars `
                --split $il `
                --merge $il `
                --seed $seed `
                --update-strategy lca `
                --respawn $respawn 2>&1 | Out-String
            $cellSecs = ((Get-Date) - $cellSw).TotalSeconds

            $structKey = if ($mode -eq 'binary') { 'binary' } else { 'itree' }
            $mv  = Get-NumberAfter $stdout 'move+update (us)' $structKey
            $vis = Get-NumberAfter $stdout 'vision cull avg (us)' $structKey
            $atk = Get-NumberAfter $stdout 'attack cull avg (us)' $structKey
            $rm  = Get-NumberAfter $stdout 'insert+remove (us)' $structKey
            $alive  = Get-IntAfter $stdout 'alive\s+(\d+)'
            $leavesPattern = if ($mode -eq 'binary') { 'binary:\s+(\d+)\s+leaves' } else { 'itree:\s+(\d+)\s+leaves' }
            $leaves = Get-IntAfter $stdout $leavesPattern
            $arena  = Get-IntAfter $stdout '(\d+)\s+arena nodes'

            if ($null -eq $mv) {
                Write-Host "  WARN: failed to parse from:"
                Write-Host $stdout
                continue
            }

            $ts  = (Get-Date).ToString('o')
            $inv = [System.Globalization.CultureInfo]::InvariantCulture
            function F([object]$x) { ([double]$x).ToString('F4', $inv) }
            $row = @(
                $ts, $pop, $il, $seed, $modeKey,
                $alive, $leaves, $arena, ([double]$cellSecs).ToString('F2', $inv),
                (F $mv[0]), (F $mv[1]), (F $mv[2]),
                (F ($(if ($vis) { $vis[0] } else { 0 }))),
                (F ($(if ($atk) { $atk[0] } else { 0 }))),
                (F ($(if ($rm)  { $rm[0]  } else { 0 })))
            ) -join ','
            Add-Content -Path $CsvPath -Value $row
            [void]$done.Add($key)
            Write-Host ("  -> mv={0:N0} vis={1:N1} | alive={2} leaves={3} | {4:N1}s" -f `
                $mv[0], $(if ($vis) { $vis[0] } else { 0 }), $alive, $leaves, $cellSecs)
        }
    }
}
$elapsed = ((Get-Date) - $swStart).TotalMinutes
Write-Host ("`nDone. Total wall: {0:N1} min. Rows in {1}" -f $elapsed, $CsvPath)
