# Sensitivity sweep: how does the optimum item_limit shift with population
# and figure size?
#
# Matrix:
#   pop          ∈ {1000, 3000, 10000}
#   item_limit   ∈ {10, 30, 50, 100, 200, 500}, merge = item_limit
#   figure_scale ∈ {0.5, 1.0, 2.0}  (drop 55 / 110 / 220 px, circle 24 / 48 / 96 px)
#   seeds        ∈ {42, 7, 123}
#   strategy     = lca, neighbors off, full sim with attacks
# Total: 3 x 6 x 3 x 3 = 162 cells.
#
# Goal: build a heuristic "optimum item_limit ≈ f(pop, figure size, world)"
# from the data rather than treating the value as a magic number.
#
# Run from repo root:
#   pwsh -File docs/sweep_sensitivity.ps1
# Resumable: existing rows in the CSV are skipped on rerun.

param(
    [string]$CsvPath = "docs/sweep_sensitivity.csv",
    [string]$BinPath = "target/release/critters_headless.exe"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

if (-not (Test-Path $BinPath)) { throw "Missing binary: $BinPath (build first)" }

$pops         = @(1000, 3000, 10000)
$itemLimits   = @(10, 30, 50, 100, 200, 500)
$figureScales = @(0.5, 1.0, 2.0)
$seeds        = @(42, 7, 123)
$warmup       = 120
$frames       = 120
$respawn      = 0.05

$csvHeader = 'timestamp,pop,item_limit,figure_scale,seed,warmup,frames,alive,leaves,arena_nodes,wall_s,mv_mean_us,mv_p50_us,mv_p95_us,vis_avg_us,atk_avg_us,ins_rm_us'
if (-not (Test-Path $CsvPath)) {
    $csvHeader | Out-File -FilePath $CsvPath -Encoding utf8
}
$done = New-Object 'System.Collections.Generic.HashSet[string]'
Import-Csv $CsvPath | ForEach-Object {
    [void]$done.Add("$($_.pop)|$($_.item_limit)|$($_.figure_scale)|$($_.seed)")
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

$totalCells = $pops.Count * $itemLimits.Count * $figureScales.Count * $seeds.Count
$cellIdx = 0
$swStart = Get-Date

foreach ($pop in $pops) {
    $perKind  = [int]($pop / 3)
    $drifters = $pop - 2 * $perKind
    $hunters  = $perKind
    $pulsars  = $perKind

    foreach ($il in $itemLimits) {
        foreach ($scale in $figureScales) {
            foreach ($seed in $seeds) {
                $cellIdx++
                $key = "$pop|$il|$scale|$seed"
                if ($done.Contains($key)) {
                    Write-Host "[$cellIdx/$totalCells] SKIP (already in CSV): $key"
                    continue
                }
                Write-Host "[$cellIdx/$totalCells] pop=$pop il=$il scale=$scale seed=$seed ..."
                $cellSw = Get-Date
                $stdout = & $BinPath `
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
                    --figure-scale $scale 2>&1 | Out-String
                $cellSecs = ((Get-Date) - $cellSw).TotalSeconds

                $mv  = Get-NumberAfter $stdout 'move+update (us)'
                $vis = Get-NumberAfter $stdout 'vision cull avg (us)'
                $atk = Get-NumberAfter $stdout 'attack cull avg (us)'
                $rm  = Get-NumberAfter $stdout 'insert+remove (us)'
                $alive  = Get-IntAfter $stdout 'alive\s+(\d+)'
                $leaves = Get-IntAfter $stdout 'binary:\s+(\d+)\s+leaves'
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
                    $ts, $pop, $il, $scale, $seed, $warmup, $frames,
                    $alive, $leaves, $arena, ([double]$cellSecs).ToString('F2', $inv),
                    (F $mv[0]), (F $mv[1]), (F $mv[2]),
                    (F ($(if ($vis) { $vis[0] } else { 0 }))),
                    (F ($(if ($atk) { $atk[0] } else { 0 }))),
                    (F ($(if ($rm)  { $rm[0]  } else { 0 })))
                ) -join ','
                Add-Content -Path $CsvPath -Value $row
                [void]$done.Add($key)
                Write-Host ("  -> mv={0:N0} vis={1:N1} atk={2:N1} | alive={3} leaves={4} arena={5} | {6:N1}s" -f `
                    $mv[0], $(if ($vis) { $vis[0] } else { 0 }),
                    $(if ($atk) { $atk[0] } else { 0 }),
                    $alive, $leaves, $arena, $cellSecs)
            }
        }
    }
}

$elapsed = ((Get-Date) - $swStart).TotalMinutes
Write-Host ("`nDone. Total wall: {0:N1} min. Rows in {1}" -f $elapsed, $CsvPath)
