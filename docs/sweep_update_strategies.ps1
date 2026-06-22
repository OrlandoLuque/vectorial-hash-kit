# Sweep: UpdateStrategy comparison under the --no-attack workload.
#
# Iterates strategy x population x item_limit x neighbors-feature x seed,
# parses each run's move+update / vision-avg / arena summary, and appends a
# row to docs/sweep_update_strategies.csv. Resumable: existing rows in the
# CSV are skipped on rerun.
#
# Run from the repo root:
#   pwsh -File docs/sweep_update_strategies.ps1
#
# Optional:
#   -Quick               Skip the 10000-pop tier (~5 min instead of ~2.5 h).
#   -CsvPath <path>      Override CSV destination.
#   -BinDir <path>       Override location of the two binaries.

param(
    [switch]$Quick,
    [string]$CsvPath = "docs/sweep_update_strategies.csv",
    [string]$BinDir  = "target/release"
)

$ErrorActionPreference = 'Stop'
# Force invariant culture so doubles serialize with "." not "," (Spanish locale).
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

# ---------- matrix ----------
$strategies = @('legacy', 'lca', 'lca-ropes')
$pops       = if ($Quick) { @(1000, 3000) } else { @(1000, 3000, 10000) }
$itemLimits = @(3, 10, 30)
$features   = @('on', 'off')
$seeds      = @(42, 7, 123)
$warmup     = 120
$frames     = 240

$binNbrs   = Join-Path $BinDir 'critters_headless_nbrs.exe'
$binNoNbrs = Join-Path $BinDir 'critters_headless_nonbrs.exe'
if (-not (Test-Path $binNbrs))   { throw "Missing binary: $binNbrs (build first)" }
if (-not (Test-Path $binNoNbrs)) { throw "Missing binary: $binNoNbrs (build first)" }

# ---------- CSV bootstrap ----------
$csvHeader = 'timestamp,pop,item_limit,merge_limit,strategy,neighbors,seed,warmup,frames,alive,leaves,arena_nodes,wall_s,mv_mean_us,mv_p50_us,mv_p95_us,vis_avg_us,atk_avg_us,ins_rm_us'
if (-not (Test-Path $CsvPath)) {
    $csvHeader | Out-File -FilePath $CsvPath -Encoding utf8
}

# Build a set of (pop,item,strategy,feature,seed) keys already in CSV.
$done = New-Object 'System.Collections.Generic.HashSet[string]'
Import-Csv $CsvPath | ForEach-Object {
    [void]$done.Add("$($_.pop)|$($_.item_limit)|$($_.strategy)|$($_.neighbors)|$($_.seed)")
}

# ---------- helpers ----------
function Get-NumberAfter([string]$lines, [string]$label) {
    # Parse a line like "binary   move+update (us)             89.3       83.8      128.5"
    # and return @(89.3, 83.8, 128.5). Returns $null if not found.
    $escLabel = [regex]::Escape($label)
    $rx = "binary\s+$escLabel.*?(\d[\d\.]*)\s+(\d[\d\.]*)\s+(\d[\d\.]*)\s*$"
    foreach ($line in ($lines -split "`r?`n")) {
        if ($line -match $rx) {
            return @([double]$Matches[1], [double]$Matches[2], [double]$Matches[3])
        }
    }
    return $null
}

function Get-FirstNumberAfter([string]$lines, [string]$pattern) {
    foreach ($line in ($lines -split "`r?`n")) {
        if ($line -match $pattern) { return [double]$Matches[1] }
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
$totalCells = 0
foreach ($pop in $pops) {
    foreach ($il in $itemLimits) {
        foreach ($s in $strategies) {
            foreach ($feat in $features) {
                if ($s -eq 'lca-ropes' -and $feat -eq 'off') { continue }
                foreach ($seed in $seeds) { $totalCells++ }
            }
        }
    }
}
Write-Host "Sweep plan: $totalCells cells. CSV: $CsvPath"

$cellIdx = 0
$swStart = Get-Date
foreach ($pop in $pops) {
    $perKind = [int]($pop / 3)
    $drifters = $pop - 2 * $perKind   # make the totals exact
    $hunters  = $perKind
    $pulsars  = $perKind
    foreach ($il in $itemLimits) {
        foreach ($s in $strategies) {
            foreach ($feat in $features) {
                if ($s -eq 'lca-ropes' -and $feat -eq 'off') { continue }
                $bin = if ($feat -eq 'on') { $binNbrs } else { $binNoNbrs }
                foreach ($seed in $seeds) {
                    $cellIdx++
                    $key = "$pop|$il|$s|$feat|$seed"
                    if ($done.Contains($key)) {
                        Write-Host "[$cellIdx/$totalCells] SKIP (already in CSV): $key"
                        continue
                    }
                    Write-Host "[$cellIdx/$totalCells] pop=$pop il=$il strat=$s nbrs=$feat seed=$seed ..."

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
                        --update-strategy $s `
                        --no-attack 2>&1 | Out-String
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

                    $ts = (Get-Date).ToString('o')
                    $inv = [System.Globalization.CultureInfo]::InvariantCulture
                    function F([object]$x) { ([double]$x).ToString('F4', $inv) }
                    $row = @(
                        $ts, $pop, $il, $il, $s, $feat, $seed, $warmup, $frames,
                        $alive, $leaves, $arena, ([double]$cellSecs).ToString('F2', $inv),
                        (F $mv[0]), (F $mv[1]), (F $mv[2]),
                        (F ($(if ($vis) { $vis[0] } else { 0 }))),
                        (F ($(if ($atk) { $atk[0] } else { 0 }))),
                        (F ($(if ($rm)  { $rm[0]  } else { 0 })))
                    ) -join ','
                    Add-Content -Path $CsvPath -Value $row
                    [void]$done.Add($key)
                    Write-Host ("  -> mv mean={0:N1} p50={1:N1} p95={2:N1} | alive={3} leaves={4} arena={5} | {6:N1}s" -f `
                        $mv[0], $mv[1], $mv[2], $alive, $leaves, $arena, $cellSecs)
                }
            }
        }
    }
}

$elapsed = ((Get-Date) - $swStart).TotalMinutes
Write-Host ("`nDone. Total wall: {0:N1} min. Rows in {1}" -f $elapsed, $CsvPath)
