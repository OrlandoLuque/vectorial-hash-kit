# Aggregate docs/sweep_update_strategies.csv: average across seeds, group by
# (pop, item_limit, strategy, neighbors). Emit a markdown-friendly table.

param(
    [string]$CsvPath = "docs/sweep_update_strategies.csv"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

$rows = Import-Csv $CsvPath
$groups = $rows | Group-Object { "$($_.pop)|$($_.item_limit)|$($_.strategy)|$($_.neighbors)" }

$agg = foreach ($g in $groups) {
    $parts = $g.Name -split '\|'
    [PSCustomObject]@{
        pop        = [int]$parts[0]
        item_limit = [int]$parts[1]
        strategy   = $parts[2]
        nbrs       = $parts[3]
        n_seeds    = $g.Count
        mv_mean    = [math]::Round((($g.Group | Measure-Object -Property mv_mean_us -Average).Average), 1)
        mv_p50     = [math]::Round((($g.Group | Measure-Object -Property mv_p50_us  -Average).Average), 1)
        mv_p95     = [math]::Round((($g.Group | Measure-Object -Property mv_p95_us  -Average).Average), 1)
        arena_avg  = [int](($g.Group | Measure-Object -Property arena_nodes -Average).Average)
        leaves_avg = [int](($g.Group | Measure-Object -Property leaves -Average).Average)
    }
}

$agg = $agg | Sort-Object pop, item_limit, strategy, nbrs

# --- Print headline table -------------------------------------------------
Write-Host "`n## Aggregated means (across 3 seeds)`n"
Write-Host "| pop  | il | strategy | nbrs | mv_mean | mv_p50 | mv_p95 | arena   | leaves |"
Write-Host "|------|----|----------|------|--------:|-------:|-------:|--------:|-------:|"
foreach ($r in $agg) {
    Write-Host ("| {0,5} | {1,2} | {2,-9} | {3,-3} | {4,7:F1} | {5,6:F1} | {6,6:F1} | {7,7} | {8,6} |" -f `
        $r.pop, $r.item_limit, $r.strategy, $r.nbrs, $r.mv_mean, $r.mv_p50, $r.mv_p95, $r.arena_avg, $r.leaves_avg)
}

# --- Cross-cuts: cost of neighbors feature  -------------------------------
Write-Host "`n## Cost of the `neighbors` feature (mv_mean: on vs off, same pop/il/strategy)`n"
Write-Host "| pop  | il | strategy | off    | on     | diff (%) |"
Write-Host "|------|----|----------|-------:|-------:|---------:|"
$byOnOff = $agg | Group-Object { "$($_.pop)|$($_.item_limit)|$($_.strategy)" }
foreach ($g in $byOnOff) {
    $off = $g.Group | Where-Object nbrs -eq 'off' | Select-Object -First 1
    $on  = $g.Group | Where-Object nbrs -eq 'on'  | Select-Object -First 1
    if ($null -eq $off -or $null -eq $on) { continue }
    $diffPct = (($on.mv_mean - $off.mv_mean) / $off.mv_mean) * 100
    $parts = $g.Name -split '\|'
    Write-Host ("| {0,5} | {1,2} | {2,-9} | {3,6:F1} | {4,6:F1} | {5,8:F1} |" -f `
        $parts[0], $parts[1], $parts[2], $off.mv_mean, $on.mv_mean, $diffPct)
}

# --- Cross-cuts: legacy vs lca (no neighbors, clean A/B) -------------------
Write-Host "`n## Legacy vs LCA (nbrs=off, isolates strategy)`n"
Write-Host "| pop  | il | legacy | lca    | lca-vs-legacy (%) |"
Write-Host "|------|----|-------:|-------:|------------------:|"
$byPopIl = $agg | Where-Object nbrs -eq 'off' | Group-Object { "$($_.pop)|$($_.item_limit)" }
foreach ($g in $byPopIl) {
    $leg = $g.Group | Where-Object strategy -eq 'legacy' | Select-Object -First 1
    $lca = $g.Group | Where-Object strategy -eq 'lca'    | Select-Object -First 1
    if ($null -eq $leg -or $null -eq $lca) { continue }
    $diffPct = (($lca.mv_mean - $leg.mv_mean) / $leg.mv_mean) * 100
    $parts = $g.Name -split '\|'
    Write-Host ("| {0,5} | {1,2} | {2,6:F1} | {3,6:F1} | {4,17:F1} |" -f `
        $parts[0], $parts[1], $leg.mv_mean, $lca.mv_mean, $diffPct)
}

# --- Cross-cuts: lca vs lca-ropes (nbrs=on, clean A/B same tree) -----------
Write-Host "`n## LCA vs LCA-ropes (nbrs=on, same tree state)`n"
Write-Host "| pop  | il | lca    | lca-ropes | ropes-vs-lca (%) |"
Write-Host "|------|----|-------:|----------:|-----------------:|"
$byPopIl = $agg | Where-Object nbrs -eq 'on' | Group-Object { "$($_.pop)|$($_.item_limit)" }
foreach ($g in $byPopIl) {
    $lca = $g.Group | Where-Object strategy -eq 'lca'       | Select-Object -First 1
    $rop = $g.Group | Where-Object strategy -eq 'lca-ropes' | Select-Object -First 1
    if ($null -eq $lca -or $null -eq $rop) { continue }
    $diffPct = (($rop.mv_mean - $lca.mv_mean) / $lca.mv_mean) * 100
    $parts = $g.Name -split '\|'
    Write-Host ("| {0,5} | {1,2} | {2,6:F1} | {3,9:F1} | {4,16:F1} |" -f `
        $parts[0], $parts[1], $lca.mv_mean, $rop.mv_mean, $diffPct)
}

Write-Host ""
