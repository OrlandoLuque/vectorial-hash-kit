# Aggregate docs/sweep_item_merge.csv. Group by (item_limit, merge_limit),
# average across seeds, and emit:
# - Per-cell totals (mv mean, vis_avg, atk_avg, ins_rm)
# - Best-case totals per item_limit (varying merge_limit)
# - Estimated per-frame total cost (mv + vis_avg*hunters_per_frame + ...)
#
# For the per-frame total we approximate the calls/frame from the workload:
# at pop=10000 with targets ~3333 each kind, each kind fires once every
# ~2.5s sim time (cooldown range avg). With dt=1/60 → ~67 vision culls per
# hunter frame ON AVERAGE, attacks proportional. We ESTIMATE the total
# directly from the per-call averages; the raw numbers below are what the
# bench observed under the actual workload.

param(
    [string]$CsvPath = "docs/sweep_item_merge.csv"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

$rows = Import-Csv $CsvPath
$groups = $rows | Group-Object { "$($_.item_limit)|$($_.merge_limit)" }

$agg = foreach ($g in $groups) {
    $parts = $g.Name -split '\|'
    [PSCustomObject]@{
        item_limit = [int]$parts[0]
        merge_limit = [int]$parts[1]
        n_seeds    = $g.Count
        mv         = [math]::Round((($g.Group | Measure-Object -Property mv_mean_us -Average).Average), 1)
        vis_avg    = [math]::Round((($g.Group | Measure-Object -Property vis_avg_us -Average).Average), 1)
        atk_avg    = [math]::Round((($g.Group | Measure-Object -Property atk_avg_us -Average).Average), 1)
        ins_rm     = [math]::Round((($g.Group | Measure-Object -Property ins_rm_us  -Average).Average), 1)
        alive_avg  = [int](($g.Group | Measure-Object -Property alive -Average).Average)
        leaves_avg = [int](($g.Group | Measure-Object -Property leaves -Average).Average)
        arena_avg  = [int](($g.Group | Measure-Object -Property arena_nodes -Average).Average)
    }
}
$agg = $agg | Sort-Object item_limit, merge_limit

Write-Host "`n## All cells: per-frame mv (us), per-call cull avgs (us), per-frame insert+remove (us)`n"
Write-Host "| il | merge | mv     | vis_avg | atk_avg | ins+rm | alive | leaves | arena   |"
Write-Host "|----|-------|-------:|--------:|--------:|-------:|------:|-------:|--------:|"
foreach ($r in $agg) {
    Write-Host ("| {0,2} | {1,5} | {2,6:F1} | {3,7:F1} | {4,7:F1} | {5,6:F1} | {6,5} | {7,6} | {8,7} |" -f `
        $r.item_limit, $r.merge_limit, $r.mv, $r.vis_avg, $r.atk_avg, $r.ins_rm, $r.alive_avg, $r.leaves_avg, $r.arena_avg)
}

# --- Per item_limit, show best merge_limit (the one minimising mv) -------
Write-Host "`n## Best merge_limit per item_limit (by mv mean)`n"
Write-Host "| il | best_merge | mv     | vs merge=il (%) |"
Write-Host "|----|-----------:|-------:|----------------:|"
$byIl = $agg | Group-Object item_limit
foreach ($g in $byIl) {
    $best = $g.Group | Sort-Object mv | Select-Object -First 1
    $tied = $g.Group | Where-Object merge_limit -eq $best.item_limit | Select-Object -First 1
    $pct = if ($tied) { (($best.mv - $tied.mv) / $tied.mv) * 100 } else { 0 }
    Write-Host ("| {0,2} | {1,10} | {2,6:F1} | {3,15:F1} |" -f $best.item_limit, $best.merge_limit, $best.mv, $pct)
}

# --- Trend across item_limit (using merge = item_limit, the diagonal) -----
Write-Host "`n## Trend across item_limit (merge = item_limit)`n"
Write-Host "| il | mv    | vis_avg | atk_avg | ins+rm | leaves |"
Write-Host "|----|------:|--------:|--------:|-------:|-------:|"
foreach ($g in $byIl) {
    $diag = $g.Group | Where-Object merge_limit -eq $g.Name -as [int] | Select-Object -First 1
    if ($null -eq $diag) {
        $diag = $g.Group | Sort-Object merge_limit -Descending | Select-Object -First 1
    }
    Write-Host ("| {0,2} | {1,5:F1} | {2,7:F1} | {3,7:F1} | {4,6:F1} | {5,6} |" -f `
        $diag.item_limit, $diag.mv, $diag.vis_avg, $diag.atk_avg, $diag.ins_rm, $diag.leaves_avg)
}

Write-Host ""
