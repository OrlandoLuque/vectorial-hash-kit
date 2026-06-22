# Aggregate docs/sweep_tree_vs_itree.csv: average across seeds, group by
# (item_limit, mode), emit a side-by-side comparison.

param(
    [string]$CsvPath = "docs/sweep_tree_vs_itree.csv"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

$rows = Import-Csv $CsvPath
$groups = $rows | Group-Object { "$($_.item_limit)|$($_.mode)" }

$agg = foreach ($g in $groups) {
    $parts = $g.Name -split '\|'
    [PSCustomObject]@{
        item_limit = [int]$parts[0]
        mode       = $parts[1]
        n          = $g.Count
        mv         = [math]::Round((($g.Group | Measure-Object -Property mv_mean_us -Average).Average), 1)
        vis        = [math]::Round((($g.Group | Measure-Object -Property vis_avg_us -Average).Average), 1)
        atk        = [math]::Round((($g.Group | Measure-Object -Property atk_avg_us -Average).Average), 1)
        rm         = [math]::Round((($g.Group | Measure-Object -Property ins_rm_us  -Average).Average), 1)
        leaves     = [int](($g.Group | Measure-Object -Property leaves -Average).Average)
        arena      = [int](($g.Group | Measure-Object -Property arena_nodes -Average).Average)
    }
}

# Use vis_n=3000 (measured earlier) for total per-frame estimate.
$visN = 3000
$atkN = 16

Write-Host "`n## Tree vs IntegerTree at pop=10000, lca, merge=item_limit (avg of 3 seeds)`n"
Write-Host "| item_limit | mode  | mv     | vis_avg | atk_avg | ins+rm | total est ms/frame | leaves | arena |"
Write-Host "|------------|-------|-------:|--------:|--------:|-------:|-------------------:|-------:|------:|"
$byIl = $agg | Group-Object item_limit | Sort-Object { [int]$_.Name }
foreach ($g in $byIl) {
    $il = [int]$g.Name
    $tree = $g.Group | Where-Object mode -eq 'tree'  | Select-Object -First 1
    $it   = $g.Group | Where-Object mode -eq 'itree' | Select-Object -First 1
    foreach ($r in @($tree, $it)) {
        if ($null -eq $r) { continue }
        $totalUs = $r.mv + $r.vis * $visN + $r.atk * $atkN + $r.rm
        $totalMs = $totalUs / 1000.0
        Write-Host ("| {0,10} | {1,-5} | {2,6:F1} | {3,7:F1} | {4,7:F1} | {5,6:F1} | {6,18:F2} | {7,6} | {8,5} |" -f `
            $il, $r.mode, $r.mv, $r.vis, $r.atk, $r.rm, $totalMs, $r.leaves, $r.arena)
    }
}

Write-Host "`n## Direct delta per item_limit (itree vs tree)`n"
Write-Host "| item_limit | mv tree | mv itree | mv delta | vis tree | vis itree | vis delta |"
Write-Host "|------------|--------:|---------:|---------:|---------:|----------:|----------:|"
foreach ($g in $byIl) {
    $il = [int]$g.Name
    $tree = $g.Group | Where-Object mode -eq 'tree'  | Select-Object -First 1
    $it   = $g.Group | Where-Object mode -eq 'itree' | Select-Object -First 1
    if ($null -eq $tree -or $null -eq $it) { continue }
    $mvDelta  = (($it.mv - $tree.mv) / $tree.mv) * 100
    $visDelta = (($it.vis - $tree.vis) / $tree.vis) * 100
    Write-Host ("| {0,10} | {1,7:F1} | {2,8:F1} | {3,7:F1}% | {4,8:F1} | {5,9:F1} | {6,8:F1}% |" -f `
        $il, $tree.mv, $it.mv, $mvDelta, $tree.vis, $it.vis, $visDelta)
}

Write-Host ""
