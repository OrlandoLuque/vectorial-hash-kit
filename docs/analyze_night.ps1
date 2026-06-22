# Aggregate all five overnight sweeps. vis_n ~= 0.30*pop, atk_n ~= 0.005*pop
# (measured earlier) for the total-per-frame estimate.

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Mean($rows, $col) { [math]::Round((($rows | Measure-Object -Property $col -Average).Average), 1) }
function MeanI($rows, $col) { [int](($rows | Measure-Object -Property $col -Average).Average) }
function Total($mv, $vis, $atk, $rm, $pop) { [math]::Round($mv + $vis*($pop*0.30) + $atk*($pop*0.005) + $rm, 0) }

Write-Host "`n################ 1. cull_walk break-even (pop=10k, il=100) ################`n"
$cw = Import-Csv "docs/sweep_cull_walk.csv"
Write-Host "| feature | cull       | mv    | vis_avg | total est µs/frame |"
Write-Host "|---------|------------|------:|--------:|-------------------:|"
foreach ($g in ($cw | Group-Object { "$($_.feature)|$($_.cull)" })) {
    $p = $g.Name -split '\|'
    $mv = Mean $g.Group mv_mean_us; $vis = Mean $g.Group vis_avg_us; $atk = Mean $g.Group atk_avg_us; $rm = Mean $g.Group ins_rm_us
    $tot = Total $mv $vis $atk $rm 10000
    Write-Host ("| {0,-7} | {1,-10} | {2,5:F0} | {3,7:F1} | {4,18:F0} |" -f $p[0], $p[1], $mv, $vis, $tot)
}

Write-Host "`n################ 2. strategy at il=100 (pop=10k) ################`n"
$st = Import-Csv "docs/sweep_strategy_il100.csv"
Write-Host "| il  | strategy   | feature | mv    | mv p95 | arena | total est |"
Write-Host "|-----|------------|---------|------:|-------:|------:|----------:|"
foreach ($g in ($st | Group-Object { "$($_.item_limit)|$($_.strategy)|$($_.feature)" } | Sort-Object { [int](($_.Name -split '\|')[0]) })) {
    $p = $g.Name -split '\|'
    $mv = Mean $g.Group mv_mean_us; $p95 = Mean $g.Group mv_p95_us; $vis = Mean $g.Group vis_avg_us; $atk = Mean $g.Group atk_avg_us; $rm = Mean $g.Group ins_rm_us
    $arena = MeanI $g.Group arena_nodes
    $tot = Total $mv $vis $atk $rm 10000
    Write-Host ("| {0,3} | {1,-10} | {2,-7} | {3,5:F0} | {4,6:F0} | {5,5} | {6,9:F0} |" -f $p[0], $p[1], $p[2], $mv, $p95, $arena, $tot)
}

Write-Host "`n################ 3. quadtree vs binary (pop=10k) ################`n"
$qb = Import-Csv "docs/sweep_quad_vs_binary.csv"
Write-Host "| il  | mode   | mv    | vis_avg | atk_avg | total est µs/frame | leaves |"
Write-Host "|-----|--------|------:|--------:|--------:|-------------------:|-------:|"
foreach ($g in ($qb | Group-Object { "$($_.item_limit)|$($_.mode)" } | Sort-Object { [int](($_.Name -split '\|')[0]) })) {
    $p = $g.Name -split '\|'
    $mv = Mean $g.Group mv_mean_us; $vis = Mean $g.Group vis_avg_us; $atk = Mean $g.Group atk_avg_us; $rm = Mean $g.Group ins_rm_us
    $lv = MeanI $g.Group leaves
    $tot = Total $mv $vis $atk $rm 10000
    Write-Host ("| {0,3} | {1,-6} | {2,5:F0} | {3,7:F1} | {4,7:F1} | {5,18:F0} | {6,6} |" -f $p[0], $p[1], $mv, $vis, $atk, $tot, $lv)
}

Write-Host "`n################ 4. movement-step (dt) sensitivity (pop=10k, il=100, --no-attack) ################`n"
$ms = Import-Csv "docs/sweep_movement_step.csv"
Write-Host "| dt      | mv mean | mv p50 | mv p95 |"
Write-Host "|---------|--------:|-------:|-------:|"
foreach ($g in ($ms | Group-Object dt | Sort-Object { [double]$_.Name })) {
    $mv = Mean $g.Group mv_mean_us; $p50 = Mean $g.Group mv_p50_us; $p95 = Mean $g.Group mv_p95_us
    Write-Host ("| {0,-7} | {1,7:F0} | {2,6:F0} | {3,6:F0} |" -f $g.Name, $mv, $p50, $p95)
}

Write-Host "`n################ 5. 30k variance + optimum (pop=30k) ################`n"
$k30 = Import-Csv "docs/sweep_30k.csv"
Write-Host "| il  | mv mean | mv stdev | vis_avg | total est ms/frame | leaves | arena |"
Write-Host "|-----|--------:|---------:|--------:|-------------------:|-------:|------:|"
foreach ($g in ($k30 | Group-Object item_limit | Sort-Object { [int]$_.Name })) {
    $mv = Mean $g.Group mv_mean_us; $vis = Mean $g.Group vis_avg_us; $atk = Mean $g.Group atk_avg_us; $rm = Mean $g.Group ins_rm_us
    $vals = $g.Group | ForEach-Object { [double]$_.mv_mean_us }
    $m = ($vals | Measure-Object -Average).Average
    $sd = [math]::Sqrt((($vals | ForEach-Object { ($_ - $m) * ($_ - $m) }) | Measure-Object -Sum).Sum / $vals.Count)
    $lv = MeanI $g.Group leaves; $ar = MeanI $g.Group arena_nodes
    $tot = Total $mv $vis $atk $rm 30000
    Write-Host ("| {0,3} | {1,7:F0} | {2,8:F1} | {3,7:F1} | {4,18:F2} | {5,6} | {6,5} |" -f $g.Name, $mv, $sd, $vis, ($tot/1000), $lv, $ar)
}
Write-Host ""
