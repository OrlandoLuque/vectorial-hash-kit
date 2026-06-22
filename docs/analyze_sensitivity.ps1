# Aggregate docs/sweep_sensitivity.csv.
# For each (pop, figure_scale) cell, find the item_limit that minimises
# the total per-frame cost (mv + vis*vis_n + atk*atk_n + ins+rm).
# We don't have vis_n/atk_n in the CSV but they scale predictably with pop.

param(
    [string]$CsvPath = "docs/sweep_sensitivity.csv"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

# Hunters fire vision_prey once per frame in the movement loop; ≈ alive_hunters.
# At 9700 alive avg the measured count was ~3000 (alive * 1/3 share); use the
# scaled estimate.
function Vis-N($pop) { return [int]($pop * 0.30) }
# Attack culls per frame are a few percent of population at most.
function Atk-N($pop) { return [int]($pop * 0.005) }

$rows = Import-Csv $CsvPath
$groups = $rows | Group-Object { "$($_.pop)|$($_.item_limit)|$($_.figure_scale)" }

$agg = foreach ($g in $groups) {
    $parts = $g.Name -split '\|'
    $pop = [int]$parts[0]
    $il  = [int]$parts[1]
    $sc  = [double]$parts[2]
    $visN = Vis-N $pop
    $atkN = Atk-N $pop
    $mv  = (($g.Group | Measure-Object -Property mv_mean_us -Average).Average)
    $vis = (($g.Group | Measure-Object -Property vis_avg_us -Average).Average)
    $atk = (($g.Group | Measure-Object -Property atk_avg_us -Average).Average)
    $rm  = (($g.Group | Measure-Object -Property ins_rm_us  -Average).Average)
    [PSCustomObject]@{
        pop          = $pop
        item_limit   = $il
        figure_scale = $sc
        mv           = [math]::Round($mv, 1)
        vis_avg      = [math]::Round($vis, 1)
        atk_avg      = [math]::Round($atk, 1)
        rm           = [math]::Round($rm, 1)
        leaves       = [int](($g.Group | Measure-Object -Property leaves -Average).Average)
        total_us     = [math]::Round($mv + $vis * $visN + $atk * $atkN + $rm, 0)
    }
}

# --- Per-(pop, scale) table with totals for each item_limit ---
Write-Host "`n## Total per-frame cost (µs) per (pop, figure_scale, item_limit)`n"
Write-Host "| pop   | scale | il=10  | il=30  | il=50  | il=100 | il=200 | il=500 |"
Write-Host "|-------|------:|-------:|-------:|-------:|-------:|-------:|-------:|"
foreach ($pop in @(1000, 3000, 10000)) {
    foreach ($scale in @(0.5, 1.0, 2.0)) {
        $vals = @()
        foreach ($il in @(10, 30, 50, 100, 200, 500)) {
            $cell = $agg | Where-Object { $_.pop -eq $pop -and $_.figure_scale -eq $scale -and $_.item_limit -eq $il } | Select-Object -First 1
            if ($null -ne $cell) { $vals += $cell.total_us } else { $vals += 0 }
        }
        # Find min and mark it
        $min = ($vals | Measure-Object -Minimum).Minimum
        $marked = $vals | ForEach-Object { if ($_ -eq $min) { "**$([int]$_)**" } else { "$([int]$_)" } }
        Write-Host ("| {0,5} | {1,5} | {2,6} | {3,6} | {4,6} | {5,6} | {6,6} | {7,6} |" -f `
            $pop, $scale, $marked[0], $marked[1], $marked[2], $marked[3], $marked[4], $marked[5])
    }
}

# --- Best item_limit per (pop, scale) ---
Write-Host "`n## Best item_limit per (pop, figure_scale)`n"
Write-Host "| pop   | scale | best il | total ms/frame | leaves | leaf side ≈ |"
Write-Host "|-------|------:|--------:|---------------:|-------:|------------:|"
$worldArea = 1024.0 * 1024.0
foreach ($pop in @(1000, 3000, 10000)) {
    foreach ($scale in @(0.5, 1.0, 2.0)) {
        $cells = $agg | Where-Object { $_.pop -eq $pop -and $_.figure_scale -eq $scale }
        if ($cells.Count -eq 0) { continue }
        $best = $cells | Sort-Object total_us | Select-Object -First 1
        $leafSide = [math]::Round([math]::Sqrt($worldArea / [math]::Max(1, $best.leaves)), 0)
        Write-Host ("| {0,5} | {1,5} | {2,7} | {3,14:F2} | {4,6} | {5,11} |" -f `
            $pop, $scale, $best.item_limit, ($best.total_us / 1000), $best.leaves, $leafSide)
    }
}

# --- Heuristic guess: ratio of optimum-leaf-size to figure-linear-size ---
Write-Host "`n## Optimum leaf side vs figure linear size`n"
Write-Host "Figure 'linear' size = DROP_SCALE * figure_scale = 110*scale."
Write-Host ""
Write-Host "| pop   | scale | figure side | leaf side | ratio leaf/figure |"
Write-Host "|-------|------:|------------:|----------:|------------------:|"
foreach ($pop in @(1000, 3000, 10000)) {
    foreach ($scale in @(0.5, 1.0, 2.0)) {
        $cells = $agg | Where-Object { $_.pop -eq $pop -and $_.figure_scale -eq $scale }
        if ($cells.Count -eq 0) { continue }
        $best = $cells | Sort-Object total_us | Select-Object -First 1
        $leafSide = [math]::Sqrt($worldArea / [math]::Max(1, $best.leaves))
        $figSide  = 110 * $scale
        $ratio    = $leafSide / $figSide
        Write-Host ("| {0,5} | {1,5} | {2,11:F0} | {3,9:F0} | {4,17:F2} |" -f `
            $pop, $scale, $figSide, $leafSide, $ratio)
    }
}

Write-Host ""
