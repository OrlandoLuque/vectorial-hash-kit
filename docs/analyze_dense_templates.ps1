# Compare dense-template sweep (8/16/32/64/128) against the
# original-template equivalent rows in sweep_sensitivity.csv (pop=10000,
# scale=1.0).

param(
    [string]$DensePath  = "docs/sweep_dense_templates.csv",
    [string]$BasePath   = "docs/sweep_sensitivity.csv"
)

$ErrorActionPreference = 'Stop'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

$visN = 3000  # 10000 pop ≈ 3000 vision calls/frame
$atkN = 50

function Aggregate-Rows([string]$Path, [scriptblock]$Predicate) {
    $rows = Import-Csv $Path | Where-Object $Predicate
    $rows | Group-Object { $_.item_limit } | ForEach-Object {
        $g = $_.Group
        $mv  = (($g | Measure-Object -Property mv_mean_us -Average).Average)
        $vis = (($g | Measure-Object -Property vis_avg_us -Average).Average)
        $atk = (($g | Measure-Object -Property atk_avg_us -Average).Average)
        $rm  = (($g | Measure-Object -Property ins_rm_us  -Average).Average)
        [PSCustomObject]@{
            item_limit = [int]$_.Name
            mv         = [math]::Round($mv, 1)
            vis        = [math]::Round($vis, 1)
            atk        = [math]::Round($atk, 1)
            rm         = [math]::Round($rm, 1)
            leaves     = [int](($g | Measure-Object -Property leaves -Average).Average)
            total      = [math]::Round($mv + $vis*$visN + $atk*$atkN + $rm, 0)
        }
    } | Sort-Object item_limit
}

$base  = Aggregate-Rows -Path $BasePath  -Predicate { $_.pop -eq '10000' -and $_.figure_scale -eq '1' }
$dense = Aggregate-Rows -Path $DensePath -Predicate { $_.pop -eq '10000' }

Write-Host "`n## Base template set (8/16/32 + rects) vs dense (+ 64, 128)`n"
Write-Host "| item_limit | base mv | base vis | base total | dense mv | dense vis | dense total | delta total |"
Write-Host "|-----------:|--------:|---------:|-----------:|---------:|----------:|------------:|--------:|"

$ils = $base | ForEach-Object { $_.item_limit }
foreach ($il in $ils) {
    $b = $base  | Where-Object { $_.item_limit -eq $il } | Select-Object -First 1
    $d = $dense | Where-Object { $_.item_limit -eq $il } | Select-Object -First 1
    if ($null -eq $b -or $null -eq $d) { continue }
    $deltaPct = (($d.total - $b.total) / $b.total) * 100
    Write-Host ("| {0,10} | {1,7:F1} | {2,8:F1} | {3,10:F0} | {4,8:F1} | {5,9:F1} | {6,11:F0} | {7,6:F1}% |" -f `
        $il, $b.mv, $b.vis, $b.total, $d.mv, $d.vis, $d.total, $deltaPct)
}

Write-Host "`n## Optimum item_limit comparison`n"
$bBest = $base  | Sort-Object total | Select-Object -First 1
$dBest = $dense | Sort-Object total | Select-Object -First 1
Write-Host ("base  optimum: il={0,3}  total={1} µs/frame" -f $bBest.item_limit, $bBest.total)
Write-Host ("dense optimum: il={0,3}  total={1} µs/frame" -f $dBest.item_limit, $dBest.total)

Write-Host ""
