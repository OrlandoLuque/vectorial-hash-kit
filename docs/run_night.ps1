# Overnight runner: every measurement-only sweep, in sequence (never in
# parallel — concurrent benches contaminate each other's cache timings).
# Every sub-script is resumable, so this is safe to relaunch after a kill.
$ErrorActionPreference = 'Continue'

$steps = @(
    @{ name='cull_walk break-even';      file='docs/sweep_cull_walk.ps1' }
    @{ name='strategy at il=100';        file='docs/sweep_strategy_il100.ps1' }
    @{ name='quadtree vs binary';        file='docs/sweep_quad_vs_binary.ps1' }
    @{ name='movement-step sensitivity'; file='docs/sweep_movement_step.ps1' }
    @{ name='30k variance';              file='docs/sweep_30k.ps1' }
)

foreach ($s in $steps) {
    Write-Host "`n=================== $($s.name) ==================="
    & powershell -ExecutionPolicy Bypass -File $s.file
}
Write-Host "`n=================== ALL NIGHT SWEEPS DONE ==================="
