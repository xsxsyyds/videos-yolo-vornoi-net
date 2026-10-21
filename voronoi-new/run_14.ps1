param(
    [string]$Python = 'E:\AAA-reseach\env\yolo26\python.exe',
    [string]$Dependencies = 'E:\AAA-reseach\videos\new-track\voronoi-test\deps',
    [int]$Workers = 3,
    [int]$Start = 0,
    [int]$Count = 0,
    [string]$OutputDir = '',
    [switch]$Preview,
    [switch]$Overwrite
)
$ErrorActionPreference = 'Stop'
$previousDeps = $env:VORONOI_DEPS
try {
    if ($Dependencies) { $env:VORONOI_DEPS = $Dependencies }
    else { Remove-Item Env:\VORONOI_DEPS -ErrorAction SilentlyContinue }
    $renderArgs = @('-u', (Join-Path $PSScriptRoot 'render_video.py'),
        '--config', (Join-Path $PSScriptRoot 'configs\14-10.20.json'),
        '--workers', $Workers, '--start', $Start)
    if ($Count -gt 0) { $renderArgs += @('--count', $Count) }
    if ($OutputDir) { $renderArgs += @('--out-dir', $OutputDir) }
    if ($Preview) { $renderArgs += '--preview' }
    if ($Overwrite) { $renderArgs += '--overwrite' }
    & $Python @renderArgs
    if ($LASTEXITCODE -ne 0) { throw "Renderer failed with exit code $LASTEXITCODE" }
}
finally { $env:VORONOI_DEPS = $previousDeps }
