param(
    [ValidateSet('train', 'eval')][string]$Phase,
    [string]$RunDir = 'results/compact_v4',
    [string]$Model = 'gpt-5.5',
    [ValidateSet('sdk', 'deepagents')][string]$Backend = 'sdk',
    [int]$Attempts = 6,
    [int]$EvalAttempts = 3,
    [ValidateSet('both', 'kb', 'baseline')][string]$EvalMode = 'both',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', 'User')
if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw 'OPENAI_API_KEY is not set for the current Windows user.'
}
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if ($Phase -eq 'train') {
    $resumeArgs = @()
    if ($Resume) { $resumeArgs += '--resume' }
    & $python run.py --phase train --run-dir $RunDir --model $Model --backend $Backend --attempts $Attempts @resumeArgs
} else {
    & $python run.py --phase eval --run-dir $RunDir --eval-mode $EvalMode --eval-attempts $EvalAttempts
}
exit $LASTEXITCODE
