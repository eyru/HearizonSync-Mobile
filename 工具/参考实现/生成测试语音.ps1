# Generate a Chinese test utterance with the built-in SAPI voice.
# Kept pure ASCII on purpose: Windows PowerShell 5.1 misreads UTF-8 without BOM,
# so the sentence is carried as base64 and decoded at runtime.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

$b64 = '5aSn5a625aW977yM5LuK5aSp5aSp5rCU5LiN6ZSZ44CC5oiR5Lus546w5Zyo5rWL6K+V5LiA5LiL5a6e5pe26K+t6Z+z6K+G5Yir57O757uf77yM55yL55yL5pat5Y+l5ZKM5qCH54K55YeG5LiN5YeG44CC'
$text = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($b64))

$dir = Join-Path $PSScriptRoot 'test-audio'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$out = Join-Path $dir 'raw.wav'

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('Microsoft Huihui Desktop')
$synth.Rate = 0
$synth.Volume = 95
$synth.SetOutputToWaveFile($out)
$synth.Speak($text)
$synth.Dispose()

Write-Output "written: $out"
