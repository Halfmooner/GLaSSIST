# Synthesize a test utterance with Windows SAPI at 16 kHz mono 16-bit.
param([string]$Out = "speech.wav", [string]$Text = "Alexa, turn on the living room light.", [int]$Rate = 0)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = $Rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($Out, $fmt)
$s.Speak($Text)
$s.Dispose()
Write-Host "voice:" ($s.Voice.Name) "->" $Out
