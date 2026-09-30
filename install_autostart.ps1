# Registers (or removes) the "MTG Advisor" scheduled task that runs run_advisor.ps1
# at logon, hidden, and restarts it if it ever dies. Current user only; no admin needed.
#
#   powershell -ExecutionPolicy Bypass -File install_autostart.ps1            # install + start now
#   powershell -ExecutionPolicy Bypass -File install_autostart.ps1 -Uninstall
param([switch]$Uninstall)

$TaskName = "MTG Advisor"

if ($Uninstall) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Removed scheduled task '$TaskName'."
    exit 0
}

$script = Join-Path $PSScriptRoot "run_advisor.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Keeps the MTG Deckbuilding Advisor server + Cloudflare tunnel running." -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Output "Installed and started '$TaskName'. Public URL will appear in advisor_url.txt."
