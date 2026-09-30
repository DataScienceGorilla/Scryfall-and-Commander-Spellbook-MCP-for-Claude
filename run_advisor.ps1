# Keeps the Deckbuilding Advisor (uvicorn) and its Cloudflare tunnel running.
#
# Checks every $CheckSecs: if the server stops answering /healthz it is restarted;
# if cloudflared exits it is restarted. The current public URL is written to
# advisor_url.txt. Run by the "MTG Advisor" scheduled task (install_autostart.ps1)
# so it starts at logon and survives reboots.
#
# Tunnel mode: if ADVISOR_TUNNEL_TOKEN is set in .env, runs a *named* Cloudflare
# tunnel (stable URL). Otherwise a quick tunnel (random trycloudflare.com URL that
# changes whenever cloudflared restarts).
param(
    [string]$Python = "C:\Users\brian\AppData\Local\Programs\Python\Python312\python.exe",
    [int]$Port = 8000,
    [int]$CheckSecs = 30,
    [switch]$NoTunnel
)

$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
Set-Location $Root
$LogFile = Join-Path $Root "supervisor.log"
$UrlFile = Join-Path $Root "advisor_url.txt"
$TunnelLog = Join-Path $Root "cf_tunnel.err.log"

function Log($msg) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} | {1}" -f (Get-Date), $msg
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

# Single instance: a second copy (e.g. task fired twice) exits immediately.
$mutex = New-Object System.Threading.Mutex($false, "Global\MTGAdvisorSupervisor")
if (-not $mutex.WaitOne(0)) { exit 0 }

function Get-EnvValue($key) {
    $envFile = Join-Path $Root ".env"
    if (-not (Test-Path $envFile)) { return $null }
    foreach ($l in Get-Content $envFile) {
        if ($l -match "^\s*$key\s*=\s*(.*)$") { return $Matches[1].Trim().Trim('"').Trim("'") }
    }
    return $null
}

function Test-Server {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -UseBasicParsing -TimeoutSec 10
        return $r.StatusCode -eq 200
    } catch { return $false }
}

function Stop-PortOwner {
    # Kill a hung server still holding the port so the restart can bind.
    try {
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
            ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    } catch {}
}

function Start-Server {
    Stop-PortOwner
    Log "starting server on port $Port"
    Start-Process -WindowStyle Hidden -WorkingDirectory $Root -FilePath $Python `
        -ArgumentList "-m", "uvicorn", "advisor_app:app", "--host", "127.0.0.1", "--port", "$Port" `
        -RedirectStandardOutput (Join-Path $Root "advisor_server.out.log") `
        -RedirectStandardError (Join-Path $Root "advisor_server.err.log") -PassThru
}

function Start-Tunnel {
    $exe = Join-Path $Root "cloudflared.exe"
    $token = Get-EnvValue "ADVISOR_TUNNEL_TOKEN"
    if ($token) {
        Log "starting named tunnel"
        $tunnelArgs = @("tunnel", "--no-autoupdate", "run", "--token", $token)
    } else {
        Log "starting quick tunnel"
        $tunnelArgs = @("tunnel", "--no-autoupdate", "--url", "http://localhost:$Port")
    }
    Start-Process -WindowStyle Hidden -WorkingDirectory $Root -FilePath $exe -ArgumentList $tunnelArgs `
        -RedirectStandardOutput (Join-Path $Root "cf_tunnel.out.log") -RedirectStandardError $TunnelLog -PassThru
}

function Update-Url {
    $named = Get-EnvValue "ADVISOR_PUBLIC_URL"
    if ($named) { $url = $named }
    else {
        $url = $null
        for ($i = 0; $i -lt 20 -and -not $url; $i++) {
            Start-Sleep -Seconds 1
            if (Test-Path $TunnelLog) {
                $m = Select-String -Path $TunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -Last 1
                if ($m) { $url = $m.Matches[0].Value }
            }
        }
    }
    if ($url) {
        Set-Content -Path $UrlFile -Value $url -Encoding ascii  # no BOM, so scripts can read it raw
        Log "public URL: $url"
    } else { Log "could not determine tunnel URL yet" }
}

Log "supervisor started (pid $PID)"
$server = $null
$tunnel = $null
$serverFails = 0

while ($true) {
    if (Test-Server) {
        $serverFails = 0
    } else {
        $serverFails++
        # A fresh start needs a few seconds before /healthz answers; only restart
        # if the process died or it has been unresponsive for 2 checks in a row.
        if (-not $server -or $server.HasExited -or $serverFails -ge 2) {
            $server = Start-Server
            $serverFails = 0
            Start-Sleep -Seconds 15
        }
    }

    # If cloudflared is installed as a Windows service (Cloudflare's "service install
    # <token>" command), it owns the tunnel - it starts at boot and Windows restarts it.
    # Stop our own connector so there's just one, and leave the service alone.
    $svc = Get-Service -Name Cloudflared -ErrorAction SilentlyContinue
    $serviceOwnsTunnel = $svc -and $svc.Status -eq "Running"
    if ($serviceOwnsTunnel) {
        # Our connectors are the ones whose command line we can read (the SYSTEM
        # service's is hidden from us), including leftovers from a previous run.
        foreach ($p in Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'") {
            if ($p.CommandLine -like "*tunnel*") {
                Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
                Log "Cloudflared service owns the tunnel; stopped our connector (pid $($p.ProcessId))"
            }
        }
        $tunnel = $null
    }

    if (-not $NoTunnel -and -not $serviceOwnsTunnel -and (-not $tunnel -or $tunnel.HasExited)) {
        # Adopt a tunnel that's already running (e.g. supervisor restarted) instead of
        # duplicating it - but only if it's the right kind (named vs quick). A quick
        # tunnel left over from before ADVISOR_TUNNEL_TOKEN was set gets replaced.
        $wantNamed = [bool](Get-EnvValue "ADVISOR_TUNNEL_TOKEN")
        $existing = $null
        foreach ($p in Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'") {
            $isNamed = $p.CommandLine -like "*--token*"
            if ($isNamed -eq $wantNamed -and -not $existing) { $existing = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue }
            else { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Log "stopped mismatched/duplicate cloudflared (pid $($p.ProcessId))" }
        }
        if ($existing) {
            $tunnel = $existing
            Log "adopted running cloudflared (pid $($existing.Id))"
        } else {
            if ($tunnel) { Log "cloudflared exited (code $($tunnel.ExitCode)); restarting" }
            $tunnel = Start-Tunnel
            Update-Url
        }
    }

    Start-Sleep -Seconds $CheckSecs
}
