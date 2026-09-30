# One-command deploy for the live advisor (brewbot.link).
#
#   powershell -ExecutionPolicy Bypass -File deploy.ps1            # deploy
#   powershell -ExecutionPolicy Bypass -File deploy.ps1 -Force     # even if an answer is mid-stream
#   powershell -ExecutionPolicy Bypass -File deploy.ps1 -Rollback  # restore the last good deploy
#
# Steps: refuse if someone's answer is in flight -> smoke test (free, no AI calls) ->
# promote advisor_ui.dev.html -> restart the server (the supervisor brings it back) ->
# verify local + public health -> snapshot as "last good". Any failure after the restart
# restores the previous last-good files and restarts again.
param(
    [switch]$Force,
    [switch]$Rollback,
    [string]$Python = "C:\Users\brian\AppData\Local\Programs\Python\Python312\python.exe",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$LastGood = Join-Path $Root ".deploy\last_good"
$DeployLog = Join-Path $Root "deploy.log"
$LiveFiles = @("advisor_app.py", "mtg_tools.py", "role_index.py", "accounts.py", "advisor_ui.html", "advisor_login.html", "advisor_signup.html")

function Say($msg, $color = "Gray") {
    Write-Host $msg -ForegroundColor $color
    Add-Content -Path $DeployLog -Value ("{0:yyyy-MM-dd HH:mm:ss} | {1}" -f (Get-Date), $msg) -Encoding utf8
}

function Get-EnvValue($key) {
    foreach ($l in Get-Content (Join-Path $Root ".env")) {
        if ($l -match "^\s*$key\s*=\s*(.*)$") { return $Matches[1].Trim().Trim('"').Trim("'") }
    }
    return $null
}

function Test-Health($url) {
    try { return (Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 10).StatusCode -eq 200 } catch { return $false }
}

function Restart-Server {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    # The supervisor restarts it within ~30 s; the app then needs a few seconds to boot.
    $deadline = (Get-Date).AddSeconds(100)
    do {
        Start-Sleep -Seconds 4
        if (Test-Health "http://127.0.0.1:$Port/healthz") { return $true }
    } until ((Get-Date) -gt $deadline)
    return $false
}

function Restore-LastGood {
    if (-not (Test-Path $LastGood)) { Say "No last-good snapshot to roll back to." "Red"; return $false }
    foreach ($f in $LiveFiles) {
        $src = Join-Path $LastGood $f
        if (Test-Path $src) { Copy-Item $src (Join-Path $Root $f) -Force }
    }
    Say "Restored last-good files; restarting..." "Yellow"
    return (Restart-Server)
}

if ($Rollback) {
    if (Restore-LastGood) { Say "ROLLBACK OK - live site is back on the last good deploy." "Green"; exit 0 }
    Say "ROLLBACK FAILED - check advisor_server.err.log" "Red"; exit 1
}

# 1. Don't cut off someone's answer: a QUERY in the last 5 min with no ANSWER after it.
if (-not $Force) {
    $recent = Get-Content (Join-Path $Root "advisor_activity.log") -Tail 40 -ErrorAction SilentlyContinue
    $open = @{}
    foreach ($line in $recent) {
        if ($line -match "^(\S+ \S+) \| QUERY\s+sid=(\w+)") { $open[$Matches[2]] = [datetime]$Matches[1] }
        elseif ($line -match "\| ANSWER sid=(\w+)") { $open.Remove($Matches[1]) }
    }
    $inflight = $open.GetEnumerator() | Where-Object { $_.Value -gt (Get-Date).AddMinutes(-5) }
    if ($inflight) {
        Say "Someone's answer is still being generated (session $(@($inflight)[0].Key)). Retry shortly, or use -Force." "Yellow"
        exit 2
    }
}

# 2. Smoke test against the UI we're about to ship.
$devUi = Join-Path $Root "advisor_ui.dev.html"
$uiToShip = if (Test-Path $devUi) { "advisor_ui.dev.html" } else { "advisor_ui.html" }
Say "Smoke testing (UI: $uiToShip)..."
& $Python (Join-Path $Root "smoke_test.py") --ui $uiToShip
if ($LASTEXITCODE -ne 0) { Say "DEPLOY ABORTED - smoke test failed; the live site was not touched." "Red"; exit 1 }

# 3. Promote the dev UI.
if ($uiToShip -eq "advisor_ui.dev.html") {
    Copy-Item $devUi (Join-Path $Root "advisor_ui.html") -Force
    Say "Promoted advisor_ui.dev.html -> advisor_ui.html"
}

# 4. Restart + verify (local, then the public URL through the tunnel).
Say "Restarting the server..."
$ok = Restart-Server
$public = Get-EnvValue "ADVISOR_PUBLIC_URL"
if ($ok -and $public) {
    $ok = $false
    for ($i = 0; $i -lt 6 -and -not $ok; $i++) { $ok = Test-Health "$public/healthz"; if (-not $ok) { Start-Sleep -Seconds 5 } }
    if (-not $ok) { Say "Server is up locally but $public isn't answering." "Red" }
}

if (-not $ok) {
    Say "DEPLOY FAILED after restart - rolling back." "Red"
    if (Restore-LastGood) { Say "Rolled back; the live site is on the previous deploy." "Yellow" }
    exit 1
}

# 5. Snapshot as last good.
New-Item -ItemType Directory -Force $LastGood | Out-Null
foreach ($f in $LiveFiles) { Copy-Item (Join-Path $Root $f) (Join-Path $LastGood $f) -Force }
$commit = (git -C $Root rev-parse --short HEAD 2>$null)
Say "DEPLOY OK - live at $(if ($public) { $public } else { "http://127.0.0.1:$Port" }) (HEAD $commit + working tree)." "Green"
