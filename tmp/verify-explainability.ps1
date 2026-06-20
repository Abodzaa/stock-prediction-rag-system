$ErrorActionPreference = "Stop"

function Receive-WebSocketMessage {
  param([System.Net.WebSockets.ClientWebSocket]$Socket)

  $buffer = New-Object byte[] 65536
  $segment = [ArraySegment[byte]]::new($buffer)
  $stream = [System.IO.MemoryStream]::new()

  do {
    $result = $Socket.ReceiveAsync($segment, [System.Threading.CancellationToken]::None).GetAwaiter().GetResult()
    if ($result.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) {
      throw "DevTools socket closed unexpectedly."
    }
    $stream.Write($buffer, 0, $result.Count)
  } while (-not $result.EndOfMessage)

  $stream.Position = 0
  $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
  $text = $reader.ReadToEnd()
  $reader.Dispose()
  $stream.Dispose()
  return $text
}

function Send-CdpCommand {
  param(
    [System.Net.WebSockets.ClientWebSocket]$Socket,
    [ref]$Id,
    [string]$Method,
    [hashtable]$Params = @{}
  )

  $Id.Value += 1
  $payload = @{
    id = $Id.Value
    method = $Method
    params = $Params
  } | ConvertTo-Json -Depth 10 -Compress

  $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
  $segment = [ArraySegment[byte]]::new($bytes)
  $Socket.SendAsync($segment, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, [System.Threading.CancellationToken]::None).GetAwaiter().GetResult()

  while ($true) {
    $message = Receive-WebSocketMessage -Socket $Socket | ConvertFrom-Json
    if ($null -ne $message.id -and $message.id -eq $Id.Value) {
      if ($message.error) {
        throw $message.error.message
      }
      return $message.result
    }
  }
}

function Invoke-Eval {
  param(
    [System.Net.WebSockets.ClientWebSocket]$Socket,
    [ref]$Id,
    [string]$Expression
  )

  $result = Send-CdpCommand -Socket $Socket -Id $Id -Method "Runtime.evaluate" -Params @{
    expression = $Expression
    awaitPromise = $true
    returnByValue = $true
  }

  if ($result.exceptionDetails) {
    throw ($result.exceptionDetails.text + " " + $result.exceptionDetails.exception.description)
  }

  return $result.result.value
}

function Save-Screenshot {
  param(
    [System.Net.WebSockets.ClientWebSocket]$Socket,
    [ref]$Id,
    [string]$Path
  )

  $result = Send-CdpCommand -Socket $Socket -Id $Id -Method "Page.captureScreenshot" -Params @{
    format = "png"
    captureBeyondViewport = $true
    fromSurface = $true
  }

  [System.IO.File]::WriteAllBytes($Path, [Convert]::FromBase64String($result.data))
}

$target = Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:9222/json/new?http://127.0.0.1:4173/"
$socket = [System.Net.WebSockets.ClientWebSocket]::new()
$socket.ConnectAsync([Uri]$target.webSocketDebuggerUrl, [System.Threading.CancellationToken]::None).GetAwaiter().GetResult()
$id = 0

Send-CdpCommand -Socket $socket -Id ([ref]$id) -Method "Page.enable" | Out-Null
Send-CdpCommand -Socket $socket -Id ([ref]$id) -Method "Runtime.enable" | Out-Null
Send-CdpCommand -Socket $socket -Id ([ref]$id) -Method "Emulation.setDeviceMetricsOverride" -Params @{
  width = 1440
  height = 2600
  deviceScaleFactor = 1
  mobile = $false
} | Out-Null
Send-CdpCommand -Socket $socket -Id ([ref]$id) -Method "Page.navigate" -Params @{ url = "http://127.0.0.1:4173/" } | Out-Null
Start-Sleep -Seconds 3

$waitForPage = @'
(async () => {
  const started = Date.now();
  while (Date.now() - started < 8000) {
    if (document.querySelector('.action-summary-banner') && document.querySelector('.actions .button-secondary')) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return false;
})()
'@

$ready = Invoke-Eval -Socket $socket -Id ([ref]$id) -Expression $waitForPage
if (-not $ready) {
  throw "App did not become ready for verification."
}

$runMismatch = @'
(async () => {
  document.querySelector('.model-comparison-accordion summary').click();
  await new Promise((resolve) => setTimeout(resolve, 150));
  [...document.querySelectorAll('.model-comparison-accordion .model-card')]
    .find((node) => /Mismatch Ridge/i.test(node.innerText))
    ?.click();
  await new Promise((resolve) => setTimeout(resolve, 150));
  document.querySelector('.actions .button-secondary').click();
  const started = Date.now();
  while (Date.now() - started < 8000) {
    const sources = document.querySelectorAll('.source-card').length;
    const note = document.querySelector('.reconciliation-note strong')?.innerText || '';
    const priceRange = [...document.querySelectorAll('.micro-card strong')].map((node) => node.innerText);
    if (sources >= 3 && /mismatch/i.test(note) && priceRange.length >= 5) {
      return JSON.stringify({
        summary: document.querySelector('.action-summary-banner .lead')?.innerText || '',
        note,
        netLean: document.querySelector('.net-lean-value')?.innerText || '',
        priceStats: [...document.querySelectorAll('.price-stat-grid strong')].map((node) => node.innerText),
        bullishCount: document.querySelectorAll('.catalyst-card:first-of-type .cite-button').length,
        bearishCount: document.querySelectorAll('.catalyst-card:nth-of-type(2) .cite-button').length
      });
    }
    await new Promise((resolve) => setTimeout(resolve, 120));
  }
  return JSON.stringify({ summary: 'timeout' });
})()
'@

$mismatchState = Invoke-Eval -Socket $socket -Id ([ref]$id) -Expression $runMismatch | ConvertFrom-Json

$outDir = Join-Path $PSScriptRoot "explainability-verification"
if (-not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Path $outDir | Out-Null
}
Save-Screenshot -Socket $socket -Id ([ref]$id) -Path (Join-Path $outDir "mismatch.png")

$captureExplanationPanel = @'
(async () => {
  const panel = document.querySelector('.explanation-panel');
  if (!panel) return false;
  panel.scrollIntoView({ behavior: 'instant', block: 'start' });
  await new Promise((resolve) => setTimeout(resolve, 200));
  return true;
})()
'@

Invoke-Eval -Socket $socket -Id ([ref]$id) -Expression $captureExplanationPanel | Out-Null
Save-Screenshot -Socket $socket -Id ([ref]$id) -Path (Join-Path $outDir "mismatch-explanation.png")

$verifyCitation = @'
(async () => {
  document.querySelector('.catalyst-card .cite-button')?.click();
  await new Promise((resolve) => setTimeout(resolve, 150));
  return document.querySelector('.source-card.is-highlighted .source-number')?.innerText || '';
})()
'@

$citationHighlight = Invoke-Eval -Socket $socket -Id ([ref]$id) -Expression $verifyCitation

$reloadForAligned = @'
(async () => {
  location.href = 'http://127.0.0.1:4173/';
  const started = Date.now();
  while (Date.now() - started < 8000) {
    if (document.querySelector('.actions .button-secondary')) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return false;
})()
'@

Invoke-Eval -Socket $socket -Id ([ref]$id) -Expression $reloadForAligned | Out-Null
Start-Sleep -Seconds 2

$runAligned = @'
(async () => {
  document.querySelector('.actions .button-secondary').click();
  const started = Date.now();
  while (Date.now() - started < 8000) {
    const note = document.querySelector('.reconciliation-note strong')?.innerText || '';
    if (/aligned/i.test(note)) {
      return JSON.stringify({
        summary: document.querySelector('.action-summary-banner .lead')?.innerText || '',
        note,
        netLean: document.querySelector('.net-lean-value')?.innerText || ''
      });
    }
    await new Promise((resolve) => setTimeout(resolve, 120));
  }
  return JSON.stringify({
    summary: document.querySelector('.action-summary-banner .lead')?.innerText || 'timeout',
    note: document.querySelector('.reconciliation-note strong')?.innerText || '',
    selected: document.querySelector('.selected-setup-banner')?.innerText || ''
  });
})()
'@

$alignedState = Invoke-Eval -Socket $socket -Id ([ref]$id) -Expression $runAligned | ConvertFrom-Json
Save-Screenshot -Socket $socket -Id ([ref]$id) -Path (Join-Path $outDir "aligned.png")

Invoke-Eval -Socket $socket -Id ([ref]$id) -Expression $captureExplanationPanel | Out-Null
Save-Screenshot -Socket $socket -Id ([ref]$id) -Path (Join-Path $outDir "aligned-explanation.png")

Send-CdpCommand -Socket $socket -Id ([ref]$id) -Method "Browser.close" | Out-Null
$socket.Dispose()

[PSCustomObject]@{
  mismatch = $mismatchState
  aligned = $alignedState
  citationHighlight = $citationHighlight
  screenshotDir = $outDir
} | ConvertTo-Json -Depth 8
