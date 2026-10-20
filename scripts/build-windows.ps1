<#
.SYNOPSIS
  Builds RLM IDE on Windows from scratch: gets the project, sets up the tools, installs dependencies, creates the
  Python venv, runs the tests, builds the app, tests that it starts, builds the installer and tests the packaged app.

.DESCRIPTION
  Works in Windows PowerShell 5.1 and PowerShell 7, without admin rights. Anything missing is set up locally:
    - the project is cloned with git, or downloaded as a zip when git is not installed;
    - Node.js (the major version from package.json "engines") is downloaded as a portable zip into -ToolsDir;
    - pnpm (the exact version from package.json "packageManager") is installed into -ToolsDir.
  The bundled Python runtime and the bundled embedder (bge-m3) are downloaded by the project's own scripts.
  A Python venv (.venv in the project) is created from the bundled runtime for the Python tests.

  Steps: project, tools, dependencies, Python runtime + venv, models, tests, build, start-up test,
  installer, packaged-app start-up test, and (with -TestInstaller) a real silent install, start-up test
  and uninstall of the installer.

  The start-up test launches the IDE with a throwaway profile and checks the window, the terminal, memory and
  an agent turn that runs a Python cell (against a local mock model; no keys or network needed).

  Logs and a summary.json go to -LogDir.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1
  Builds the project this script lives in, or clones it into $HOME\rlm-ide when run on its own.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-windows.ps1 -ProjectDir D:\src\rlm-ide -TestInstaller
  Also installs the built installer silently into a temporary folder, tests the installed app and uninstalls it.
#>
[CmdletBinding()]
param(
  # Where the project is (or is cloned to). Default: the checkout this script is in, else $HOME\rlm-ide.
  [string]$ProjectDir = "",
  [string]$RepoUrl = "https://github.com/adamsmith988/rlm-ide.git",
  [string]$Branch = "master",
  # Pull the latest commits when the project is already there (git checkouts only).
  [switch]$Update,
  # Portable Node.js and pnpm go here when the installed ones do not fit. Default: <ProjectDir>\.tools
  [string]$ToolsDir = "",
  # Use a portable Node.js even when a suitable one is installed.
  [switch]$PortableNode,
  [string]$NodeDistUrl = "https://nodejs.org/dist",
  # Python packages installed into the venv for the Python tests.
  [string[]]$PythonTestPackages = @("pytest"),
  [switch]$SkipTests,
  [switch]$SkipStartTest,
  [switch]$SkipInstaller,
  # Silently installs the built installer for the current user, tests the installed app, then uninstalls it.
  [switch]$TestInstaller,
  # Run -TestInstaller even though RLM IDE is already installed for this user (the test would replace it).
  [switch]$Force,
  # Remove previous build output (apps\desktop\out and this version's installer folder) and the venv first.
  [switch]$Clean,
  [int]$StartTestTimeoutSec = 300,
  [int]$MinFreeGB = 12,
  # Default: %LOCALAPPDATA%\rlm-ide-build\<timestamp>
  [string]$LogDir = ""
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$script:Steps = New-Object System.Collections.ArrayList
$script:Artifacts = [ordered]@{}
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $LogDir) { $LogDir = Join-Path (Join-Path $env:LOCALAPPDATA "rlm-ide-build") $stamp }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Banner([string]$Text) {
  Write-Host ""
  Write-Host "==> $Text" -ForegroundColor Cyan
}

function Add-ToPath([string]$Dir) {
  $env:PATH = "$Dir;$env:PATH"
}

# Runs a native command, streaming its output to the console and a log file, and fails on a non-zero exit code.
# Windows PowerShell 5.1 turns native stderr into errors under -ErrorAction Stop, so it is relaxed while the command runs.
function Invoke-Native {
  param([Parameter(Mandatory)][string]$File, [string[]]$Arguments = @(), [string]$Log, [string]$WorkDir = "")
  if (-not $Log) { $Log = Join-Path $LogDir "commands.log" }
  [IO.File]::AppendAllText($Log, "`r`n> $File $($Arguments -join ' ')`r`n")
  $prevPref = $ErrorActionPreference
  $prevDir = Get-Location
  $ErrorActionPreference = "Continue"
  try {
    if ($WorkDir) { Set-Location $WorkDir }
    & $File @Arguments 2>&1 | ForEach-Object {
      $line = "$_"
      [IO.File]::AppendAllText($Log, "$line`r`n")
      Write-Host $line
    }
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prevPref
    Set-Location $prevDir
  }
  if ($code -ne 0) { throw "$([IO.Path]::GetFileName($File)) $($Arguments -join ' ') exited with code $code (log: $Log)" }
}

# Prefers .exe/.cmd shims: a .ps1 shim can be blocked by the execution policy.
function Find-Tool([string]$Name) {
  foreach ($candidate in @("$Name.exe", "$Name.cmd", $Name)) {
    $cmd = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
  }
  return $null
}

function Invoke-Step {
  param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][scriptblock]$Action)
  Write-Banner $Name
  $entry = [ordered]@{ name = $Name; status = "running"; seconds = 0; detail = "" }
  [void]$script:Steps.Add($entry)
  $sw = [Diagnostics.Stopwatch]::StartNew()
  try {
    $detail = & $Action
    $entry.status = "ok"
    if ($detail) { $entry.detail = ($detail | Out-String).Trim() }
  } catch {
    $entry.status = "failed"
    $entry.detail = $_.Exception.Message
    throw
  } finally {
    $entry.seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
  }
  Write-Host ("    done in {0} s{1}" -f $entry.seconds, $(if ($entry.detail) { " - $($entry.detail)" } else { "" })) -ForegroundColor Green
}

function Skip-Step([string]$Name, [string]$Why) {
  Write-Banner "$Name (skipped: $Why)"
  [void]$script:Steps.Add([ordered]@{ name = $Name; status = "skipped"; seconds = 0; detail = $Why })
}

function Get-Download([string]$Url, [string]$Dest) {
  Write-Host "    downloading $Url"
  Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
}

function Get-InstalledRlmIde {
  $keys = @("HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*")
  foreach ($k in $keys) {
    $hit = Get-ItemProperty $k -ErrorAction SilentlyContinue | Where-Object { $_.PSObject.Properties["DisplayName"] -and $_.DisplayName -like "RLM IDE*" } | Select-Object -First 1
    if ($hit) { return $hit }
  }
  return $null
}

function Invoke-StartTest([string]$Label, [string]$Exe) {
  $out = Join-Path $LogDir "start-test-$Label"
  $smokeArgs = @("scripts\smoke-start.mjs", "--out", $out, "--timeout-sec", "$StartTestTimeoutSec")
  if ($Exe) { $smokeArgs += @("--exe", $Exe) }
  Invoke-Native -File $script:Node -Arguments $smokeArgs -Log (Join-Path $LogDir "start-test-$Label.log") -WorkDir $ProjectDir
  $result = Get-Content (Join-Path $out "smoke.json") -Raw | ConvertFrom-Json
  $passed = @($result.checks | Where-Object { $_.ok }).Count
  return "$passed/$(@($result.checks).Count) checks passed (screenshot: $(Join-Path $out 'smoke.png'))"
}

$failed = $null
try {
  Invoke-Step "Check this machine" {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw "This script builds the Windows app and must run on Windows." }
    if (-not [Environment]::Is64BitOperatingSystem) { throw "A 64-bit Windows is required." }
    if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") { Write-Warning "This builds the x64 app; on $($env:PROCESSOR_ARCHITECTURE) it runs under emulation." }
    if (-not $script:ProjectDir) {
      $script:ProjectDir = Join-Path $HOME "rlm-ide"
      if ($PSCommandPath) {
        $here = Split-Path -Parent $PSCommandPath
        $candidate = Split-Path -Parent $here
        $pkg = Join-Path $candidate "package.json"
        if ((Split-Path -Leaf $here) -eq "scripts" -and (Test-Path $pkg) -and ((Get-Content $pkg -Raw) -match '"name":\s*"rlm-ide-monorepo"')) { $script:ProjectDir = $candidate }
      }
    }
    $script:ProjectDir = [IO.Path]::GetFullPath($script:ProjectDir)
    if (-not $script:ToolsDir) { $script:ToolsDir = Join-Path $script:ProjectDir ".tools" }
    $drive = [IO.Path]::GetPathRoot($script:ProjectDir)
    $freeGB = [math]::Round((New-Object IO.DriveInfo $drive).AvailableFreeSpace / 1GB, 1)
    if ($freeGB -lt $MinFreeGB) { Write-Warning "Only $freeGB GB free on $drive; a full build needs about $MinFreeGB GB." }
    "PowerShell $($PSVersionTable.PSVersion) $($PSVersionTable.PSEdition), project $script:ProjectDir, $freeGB GB free, logs $LogDir"
  }

  Invoke-Step "Get the project" {
    $git = Find-Tool "git"
    if (Test-Path (Join-Path $ProjectDir "package.json")) {
      if ($Update) {
        if (-not ($git -and (Test-Path (Join-Path $ProjectDir ".git")))) { throw "-Update needs git and a git checkout in $ProjectDir." }
        Invoke-Native -File $git -Arguments @("-C", $ProjectDir, "pull", "--ff-only")
      }
      return "using the existing project"
    }
    if ((Test-Path $ProjectDir) -and @(Get-ChildItem -Force $ProjectDir).Count -gt 0) { throw "$ProjectDir exists, is not empty and is not the project." }
    if ($git) {
      Invoke-Native -File $git -Arguments @("clone", "--depth", "1", "--branch", $Branch, $RepoUrl, $ProjectDir)
      return "cloned $RepoUrl ($Branch)"
    }
    $zipUrl = ($RepoUrl -replace '\.git$', '') + "/archive/refs/heads/$Branch.zip"
    $zip = Join-Path $env:TEMP "rlm-ide-$stamp.zip"
    $unz = Join-Path $env:TEMP "rlm-ide-$stamp"
    Get-Download $zipUrl $zip
    Expand-Archive -Path $zip -DestinationPath $unz -Force
    $inner = Get-ChildItem $unz -Directory | Select-Object -First 1
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ProjectDir) | Out-Null
    Move-Item $inner.FullName $ProjectDir
    Remove-Item $zip -Force
    Remove-Item $unz -Recurse -Force
    "downloaded $zipUrl (git is not installed)"
  }

  $pkgJson = Get-Content (Join-Path $ProjectDir "package.json") -Raw | ConvertFrom-Json
  if ($pkgJson.name -ne "rlm-ide-monorepo") { throw "$ProjectDir does not hold the RLM IDE project." }
  $desktopPkg = Get-Content (Join-Path $ProjectDir "apps\desktop\package.json") -Raw | ConvertFrom-Json
  $version = $desktopPkg.version
  $releaseDir = Join-Path $ProjectDir "apps\desktop\release\$version"
  $venv = Join-Path $ProjectDir ".venv"

  if ($Clean) {
    Invoke-Step "Remove previous build output" {
      foreach ($p in @((Join-Path $ProjectDir "apps\desktop\out"), $releaseDir, $venv)) {
        if (Test-Path $p) { Remove-Item $p -Recurse -Force }
      }
      "removed apps\desktop\out, release\$version and .venv"
    }
  }

  Invoke-Step "Set up Node.js and pnpm" {
    $nodeMajor = [int]([regex]::Match([string]$pkgJson.engines.node, '\d+').Value)
    $pnpmVersion = ([string]$pkgJson.packageManager -split '@')[1] -replace '\+.*$', ''
    $node = $null
    if (-not $PortableNode) {
      $node = Find-Tool "node"
      if ($node) {
        $have = (& $node -p "process.versions.node").Trim()
        if ([int]($have -split '\.')[0] -lt $nodeMajor) {
          Write-Host "    installed Node.js $have is older than $nodeMajor; using a portable one"
          $node = $null
        }
      }
    }
    if (-not $node) {
      $index = Invoke-WebRequest -Uri "$NodeDistUrl/index.json" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json
      $release = $index | Where-Object { $_.version -like "v$nodeMajor.*" -and $_.files -contains "win-x64-zip" } | Select-Object -First 1
      if (-not $release) { throw "No Node.js $nodeMajor release for win-x64 at $NodeDistUrl." }
      $name = "node-$($release.version)-win-x64"
      $nodeDir = Join-Path $ToolsDir $name
      if (-not (Test-Path (Join-Path $nodeDir "node.exe"))) {
        New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
        $zip = Join-Path $ToolsDir "$name.zip"
        Get-Download "$NodeDistUrl/$($release.version)/$name.zip" $zip
        $sums = (Invoke-WebRequest -Uri "$NodeDistUrl/$($release.version)/SHASUMS256.txt" -UseBasicParsing).Content
        $expected = ([regex]::Match($sums, "([0-9a-f]{64})\s+$([regex]::Escape("$name.zip"))")).Groups[1].Value
        $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if (-not $expected -or $actual -ne $expected) { throw "Checksum mismatch for $name.zip (expected $expected, got $actual)." }
        Expand-Archive -Path $zip -DestinationPath $ToolsDir -Force
        Remove-Item $zip -Force
      }
      Add-ToPath $nodeDir
      $node = Join-Path $nodeDir "node.exe"
    } else {
      Add-ToPath (Split-Path -Parent $node)
    }
    $script:Node = $node

    $pnpm = Find-Tool "pnpm"
    $havePnpm = $null
    if ($pnpm) {
      try { $havePnpm = (& $pnpm --version | Out-String).Trim() } catch { $havePnpm = $null }
    }
    if ($havePnpm -ne $pnpmVersion) {
      $pnpmDir = Join-Path $ToolsDir "pnpm-$pnpmVersion"
      if (-not (Test-Path (Join-Path $pnpmDir "pnpm.cmd"))) {
        $npm = Join-Path (Split-Path -Parent $node) "npm.cmd"
        if (-not (Test-Path $npm)) { $npm = Find-Tool "npm" }
        Invoke-Native -File $npm -Arguments @("install", "--global", "--prefix", $pnpmDir, "--no-fund", "--no-audit", "pnpm@$pnpmVersion")
      }
      Add-ToPath $pnpmDir
      $pnpm = Join-Path $pnpmDir "pnpm.cmd"
    }
    $script:Pnpm = $pnpm
    $env:npm_config_update_notifier = "false"
    "Node.js $((& $node --version).Trim()) ($node), pnpm $((& $pnpm --version).Trim())"
  }

  Invoke-Step "Install dependencies" {
    Invoke-Native -File $Pnpm -Arguments @("install", "--frozen-lockfile") -Log (Join-Path $LogDir "install.log") -WorkDir $ProjectDir
  }

  Invoke-Step "Bundled Python runtime and venv" {
    Invoke-Native -File $Node -Arguments @("scripts\fetch-python.mjs", "--platform", "win32", "--arch", "x64") -WorkDir $ProjectDir
    $python = Join-Path $ProjectDir "resources\python\win32-x64\python.exe"
    $venvPython = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) { Invoke-Native -File $python -Arguments @("-m", "venv", $venv) }
    Invoke-Native -File $venvPython -Arguments (@("-m", "pip", "install", "--disable-pip-version-check", "--quiet") + $PythonTestPackages)
    "$((& $venvPython --version).Trim()) in $venv"
  }

  Invoke-Step "Bundled embedder model (bge-m3)" {
    Invoke-Native -File $Node -Arguments @("--experimental-strip-types", "scripts\fetch-models.mjs", "--only", "builtin:bge-m3") -Log (Join-Path $LogDir "models.log") -WorkDir $ProjectDir
  }

  if ($SkipTests) { Skip-Step "Tests" "-SkipTests" }
  else {
    Invoke-Step "Tests (typecheck, unit tests, Python tests)" {
      $log = Join-Path $LogDir "tests.log"
      Invoke-Native -File $Pnpm -Arguments @("typecheck") -Log $log -WorkDir $ProjectDir
      Invoke-Native -File $Pnpm -Arguments @("test") -Log $log -WorkDir $ProjectDir
      Invoke-Native -File (Join-Path $venv "Scripts\python.exe") -Arguments @("-m", "pytest", "python\rlm_worker\tests", "-q") -Log $log -WorkDir $ProjectDir
      "all passed (log: $log)"
    }
  }

  Invoke-Step "Build the app" {
    Invoke-Native -File $Pnpm -Arguments @("build") -Log (Join-Path $LogDir "build.log") -WorkDir $ProjectDir
  }

  if ($SkipStartTest) { Skip-Step "Start-up test (development build)" "-SkipStartTest" }
  else { Invoke-Step "Start-up test (development build)" { Invoke-StartTest "dev" "" } }

  if ($SkipInstaller) { Skip-Step "Windows installer" "-SkipInstaller" }
  else {
    Invoke-Step "Windows installer" {
      Invoke-Native -File $Pnpm -Arguments @("--filter", "@rlm/desktop", "exec", "electron-builder", "--win", "--x64", "--config", "electron-builder.yml", "--publish", "never") -Log (Join-Path $LogDir "installer.log") -WorkDir $ProjectDir
      $installer = Get-ChildItem $releaseDir -Filter "*.exe" -File | Where-Object { $_.Name -like "*Setup*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
      if (-not $installer) { throw "electron-builder finished but no installer is in $releaseDir." }
      $script:Artifacts["installer"] = $installer.FullName
      $script:Artifacts["installerSha256"] = (Get-FileHash $installer.FullName -Algorithm SHA256).Hash.ToLower()
      $script:Artifacts["installerMB"] = [math]::Round($installer.Length / 1MB)
      $script:Artifacts["unpackedApp"] = Join-Path $releaseDir "win-unpacked\$($desktopPkg.productName).exe"
      "$($installer.Name), $($script:Artifacts['installerMB']) MB"
    }

    if ($SkipStartTest) { Skip-Step "Start-up test (packaged app)" "-SkipStartTest" }
    else { Invoke-Step "Start-up test (packaged app)" { Invoke-StartTest "packaged" $script:Artifacts["unpackedApp"] } }

    if (-not $TestInstaller) { Skip-Step "Installer test" "pass -TestInstaller to install, test and uninstall it" }
    else {
      Invoke-Step "Installer test (silent install, start-up test, uninstall)" {
        $existing = Get-InstalledRlmIde
        if ($existing -and -not $Force) { throw "RLM IDE is already installed for this user ($($existing.InstallLocation)); the test would replace it. Pass -Force to run it anyway." }
        $productName = $desktopPkg.productName
        $dir = Join-Path $env:TEMP "rlm-ide-install-test-$stamp"
        # NSIS takes /D= last and unquoted, even with spaces in the path; one string keeps both PowerShells from quoting it.
        $p = Start-Process -FilePath $script:Artifacts["installer"] -ArgumentList "/S /currentuser /D=$dir" -Wait -PassThru
        if ($p.ExitCode -ne 0) { throw "The installer exited with code $($p.ExitCode)." }
        $exe = Join-Path $dir "$productName.exe"
        if (-not (Test-Path $exe)) { throw "The installer did not put $productName.exe in $dir." }
        $entry = Get-InstalledRlmIde
        if (-not $entry) { throw "The installer did not register an uninstall entry." }
        $shortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "$productName.lnk"
        $hadShortcut = Test-Path $shortcut
        $started = Invoke-StartTest "installed" $exe

        $uninstaller = Join-Path $dir "Uninstall $productName.exe"
        if (-not (Test-Path $uninstaller)) { throw "No uninstaller at $uninstaller." }
        # The NSIS uninstaller copies itself to TEMP and relaunches, so wait for its effects, not the process.
        Start-Process -FilePath $uninstaller -ArgumentList "/S /currentuser" -Wait | Out-Null
        $until = (Get-Date).AddSeconds(120)
        while (((Test-Path $exe) -or (Get-InstalledRlmIde)) -and (Get-Date) -lt $until) { Start-Sleep -Seconds 2 }
        if (Test-Path $exe) { throw "Uninstalling left $exe behind." }
        if (Get-InstalledRlmIde) { throw "Uninstalling left the uninstall entry behind." }
        if (Test-Path $shortcut) { throw "Uninstalling left the desktop shortcut behind." }
        if (Test-Path $dir) { Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue }
        "installed to $dir (desktop shortcut: $hadShortcut), $started, uninstalled cleanly"
      }
    }
  }
} catch {
  $failed = $_
}

$summary = [ordered]@{
  ok         = ($null -eq $failed)
  project    = $ProjectDir
  powershell = "$($PSVersionTable.PSVersion) $($PSVersionTable.PSEdition)"
  steps      = $script:Steps
  artifacts  = $script:Artifacts
  error      = $(if ($failed) { $failed.Exception.Message } else { $null })
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $LogDir "summary.json") -Encoding UTF8

Write-Host ""
Write-Host "Summary" -ForegroundColor Cyan
foreach ($s in $script:Steps) {
  $color = switch ($s.status) { "ok" { "Green" } "skipped" { "DarkGray" } default { "Red" } }
  Write-Host ("  {0,-8} {1,-58} {2,7} s" -f $s.status.ToUpper(), $s.name, $s.seconds) -ForegroundColor $color
}
foreach ($k in $script:Artifacts.Keys) { Write-Host ("  {0}: {1}" -f $k, $script:Artifacts[$k]) }
Write-Host "  logs: $LogDir"
if ($failed) {
  Write-Host ""
  Write-Host "BUILD FAILED: $($failed.Exception.Message)" -ForegroundColor Red
  exit 1
}
Write-Host ""
Write-Host "BUILD SUCCEEDED" -ForegroundColor Green
exit 0
