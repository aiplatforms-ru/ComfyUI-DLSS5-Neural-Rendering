[CmdletBinding()]
param(
    [string] $NgxSdkPath = "",
    [ValidateSet("Release", "Debug")]
    [string] $Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$NativeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$NodeRoot = Split-Path -Parent $NativeRoot
$OutputDirectory = Join-Path $NodeRoot "bin"
$ArtifactDirectory = Join-Path $NativeRoot ".deps\build-artifacts"

if (-not $NgxSdkPath) {
    $NgxSdkPath = Join-Path $NativeRoot ".deps\DLSS-310.7.0"
    if (-not (Test-Path -LiteralPath (Join-Path $NgxSdkPath "include\nvsdk_ngx.h"))) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $NgxSdkPath) | Out-Null
        git clone --depth 1 --branch v310.7.0 https://github.com/NVIDIA/DLSS.git $NgxSdkPath
        if ($LASTEXITCODE -ne 0) {
            throw "Could not download the official NVIDIA DLSS SDK."
        }
    }
}

$NgxSdkPath = (Resolve-Path -LiteralPath $NgxSdkPath).Path
$NgxHeader = Join-Path $NgxSdkPath "include\nvsdk_ngx.h"
$NgxLibrary = Join-Path $NgxSdkPath "lib\Windows_x86_64\x64\nvsdk_ngx_s.lib"
if (-not (Test-Path -LiteralPath $NgxHeader) -or -not (Test-Path -LiteralPath $NgxLibrary)) {
    throw "NgxSdkPath must point to the official NVIDIA DLSS SDK 310.7.0 tree."
}

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $VsWhere)) {
    throw "Visual Studio 2022 Build Tools with the Desktop C++ workload is required."
}
$VisualStudio = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VisualStudio) {
    throw "Visual Studio 2022 C++ Build Tools were not found."
}
$DeveloperCommand = Join-Path $VisualStudio "Common7\Tools\VsDevCmd.bat"

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactDirectory | Out-Null
$Source = Join-Path $NativeRoot "dlssnr_worker.cpp"
$Output = Join-Path $OutputDirectory "dlssnr_worker.exe"
$Object = Join-Path $ArtifactDirectory "dlssnr_worker.obj"
$Optimization = if ($Configuration -eq "Debug") { "/Od /Zi" } else { "/O2 /GL" }
$CompileCommand = @(
    'call "' + $DeveloperCommand + '" -arch=x64 -host_arch=x64 &&',
    'cl.exe /nologo /std:c++20 /EHsc /W4 /permissive- /utf-8 /MT ' + $Optimization,
    '/I"' + (Join-Path $NgxSdkPath 'include') + '"',
    '/Fo"' + $Object + '"',
    '"' + $Source + '"',
    '"' + $NgxLibrary + '"',
    'd3d12.lib d3dcompiler.lib dxgi.lib advapi32.lib user32.lib',
    '/link /OUT:"' + $Output + '"'
) -join ' '

& cmd.exe /d /s /c $CompileCommand
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Output)) {
    throw "Native backend compilation failed."
}

Write-Host "Built $Output"
