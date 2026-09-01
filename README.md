# ComfyUI DLSS 5 Neural Rendering

> [!WARNING]
> Experimental Windows/NVIDIA proof of concept. It uses an experimental NGX
> feature contract that may change between runtime or driver versions. This
> project is not affiliated with or endorsed by NVIDIA.

An experimental Windows custom node that invokes NVIDIA's DLSS Neural Rendering
feature 18 as a headless Direct3D 12 post-process. It is a normal ComfyUI node:
connect an `IMAGE` or video-frame batch and receive an `IMAGE` batch back.

The implementation has no Unity, ReShade, game, window, swapchain, or geometry
dependency. A short-lived native worker is managed automatically by the Python
node so an experimental NGX runtime failure cannot take the ComfyUI process down.

## Requirements

- Windows 10/11, a supported NVIDIA RTX GPU, and a compatible NVIDIA driver.
- An NVIDIA-signed `nvngx_dlssnr.dll`. The DLL is **not** redistributed here.
- The included `bin/dlssnr_worker.exe`. Release packages can ship it prebuilt;
  source checkouts can build it with Visual Studio 2022 Build Tools.
- The Python packages in `requirements.txt`, which add the depth-model loader.

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/aiplatforms-ru/ComfyUI-DLSS5-Neural-Rendering.git
cd ComfyUI-DLSS5-Neural-Rendering
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

The relative Python path above is for the Windows portable build. For a venv
installation, run `python -m pip install -r requirements.txt` in the same
directory. Restart ComfyUI after installation.

Place a legally obtained, NVIDIA-signed `nvngx_dlssnr.dll` in `runtime/`, set
`COMFYUI_DLSSNR_RUNTIME`, or select its directory with `runtime_directory`.
The DLL is deliberately not included in this repository.

On first use, the node downloads the Apache-2.0 Depth Anything V2 Small model
into `ComfyUI/models/dlssnr/`. Motion does not download or load a neural model.

The node looks for the runtime in this order:

1. `runtime_directory` in the node (folder or full DLL path);
2. the `COMFYUI_DLSSNR_RUNTIME` environment variable;
3. this custom node's `runtime/` directory;
4. the newest `%LOCALAPPDATA%/RHI/DLSS-NR/*/` directory, when RHI is installed.

## Build the native backend

From PowerShell:

```powershell
./native/build_backend.ps1
```

The script obtains the official NVIDIA DLSS 310.7.0 SDK from
<https://github.com/NVIDIA/DLSS> when it is not already cached, then creates
`bin/dlssnr_worker.exe`. To use an existing SDK tree:

```powershell
./native/build_backend.ps1 -NgxSdkPath C:\SDKs\DLSS-310.7.0
```

## Inputs and video behavior

- `independent frames` runs Depth Anything V2 Small separately on every image,
  normalizes each guide independently, and resets NGX history for every frame.
- `video sequence` is strictly causal. For frame N, Depth Anything sees only N.
  The native D3D12 worker compares N with its previous luminance pyramid and
  produces current-to-previous pixel motion immediately before NGX. NGX keeps
  its own internal history. No future frame, full-video normalization, or
  multi-frame depth window is used.
- Motion is a GPU-resident coarse-to-fine patch-matching pipeline. It searches
  from 1/128 through 1/64, 1/32, 1/16, and 1/8 resolution, applies median and
  confidence-weighted depth-aware filtering, then expands to full-resolution
  `RG16_FLOAT` pixel vectors. `balanced`, `high`, and `maximum` select increasing
  coarse search radii.
- There are no depth or motion-vector sockets to configure. The node constructs
  both guide buffers internally and sends each frame to NGX immediately.
- `motion_quality` controls the shader search radius. The confidence threshold
  rejects vectors that fail pattern, spatial-consistency, and temporal checks.
- `depth_temporal_stability` smooths only the causal running normalization range;
  it does not feed previous or future images into the depth network.
- `scene_cut_sensitivity` controls when DLSS history and depth-range history reset.
- `depth_preview`, `motion_preview`, and `motion_confidence` are inspectable IMAGE
  outputs generated from the exact guide arrays sent to the native worker. They
  are allocated only when connected, so unused diagnostics consume no video-sized
  batch. Depth is grayscale; motion hue shows direction and brightness magnitude.
- `adapter_index = -1` automatically selects the first suitable NVIDIA adapter.
  Non-negative indices select among NVIDIA adapters in high-performance order.

DLSS Neural Rendering was designed for raster inputs with true depth and motion.
Unity obtains those buffers directly from its scene renderer. An encoded video
contains neither buffer, so this node reconstructs depth with Depth Anything and
motion with the native D3D12 optical-flow shader rather than substituting flat or
zero placeholders.

## Technical notes

Color and output use `RGBA16_FLOAT`, motion uses `RG16_FLOAT`, and depth uses
`R32_FLOAT`. Only one frame's color and depth are staged in shared memory at a
time. Motion, its luminance pyramid, and its one-frame history remain on the D3D12
GPU. Motion/confidence are read back only when their diagnostic outputs are
connected. The worker keeps one D3D12 device and one feature instance alive for
the sequence, exactly so both flow and NGX history persist between frames.

The feature-18 contract is experimental and not part of the stable public NGX
API. Runtime 310.x is the currently tested ABI. The architecture is informed by
the public direct-D3D12 `UnityDLSSNR` proof of concept and NVIDIA's public NGX
SDK. The worker links the public NGX SDK under NVIDIA's SDK terms; see
`THIRD_PARTY_NOTICES.md`. The proprietary experimental runtime DLL is not
redistributed.

## Licensing

Original source code is available under the MIT License. The prebuilt worker
also incorporates NVIDIA NGX SDK object code and remains subject to NVIDIA's
RTX SDK terms. Depth model weights and all other referenced projects retain
their own licenses. See `THIRD_PARTY_NOTICES.md` before redistributing a build.

References:

- <https://github.com/Kuan-Mi/UnityDLSSNR>
- <https://github.com/DepthAnything/Depth-Anything-V2>
- <https://github.com/umar-afzaal/LumeniteFX>
- <https://gpuopen.com/manuals/fidelityfx_sdk/techniques/optical-flow/>
- <https://github.com/jlrouzies-fr/DLSS5-Feeder>
- <https://github.com/NVIDIA/DLSS>
- <https://www.nvidia.com/en-us/geforce/news/dlss5-breakthrough-in-visual-fidelity-for-games/>

