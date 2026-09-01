# Third-party notices

The native backend links the NVIDIA NGX SDK from the public
[NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) distribution.

> This software contains source code provided by NVIDIA Corporation.

Use and redistribution of the NVIDIA SDK portions are governed by the
[`NVIDIA RTX SDKs LICENSE`](https://github.com/NVIDIA/DLSS/blob/v310.7.0/LICENSE.txt)
supplied with SDK version 310.7.0. The MIT license in this repository does not
relicense NVIDIA code incorporated into the prebuilt worker. The experimental
`nvngx_dlssnr.dll` is not part of this custom node and is not redistributed.

The automatic depth path uses Depth Anything V2 Small, distributed under
[Apache-2.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE).
Its weights are downloaded from the official model repository on first use.

The native causal motion implementation is an independently written D3D12
coarse-to-fine optical-flow pipeline. Its design was informed by the LumeniteFX
QuantMotion/LumaFlow shader supplied for this private integration and by the
public AMD FidelityFX Optical Flow documentation. LumeniteFX is authored by
Afzaal (Kaidō) and distributed under the AGNYA license; no LumeniteFX source is
redistributed as a ReShade shader by this node.

