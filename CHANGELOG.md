# Changelog

## 0.2.0 - 2026-09-11

- Add 1-5 real chained DLSS Neural Rendering passes in one node.
- Recompute depth, motion, and NGX history for every pass.
- Add optional aspect-ratio-preserving input scaling by target megapixels.
- Replace Depth Anything V2 Small with ComfyUI-native Depth Anything 3 Mono Large.
- Reuse `models/geometry_estimation/depth_anything_3_mono_large.safetensors`
  without a separate Transformers download or model copy.

## 0.1.1 - 2026-09-11

- Store and discover `nvngx_dlssnr.dll` under `ComfyUI/models/dlssnr/`.
- Remove automatic dependency on the RHI application cache.
- Keep explicit path and environment-variable overrides for advanced setups.

## 0.1.0 - 2026-09-01

- Initial public experimental release.
- Headless Direct3D 12 DLSS Neural Rendering worker.
- Depth Anything V2 Small depth estimation.
- Causal GPU optical flow with confidence, scene-cut resets, and diagnostics.
- Independent-image and video-sequence processing modes.
