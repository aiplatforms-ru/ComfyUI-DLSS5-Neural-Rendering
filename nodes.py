from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
from multiprocessing import shared_memory
from pathlib import Path

import torch
import torch.nn.functional as F
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


_NODE_ROOT = Path(__file__).resolve().parent
_WORKER = _NODE_ROOT / "bin" / "dlssnr_worker.exe"
_RUNTIME_NAME = "nvngx_dlssnr.dll"
_IMAGE_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _models_root() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.models_dir) / "dlssnr"
    except ImportError:
        root = _NODE_ROOT / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _inference_device(memory_required: int) -> torch.device:
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        # Do not unload every ComfyUI model: that can move tens of gigabytes
        # from VRAM into system RAM/pagefile. Ask ComfyUI to make only the
        # amount needed by the two small causal guide networks and NGX.
        model_management.free_memory(memory_required, device)
        return device
    except (ImportError, AttributeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _release_inference_memory() -> None:
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except (ImportError, AttributeError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _progress(total: int):
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(total)
    except (ImportError, TypeError):
        return None


def _progress_step(progress) -> None:
    if progress is not None:
        progress.update(1)


def _version_key(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", path.name)
    return tuple(int(number) for number in numbers) if numbers else (0,)


def _find_runtime(configured_path: str) -> Path:
    candidates: list[Path] = []
    if configured_path.strip():
        candidates.append(Path(os.path.expandvars(configured_path.strip())).expanduser())

    environment_path = os.environ.get("COMFYUI_DLSSNR_RUNTIME", "").strip()
    if environment_path:
        candidates.append(Path(os.path.expandvars(environment_path)).expanduser())

    candidates.append(_NODE_ROOT / "runtime")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        rhi_root = Path(local_app_data) / "RHI" / "DLSS-NR"
        if rhi_root.is_dir():
            candidates.extend(
                sorted(
                    (entry for entry in rhi_root.iterdir() if entry.is_dir()),
                    key=_version_key,
                    reverse=True,
                )
            )

    checked: list[str] = []
    for candidate in candidates:
        dll = candidate if candidate.name.lower() == _RUNTIME_NAME else candidate / _RUNTIME_NAME
        checked.append(str(dll))
        if dll.is_file():
            return dll.resolve()

    locations = "\n  - ".join(checked)
    raise FileNotFoundError(
        f"{_RUNTIME_NAME} was not found. Put the NVIDIA-signed DLL in "
        f"'{_NODE_ROOT / 'runtime'}', set COMFYUI_DLSSNR_RUNTIME, or enter its folder "
        f"in the node. Checked:\n  - {locations}"
    )


def _multiple_of(value: float, multiple: int, minimum: int) -> int:
    return max(minimum, int(round(value / multiple)) * multiple)


def _motion_preview(motion: torch.Tensor) -> torch.Tensor:
    flow = motion.float()
    magnitude = torch.linalg.vector_norm(flow, dim=-1)
    sample = magnitude.flatten()
    if sample.numel() > 1_000_000:
        sample = sample[:: (sample.numel() + 999_999) // 1_000_000]
    scale = torch.quantile(sample, 0.98).clamp_min(1.0)
    value = (magnitude / scale).clamp(0.0, 1.0)
    hue = torch.remainder(torch.atan2(flow[..., 1], flow[..., 0]) / (2.0 * torch.pi), 1.0)
    hue_sector = hue * 6.0
    sector = torch.floor(hue_sector).to(torch.int64) % 6
    fraction = hue_sector - torch.floor(hue_sector)
    zero = torch.zeros_like(value)
    rising = value * fraction
    falling = value * (1.0 - fraction)
    red = torch.where((sector == 0) | (sector == 5), value, zero)
    green = torch.where(
        (sector == 1) | (sector == 2),
        value,
        torch.where(sector == 0, rising, torch.where(sector == 3, falling, zero)),
    )
    blue = torch.where(
        (sector == 3) | (sector == 4),
        value,
        torch.where(sector == 2, rising, torch.where(sector == 5, falling, zero)),
    )
    red = torch.where(sector == 1, falling, torch.where(sector == 4, rising, red))
    return torch.stack((red, green, blue), dim=-1).clamp_(0.0, 1.0)


def _connected_output_indices(hidden) -> set[int]:
    """Find which outputs of this node are referenced by the current prompt."""
    if hidden is None or hidden.prompt is None or hidden.unique_id is None:
        return set()
    prompt = hidden.prompt
    if not isinstance(prompt, dict):
        return set()
    source_id = str(hidden.unique_id)
    connected: set[int] = set()
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and str(value[0]) == source_id
                and isinstance(value[1], int)
            ):
                connected.add(value[1])
    return connected


def _prepare_frame(
    images: torch.Tensor,
    frame_index: int,
    rgba: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prepare one frame only; never materialize a second full video batch."""
    source = images[frame_index].detach()
    if source.device.type != "cpu" or source.dtype != torch.float32:
        source = source.to(device="cpu", dtype=torch.float32)
    rgb = source[..., :3]
    if rgb.shape[-1] == 1:
        rgb = rgb.expand(*rgb.shape[:-1], 3)
    rgba[..., :3].copy_(rgb)
    rgba[..., 3].fill_(1.0)
    alpha = source[..., 3] if source.shape[-1] == 4 else None
    return rgb, alpha


class _CausalGuideEstimator:
    """Current-frame depth and a tiny causal scene-cut detector.

    Motion is deliberately not estimated in Python. The native D3D12 worker
    reconstructs current-to-previous flow and keeps its one-frame history on
    the GPU, immediately before the NGX dispatch.
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        temporal_mode: str,
        scene_cut_sensitivity: float,
        depth_range_stability: float,
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError(
                "Neural depth requires transformers. Install this node's requirements.txt "
                "in the ComfyUI Python environment and retry."
            ) from error

        self.height = height
        self.width = width
        self.video = temporal_mode == "video sequence"
        self.scene_cut_sensitivity = scene_cut_sensitivity
        self.depth_range_stability = depth_range_stability
        self.device = _inference_device(2 * 1024**3)
        self.previous_scene_luma: torch.Tensor | None = None
        self.depth_low: float | None = None
        self.depth_high: float | None = None

        cache_dir = _models_root() / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.depth_processor = AutoImageProcessor.from_pretrained(
            _IMAGE_DEPTH_MODEL,
            cache_dir=str(cache_dir),
        )
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            _IMAGE_DEPTH_MODEL,
            cache_dir=str(cache_dir),
        ).eval().to(self.device)

    def close(self) -> None:
        self.previous_scene_luma = None
        self.depth_model = None
        self.depth_processor = None
        _release_inference_memory()

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _scene_reset(self, rgb: torch.Tensor) -> bool:
        if not self.video:
            return True
        current = F.interpolate(
            rgb.permute(2, 0, 1).unsqueeze(0),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        luma = (
            current[0].mul(0.2126)
            + current[1].mul(0.7152)
            + current[2].mul(0.0722)
        ).contiguous()
        previous = self.previous_scene_luma
        self.previous_scene_luma = luma
        if previous is None:
            return True

        raw_error = (luma - previous).abs().mean().item()
        structural_error = (
            (luma - luma.mean()) - (previous - previous.mean())
        ).abs().mean().item()
        raw_threshold = 0.30 - 0.14 * self.scene_cut_sensitivity
        structural_threshold = 0.22 - 0.10 * self.scene_cut_sensitivity
        return raw_error > raw_threshold and structural_error > structural_threshold

    def _depth(self, rgb: torch.Tensor, reset: bool) -> torch.Tensor:
        image = rgb.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        inputs = self.depth_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode(), self._autocast():
            prediction = self.depth_model(pixel_values=pixel_values).predicted_depth[0].float()
        prediction = torch.nan_to_num(prediction)
        sample = prediction.flatten()
        if sample.numel() > 1_000_000:
            sample = sample[:: (sample.numel() + 999_999) // 1_000_000]
        current_low, current_high = torch.quantile(
            sample,
            torch.tensor([0.02, 0.98], device=sample.device),
        ).tolist()
        if reset or self.depth_low is None or self.depth_high is None:
            low, high = current_low, current_high
        else:
            history = self.depth_range_stability
            low = current_low * (1.0 - history) + self.depth_low * history
            high = current_high * (1.0 - history) + self.depth_high * history
        self.depth_low, self.depth_high = low, high
        if high - low < 1e-6:
            normalized = torch.full_like(prediction, 0.5)
        else:
            normalized = prediction.sub(low).div(high - low).clamp_(0.0, 1.0)
        depth = F.interpolate(
            normalized[None, None],
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        return depth.to(device="cpu", dtype=torch.float32).contiguous()

    def process(self, rgb: torch.Tensor) -> tuple[torch.Tensor, bool]:
        reset = self._scene_reset(rgb)
        return self._depth(rgb, reset), reset


class _WorkerSession:
    def __init__(
        self,
        *,
        runtime: Path,
        width: int,
        height: int,
        adapter_index: int,
        preset: int,
        style: int,
        intensity: float,
        tone: float,
        structure: float,
        skin: float,
        auto_mask: bool,
        ui_correction: bool,
        depth_inverted: bool,
        motion_quality: str,
        motion_confidence: float,
        diagnostics: bool,
    ) -> None:
        if os.name != "nt":
            raise RuntimeError("DLSS Neural Rendering requires Windows and Direct3D 12.")
        if not _WORKER.is_file():
            raise FileNotFoundError(
                f"Native backend is missing: {_WORKER}. Run native/build_backend.ps1 first."
            )

        pixels = width * height
        self._pixels = pixels
        self._diagnostics = diagnostics
        self._input_size = pixels * (8 + 4)
        self._output_size = pixels * (8 + (6 if diagnostics else 0))
        self._input = shared_memory.SharedMemory(create=True, size=self._input_size)
        self._output = shared_memory.SharedMemory(create=True, size=self._output_size)
        self._process: subprocess.Popen[str] | None = None

        command = [
            str(_WORKER),
            "--runtime-dll", str(runtime),
            "--input-map", self._input.name,
            "--output-map", self._output.name,
            "--width", str(width),
            "--height", str(height),
            "--adapter", str(adapter_index),
            "--preset", str(preset),
            "--style", str(style),
            "--intensity", str(intensity),
            "--tone", str(tone),
            "--structure", str(structure),
            "--skin", str(skin),
            "--auto-mask", "1" if auto_mask else "0",
            "--ui-correction", "1" if ui_correction else "0",
            "--depth-inverted", "1" if depth_inverted else "0",
            "--motion-quality", str(["balanced", "high", "maximum"].index(motion_quality)),
            "--motion-confidence", str(motion_confidence),
            "--diagnostics", "1" if diagnostics else "0",
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=creation_flags,
            )
            ready = self._read_line()
            if not ready.startswith("READY "):
                raise RuntimeError(f"DLSS-NR backend did not start: {ready}")
            self.adapter_name = ready.removeprefix("READY ").strip()
        except Exception:
            self.close()
            raise

    def _read_line(self) -> str:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("DLSS-NR backend is not running.")
        line = self._process.stdout.readline().strip()
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"DLSS-NR backend stopped unexpectedly (exit code {code}).")
        if line.startswith("ERROR "):
            raise RuntimeError(line.removeprefix("ERROR "))
        return line

    def process_into(
        self,
        color: torch.Tensor,
        depth: torch.Tensor,
        destination: torch.Tensor,
        motion_destination: torch.Tensor | None = None,
        confidence_destination: torch.Tensor | None = None,
        *,
        reset: bool,
    ) -> None:
        color_bytes = memoryview(color.numpy()).cast("B")
        depth_bytes = memoryview(depth.numpy()).cast("B")
        color_end = color_bytes.nbytes
        if color_end + depth_bytes.nbytes != self._input_size:
            raise RuntimeError("Internal DLSS-NR shared-memory layout mismatch.")

        self._input.buf[:color_end] = color_bytes
        self._input.buf[color_end:self._input_size] = depth_bytes

        if self._process is None or self._process.stdin is None:
            raise RuntimeError("DLSS-NR backend is not running.")
        self._process.stdin.write(f"PROCESS {1 if reset else 0}\n")
        self._process.stdin.flush()
        response = self._read_line()
        if response != "OK":
            raise RuntimeError(f"Unexpected DLSS-NR backend response: {response}")

        # Copy directly into the final preallocated video tensor. The shared
        # view becomes invalid on the next dispatch, but no second frame-sized
        # Python bytes/float32 allocation is created here.
        values = torch.frombuffer(
            self._output.buf,
            dtype=torch.float16,
            count=self._output_size // 2,
        )
        color_values = values[: self._pixels * 4].reshape(color.shape)
        destination.copy_(color_values[..., :3])
        if self._diagnostics:
            motion_start = self._pixels * 4
            confidence_start = motion_start + self._pixels * 2
            if motion_destination is not None:
                motion_destination.copy_(
                    values[motion_start:confidence_start].reshape(*color.shape[:2], 2)
                )
            if confidence_destination is not None:
                confidence_destination.copy_(
                    values[confidence_start:].reshape(*color.shape[:2])
                )
        del values, color_values, color_bytes, depth_bytes

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                    process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            finally:
                if process.stdin:
                    process.stdin.close()
                if process.stdout:
                    process.stdout.close()

        for mapping in (getattr(self, "_input", None), getattr(self, "_output", None)):
            if mapping is not None:
                mapping.close()
                try:
                    mapping.unlink()
                except FileNotFoundError:
                    pass

    def __enter__(self) -> "_WorkerSession":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class Dlss5NeuralRendering(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Dlss5NeuralRendering",
            display_name="DLSS 5 Neural Rendering",
            category="image/postprocessing",
            description=(
                "Runs NVIDIA DLSS Neural Rendering feature 18 as a headless D3D12 post-process. "
                "An IMAGE batch can be treated as independent pictures or a temporal video sequence."
            ),
            search_aliases=["dlss", "dlss 5", "neural rendering", "nvidia", "video enhance"],
            inputs=[
                io.Image.Input("images", tooltip="Image or video-frame batch in display-referred RGB."),
                io.Combo.Input(
                    "temporal_mode",
                    options=["independent frames", "video sequence"],
                    default="independent frames",
                    tooltip=(
                        "Independent resets NGX for every image. Video is strictly causal: depth uses "
                        "only the current frame, motion uses current and previous, and NGX keeps history."
                    ),
                ),
                io.Combo.Input(
                    "motion_quality",
                    options=["balanced", "high", "maximum"],
                    default="balanced",
                    tooltip=(
                        "Controls the native D3D12 pyramidal search radius. Motion is calculated "
                        "on the GPU immediately before NGX; no RAFT model or CPU round-trip is used."
                    ),
                ),
                io.Float.Input(
                    "motion_confidence_threshold",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Shader-estimated motion confidence below this value is rejected. "
                        "Use 0 to keep every estimated vector."
                    ),
                ),
                io.Float.Input(
                    "depth_temporal_stability",
                    default=0.35,
                    min=0.0,
                    max=0.75,
                    step=0.01,
                    tooltip=(
                        "Causal smoothing of Depth Anything's normalization range using past values only. "
                        "The depth network itself always receives one current frame."
                    ),
                ),
                io.Float.Input(
                    "scene_cut_sensitivity",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Higher values reset DLSS history more readily at shot changes.",
                ),
                io.Combo.Input(
                    "preset", options=["default", "preset 1", "preset 2", "preset 3"], default="default"
                ),
                io.Combo.Input("style", options=["default", "natural", "cinematic"], default="default"),
                io.Float.Input("intensity", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Float.Input("local_tone", default=1.0, min=0.0, max=2.0, step=0.05, advanced=True),
                io.Float.Input("local_structure", default=1.0, min=0.0, max=2.0, step=0.05, advanced=True),
                io.Float.Input("skin_structure", default=-1.0, min=-1.0, max=2.0, step=0.05, advanced=True),
                io.Boolean.Input("auto_mask", default=False, advanced=True),
                io.Boolean.Input("ui_correction", default=False, advanced=True),
                io.Boolean.Input(
                    "depth_inverted",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "True means near=1/far=0 (reversed-Z), matching the node's automatic depth. "
                        "Disable only when a connected depth input uses near=0/far=1."
                    ),
                ),
                io.Int.Input(
                    "adapter_index",
                    default=-1,
                    min=-1,
                    max=15,
                    step=1,
                    advanced=True,
                    tooltip="-1 selects the first suitable NVIDIA GPU; 0..15 selects a NVIDIA adapter explicitly.",
                ),
                io.String.Input(
                    "runtime_directory",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Folder containing nvngx_dlssnr.dll, or the full DLL path. Empty uses the node's "
                        "runtime folder, COMFYUI_DLSSNR_RUNTIME, then an installed RHI cache."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Image.Output("depth_preview", tooltip="Exact depth guides; only allocated when connected."),
                io.Image.Output("motion_preview", tooltip="Exact motion guides; only allocated when connected."),
                io.Image.Output(
                    "motion_confidence",
                    tooltip="D3D12 optical-flow confidence; only read back when connected.",
                ),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        temporal_mode: str,
        motion_quality: str,
        motion_confidence_threshold: float,
        depth_temporal_stability: float,
        scene_cut_sensitivity: float,
        preset: str,
        style: str,
        intensity: float,
        local_tone: float,
        local_structure: float,
        skin_structure: float,
        auto_mask: bool,
        ui_correction: bool,
        depth_inverted: bool,
        adapter_index: int,
        runtime_directory: str = "",
    ) -> io.NodeOutput:
        runtime = _find_runtime(runtime_directory)
        if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"images must be BHWC with 1, 3, or 4 channels; got {tuple(images.shape)}."
            )
        frames, height, width, source_channels = images.shape
        preset_id = ["default", "preset 1", "preset 2", "preset 3"].index(preset)
        style_id = ["default", "natural", "cinematic"].index(style)
        output_channels = 4 if source_channels == 4 else 3
        # NGX already returns FP16. Keeping the only result batch in FP16 avoids
        # manufacturing a second full-size float32 copy of the video.
        output = torch.empty(
            (frames, height, width, output_channels),
            device="cpu",
            dtype=torch.float16,
        )
        connected = _connected_output_indices(getattr(cls, "hidden", None))
        depth_debug = (
            torch.empty((frames, height, width, 3), dtype=torch.float16)
            if 1 in connected else None
        )
        motion_debug = (
            torch.empty((frames, height, width, 3), dtype=torch.float16)
            if 2 in connected else None
        )
        confidence_debug = (
            torch.empty((frames, height, width, 3), dtype=torch.float16)
            if 3 in connected else None
        )
        diagnostics = motion_debug is not None or confidence_debug is not None
        motion_raw = (
            torch.empty((height, width, 2), dtype=torch.float16) if diagnostics else None
        )
        confidence_raw = (
            torch.empty((height, width), dtype=torch.float16) if diagnostics else None
        )
        rgba = torch.empty((height, width, 4), dtype=torch.float16)
        progress = _progress(frames)
        resets: list[int] = []
        depth_min, depth_max = float("inf"), float("-inf")
        motion_mean_sum = 0.0
        motion_max = 0.0
        motion_nonzero_sum = 0.0
        confidence_mean_sum = 0.0

        estimator = _CausalGuideEstimator(
            height=height,
            width=width,
            temporal_mode=temporal_mode,
            scene_cut_sensitivity=scene_cut_sensitivity,
            depth_range_stability=depth_temporal_stability,
        )
        try:
            with _WorkerSession(
                runtime=runtime,
                width=width,
                height=height,
                adapter_index=adapter_index,
                preset=preset_id,
                style=style_id,
                intensity=intensity,
                tone=local_tone,
                structure=local_structure,
                skin=skin_structure,
                auto_mask=auto_mask,
                ui_correction=ui_correction,
                depth_inverted=depth_inverted,
                motion_quality=motion_quality,
                motion_confidence=motion_confidence_threshold,
                diagnostics=diagnostics,
            ) as worker:
                for frame_index in range(frames):
                    rgb, alpha = _prepare_frame(images, frame_index, rgba)
                    depth, reset = estimator.process(rgb)
                    if not depth_inverted:
                        depth = 1.0 - depth
                    if reset:
                        resets.append(frame_index)
                    destination = output[frame_index, ..., :3]
                    worker.process_into(
                        rgba,
                        depth.unsqueeze(-1).contiguous(),
                        destination,
                        motion_raw,
                        confidence_raw,
                        reset=reset,
                    )
                    torch.nan_to_num_(destination, nan=0.0, posinf=1.0, neginf=0.0)
                    destination.clamp_(0.0, 1.0)
                    if alpha is not None:
                        output[frame_index, ..., 3].copy_(alpha)

                    if depth_debug is not None:
                        depth_debug[frame_index].copy_(depth.unsqueeze(-1).expand(-1, -1, 3))
                    if motion_debug is not None and motion_raw is not None:
                        motion_debug[frame_index].copy_(_motion_preview(motion_raw))
                    if confidence_debug is not None and confidence_raw is not None:
                        confidence_debug[frame_index].copy_(
                            confidence_raw.unsqueeze(-1).expand(-1, -1, 3)
                        )

                    # Logging must not manufacture another full-resolution
                    # float32 motion image on every frame.
                    sampled_depth = depth[::8, ::8]
                    depth_min = min(depth_min, sampled_depth.min().item())
                    depth_max = max(depth_max, sampled_depth.max().item())
                    if motion_raw is not None and confidence_raw is not None:
                        sampled_motion = motion_raw[::8, ::8].float()
                        magnitude = torch.linalg.vector_norm(sampled_motion, dim=-1)
                        motion_mean_sum += magnitude.mean().item()
                        motion_max = max(motion_max, magnitude.max().item())
                        motion_nonzero_sum += (magnitude > 0.01).float().mean().item()
                        confidence_mean_sum += confidence_raw[::8, ::8].float().mean().item()
                    _progress_step(progress)
        finally:
            estimator.close()

        divisor = max(frames, 1)
        if diagnostics:
            logging.info(
                "[DLSS5-Neural-Rendering] causal guides: Depth Anything V2 Small %.4f..%.4f; "
                "D3D12 flow mean %.4f px max %.4f px nonzero %.2f%%; confidence %.4f; resets %s",
                depth_min,
                depth_max,
                motion_mean_sum / divisor,
                motion_max,
                motion_nonzero_sum * 100.0 / divisor,
                confidence_mean_sum / divisor,
                resets,
            )
        else:
            logging.info(
                "[DLSS5-Neural-Rendering] causal guides: Depth Anything V2 Small %.4f..%.4f; "
                "D3D12 flow remained GPU-resident; resets %s",
                depth_min,
                depth_max,
                resets,
            )
        dummy = torch.zeros((1, 1, 1, 3), dtype=torch.float16)
        return io.NodeOutput(
            output,
            depth_debug if depth_debug is not None else dummy,
            motion_debug if motion_debug is not None else dummy,
            confidence_debug if confidence_debug is not None else dummy,
        )


class DlssNeuralRenderingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Dlss5NeuralRendering]
