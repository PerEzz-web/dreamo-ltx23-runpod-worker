import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageOps


def _unique_sorted(paths):
    seen = set()
    out = []
    for p in paths:
        p = Path(p)
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def _first_file(base: Path, patterns, required_name: str) -> str:
    matches = []

    for pattern in patterns:
        matches.extend(base.glob(pattern))
        matches.extend(base.rglob(pattern))

    matches = [
        p for p in _unique_sorted(matches)
        if p.is_file()
    ]

    if not matches:
        raise RuntimeError(
            f"Could not find {required_name} under {base}. "
            f"Tried patterns: {patterns}"
        )

    return str(matches[0])


def _first_dir(base: Path, patterns, required_name: str) -> str:
    matches = []

    for pattern in patterns:
        matches.extend(base.glob(pattern))
        matches.extend(base.rglob(pattern))

    matches = [
        p for p in _unique_sorted(matches)
        if p.is_dir()
    ]

    if not matches:
        raise RuntimeError(
            f"Could not find {required_name} under {base}. "
            f"Tried patterns: {patterns}"
        )

    return str(matches[0])


def discover_runtime_files(model_dir: str) -> Dict[str, str]:
    """
    Finds the files inside the cached Dreamo HF model snapshot.

    Preferred production checkpoint:
      checkpoints/ltx-2.3-22b-distilled-1.1.safetensors

    This is the BF16 distilled checkpoint. We then use:
      LTX_QUANTIZATION=fp8-cast

    Avoid using the FP8 checkpoint with this native loader path because it failed with:
      KeyError: attn1.to_gate_logits.input_scale
    """

    base = Path(model_dir)

    if not base.exists():
        raise RuntimeError(f"model_dir does not exist: {model_dir}")

    # Prefer explicit env var, defaulting to BF16 distilled checkpoint.
    checkpoint_glob = os.environ.get(
        "LTX_CHECKPOINT_GLOB",
        "checkpoints/ltx-2.3-22b-distilled-1.1.safetensors",
    )

    checkpoint_candidates = []
    checkpoint_candidates.extend(base.glob(checkpoint_glob))
    checkpoint_candidates.extend(base.rglob(checkpoint_glob))

    # Fallbacks: still prefer non-FP8 distilled files.
    if not checkpoint_candidates:
        fallback_patterns = [
            "checkpoints/*distilled-1.1.safetensors",
            "**/*distilled-1.1.safetensors",
            "checkpoints/*distilled*.safetensors",
            "**/*distilled*.safetensors",
        ]

        for pattern in fallback_patterns:
            checkpoint_candidates.extend(base.glob(pattern))
            checkpoint_candidates.extend(base.rglob(pattern))

    checkpoint_candidates = [
        p for p in _unique_sorted(checkpoint_candidates)
        if p.is_file()
        and "lora" not in p.name.lower()
        and "fp8" not in p.name.lower()
    ]

    # Only allow FP8 checkpoint if explicitly requested.
    if not checkpoint_candidates and os.environ.get("ALLOW_FP8_CHECKPOINT", "0") == "1":
        for pattern in [
            "checkpoints/*distilled-fp8*.safetensors",
            "**/*distilled-fp8*.safetensors",
        ]:
            checkpoint_candidates.extend(base.glob(pattern))
            checkpoint_candidates.extend(base.rglob(pattern))

        checkpoint_candidates = [
            p for p in _unique_sorted(checkpoint_candidates)
            if p.is_file() and "lora" not in p.name.lower()
        ]

    if not checkpoint_candidates:
        available = []
        for p in base.rglob("*.safetensors"):
            available.append(str(p.relative_to(base)))
        available = sorted(available)[:100]

        raise RuntimeError(
            f"Could not find compatible BF16 LTX distilled checkpoint under {base}. "
            "Expected checkpoints/ltx-2.3-22b-distilled-1.1.safetensors. "
            f"Available safetensors files: {available}"
        )

    checkpoint_path = str(checkpoint_candidates[0])

    upscaler_path = _first_file(
        base,
        [
            "upscalers/*spatial-upscaler*.safetensors",
            "**/*spatial-upscaler*.safetensors",
        ],
        "spatial upscaler",
    )

    gemma_root = _first_dir(
        base,
        [
            "text_encoders/*gemma*",
            "**/*gemma*",
        ],
        "Gemma text encoder folder",
    )

    return {
        "model_dir": str(base),
        "checkpoint_path": checkpoint_path,
        "spatial_upsampler_path": upscaler_path,
        "gemma_root": gemma_root,
    }

def native_environment_check(model_dir: str) -> Dict:
    """
    Lightweight native check.
    It verifies:
      - model files exist
      - ltx_core / ltx_pipelines imports work
      - the ltx_pipelines.distilled module is available
    It does NOT load the full 22B model.
    """

    result = {
        "model_dir": model_dir,
        "model_dir_exists": Path(model_dir).exists(),
    }

    try:
        result["runtime_files"] = discover_runtime_files(model_dir)
        result["files_ok"] = True
    except Exception as exc:
        result["files_ok"] = False
        result["files_error"] = str(exc)

    code = (
        "import ltx_core\n"
        "import ltx_pipelines\n"
        "import ltx_pipelines.distilled\n"
        "print('ltx imports ok')\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )

    result["imports_ok"] = completed.returncode == 0
    result["import_output"] = completed.stdout[-4000:]

    return result


class DreamoLtxPipeline:
    """
    Dreamo adapter.

    smoke mode:
      Creates a simple silent MP4 from the input image.
      This tests Runpod + signed URLs only.

    native mode:
      Calls the official LTX distilled pipeline through:
        python -m ltx_pipelines.distilled

      The official pipeline generates internally at 832x512.
      Then this adapter center-crops to final 832x480 and removes audio.
    """

    def __init__(self, model_dir: str, mode: str = "smoke"):
        self.model_dir = model_dir
        self.mode = mode
        self.loaded_at = time.time()
        self.runtime_files: Optional[Dict[str, str]] = None

        if self.mode == "native":
            self._load_native_pipeline()
        elif self.mode == "smoke":
            print("[DreamoLtxPipeline] Smoke mode enabled. No LTX model will be loaded.")
        else:
            raise ValueError(f"Unknown DREAMO_PIPELINE_MODE: {self.mode}")

    def _load_native_pipeline(self):
        """
        This validates imports and model files.

        It intentionally does NOT load the huge model weights here.
        Actual LTX loading happens inside the subprocess during generation.
        This is simpler and safer for no-warm-worker Serverless testing.
        """

        check = native_environment_check(self.model_dir)

        if not check.get("files_ok"):
            raise RuntimeError(f"Native LTX file check failed: {check}")

        if not check.get("imports_ok"):
            raise RuntimeError(f"Native LTX import check failed: {check}")

        self.runtime_files = check["runtime_files"]

        print("[DreamoLtxPipeline] Native LTX environment ready:")
        print(json.dumps(self.runtime_files, indent=2))

    def generate_mp4(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        num_frames: int,
        seed: Optional[int],
        width: int,
        height: int,
        ltx_width: int,
        ltx_height: int,
        fps: int,
    ) -> Dict:
        if self.mode == "smoke":
            return self._generate_smoke_mp4(
                image_path=image_path,
                output_path=output_path,
                num_frames=num_frames,
                width=width,
                height=height,
                fps=fps,
            )

        return self._generate_native_mp4(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            num_frames=num_frames,
            seed=seed,
            width=width,
            height=height,
            ltx_width=ltx_width,
            ltx_height=ltx_height,
            fps=fps,
        )

    def _generate_native_mp4(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        num_frames: int,
        seed: Optional[int],
        width: int,
        height: int,
        ltx_width: int,
        ltx_height: int,
        fps: int,
    ) -> Dict:
        """
        Real LTX generation.

        Steps:
          1. Run official distilled LTX pipeline at 832x512.
          2. Save internal MP4.
          3. Center-crop to final 832x480.
          4. Remove audio with ffmpeg -an.
        """

        if self.runtime_files is None:
            self.runtime_files = discover_runtime_files(self.model_dir)

        runtime = self.runtime_files

        work_dir = Path(output_path).parent
        internal_output_path = str(work_dir / "internal_ltx_832x512.mp4")

        seed = int(seed if seed is not None else int(time.time()) % 2147483647)

        image_strength = os.environ.get("LTX_IMAGE_STRENGTH", "0.7")
        image_crf = os.environ.get("LTX_IMAGE_CRF", "33")
        offload_mode = os.environ.get("LTX_OFFLOAD_MODE", "cpu")
        quantization = os.environ.get("LTX_QUANTIZATION", "none").strip().lower()
        timeout_seconds = int(os.environ.get("LTX_SUBPROCESS_TIMEOUT_SECONDS", "1800"))

        cmd = [
            sys.executable,
            "-m",
            "ltx_pipelines.distilled",

            "--distilled-checkpoint-path",
            runtime["checkpoint_path"],

            "--spatial-upsampler-path",
            runtime["spatial_upsampler_path"],

            "--gemma-root",
            runtime["gemma_root"],

            "--prompt",
            prompt,

            "--output-path",
            internal_output_path,

            "--seed",
            str(seed),

            "--height",
            str(ltx_height),

            "--width",
            str(ltx_width),

            "--num-frames",
            str(num_frames),

            "--frame-rate",
            str(float(fps)),

            "--image",
            image_path,
            "0",
            image_strength,
            image_crf,

            "--offload",
            offload_mode,
        ]

        if quantization not in ("", "none", "null", "false", "0"):
            cmd.extend(["--quantization", quantization])

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )

        started = time.perf_counter()

        completed = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            env=env,
        )

        ltx_seconds = round(time.perf_counter() - started, 3)

        if completed.returncode != 0:
            raise RuntimeError(
                "LTX native generation failed.\n\n"
                f"Return code: {completed.returncode}\n\n"
                "Last logs:\n"
                f"{completed.stdout[-8000:]}"
            )

        if not Path(internal_output_path).is_file():
            raise RuntimeError(
                f"LTX command completed but internal MP4 was not created: {internal_output_path}"
            )

        internal_size = Path(internal_output_path).stat().st_size

        crop_started = time.perf_counter()
        crop_internal_to_final(
            input_path=internal_output_path,
            output_path=output_path,
            width=width,
            height=height,
            ltx_width=ltx_width,
            ltx_height=ltx_height,
            fps=fps,
        )
        crop_seconds = round(time.perf_counter() - crop_started, 3)

        if not Path(output_path).is_file():
            raise RuntimeError(f"Final MP4 was not created: {output_path}")

        final_size = Path(output_path).stat().st_size

        return {
            "mode": "native",
            "engine": "ltx_pipelines.distilled subprocess",
            "ltx_generation_seconds": ltx_seconds,
            "crop_finalize_seconds": crop_seconds,
            "internal_output_path": internal_output_path,
            "internal_mp4_size_bytes": internal_size,
            "final_mp4_size_bytes": final_size,
            "runtime_files": runtime,
            "settings": {
                "ltx_width": ltx_width,
                "ltx_height": ltx_height,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
                "offload_mode": offload_mode,
                "quantization": quantization,
                "image_strength": image_strength,
                "image_crf": image_crf,
            },
        }

    def _generate_smoke_mp4(
        self,
        image_path: str,
        output_path: str,
        num_frames: int,
        width: int,
        height: int,
        fps: int,
    ) -> Dict:
        tmp_dir = Path(output_path).parent / "smoke_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img = center_crop_resize(img, width, height)

        for i in range(num_frames):
            frame = img.copy()
            frame_path = tmp_dir / f"frame_{i:05d}.png"
            frame.save(frame_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%05d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]

        subprocess.check_call(cmd)

        return {
            "mode": "smoke",
            "note": "Static test video only. This is not LTX output.",
        }


def center_crop_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    target_ratio = width / height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_h = height
        new_w = int(round(height * src_ratio))
    else:
        new_w = width
        new_h = int(round(width / src_ratio))

    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)

    return image.crop((left, top, left + width, top + height))


def crop_internal_to_final(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    ltx_width: int,
    ltx_height: int,
    fps: int,
):
    """
    Crops internal 832x512 video to final 832x480.
    Also removes audio using -an.
    """

    crop_x = max(0, (ltx_width - width) // 2)
    crop_y = max(0, (ltx_height - height) // 2)

    vf = f"crop={width}:{height}:{crop_x}:{crop_y},fps={fps}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    subprocess.check_call(cmd)
