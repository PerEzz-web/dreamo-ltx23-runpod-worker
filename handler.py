import gc
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import runpod
import torch

from dreamo_ltx_pipeline import DreamoLtxPipeline


MODEL_NAME = os.environ.get("MODEL_NAME", "PerEzz/dreamo-ltx23-runtime-a6000")
HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")

DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "832"))
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "480"))
LTX_WIDTH = int(os.environ.get("LTX_WIDTH", "832"))
LTX_HEIGHT = int(os.environ.get("LTX_HEIGHT", "512"))
DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "24"))
DEFAULT_NUM_FRAMES = int(os.environ.get("DEFAULT_NUM_FRAMES", "73"))
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "49"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "97"))

VALID_NUM_FRAMES = {49, 73, 97}

_PIPELINE = None
_PIPELINE_INFO: Dict[str, Any] = {}


def now() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def bytes_to_gb(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 3), 3)


def gpu_max_memory_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return bytes_to_gb(torch.cuda.max_memory_allocated())


def gpu_snapshot() -> Dict[str, Any]:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "nvidia_smi": run_cmd([
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true"
        ]),
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated_gb"] = bytes_to_gb(torch.cuda.memory_allocated())
        info["memory_reserved_gb"] = bytes_to_gb(torch.cuda.memory_reserved())

    return info


def clean_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def resolve_snapshot_path(model_id: str) -> str:
    if "/" not in model_id:
        raise ValueError(f"MODEL_NAME must look like 'org/repo', got: {model_id}")

    org, repo = model_id.split("/", 1)

    expected_names = [
        f"models--{org}--{repo}",
        f"models--{org.lower()}--{repo}",
        f"models--{org}--{repo.lower()}",
        f"models--{org.lower()}--{repo.lower()}",
    ]

    tried_paths = []

    for folder_name in expected_names:
        model_root = os.path.join(HF_CACHE_ROOT, folder_name)
        tried_paths.append(model_root)

        refs_main = os.path.join(model_root, "refs", "main")
        snapshots_dir = os.path.join(model_root, "snapshots")

        if os.path.isfile(refs_main):
            with open(refs_main, "r", encoding="utf-8") as f:
                snapshot_hash = f.read().strip()

            candidate = os.path.join(snapshots_dir, snapshot_hash)

            if os.path.isdir(candidate):
                return candidate

        if os.path.isdir(snapshots_dir):
            versions = [
                d for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))
            ]

            versions.sort()

            if versions:
                return os.path.join(snapshots_dir, versions[0])

    available_root = []
    available_hf_root = []

    if os.path.isdir("/runpod-volume"):
        try:
            available_root = sorted(os.listdir("/runpod-volume"))[:50]
        except Exception as exc:
            available_root = [f"Could not list /runpod-volume: {exc}"]
    else:
        available_root = ["/runpod-volume does not exist"]

    if os.path.isdir(HF_CACHE_ROOT):
        try:
            available_hf_root = sorted(os.listdir(HF_CACHE_ROOT))[:50]
        except Exception as exc:
            available_hf_root = [f"Could not list HF_CACHE_ROOT: {exc}"]
    else:
        available_hf_root = [f"HF_CACHE_ROOT does not exist: {HF_CACHE_ROOT}"]

    raise RuntimeError(
        "Cached model not found.\n"
        f"MODEL_NAME={model_id}\n"
        f"HF_CACHE_ROOT={HF_CACHE_ROOT}\n"
        f"Tried paths={tried_paths}\n"
        f"/runpod-volume listing={available_root}\n"
        f"HF cache root listing={available_hf_root}\n"
        "This means the Runpod cached model is not mounted, not downloaded yet, "
        "or mounted in a different path."
    )


def get_model_dir() -> Tuple[str, Dict[str, Any]]:
    local_override = os.environ.get("LOCAL_MODEL_DIR")
    use_cache = os.environ.get("USE_RUNPOD_CACHE", "1") == "1"
    allow_download = os.environ.get("ALLOW_HF_DOWNLOAD", "0") == "1"

    if local_override:
        if not os.path.isdir(local_override):
            raise RuntimeError(f"LOCAL_MODEL_DIR does not exist: {local_override}")

        return local_override, {
            "source": "LOCAL_MODEL_DIR",
            "path": local_override,
        }

    if use_cache:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        path = resolve_snapshot_path(MODEL_NAME)

        return path, {
            "source": "runpod_cached_model",
            "model_name": MODEL_NAME,
            "path": path,
        }

    if allow_download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        return MODEL_NAME, {
            "source": "runtime_hf_download_dev_only",
            "model_name": MODEL_NAME,
        }

    # Smoke mode does not need a model directory.
    if os.environ.get("DREAMO_PIPELINE_MODE", "smoke") == "smoke":
        return "/tmp/no_model_needed_for_smoke_mode", {
            "source": "smoke_mode_no_model",
            "path": "/tmp/no_model_needed_for_smoke_mode",
        }

    raise RuntimeError(
        "No model source available. Use USE_RUNPOD_CACHE=1 in production."
    )


def get_pipeline() -> Tuple[DreamoLtxPipeline, Dict[str, Any]]:
    global _PIPELINE, _PIPELINE_INFO

    if _PIPELINE is not None:
        return _PIPELINE, _PIPELINE_INFO

    t0 = now()

    mode = os.environ.get("DREAMO_PIPELINE_MODE", "smoke")
    model_dir, model_info = get_model_dir()

    pipeline = DreamoLtxPipeline(model_dir=model_dir, mode=mode)

    _PIPELINE = pipeline
    _PIPELINE_INFO = {
        "model": model_info,
        "mode": mode,
        "load_seconds": elapsed(t0),
        "gpu_after_load": gpu_snapshot(),
    }

    return _PIPELINE, _PIPELINE_INFO


def preflight() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "success",
        "model_name": MODEL_NAME,
        "hf_cache_root": HF_CACHE_ROOT,
        "use_runpod_cache": os.environ.get("USE_RUNPOD_CACHE", "1"),
        "allow_hf_download": os.environ.get("ALLOW_HF_DOWNLOAD", "0"),
        "dreamo_pipeline_mode": os.environ.get("DREAMO_PIPELINE_MODE", "smoke"),
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "ltx_width": LTX_WIDTH,
        "ltx_height": LTX_HEIGHT,
        "fps": DEFAULT_FPS,
        "valid_num_frames": sorted(list(VALID_NUM_FRAMES)),
        "gpu": gpu_snapshot(),
        "cache_debug": {
            "exists_runpod_volume": os.path.isdir("/runpod-volume"),
            "exists_hf_cache_root": os.path.isdir(HF_CACHE_ROOT),
            "hf_home_env": os.environ.get("HF_HOME"),
            "hf_hub_cache_env": os.environ.get("HF_HUB_CACHE"),
            "transformers_cache_env": os.environ.get("TRANSFORMERS_CACHE"),
            "runpod_volume_listing": sorted(os.listdir("/runpod-volume"))[:50] if os.path.isdir("/runpod-volume") else [],
            "hf_cache_root_listing": sorted(os.listdir(HF_CACHE_ROOT))[:50] if os.path.isdir(HF_CACHE_ROOT) else [],
        },
    }

    try:
        model_dir, model_info = get_model_dir()
        result["model_info"] = model_info

        if os.path.isdir(model_dir):
            total_size = 0
            file_count = 0
            sample_files = []

            for root, _, files in os.walk(model_dir):
                for file in files:
                    file_count += 1
                    p = os.path.join(root, file)

                    try:
                        total_size += os.path.getsize(p)
                    except OSError:
                        pass

                    if len(sample_files) < 30:
                        sample_files.append(os.path.relpath(p, model_dir))

            result["cached_file_count"] = file_count
            result["cached_size_gb"] = bytes_to_gb(total_size)
            result["sample_files"] = sample_files

    except Exception as exc:
        result["model_error"] = str(exc)

    return result


def require_text(job_input: Dict[str, Any], key: str) -> str:
    value = job_input.get(key)

    if value is None or str(value).strip() == "":
        raise ValueError(f"{key} is required")

    return str(value).strip()


def parse_num_frames(value: Any) -> int:
    if value is None:
        value = DEFAULT_NUM_FRAMES

    try:
        num_frames = int(value)
    except Exception:
        raise ValueError("num_frames must be an integer: 49, 73, or 97")

    if num_frames not in VALID_NUM_FRAMES:
        raise ValueError("num_frames must be one of: 49, 73, 97")

    if num_frames < MIN_FRAMES or num_frames > MAX_FRAMES:
        raise ValueError(f"num_frames must be between {MIN_FRAMES} and {MAX_FRAMES}")

    return num_frames


def download_file(url: str, dest_path: str, timeout_seconds: int = 60):
    with requests.get(url, stream=True, timeout=(15, timeout_seconds)) as response:
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def upload_file_to_signed_url(local_path: str, upload_url: str):
    file_size = os.path.getsize(local_path)

    headers = {
        "content-type": "video/mp4",
        "content-length": str(file_size),
    }

    with open(local_path, "rb") as f:
        data = f.read()

    response = requests.put(
        upload_url,
        data=data,
        headers=headers,
        timeout=(15, 600),
    )

    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"Signed upload failed with HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )


def generate(job_input: Dict[str, Any]) -> Dict[str, Any]:
    total_start = now()

    job_id = str(job_input.get("job_id") or uuid.uuid4())

    prompt = require_text(job_input, "prompt")
    image_url = require_text(job_input, "image_url")
    upload_url = require_text(job_input, "upload_url")

    output_url = str(job_input.get("output_url") or "").strip() or None
    num_frames = parse_num_frames(job_input.get("num_frames"))

    seed = job_input.get("seed")

    if seed is None:
        seed = int(time.time() * 1000) % 2147483647
    else:
        seed = int(seed)

    work_dir = Path("/tmp/dreamo_ltx_jobs") / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = str(work_dir / "input_image.png")
    final_mp4_path = str(work_dir / "output_832x480.mp4")

    metrics: Dict[str, Any] = {}

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t = now()
        download_file(image_url, input_image_path)
        metrics["download_image_seconds"] = elapsed(t)

        t = now()
        pipeline, pipeline_info = get_pipeline()
        metrics["load_pipeline_seconds"] = pipeline_info.get("load_seconds", 0)

        pipeline_extra = pipeline.generate_mp4(
            image_path=input_image_path,
            prompt=prompt,
            output_path=final_mp4_path,
            num_frames=num_frames,
            seed=seed,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            ltx_width=LTX_WIDTH,
            ltx_height=LTX_HEIGHT,
            fps=DEFAULT_FPS,
        )

        metrics["generate_encode_and_finalize_seconds"] = elapsed(t)

        if not os.path.isfile(final_mp4_path):
            raise RuntimeError(f"Expected MP4 was not created: {final_mp4_path}")

        metrics["local_mp4_size_bytes"] = os.path.getsize(final_mp4_path)

        t = now()
        upload_file_to_signed_url(final_mp4_path, upload_url)
        metrics["upload_seconds"] = elapsed(t)

        metrics["total_seconds"] = elapsed(total_start)
        metrics["gpu_max_memory_gb"] = gpu_max_memory_gb()

        return {
            "status": "success",
            "job_id": job_id,
            "video_url": output_url,
            "seed": seed,
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "ltx_width": LTX_WIDTH,
            "ltx_height": LTX_HEIGHT,
            "fps": DEFAULT_FPS,
            "num_frames": num_frames,
            "estimated_duration_seconds": round(num_frames / DEFAULT_FPS, 3),
            "metrics": metrics,
            "pipeline": {
                "mode": os.environ.get("DREAMO_PIPELINE_MODE", "smoke"),
                "extra": pipeline_extra,
            },
        }

    except Exception:
        clean_cuda()
        raise


def handler(job):
    job_input = job.get("input", {}) or {}
    action = str(job_input.get("action") or "generate").lower()
    job_id = str(job_input.get("job_id") or "")

    try:
        if action == "health":
            return {
                "status": "success",
                "action": "health",
                "message": "Dreamo LTX Runpod worker is alive",
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "ltx_width": LTX_WIDTH,
                "ltx_height": LTX_HEIGHT,
                "fps": DEFAULT_FPS,
                "valid_num_frames": sorted(list(VALID_NUM_FRAMES)),
                "gpu": gpu_snapshot(),
            }

        if action == "preflight":
            return preflight()

        if action == "native_env_check":
            model_dir, model_info = get_model_dir()
            from dreamo_ltx_pipeline import native_environment_check

            return {
                "status": "success",
                "action": "native_env_check",
                "model_info": model_info,
                "native": native_environment_check(model_dir),
            }

        if action == "load_model":
            pipeline, info = get_pipeline()

            return {
                "status": "success",
                "action": "load_model",
                "pipeline_loaded": pipeline is not None,
                "pipeline_info": info,
            }

        if action == "generate":
            return generate(job_input)

        raise ValueError(
            "Unknown action. Production should omit action or use action=generate. "
            "Debug actions: health, preflight, load_model."
        )

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "job_id": job_id or None,
            "traceback": traceback.format_exc()[-4000:],
        }


if __name__ == "__main__":
    if "--test_input" in sys.argv:
        idx = sys.argv.index("--test_input")
        payload = json.loads(sys.argv[idx + 1])
        print(json.dumps(handler(payload), indent=2))
    else:
        runpod.serverless.start({"handler": handler})
