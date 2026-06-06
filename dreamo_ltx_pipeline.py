import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageOps


class DreamoLtxPipeline:
    """
    Dreamo adapter around the real LTX 2.3 pipeline.

    smoke mode:
      Creates a simple silent MP4 from the input image.
      This tests Runpod + signed URLs only.
      It does not use LTX.

    native mode:
      Real LTX generation.
      Developer must wire the actual Dreamo native LTX code.
    """

    def __init__(self, model_dir: str, mode: str = "smoke"):
        self.model_dir = model_dir
        self.mode = mode
        self.loaded_at = time.time()
        self.native_pipe = None

        if self.mode == "native":
            self._load_native_pipeline()
        elif self.mode == "smoke":
            print("[DreamoLtxPipeline] Smoke mode enabled. No LTX model will be loaded.")
        else:
            raise ValueError(f"Unknown DREAMO_PIPELINE_MODE: {self.mode}")

    def _load_native_pipeline(self):
        """
        TODO FOR DEVELOPER:

        Replace this placeholder with the real Dreamo native LTX loading code.

        Required:
        - Load only from self.model_dir.
        - Do not download from Hugging Face at runtime.
        - Use the Dreamo runtime bundle cached by Runpod.
        - Load LTX distilled checkpoint.
        - Load spatial upscaler if required.
        - Load Gemma text encoder/tokenizer/config files.
        - Keep fixed settings:
            internal render: 832x512
            final output: 832x480
            fps: 24
            frames: 49 / 73 / 97
        """

        raise RuntimeError(
            "Native Dreamo LTX pipeline is not wired yet. "
            "Use DREAMO_PIPELINE_MODE=smoke for signed URL testing, "
            "or implement _load_native_pipeline() and _generate_native_mp4()."
        )

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
        TODO FOR DEVELOPER:

        Replace this with the real LTX generation.

        Required final behavior:
        - Generate internally at 832x512.
        - Center-crop final video to 832x480.
        - Save final silent MP4 to output_path.
        - No audio stream.
        """

        raise RuntimeError(
            "Native generation is not implemented yet. "
            "Implement _generate_native_mp4() with the real Dreamo LTX pipeline."
        )

    def _generate_smoke_mp4(
        self,
        image_path: str,
        output_path: str,
        num_frames: int,
        width: int,
        height: int,
        fps: int,
    ) -> Dict:
        """
        Creates a silent MP4 from the input image.
        This is only for testing the Runpod + signed URL flow.
        It is not real LTX generation.
        """

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
            "note": "Static test video only. This is not LTX output."
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


def crop_internal_832x512_to_final_832x480(input_path: str, output_path: str, fps: int):
    """
    Crops internal 832x512 video to final 832x480.
    512 - 480 = 32px.
    Center crop means remove 16px from top and 16px from bottom.
    """

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"crop=832:480:0:16,fps={fps}",
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
