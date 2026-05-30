"""Synthetic container code image generator.

Produces realistic-looking images of ISO 6346 codes on container-side
backgrounds, with configurable noise, blur, rotation, and occlusion.

USE CASE: pipeline testing only. Generates images for end-to-end tests
of the OCR -> recovery -> database flow, where we control the ground
truth. Not for training the OCR model — synthetic-to-real domain gap is
a known issue.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .iso6346 import generate_valid_code


# Container side colors (RGB) — typical shipping container palette.
CONTAINER_COLORS: dict[str, tuple[int, int, int]] = {
    "rust_red": (139, 47, 39),
    "navy_blue": (30, 55, 90),
    "forest_green": (44, 85, 50),
    "industrial_grey": (95, 95, 100),
    "weathered_white": (210, 205, 195),
    "ochre": (170, 120, 50),
}

# Text color is usually high-contrast: white on dark, black on light.
TEXT_COLORS_DARK_BG = (240, 240, 235)
TEXT_COLORS_LIGHT_BG = (25, 25, 25)


@dataclass
class GenerationConfig:
    """Knobs for the image generator.

    All "intensity" values are in [0, 1]; 0 = clean, 1 = maximum effect.
    """
    width: int = 400
    height: int = 100
    noise_intensity: float = 0.0       # gaussian pixel noise
    blur_radius: float = 0.0           # gaussian blur radius in pixels
    rotation_degrees: float = 0.0      # rotation; 0 = horizontal
    brightness_jitter: float = 0.0     # ±intensity * 50 to base channel
    occlusion_intensity: float = 0.0   # 0 = no occlusion; 1 = up to ~25% covered
    add_streak_marks: bool = False     # rust streaks / weathering


@dataclass(frozen=True)
class GeneratedImage:
    """Output of the generator: image plus ground truth."""
    image: np.ndarray              # (H, W, 3) uint8 RGB
    true_code: str                 # the ISO 6346 code rendered in the image
    background_color_name: str     # which container color was used


class ContainerImageGenerator:
    """Renders synthetic ISO 6346 container code images.

    The generator is stateful with respect to its random number generator —
    pass a seeded random.Random for reproducibility (NFR: reproducible
    simulation given fixed seed).
    """

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        font_path: Optional[str] = None,
    ):
        """
        Args:
            rng: random.Random instance for reproducibility.
            font_path: path to a TTF font. Defaults to PIL's DejaVuSans-Bold
                if available, otherwise falls back to PIL's default font
                (which is small but usable for tests).
        """
        self._rng = rng or random.Random()
        self._font_path = font_path or self._find_default_font()

    @staticmethod
    def _find_default_font() -> Optional[str]:
        """Locate a usable bold font on the system."""
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                return path
        return None  # PIL will fall back to its default bitmap font

    def generate(
        self,
        config: Optional[GenerationConfig] = None,
        code_override: Optional[str] = None,
    ) -> GeneratedImage:
        """Generate one synthetic container code image.

        Args:
            config: generation parameters. Defaults to clean (no noise/blur).
            code_override: if set, render this code instead of a random one.
                Useful for end-to-end tests where the caller wants to verify
                a specific code is recognized.

        Returns:
            GeneratedImage with the rendered array and ground-truth code.
        """
        config = config or GenerationConfig()
        true_code = code_override or generate_valid_code(self._rng)

        # 1. Pick a container color.
        bg_name = self._rng.choice(list(CONTAINER_COLORS.keys()))
        bg_color = CONTAINER_COLORS[bg_name]

        # 2. Apply brightness jitter to the base color.
        if config.brightness_jitter > 0:
            jitter = int(self._rng.uniform(-50, 50) * config.brightness_jitter)
            bg_color = tuple(np.clip(c + jitter, 0, 255) for c in bg_color)

        # 3. Build the base canvas.
        img = Image.new("RGB", (config.width, config.height), bg_color)
        draw = ImageDraw.Draw(img)

        # 4. Add subtle texture (helps the image look less flat).
        self._add_texture(img, intensity=0.15)

        # 5. Optional weathering / streak marks.
        if config.add_streak_marks:
            self._draw_streaks(img, bg_color)

        # 6. Render the code text.
        text_color = self._pick_text_color(bg_color)
        font = self._load_font(config.height)
        self._draw_centered_text(draw, true_code, font, config.width, config.height, text_color)

        # 7. Apply occlusions (simulate dirt smudges, label damage).
        if config.occlusion_intensity > 0:
            self._add_occlusions(img, config.occlusion_intensity, bg_color)

        # 8. Apply blur (camera defocus / motion).
        if config.blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))

        # 9. Apply rotation (camera misalignment).
        if abs(config.rotation_degrees) > 0.01:
            img = img.rotate(
                config.rotation_degrees,
                resample=Image.Resampling.BILINEAR,
                fillcolor=bg_color,
                expand=False,
            )

        # 10. Add pixel-level Gaussian noise (sensor noise / low light).
        arr = np.array(img, dtype=np.float32)
        if config.noise_intensity > 0:
            sigma = config.noise_intensity * 30.0
            noise = self._gaussian_noise(arr.shape, sigma)
            arr = arr + noise

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return GeneratedImage(image=arr, true_code=true_code, background_color_name=bg_name)

    # --- Helpers ---

    def _load_font(self, image_height: int) -> ImageFont.ImageFont:
        """Pick a font size that fits the image height."""
        # Target text height around 60% of image height.
        font_size = max(int(image_height * 0.6), 12)
        if self._font_path:
            try:
                return ImageFont.truetype(self._font_path, size=font_size)
            except OSError:
                pass
        return ImageFont.load_default()

    def _pick_text_color(self, bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
        """Pick a high-contrast text color based on background luminance."""
        # Standard relative luminance.
        r, g, b = bg_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return TEXT_COLORS_LIGHT_BG if luminance > 127 else TEXT_COLORS_DARK_BG

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw `text` centered in (width, height)."""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) // 2 - bbox[0]
        y = (height - text_h) // 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=color)

    def _add_texture(self, img: Image.Image, intensity: float) -> None:
        """Overlay subtle noise to make the surface look less uniform."""
        if intensity <= 0:
            return
        arr = np.array(img, dtype=np.float32)
        noise = self._gaussian_noise(arr.shape, sigma=intensity * 30)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(arr))

    def _add_occlusions(
        self,
        img: Image.Image,
        intensity: float,
        bg_color: tuple[int, int, int],
    ) -> None:
        """Draw random dark/light blobs that partially cover the text."""
        draw = ImageDraw.Draw(img, "RGBA")
        n_blobs = int(1 + intensity * 4)
        for _ in range(n_blobs):
            w = self._rng.randint(20, int(img.width * 0.25))
            h = self._rng.randint(10, int(img.height * 0.5))
            x = self._rng.randint(0, img.width - w)
            y = self._rng.randint(0, img.height - h)
            # Smudge color: darker or lighter than bg, semi-transparent.
            smudge = tuple(
                int(np.clip(c + self._rng.choice([-60, 60]), 0, 255)) for c in bg_color
            ) + (int(120 * intensity),)
            draw.ellipse([x, y, x + w, y + h], fill=smudge)

    def _draw_streaks(self, img: Image.Image, bg_color: tuple[int, int, int]) -> None:
        """Vertical rust/water streaks running down the container side."""
        draw = ImageDraw.Draw(img, "RGBA")
        n_streaks = self._rng.randint(2, 5)
        for _ in range(n_streaks):
            x = self._rng.randint(0, img.width - 1)
            streak_w = self._rng.randint(1, 3)
            # Slightly darker, brown-tinted streak.
            color = (
                int(np.clip(bg_color[0] + 20, 0, 255)),
                int(np.clip(bg_color[1] - 10, 0, 255)),
                int(np.clip(bg_color[2] - 15, 0, 255)),
                70,  # alpha
            )
            draw.rectangle([x, 0, x + streak_w, img.height], fill=color)

    def _gaussian_noise(self, shape: tuple[int, ...], sigma: float) -> np.ndarray:
        """Reproducible Gaussian noise using the generator's RNG."""
        # numpy RNG seeded from our random.Random for reproducibility.
        seed = self._rng.randint(0, 2**32 - 1)
        return np.random.default_rng(seed).normal(0, sigma, size=shape)


# --- Convenience presets ---

CLEAN = GenerationConfig()
"""Pristine image — for sanity tests."""

MODERATE = GenerationConfig(
    noise_intensity=0.15,
    blur_radius=0.5,
    rotation_degrees=2.0,
    brightness_jitter=0.3,
    add_streak_marks=True,
)
"""Realistic depot conditions — what your OCR should handle in production."""

HARSH = GenerationConfig(
    noise_intensity=0.35,
    blur_radius=1.2,
    rotation_degrees=8.0,
    brightness_jitter=0.6,
    occlusion_intensity=0.4,
    add_streak_marks=True,
)
"""Stress-test conditions — partially occluded, blurred, misaligned."""


def save_image(generated: GeneratedImage, path: str | Path) -> None:
    """Convenience: write a GeneratedImage to disk as PNG."""
    Image.fromarray(generated.image).save(str(path))