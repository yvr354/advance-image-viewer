"""
Multi-illumination fusion engine.

Modes:
  RGB Composite       — assign each image to R/G/B channel with weights
  Max / Min / Average — pixel-wise statistics across all images
  Photometric Stereo  — Woodham (1980) least-squares normal estimation
                        Outputs: normal map, albedo, gradient magnitude, height map
  RTI Relight         — biquadratic polynomial per pixel (PTM)
                        Fit once, relight interactively from any angle
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class FusionEntry:
    image: np.ndarray        # grayscale float32, range [0, 1]
    path:  str  = ""
    azimuth:   float = 0.0   # degrees, 0 = right, CCW positive
    elevation: float = 45.0  # degrees from horizontal, 0–90
    weight: float = 1.0
    assign_r: bool = False
    assign_g: bool = False
    assign_b: bool = False


class IlluminationFusion:
    MAX_INPUTS = 16

    def __init__(self):
        self.entries: List[FusionEntry] = []
        # RTI coefficients cached after fit — (6, H*W)
        self._rti_coeff: Optional[np.ndarray] = None
        self._rti_shape: Optional[tuple] = None

    # ── Entry management ──────────────────────────────────────────────────────

    def add_image(self, image: np.ndarray, path: str = "") -> int:
        if len(self.entries) >= self.MAX_INPUTS:
            raise ValueError(f"Maximum {self.MAX_INPUTS} input images")
        e = FusionEntry(image=self._to_f32(image), path=path)
        self.entries.append(e)
        self._rti_coeff = None   # invalidate RTI cache
        return len(self.entries) - 1

    def remove_image(self, index: int):
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            self._rti_coeff = None

    def clear(self):
        self.entries.clear()
        self._rti_coeff = None

    def set_angles(self, index: int, azimuth: float, elevation: float):
        if 0 <= index < len(self.entries):
            self.entries[index].azimuth   = azimuth
            self.entries[index].elevation = elevation
            self._rti_coeff = None

    def set_assignment(self, index: int, r: bool, g: bool, b: bool, weight: float = 1.0):
        if 0 <= index < len(self.entries):
            e = self.entries[index]
            e.assign_r, e.assign_g, e.assign_b, e.weight = r, g, b, weight

    # ── Simple fusion modes ───────────────────────────────────────────────────

    def rgb_composite(self) -> Optional[np.ndarray]:
        """Weighted R/G/B channel assignment → colour image."""
        if not self.entries:
            return None
        h, w = self.entries[0].image.shape[:2]
        r = np.zeros((h, w), np.float64)
        g = np.zeros((h, w), np.float64)
        b = np.zeros((h, w), np.float64)
        rc = gc = bc = 0
        for e in self.entries:
            img = self._fit(e.image, h, w).astype(np.float64) * e.weight
            if e.assign_r: r += img; rc += 1
            if e.assign_g: g += img; gc += 1
            if e.assign_b: b += img; bc += 1
        if rc: r /= rc
        if gc: g /= gc
        if bc: b /= bc
        out = np.stack([r, g, b], axis=2)
        return self._norm8(out)

    def max_fusion(self) -> Optional[np.ndarray]:
        return self._pixel_stat("max")

    def min_fusion(self) -> Optional[np.ndarray]:
        return self._pixel_stat("min")

    def average_fusion(self) -> Optional[np.ndarray]:
        return self._pixel_stat("avg")

    def superposition(self) -> Optional[np.ndarray]:
        """
        A + B + C + ... (sum, then normalise).
        Any pixel bright in ANY light accumulates. Works with 2, 3, 4+ images.
        Defect visible in 3 lights = 3× stronger than defect in 1 light.
        """
        if not self.entries:
            return None
        h, w = self.entries[0].image.shape[:2]
        stack = np.stack([self._fit(e.image, h, w) for e in self.entries], axis=0)
        return self._norm8(stack.sum(axis=0).astype(np.float64))

    def multiply(self) -> Optional[np.ndarray]:
        """
        A × B × C × ... (product, then normalise).
        Only pixels bright in ALL images survive — extremely selective.
        Any image where the pixel is dark kills the result for that pixel.
        Use to confirm defects that appear under every lighting angle.
        Works with 2, 3, 4+ images.
        """
        if not self.entries:
            return None
        h, w = self.entries[0].image.shape[:2]
        result = np.ones((h, w), dtype=np.float64)
        for e in self.entries:
            result *= self._fit(e.image, h, w).astype(np.float64) / 255.0
        return self._norm8(result)

    def range_fusion(self) -> Optional[np.ndarray]:
        """
        Max − Min across all images.
        Shows which pixels change the MOST between lighting angles.
        Flat uniform surface = same in all lights = Max≈Min → black (no defect).
        Scratch / pit = very different per light = large Max-Min → bright white.
        Best multi-image operation — works with 3, 4, 5+ images automatically.
        No need to pick A and B.
        """
        if not self.entries:
            return None
        h, w = self.entries[0].image.shape[:2]
        stack = np.stack([self._fit(e.image, h, w) for e in self.entries], axis=0).astype(np.float64)
        return self._norm8(stack.max(axis=0) - stack.min(axis=0))

    def difference(self, idx_a: int = 0, idx_b: int = 1) -> Optional[np.ndarray]:
        """
        |A − B| per pixel — absolute difference between two images.
        Flat surface (identical in both lights) → black.
        Defect (reacts differently to each light) → bright.
        """
        if idx_a >= len(self.entries) or idx_b >= len(self.entries):
            return None
        h, w = self.entries[0].image.shape[:2]
        a = self._fit(self.entries[idx_a].image, h, w).astype(np.float64)
        b = self._fit(self.entries[idx_b].image, h, w).astype(np.float64)
        diff = np.abs(a - b)
        return self._norm8(diff)

    def _pixel_stat(self, mode: str) -> Optional[np.ndarray]:
        if not self.entries:
            return None
        h, w = self.entries[0].image.shape[:2]
        stack = np.stack([self._fit(e.image, h, w) for e in self.entries], axis=0)
        if mode == "max":
            out = stack.max(axis=0)
        elif mode == "min":
            out = stack.min(axis=0)
        else:
            out = stack.mean(axis=0)
        return self._norm8(out)

    # ── Photometric Stereo ────────────────────────────────────────────────────

    def photometric_stereo(
        self,
        output: Literal["normal_map", "albedo", "gradient", "height"] = "gradient"
    ) -> Optional[np.ndarray]:
        """
        Woodham (1980) least-squares photometric stereo.

        Requires ≥ 3 images with known light directions (azimuth + elevation).
        Returns one of: normal_map (RGB), albedo, gradient magnitude, height map.
        """
        if len(self.entries) < 3:
            return None

        h, w   = self.entries[0].image.shape[:2]
        N      = len(self.entries)
        L      = self._light_matrix()          # (N, 3)
        I      = self._image_stack(h, w)       # (N, H*W)

        # Solve  g = (L^T L)^{-1} L^T I  via pseudo-inverse (one shot)
        L_pinv = np.linalg.pinv(L)             # (3, N)
        g      = L_pinv @ I                    # (3, H*W)  — ρ·n per pixel

        # Albedo = ‖g‖
        rho    = np.linalg.norm(g, axis=0)     # (H*W)
        safe   = rho + 1e-8

        # Unit surface normals
        nx, ny, nz = g[0] / safe, g[1] / safe, g[2] / safe

        if output == "normal_map":
            normals = np.stack([nx, ny, nz], axis=1).reshape(h, w, 3)
            rgb     = ((normals + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if output == "albedo":
            return self._norm8(rho.reshape(h, w))

        if output == "gradient":
            # Gradient magnitude of the surface slope
            nz_safe = nz.clip(-1, -1e-4) if (nz < 0).any() else nz.clip(1e-4, 1)
            gx      = -nx / nz_safe
            gy      = -ny / nz_safe
            mag     = np.sqrt(gx ** 2 + gy ** 2).reshape(h, w)
            return self._norm8(mag)

        if output == "height":
            nz_safe = nz.copy()
            nz_safe[np.abs(nz_safe) < 1e-4] = 1e-4
            gx = (-nx / nz_safe).reshape(h, w)
            gy = (-ny / nz_safe).reshape(h, w)
            z  = self._frankot_chellappa(gx, gy)
            return self._norm8(z)

        return None

    # ── RTI / Polynomial Texture Mapping ─────────────────────────────────────

    def rti_fit(self) -> bool:
        """
        Fit biquadratic PTM polynomial per pixel.
        Stores coefficients internally; call rti_relight() afterwards.

        Per-pixel model:  I(u,v) = c0·u² + c1·v² + c2·uv + c3·u + c4·v + c5
        where (u,v) are normalised light direction components.

        Returns True on success.
        """
        if len(self.entries) < 6:
            return False

        h, w   = self.entries[0].image.shape[:2]
        N      = len(self.entries)
        I      = self._image_stack(h, w)       # (N, H*W)

        u, v   = self._uv_vectors()            # (N,), (N,)
        A      = np.column_stack([
            u**2, v**2, u*v, u, v, np.ones(N)
        ]).astype(np.float32)                  # (N, 6)

        # Solve A @ C = I  →  C = pinv(A) @ I
        A_pinv           = np.linalg.pinv(A)   # (6, N)
        self._rti_coeff  = (A_pinv @ I).astype(np.float32)   # (6, H*W)
        self._rti_shape  = (h, w)
        return True

    def rti_relight(self, azimuth: float, elevation: float) -> Optional[np.ndarray]:
        """
        Evaluate PTM polynomial for a given light direction.
        Returns relit grayscale image as uint8.
        """
        if self._rti_coeff is None:
            return None
        h, w = self._rti_shape
        az_r = np.radians(azimuth)
        el_r = np.radians(elevation)
        u = np.cos(el_r) * np.cos(az_r)
        v = np.cos(el_r) * np.sin(az_r)
        poly = np.array([u**2, v**2, u*v, u, v, 1.0], dtype=np.float32)  # (6,)
        relit = (self._rti_coeff.T @ poly).reshape(h, w)
        return self._norm8(relit)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _light_matrix(self) -> np.ndarray:
        """Convert az/el angles to unit light direction vectors (N, 3)."""
        rows = []
        for e in self.entries:
            az = np.radians(e.azimuth)
            el = np.radians(e.elevation)
            lx = np.cos(el) * np.cos(az)
            ly = np.cos(el) * np.sin(az)
            lz = np.sin(el)
            rows.append([lx, ly, lz])
        return np.array(rows, dtype=np.float64)

    def _uv_vectors(self):
        """Return (u, v) normalised light components for RTI polynomial."""
        L  = self._light_matrix()
        norms = np.linalg.norm(L, axis=1, keepdims=True) + 1e-8
        Ln = L / norms
        return Ln[:, 0], Ln[:, 1]

    def _image_stack(self, h: int, w: int) -> np.ndarray:
        """Stack all images into (N, H*W) float64."""
        return np.stack(
            [self._fit(e.image, h, w).ravel() for e in self.entries],
            axis=0
        ).astype(np.float64)

    @staticmethod
    def _frankot_chellappa(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        Frankot-Chellappa (1988) FFT-based gradient integration.
        Returns height field (float64, not normalised).
        """
        H, W = gx.shape
        fy = np.fft.fftfreq(H)[:, None]
        fx = np.fft.fftfreq(W)[None, :]
        denom = (fx**2 + fy**2) + 1e-8
        denom[0, 0] = 1.0
        Z_fft = (-1j * fx * np.fft.fft2(gx) - 1j * fy * np.fft.fft2(gy)) / denom
        Z_fft[0, 0] = 0.0
        return np.real(np.fft.ifft2(Z_fft))

    @staticmethod
    def _to_f32(image: np.ndarray) -> np.ndarray:
        """Convert any image to grayscale float32 [0, 1]."""
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        if image.ndim == 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return image.astype(np.float32) / 255.0

    @staticmethod
    def _fit(image: np.ndarray, h: int, w: int) -> np.ndarray:
        if image.shape[:2] == (h, w):
            return image
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _norm8(arr: np.ndarray) -> np.ndarray:
        """Normalise float array to uint8 [0, 255]."""
        out = cv2.normalize(arr.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX)
        return out.astype(np.uint8)
