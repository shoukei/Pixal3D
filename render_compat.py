"""Metal shims for CUDA-only render libraries on macOS.

setup_macos.sh installs `mtldiffrast` (Metal nvdiffrast) but not
`nvdiffrast` or `nvdiffrec_render`. Pixal3D's renderers import those
CUDA packages directly, so this module:

1. Registers `nvdiffrast` / `nvdiffrast.torch` in sys.modules, pointing at
   the installed `mtldiffrast.torch` (same API: rasterize, DepthPeeler,
   interpolate, texture, ...). `RasterizeCudaContext` is aliased to
   `MtlRasterizeContext`.
2. Provides a pure-torch `nvdiffrec_render.light.EnvironmentLight`
   substitute implementing the subset used by `EnvMap`
   (`cubemap`/`base`, `build_mips()`, `shade(...)`, `sample(...)`),
   with a simple analytic diffuse + GGX-ish specular model.

Usage: `import render_compat` (idempotent). No-op where the real packages
exist (CUDA machines).
"""

import importlib.util
import os
import sys
import torch
import torch.nn.functional as F


def _norm(x):
    return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# 1. nvdiffrast -> mtldiffrast
# ---------------------------------------------------------------------------
if importlib.util.find_spec("nvdiffrast") is None:
    try:
        import mtldiffrast.torch as _mtl_dr
    except ImportError as _e:
        raise ImportError(
            "nvdiffrast is not installed and mtldiffrast is unavailable "
            "for the Metal shim"
        ) from _e

    if not hasattr(_mtl_dr, "RasterizeCudaContext"):
        _mtl_dr.RasterizeCudaContext = _mtl_dr.MtlRasterizeContext

    _pkg = type(sys)("nvdiffrast")
    _pkg.__path__ = []
    _pkg.torch = _mtl_dr
    sys.modules["nvdiffrast"] = _pkg
    sys.modules["nvdiffrast.torch"] = _mtl_dr
    print("[render_compat] Aliased nvdiffrast.torch -> mtldiffrast.torch (Metal)")

# ---------------------------------------------------------------------------
# 2. nvdiffrec_render.light.EnvironmentLight substitute
# ---------------------------------------------------------------------------
if importlib.util.find_spec("nvdiffrec_render") is None:

    class EnvironmentLight:
        """Pure-torch env light over a cubemap face tensor [6, H, W, 3].

        Face order follows nvdiffrast: +X, -X, +Y, -Y, +Z, -Z.
        Only the API subset used by Pixal3D's EnvMap is implemented.
        """

        def __init__(self, cubemap):
            self.cubemap = cubemap
            self.base = cubemap
            self.mip_levels = [cubemap]

        def build_mips(self):
            levels = [self.cubemap]
            c = self.cubemap
            while c.shape[1] > 16:
                n = c.shape[1] // 2
                img = c.permute(0, 3, 1, 2).reshape(6, c.shape[3], c.shape[1], c.shape[2])
                img = F.interpolate(img, size=(n, n), mode="bilinear", align_corners=False)
                c = img.reshape(6, n, n, c.shape[3])
                levels.append(c)
            self.mip_levels = levels

        def _sample(self, directions, mip=0):
            """Sample the cubemap at unit directions (..., 3) -> (..., 3)."""
            d = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ax = d.abs()
            device = d.device

            face = torch.full(d.shape[:-1], 0, dtype=torch.long, device=device)  # +X
            face = torch.where(d[..., 0] < 0, torch.full_like(face, 1), face)
            dom_y = ax[..., 1] >= ax[..., 0]
            face = torch.where(dom_y, torch.where(d[..., 1] >= 0,
                                                  torch.full_like(face, 2), torch.full_like(face, 3)), face)
            dom_z = (ax[..., 2] >= ax[..., 0]) & (ax[..., 2] >= ax[..., 1])
            face = torch.where(dom_z, torch.where(d[..., 2] >= 0,
                                                  torch.full_like(face, 4), torch.full_like(face, 5)), face)

            u = torch.where(
                face == 0, -d[..., 2] / d[..., 0].clamp(min=1e-8),
                torch.where(face == 1, d[..., 2] / (-d[..., 0]).clamp(min=1e-8),
                torch.where(face == 2, d[..., 0] / d[..., 1].clamp(min=1e-8),
                torch.where(face == 3, d[..., 0] / (-d[..., 1]).clamp(min=1e-8),
                torch.where(face == 4, d[..., 0] / d[..., 2].clamp(min=1e-8),
                                  d[..., 0] / (-d[..., 2]).clamp(min=1e-8))))))
            v = torch.where(
                face == 0, -d[..., 1] / d[..., 0].clamp(min=1e-8),
                torch.where(face == 1, -d[..., 1] / (-d[..., 0]).clamp(min=1e-8),
                torch.where(face == 2, d[..., 2] / d[..., 1].clamp(min=1e-8),
                torch.where(face == 3, d[..., 2] / (-d[..., 1]).clamp(min=1e-8),
                torch.where(face == 4, -d[..., 1] / d[..., 2].clamp(min=1e-8),
                                  -d[..., 1] / (-d[..., 2]).clamp(min=1e-8))))))

            u = (u + 1) * 0.5
            v = (1 - v) * 0.5

            c = self.mip_levels[min(mip, len(self.mip_levels) - 1)] if self.mip_levels else self.cubemap
            H = c.shape[1]
            px = (u.reshape(-1) * (H - 1)).round().long().clamp(0, H - 1)
            py = (v.reshape(-1) * (H - 1)).round().long().clamp(0, H - 1)
            vals = c[face.reshape(-1), py, px]
            return vals.reshape(*d.shape[:-1], 3)

        def sample(self, directions):
            return self._sample(directions, mip=0)

        def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
            """Diffuse + specular approximation matching nvdiffrec's signature.

            kd: albedo (base color) (..., 3); ks: ORM (..., 3) with
            ks[...,1]=roughness, ks[...,2]=metallic (per the pixal3d renderer).
            Output (..., 3).
            """
            N = _norm(gb_normal)
            V = _norm(view_pos - gb_pos)
            if os.environ.get("RENDER_DEBUG"):
                print("[shade shapes]", gb_pos.shape, gb_normal.shape, kd.shape, ks.shape, view_pos.shape)
            noV = (N * V).sum(-1, keepdim=True).clamp(min=0)

            if ks.shape[-1] == 3:
                roughness = ks[..., 1:2].clamp(min=0.0, max=1.0)
                metallic = ks[..., 2:3].clamp(min=0.0, max=1.0)
            else:
                roughness = torch.zeros_like(noV)
                metallic = ks.clamp(min=0.0, max=1.0)

            # Diffuse: a handful of hemisphere taps around the normal.
            dev, dt = N.device, N.dtype
            offs = torch.tensor([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            ], device=dev, dtype=dt)
            basis = self._orthonormal_basis(N)  # (..., 3, 3): rows = T, B, N
            taps = torch.einsum("...ab,jb->...ja", basis, offs) * 0.7071
            taps = _norm(taps + N[..., None, :] * 0.5)  # bias toward hemisphere
            diff = kd * self._sample(taps.reshape(*N.shape[:-1], 6, 3), mip=2).mean(dim=-2) / 3.14159
            rgb = diff * (0.5 + 0.5 * noV)

            if specular:
                Hh = _norm(V + N)
                spec_col = self._sample(Hh, mip=1)
                f0 = 0.04 + (1.0 - 0.04) * metallic
                F_schlick = f0 + (1.0 - f0) * (1.0 - noV).pow(2)
                spec = spec_col * F_schlick * (0.25 + 0.25 * noV)
                spec = spec * (1.0 - 0.5 * roughness)
                rgb = rgb + spec

            return rgb

        @staticmethod
        def _orthonormal_basis(N):
            ref = torch.zeros_like(N)
            ref[..., 2] = 1.0
            alt = torch.zeros_like(N)
            alt[..., 0] = 1.0
            ref = torch.where((N * ref).sum(-1, keepdim=True).abs() > 0.9, alt, ref)
            T = _norm(torch.cross(ref, N, dim=-1))
            B = torch.cross(N, T, dim=-1)
            return torch.stack([T, B, N], dim=-1)  # rows

    class _LightModule:
        EnvironmentLight = EnvironmentLight

    class _NvPkg:
        pass

    _NvPkg.light = _LightModule
    sys.modules["nvdiffrec_render"] = _NvPkg
    sys.modules["nvdiffrec_render.light"] = _LightModule
    print("[render_compat] Installed pure-torch nvdiffrec_render.light.EnvironmentLight substitute")
