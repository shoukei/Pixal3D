"""Compatibility shim for the EasternJournalist/utils3d fork installed on macOS.

setup_macos.sh installs `git+https://github.com/EasternJournalist/utils3d.git`
(because MoGe needs the newer `utils3d.pt` API). That fork does NOT expose
`utils3d.torch.intrinsics_from_fov_xy` or `utils3d.torch.get_image_rays`,
which Pixal3D's renderer/pipeline code expects from the official utils3d 0.0.2
(LDYang694/Storages wheel).

This module injects byte-for-byte equivalent implementations (taken from the
official utils3d 0.0.2 wheel: utils3d/torch/transforms.py and
utils3d/torch/nerf.py) into the installed `utils3d.torch` module so the rest of
the codebase works unchanged.

Usage: `import utils3d_compat` (idempotent).
"""

import torch
import utils3d
import utils3d.torch as _u3t


def _intrinsics_from_focal_center(fx, fy, cx, cy):
    if any(torch.is_tensor(x) for x in (fx, fy, cx, cy)):
        dtype = next(x.dtype for x in (fx, fy, cx, cy) if torch.is_tensor(x))
    else:
        dtype = torch.float32
    fx, fy, cx, cy = torch.broadcast_tensors(
        *(x if torch.is_tensor(x) else torch.tensor(x) for x in (fx, fy, cx, cy))
    )
    ret = torch.zeros(fx.shape + (3, 3), dtype=dtype, device=fx.device)
    ret[..., 0, 0] = fx
    ret[..., 1, 1] = fy
    ret[..., 0, 2] = cx
    ret[..., 1, 2] = cy
    ret[..., 2, 2] = 1.
    return ret


def intrinsics_from_fov_xy(fov_x, fov_y):
    """Normalized OpenCV intrinsics from FOV on x and y axis. [..., 3, 3]"""
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    cx = cy = 0.5
    return _intrinsics_from_focal_center(focal_x, focal_y, cx, cy)


def _image_uv(height, width):
    """Normalized pixel-center uv grid. Returns (height, width, 2) in [0, 1]."""
    device = "cpu"
    ys = torch.arange(height, dtype=torch.float32) + 0.5
    xs = torch.arange(width, dtype=torch.float32) + 0.5
    uy = ys / height
    vx = xs / width
    grid_y, grid_x = torch.meshgrid(uy, vx, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2)


def get_image_rays(extrinsics, intrinsics, width, height):
    """
    Args:
        extrinsics: (..., 4, 4) world-to-camera (OpenCV)
        intrinsics: (..., 3, 3) normalized intrinsics
        width, height: render size in pixels

    Returns:
        rays_o: (..., 1, 1, 3) camera center in world space
        rays_d: (..., height, width, 3) world-space directions (NOT normalized)
    """
    uv = _image_uv(height, width).to(extrinsics).reshape(1, -1, 2)
    uvz = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1).to(extrinsics)
    inv_transformation = (intrinsics @ extrinsics[..., :3, :3]).inverse()
    inv_extrinsics = extrinsics.inverse()
    rays_d = uvz @ inv_transformation.transpose(-1, -2)
    rays_o = inv_extrinsics[..., None, :3, 3]
    rays_o = rays_o.unflatten(-2, (1, 1))
    rays_d = rays_d.unflatten(-2, (height, width))
    return rays_o, rays_d


_applied = []
for _name, _fn in {
    "intrinsics_from_fov_xy": intrinsics_from_fov_xy,
    "get_image_rays": get_image_rays,
}.items():
    if not hasattr(_u3t, _name):
        setattr(_u3t, _name, _fn)
        _applied.append(_name)

if _applied:
    print(f"[utils3d_compat] Patched missing utils3d.torch functions: {_applied}")
