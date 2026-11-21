"""Patch the pinned third-party Metal dependency build scripts.

setup_macos.sh checks the four Metal deps (mtlgemm, mtlmesh, mtlbvh,
mtldiffrast) out at pinned upstream commits, so this fix cannot live in the
dep repos as a local commit: a future re-checkout or pinned-revision bump
would silently drop it. Instead the patch is applied here, after the
checkout and before the pip installs, and is idempotent — an already
patched setup.py is left untouched.

The pinned setup.py files hardcode `-std=metal4.0`. A metallib compiled with
a language version newer than the host OS supports is rejected at load time
("This library is using language version 4.0 which is not supported on this
OS"), so on macOS 15 every Metal kernel would fail at first use. The patch
inserts an OS-adaptive default — metal4.0 on macOS 26+, metal3.2 on 15,
metal3.1 on 14, else metal3.0 — overridable with METAL_STD=metal3.2 etc. All
shaders in these deps compile cleanly under metal3.2.
"""

from __future__ import annotations

from pathlib import Path

DEPS_ROOT = Path(__file__).resolve().parent.parent / "deps"

# (dep name, text before the insertion point, text after it, blank lines
# separating them in the pristine file)
# The inserted block sits between head and tail at module level; every dep
# already imports os and subprocess at the top of setup.py.
PATCHES = (
    ("mtlgemm", 'BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")',
     'if BUILD_TARGET == "auto":', 1),
    ("mtlmesh", 'BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")',
     "def _detect_backend():", 2),
    ("mtlbvh", 'ROOT = os.path.dirname(os.path.abspath(__file__))',
     "class MetalBuildExt(_build_ext):", 2),
    ("mtldiffrast", 'ROOT = os.path.dirname(os.path.abspath(__file__))',
     "class MetalBuildExt(_build_ext):", 2),
)

METAL_STD_BLOCK = '''def _default_metal_std():
    """Newest Metal language version this OS can load.

    A metallib is rejected at load time if it was compiled with a language
    version newer than the OS supports (e.g. "This library is using language
    version 4.0 which is not supported on this OS"), so default to the
    newest version the current OS can run:
      - metal4.0 requires macOS 26 (Tahoe)
      - metal3.2 requires macOS 15 (Sequoia)
      - metal3.1 requires macOS 14 (Sonoma)
      - metal3.0 requires macOS 13 (Ventura)
    Override with METAL_STD=metal3.2 etc.
    """
    try:
        out = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
        major = int(out.split(".")[0])
    except (OSError, ValueError):
        major = 0
    if major >= 26:
        return "metal4.0"
    if major >= 15:
        return "metal3.2"
    if major >= 14:
        return "metal3.1"
    return "metal3.0"


METAL_STD = os.environ.get("METAL_STD") or _default_metal_std()'''

HARDCODED_FLAG = '"-std=metal4.0", "-O2",'
ADAPTIVE_FLAG = 'f"-std={METAL_STD}", "-O2",'


def patch_setup(dep: str, head: str, tail: str, blanks: int) -> None:
    path = DEPS_ROOT / dep / "setup.py"
    if not path.is_file():
        raise RuntimeError(f"Missing {path}; run the setup_macos.sh clone step first")
    text = path.read_text(encoding="utf-8")
    if "METAL_STD" in text:
        print(f"  {dep}: already patched, skipping")
        return

    sep = "\n" * (blanks + 1)
    old = f"{head}{sep}{tail}"
    new = f"{head}{sep}{METAL_STD_BLOCK}{sep}{tail}"
    if text.count(old) != 1 or text.count(HARDCODED_FLAG) != 1:
        raise RuntimeError(
            f"Patch anchors not found exactly once in {path}; the pinned "
            f"revision likely changed. Update patches/metal_std.py."
        )

    text = text.replace(old, new, 1)
    text = text.replace(HARDCODED_FLAG, ADAPTIVE_FLAG, 1)
    path.write_text(text, encoding="utf-8")
    print(f"  patched {dep}/setup.py (adaptive Metal standard)")


def main() -> None:
    print("Applying Pixal3D macOS Metal build patches...")
    for dep, head, tail, blanks in PATCHES:
        patch_setup(dep, head, tail, blanks)
    print("All Pixal3D Metal build patches applied.")


if __name__ == "__main__":
    main()
