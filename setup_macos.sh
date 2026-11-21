#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Pixal3D Metal setup requires Apple Silicon (arm64)." >&2
  exit 1
fi

DEPS_DIR="$ROOT_DIR/deps"
mkdir -p "$DEPS_DIR"

clone_pinned() {
  local url="$1" dir="$2" commit="$3"
  if [[ ! -d "$DEPS_DIR/$dir/.git" ]]; then
    git clone "$url" "$DEPS_DIR/$dir"
  fi
  git -C "$DEPS_DIR/$dir" fetch --depth 1 origin "$commit"
  git -C "$DEPS_DIR/$dir" checkout --detach "$commit"
}

clone_pinned https://github.com/pedronaugusto/mtlbvh.git mtlbvh 23f441c470ce1f537e1fd836f3ffb5b8245f7975
clone_pinned https://github.com/pedronaugusto/mtldiffrast.git mtldiffrast 4668cd91cb6d27f5e264731f94a06841fbf7aab8
clone_pinned https://github.com/pedronaugusto/mtlmesh.git mtlmesh 212079e55772cff3d648a21372392c37e0643f3b
clone_pinned https://github.com/pedronaugusto/mtlgemm.git mtlgemm 867aec8234299a7fe1ede7f802c8debe5a939a82
clone_pinned https://github.com/pedronaugusto/trellis2-apple.git trellis2-apple 6055b868734af6e12769d229d90580e775fae9f0

if [[ ! -d .venv ]]; then
  uv venv .venv --python python3.11
fi
PYTHON="$ROOT_DIR/.venv/bin/python"
pip_install() {
  uv pip install --python "$PYTHON" "$@"
}

pip_install "torch>=2.13,<2.14" "torchvision>=0.28,<0.29" \
  setuptools wheel pybind11
pip_install -r requirements.txt
pip_install -r requirements-macos.txt
# MoGe currently requires the newer utils3d.pt API. The older 0.0.2 wheel
# mentioned by the Pixal3D model card does not provide that module.
pip_install git+https://github.com/EasternJournalist/utils3d.git

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"

# Prebuilt interpreters (e.g. Homebrew Python) bake the SDK that was active
# at build time into CFLAGS/LDSHARED (-isysroot ...). If xcode-select now
# points at a different toolchain (full Xcode vs Command Line Tools), the
# build mixes headers from both SDKs and libc++ compilation fails. Re-anchor
# any stale -isysroot to the active SDK.
ACTIVE_SDK="$(xcrun --show-sdk-path)"
align_sdk_flag() {
  local name="$1" val="${!1:-}"
  if [[ -z "$val" ]]; then
    val="$("$PYTHON" -c 'import sysconfig,sys; print(sysconfig.get_config_var(sys.argv[1]) or "")' "$name")"
  fi
  if [[ "$val" == *"-isysroot "* && "$val" != *"-isysroot $ACTIVE_SDK"* ]]; then
    export "$name"="$(printf '%s' "$val" | sed "s|-isysroot [^ ]*|-isysroot $ACTIVE_SDK|")"
    echo "Aligned $name -isysroot to: $ACTIVE_SDK"
  fi
}
for flag_var in CFLAGS CXXFLAGS LDSHARED LDCXXSHARED; do align_sdk_flag "$flag_var"; done

# CPATH prepends its directories to the compiler search path in front of the
# sysroot, so it can reintroduce the exact mixing the block above removes
# (e.g. CPATH baked to one SDK's headers while the active toolchain is
# another): math.h's isfinite/isinf/isnan macros then leak into
# <simd/math.h> and the local C++ builds fail. Drop it and pin the default
# SDK of the clang shim to the active one as well.
unset CPATH
export SDKROOT="$ACTIVE_SDK"

pip_install --no-build-isolation "$DEPS_DIR/mtlbvh"
pip_install --no-build-isolation "$DEPS_DIR/mtldiffrast"
pip_install --no-build-isolation "$DEPS_DIR/mtlmesh"
pip_install --no-build-isolation "$DEPS_DIR/mtlgemm"
pip_install --no-build-isolation "$DEPS_DIR/trellis2-apple/o-voxel"

"$PYTHON" patches/mps_compat.py
"$PYTHON" scripts/doctor.py

echo
echo "Pixal3D macOS setup complete."
echo "Activate: source .venv/bin/activate"
echo "Run:      python inference.py --image assets/images/0_img.png --output output.glb"
echo "Low RAM:  python inference.py --image assets/images/0_img.png --output output.glb --low_vram"
