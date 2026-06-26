#!/usr/bin/env bash

set -euo pipefail

BASE_URL="https://www.deepdetect.com/download/wheels"
REMOTE_HOST="www.deepdetect.com"
REMOTE_DIR="/var/www/deepdetect/public/download/wheels"
UPLOAD_METHOD="rsync"
WHEEL_DIR="dist/python/release"
INDEX_DIR="dist/python/wheelhouse"
ARTIFACT_DIR="files"
CPU_NAME="deepdetect-cpu"
GPU_NAME="deepdetect-gpu"
GPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}"
JOBS="$(nproc 2>/dev/null || echo 4)"
if [ -n "${CMAKE_COMMAND:-}" ]; then
    CMAKE_COMMAND="${CMAKE_COMMAND}"
elif [ -x /usr/bin/cmake ]; then
    CMAKE_COMMAND="/usr/bin/cmake"
else
    CMAKE_COMMAND="cmake"
fi
DRY_RUN=0
SKIP_BUILD=0
BUILD_CPU=1
BUILD_GPU=1
FORCE_VERSION=""
UPLOAD_USER=""
CPU_PYTHON=""
GPU_PYTHON=""
TORCHVISION_SOURCE_DIR=""

usage() {
    cat >&2 <<EOF
Usage: $0 --user USER [options]

Build CPU/GPU Python wheels, generate pip simple indexes with sha256 links,
and upload them under ${BASE_URL}.

Options:
  --user USER                  SSH username used for upload (required)
  --host HOST                  Upload SSH host (default: ${REMOTE_HOST})
  --remote-dir DIR             Remote wheel root (default: ${REMOTE_DIR})
  --base-url URL               Public wheel root URL (default: ${BASE_URL})
  --upload-method rsync|scp    Upload transport (default: ${UPLOAD_METHOD})
  --version VERSION            Override wheel version
  --variant cpu|gpu|all        Build/stage selected wheel variant(s) (default: all)
  --cpu-only                   Build/stage only CPU wheels
  --gpu-only                   Build/stage only GPU wheels
  --skip-build                 Reuse existing wheels from ${WHEEL_DIR}
  --dry-run                    Build and stage locally, but do not upload or fetch remote indexes
  --no-upload                  Alias for --dry-run
  --local                      Alias for --dry-run
  --jobs N                     Parallel native build jobs (default: ${JOBS})
  --cmake PATH                 CMake executable (default: ${CMAKE_COMMAND})
  --cpu-python PATH            Existing CPU build Python interpreter
  --gpu-python PATH            Existing GPU build Python interpreter
  --torchvision-source-dir DIR Existing torchvision 0.27.1 source checkout
  --cuda-architectures LIST    CMake CUDA architectures (default: ${CUDA_ARCHITECTURES})
  --gpu-torch-index-url URL    PyTorch GPU wheel index (default: ${GPU_TORCH_INDEX_URL})
  -h, --help                   Show this help
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

normalize_name() {
    printf '%s' "$1" | tr '[:upper:]_.' '[:lower:]--'
}

html_escape() {
    sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/"/\&quot;/g'
}

project_version() {
    python3 - <<'PY'
import tomllib
from pathlib import Path

with Path("bindings/python/pyproject.toml").open("rb") as handle:
    print(tomllib.load(handle)["project"]["version"])
PY
}

git_safe() {
    git -c "safe.directory=${repo_root}" "$@"
}

computed_version() {
    if [ -n "$FORCE_VERSION" ]; then
        printf '%s\n' "$FORCE_VERSION"
        return
    fi

    local exact_tag base count hash
    exact_tag="$(git_safe describe --tags --exact-match 2>/dev/null || true)"
    if [ -n "$exact_tag" ]; then
        printf '%s\n' "${exact_tag#v}"
        return
    fi

    base="$(node -pe "require('./package.json').version" 2>/dev/null || project_version)"
    count="$(git_safe rev-list --count HEAD)"
    hash="$(git_safe rev-parse --short=8 HEAD)"
    printf '%s.dev%s+g%s\n' "$base" "$count" "$hash"
}

wheel_glob() {
    local variant="$1"
    local project normalized
    case "$variant" in
        cpu) project="$CPU_NAME" ;;
        gpu) project="$GPU_NAME" ;;
        *) die "unknown wheel variant: $variant" ;;
    esac
    normalized="$(printf '%s' "$project" | tr '-' '_')"
    printf '%s/%s/%s-*.whl\n' "$WHEEL_DIR" "$variant" "$normalized"
}

write_project_index() {
    local project="$1"
    local variant="$2"
    local project_dir="$INDEX_DIR/simple/$(normalize_name "$project")"
    local root_index="$INDEX_DIR/simple/index.html"
    local wheel

    mkdir -p "$project_dir" "$INDEX_DIR/$ARTIFACT_DIR/$variant" "$INDEX_DIR/simple"
    cp -f "$WHEEL_DIR/$variant"/*.whl "$INDEX_DIR/$ARTIFACT_DIR/$variant/"

    {
        printf '<!doctype html>\n<html><body>\n'
        for wheel in "$INDEX_DIR/$ARTIFACT_DIR/$variant"/*.whl; do
            local filename hash href escaped_filename escaped_href
            filename="$(basename "$wheel")"
            hash="$(sha256sum "$wheel" | awk '{print $1}')"
            href="../../$ARTIFACT_DIR/$variant/$filename#sha256=$hash"
            escaped_filename="$(printf '%s' "$filename" | html_escape)"
            escaped_href="$(printf '%s' "$href" | html_escape)"
            printf '<a href="%s">%s</a><br>\n' "$escaped_href" "$escaped_filename"
        done
        printf '</body></html>\n'
    } > "$project_dir/index.html"

    {
        printf '<!doctype html>\n<html><body>\n'
        printf '<a href="%s/">%s</a><br>\n' \
            "$(normalize_name "$CPU_NAME")" "$(normalize_name "$CPU_NAME")"
        printf '<a href="%s/">%s</a><br>\n' \
            "$(normalize_name "$GPU_NAME")" "$(normalize_name "$GPU_NAME")"
        printf '</body></html>\n'
    } > "$root_index"
}

write_root_index() {
    local root_index="$INDEX_DIR/simple/index.html"
    local cpu_index_dir="$INDEX_DIR/simple/$(normalize_name "$CPU_NAME")"
    local gpu_index_dir="$INDEX_DIR/simple/$(normalize_name "$GPU_NAME")"

    mkdir -p "$INDEX_DIR/simple"
    {
        printf '<!doctype html>\n<html><body>\n'
        if [ "$BUILD_CPU" -eq 1 ] || [ -d "$cpu_index_dir" ]; then
            printf '<a href="%s/">%s</a><br>\n' \
                "$(normalize_name "$CPU_NAME")" "$(normalize_name "$CPU_NAME")"
        fi
        if [ "$BUILD_GPU" -eq 1 ] || [ -d "$gpu_index_dir" ]; then
            printf '<a href="%s/">%s</a><br>\n' \
                "$(normalize_name "$GPU_NAME")" "$(normalize_name "$GPU_NAME")"
        fi
        printf '</body></html>\n'
    } > "$root_index"
}

fetch_existing_wheelhouse() {
    [ "$DRY_RUN" -eq 0 ] || return 0
    [ "$UPLOAD_METHOD" = "rsync" ] || return 0

    require_cmd rsync
    mkdir -p "$INDEX_DIR"
    if ssh "${UPLOAD_USER}@${REMOTE_HOST}" "test -d '${REMOTE_DIR}'"; then
        rsync -a "${UPLOAD_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" "$INDEX_DIR"/
    fi
}

upload_wheelhouse() {
    [ "$DRY_RUN" -eq 0 ] || {
        echo "Dry run: upload skipped"
        return
    }

    case "$UPLOAD_METHOD" in
        rsync)
            require_cmd rsync
            rsync -av "$INDEX_DIR"/ "${UPLOAD_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
            ;;
        scp)
            require_cmd ssh
            require_cmd scp
            scp -r "$INDEX_DIR"/. "${UPLOAD_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
            ;;
        *)
            die "unsupported upload method: $UPLOAD_METHOD"
            ;;
    esac
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --user)
            UPLOAD_USER="${2:-}"
            shift 2
            ;;
        --host)
            REMOTE_HOST="${2:-}"
            shift 2
            ;;
        --remote-dir)
            REMOTE_DIR="${2:-}"
            shift 2
            ;;
        --base-url)
            BASE_URL="${2:-}"
            shift 2
            ;;
        --upload-method)
            UPLOAD_METHOD="${2:-}"
            shift 2
            ;;
        --version)
            FORCE_VERSION="${2:-}"
            shift 2
            ;;
        --variant)
            case "${2:-}" in
                all)
                    BUILD_CPU=1
                    BUILD_GPU=1
                    ;;
                cpu)
                    BUILD_CPU=1
                    BUILD_GPU=0
                    ;;
                gpu)
                    BUILD_CPU=0
                    BUILD_GPU=1
                    ;;
                *)
                    die "--variant must be one of: cpu, gpu, all"
                    ;;
            esac
            shift 2
            ;;
        --cpu-only)
            BUILD_CPU=1
            BUILD_GPU=0
            shift
            ;;
        --gpu-only)
            BUILD_CPU=0
            BUILD_GPU=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --dry-run|--no-upload|--local)
            DRY_RUN=1
            shift
            ;;
        --jobs)
            JOBS="${2:-}"
            shift 2
            ;;
        --cmake)
            CMAKE_COMMAND="${2:-}"
            shift 2
            ;;
        --cpu-python)
            CPU_PYTHON="${2:-}"
            shift 2
            ;;
        --gpu-python)
            GPU_PYTHON="${2:-}"
            shift 2
            ;;
        --torchvision-source-dir)
            TORCHVISION_SOURCE_DIR="${2:-}"
            shift 2
            ;;
        --cuda-architectures)
            CUDA_ARCHITECTURES="${2:-}"
            shift 2
            ;;
        --gpu-torch-index-url)
            GPU_TORCH_INDEX_URL="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unsupported argument: $1"
            ;;
    esac
done

[ "$BUILD_CPU" -eq 1 ] || [ "$BUILD_GPU" -eq 1 ] || die "at least one wheel variant must be selected"
[ -n "$UPLOAD_USER" ] || [ "$DRY_RUN" -eq 1 ] || die "--user is required"

require_cmd git
require_cmd python3
require_cmd sha256sum

version="$(computed_version)"
echo "Wheel version: $version"

if [ "$SKIP_BUILD" -eq 0 ]; then
    build_args=()
    if [ "$BUILD_CPU" -eq 0 ]; then
        build_args+=(--skip-cpu)
    fi
    if [ "$BUILD_GPU" -eq 0 ]; then
        build_args+=(--skip-gpu)
    fi
    if [ -n "$CPU_PYTHON" ]; then
        build_args+=(--cpu-python "$CPU_PYTHON")
    fi
    if [ -n "$GPU_PYTHON" ]; then
        build_args+=(--gpu-python "$GPU_PYTHON")
    fi
    if [ -n "$TORCHVISION_SOURCE_DIR" ]; then
        build_args+=(--torchvision-source-dir "$TORCHVISION_SOURCE_DIR")
    fi

    python3 bindings/python/scripts/build_release_wheels.py \
        --distribution-version "$version" \
        --wheel-dir "$WHEEL_DIR" \
        --cmake "$CMAKE_COMMAND" \
        --jobs "$JOBS" \
        --gpu-torch-index-url "$GPU_TORCH_INDEX_URL" \
        --cuda-architectures "$CUDA_ARCHITECTURES" \
        "${build_args[@]}"
fi

if [ "$BUILD_CPU" -eq 1 ]; then
    cpu_glob="$(wheel_glob cpu)"
    compgen -G "$cpu_glob" >/dev/null || die "no CPU wheels found matching $cpu_glob"
fi
if [ "$BUILD_GPU" -eq 1 ]; then
    gpu_glob="$(wheel_glob gpu)"
    compgen -G "$gpu_glob" >/dev/null || die "no GPU wheels found matching $gpu_glob"
fi

rm -rf "$INDEX_DIR"
fetch_existing_wheelhouse
if [ "$BUILD_CPU" -eq 1 ]; then
    write_project_index "$CPU_NAME" cpu
fi
if [ "$BUILD_GPU" -eq 1 ]; then
    write_project_index "$GPU_NAME" gpu
fi
write_root_index

cat <<EOF

Generated simple indexes:
EOF
if [ "$BUILD_CPU" -eq 1 ]; then
    echo "  ${INDEX_DIR}/simple/$(normalize_name "$CPU_NAME")/index.html"
fi
if [ "$BUILD_GPU" -eq 1 ]; then
    echo "  ${INDEX_DIR}/simple/$(normalize_name "$GPU_NAME")/index.html"
fi

cat <<EOF

Install commands after upload:
EOF
if [ "$BUILD_CPU" -eq 1 ]; then
    echo "  python -m pip install --extra-index-url ${BASE_URL}/simple ${CPU_NAME}"
fi
if [ "$BUILD_GPU" -eq 1 ]; then
    echo "  python -m pip install --extra-index-url ${BASE_URL}/simple ${GPU_NAME}"
fi

cat <<EOF

Direct index URLs:
EOF
if [ "$BUILD_CPU" -eq 1 ]; then
    echo "  ${BASE_URL}/simple/$(normalize_name "$CPU_NAME")/"
fi
if [ "$BUILD_GPU" -eq 1 ]; then
    echo "  ${BASE_URL}/simple/$(normalize_name "$GPU_NAME")/"
fi

upload_wheelhouse
