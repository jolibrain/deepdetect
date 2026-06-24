#!/usr/bin/env bash

set -euo pipefail

DEFAULT_IMAGES=(cpu gpu gpu_tensorrt)
WHEEL_BASE_URL="https://www.deepdetect.com/download/wheels"

usage() {
    local exit_code="${1:-1}"
    cat >&2 <<EOF
Usage: $0 [options] [major|minor|patch|premajor|preminor|prepatch|prerelease]

Prepare, build, test, and publish a DeepDetect release.

Options:
  --dry-run                  Build/test locally and show the release preview, but publish nothing
  --yes                      Do not ask for final confirmation before publishing
  --user USER                SSH username used by ci/release_wheels.sh for upload
  --skip-docker              Skip Docker image build/push
  --skip-wheels              Skip wheel build/upload
  --keep-stage               Keep the temporary staging clone on success
  --stage-root DIR           Parent directory for the temporary staging clone
  --jobs N                   Wheel native build jobs
  --cmake PATH               CMake executable for wheel builds
  --cuda-architectures LIST  CMake CUDA architectures for GPU wheels
  --gpu-torch-index-url URL  PyTorch GPU wheel index URL
  -h, --help                 Show this help
EOF
    exit "$exit_code"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

warn() {
    echo "Warning: $*" >&2
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

run() {
    echo "+ $*"
    "$@"
}

sync_dirty_release_tools() {
    local status
    status="$(git status --porcelain -- \
        ci/release_all.sh \
        ci/build-docker-images.sh \
        ci/release_wheels.sh \
        ci/release.sh)"

    if [ -z "$status" ]; then
        return
    fi

    warn "dry-run is syncing uncommitted release helper changes into the staging clone"
    mkdir -p "$stage_repo/ci"
    for tool in \
        ci/release_all.sh \
        ci/build-docker-images.sh \
        ci/release_wheels.sh \
        ci/release.sh
    do
        if [ -e "$repo_root/$tool" ]; then
            cp -p "$repo_root/$tool" "$stage_repo/$tool"
        fi
    done
}

ensure_standard_version() {
    if [ -x node_modules/.bin/standard-version ]; then
        return
    fi

    local cache_dir="${RELEASE_YARN_CACHE_DIR:-${TMPDIR:-/tmp}/deepdetect-release-yarn-cache}"
    local log_file="$cache_dir/yarn-install.log"
    mkdir -p "$cache_dir"
    echo "+ env YARN_CACHE_FOLDER=$cache_dir yarn install --no-lockfile --silent"
    if ! env YARN_CACHE_FOLDER="$cache_dir" yarn install --no-lockfile --silent >"$log_file" 2>&1; then
        cat "$log_file"
        return 1
    fi
}

extract_release_notes() {
    awk -v tag="$1" '
        $0 ~ "^##[[:space:]]*\\[" tag "\\]" || $0 ~ "^###[[:space:]]*\\[" tag "\\]" {
            in_section = 1
            next
        }
        in_section && ($0 ~ "^##[[:space:]]*\\[" || $0 ~ "^###[[:space:]]*\\[") {
            exit
        }
        in_section {
            print
        }
    ' CHANGELOG.md | sed '/./,$!d'
}

append_artifact_notes() {
    local tag="$1"
    local version="${tag#v}"

    cat <<EOF

### Docker images

* CPU version: \`docker pull docker.jolibrain.com/deepdetect_cpu:$tag\`
* GPU (CUDA only): \`docker pull docker.jolibrain.com/deepdetect_gpu:$tag\`
* GPU (CUDA and TensorRT): \`docker pull docker.jolibrain.com/deepdetect_gpu_tensorrt:$tag\`
* All images are available from https://docker.jolibrain.com/.

### Python wheels

* CPU wheel: \`python -m pip install --extra-index-url ${WHEEL_BASE_URL}/simple deepdetect-cpu==$version\`
* GPU wheel: \`python -m pip install --extra-index-url ${WHEEL_BASE_URL}/simple deepdetect-gpu==$version\`
* Wheel indexes: ${WHEEL_BASE_URL}/simple/deepdetect-cpu/ and ${WHEEL_BASE_URL}/simple/deepdetect-gpu/
EOF
}

kind=""
dry_run=0
assume_yes=0
skip_docker=0
skip_wheels=0
keep_stage=0
stage_root="${TMPDIR:-/tmp}"
upload_user=""
wheel_extra_args=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        major|minor|patch|premajor|preminor|prepatch|prerelease)
            [ -z "$kind" ] || die "release kind already set to $kind"
            kind="$1"
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --yes)
            assume_yes=1
            shift
            ;;
        --user)
            upload_user="${2:-}"
            [ -n "$upload_user" ] || die "--user requires a value"
            shift 2
            ;;
        --skip-docker)
            skip_docker=1
            shift
            ;;
        --skip-wheels)
            skip_wheels=1
            shift
            ;;
        --keep-stage)
            keep_stage=1
            shift
            ;;
        --stage-root)
            stage_root="${2:-}"
            [ -n "$stage_root" ] || die "--stage-root requires a value"
            shift 2
            ;;
        --jobs|--cmake|--cuda-architectures|--gpu-torch-index-url)
            [ -n "${2:-}" ] || die "$1 requires a value"
            wheel_extra_args+=("$1" "$2")
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            die "unsupported argument: $1"
            ;;
    esac
done

kind="${kind:-minor}"

[ "$skip_wheels" -eq 1 ] || [ "$dry_run" -eq 1 ] || [ -n "$upload_user" ] || die "--user is required unless --dry-run or --skip-wheels is used"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

cd "$repo_root"

require_cmd git
require_cmd node
require_cmd yarn
[ "$skip_docker" -eq 1 ] || require_cmd docker
[ "$skip_wheels" -eq 1 ] || require_cmd python3
[ "$dry_run" -eq 1 ] || require_cmd gh

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git repository"
[ -f package.json ] || die "package.json not found in $repo_root"
[ -f CHANGELOG.md ] || die "CHANGELOG.md not found in $repo_root"
origin_url="$(git remote get-url origin 2>/dev/null)" || die "git remote 'origin' is not configured"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$current_branch" != "master" ]; then
    if [ "$dry_run" -eq 1 ]; then
        warn "running dry-run from branch $current_branch instead of master"
    else
        die "release must run from master, current branch is $current_branch"
    fi
fi

if [ -n "$(git status --porcelain)" ]; then
    if [ "$dry_run" -eq 1 ]; then
        warn "working tree is not clean; dry-run will sync release helper changes only"
    else
        die "working tree must be clean before running a release"
    fi
fi

if [ "$dry_run" -eq 0 ]; then
    run git fetch origin master --tags
    local_head="$(git rev-parse HEAD)"
    remote_head="$(git rev-parse origin/master)"
    [ "$local_head" = "$remote_head" ] || die "local master is not equal to origin/master"
fi

stage_parent="$(mktemp -d "$stage_root/deepdetect-release.XXXXXX")"
stage_repo="$stage_parent/repo"
note_file="$stage_parent/release-notes.md"
RELEASE_YARN_CACHE_DIR="$stage_parent/yarn-cache"
success=0

cleanup() {
    local status=$?
    if [ "$status" -eq 0 ]; then
        success=1
    fi
    if [ "$keep_stage" -eq 1 ] || [ "$success" -eq 0 ]; then
        echo "Staging clone kept at $stage_repo"
    else
        rm -rf "$stage_parent"
    fi
}
trap cleanup EXIT

run git clone --local --no-hardlinks "$repo_root" "$stage_repo"
if [ "$dry_run" -eq 1 ]; then
    sync_dirty_release_tools
fi
cd "$stage_repo"
run git remote set-url origin "$origin_url"
run git checkout "$current_branch"

release_parent="$(git rev-parse HEAD)"

ensure_standard_version
rm -f CHANGELOG.md
run yarn run standard-version -r "$kind"

version="$(node -pe "require('./package.json').version")"
git_tag="v$version"
release_commit="$(git rev-parse HEAD)"

[ "$(git rev-list -n 1 "$git_tag")" = "$release_commit" ] || die "$git_tag does not point to the release commit"
extract_release_notes "$version" > "$note_file"
[ -s "$note_file" ] || die "generated release notes are empty"
append_artifact_notes "$git_tag" >> "$note_file"

cat <<EOF

Prepared release:
  version:        $version
  tag:            $git_tag
  release commit: $release_commit
  staging clone:  $stage_repo
  release notes:  $note_file
EOF

if [ "$skip_docker" -eq 0 ]; then
    run ci/build-docker-images.sh --tag "$git_tag" --release --no-push "${DEFAULT_IMAGES[@]}"
else
    echo "Docker build skipped."
fi

if [ "$skip_wheels" -eq 0 ]; then
    run ci/release_wheels.sh --version "$version" --dry-run "${wheel_extra_args[@]}"
else
    echo "Wheel build skipped."
fi

cat <<EOF

Release notes preview:
-------------------------------------------------------------------------------
$(cat "$note_file")
-------------------------------------------------------------------------------
EOF

if [ "$dry_run" -eq 1 ]; then
    cat <<EOF

Dry run complete. No Docker images, wheels, git refs, or GitHub releases were published.
EOF
    exit 0
fi

if [ "$assume_yes" -eq 0 ]; then
    printf "Publish Docker images, wheels, git tag, and draft GitHub release for %s? [y/N] " "$git_tag"
    read -r answer
    case "$answer" in
        y|Y|yes|YES)
            ;;
        *)
            die "release publication aborted"
            ;;
    esac
fi

run git fetch origin master --tags
remote_head="$(git rev-parse origin/master)"
[ "$remote_head" = "$release_parent" ] || die "origin/master moved since staging; aborting before publication"

if [ "$skip_docker" -eq 0 ]; then
    run ci/build-docker-images.sh --tag "$git_tag" --push-only "${DEFAULT_IMAGES[@]}"
fi

if [ "$skip_wheels" -eq 0 ]; then
    run ci/release_wheels.sh --version "$version" --skip-build --user "$upload_user" "${wheel_extra_args[@]}"
fi

run git push origin "HEAD:master"
run git push origin "$git_tag"

run gh release create "$git_tag" \
    --title "DeepDetect $git_tag" \
    --notes-file "$note_file" \
    --draft \
    --verify-tag

cat <<EOF

Draft GitHub release created for $git_tag.
EOF
