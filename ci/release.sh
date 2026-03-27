#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 [--dry-run] [major|minor|patch|premajor|preminor|prepatch|prerelease]" >&2
    exit 1
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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
kind=
dry_run=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        major|minor|patch|premajor|preminor|prepatch|prerelease)
            [ -z "${kind:-}" ] || die "release kind already set to $kind"
            kind="$1"
            ;;
        --dry-run)
            dry_run=1
            ;;
        -h|--help)
            usage
            ;;
        *)
            die "unsupported argument: $1"
            ;;
    esac
    shift
done

kind="${kind:-minor}"

cd "$repo_root"

require_cmd git
require_cmd node
require_cmd yarn

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git repository"
[ -f package.json ] || die "package.json not found in $repo_root"
[ -f CHANGELOG.md ] || die "CHANGELOG.md not found in $repo_root"
git remote get-url origin >/dev/null 2>&1 || die "git remote 'origin' is not configured"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$current_branch" = "master" ] || {
    if [ "$dry_run" -eq 1 ]; then
        warn "running dry-run from branch $current_branch instead of master"
    else
        die "release must run from master, current branch is $current_branch"
    fi
}

if [ -n "$(git status --porcelain)" ]; then
    if [ "$dry_run" -eq 1 ]; then
        warn "working tree is not clean; continuing because --dry-run does not modify files"
    else
        die "working tree must be clean before running a release"
    fi
fi

note_file="$(mktemp "${TMPDIR:-/tmp}/deepdetect-release-note.XXXXXX.md")"
dry_run_output_file="$(mktemp "${TMPDIR:-/tmp}/deepdetect-release-dry-run.XXXXXX.log")"
keep_note_file=0

cleanup() {
    if [ "$keep_note_file" -eq 0 ]; then
        rm -f "$note_file"
    fi
    rm -f "$dry_run_output_file"
}

trap cleanup EXIT

if [ "$dry_run" -eq 1 ]; then
    yarn run standard-version --dry-run -r "$kind" | tee "$dry_run_output_file"

    git_tag="$(sed -n 's/^.*tagging release \(v[^[:space:]]\+\).*$/\1/p' "$dry_run_output_file" | tail -n 1)"
    if [ -n "$git_tag" ]; then
        tag="${git_tag#v}"
        cat <<EOF

Dry run only: tag push and GitHub release creation were skipped.

The draft GitHub release would be created for $git_tag with this Docker images appendix:
### Docker images:

* CPU version: \`docker pull docker.jolibrain.com/deepdetect_cpu:v$tag\`
* GPU (CUDA only): \`docker pull docker.jolibrain.com/deepdetect_gpu:v$tag\`
* GPU (CUDA and TensorRT): \`docker pull docker.jolibrain.com/deepdetect_gpu_tensorrt:v$tag\`
* All images available from https://docker.jolibrain.com/, list images with \`curl -X GET https://docker.jolibrain.com/v2/_catalog\`
EOF
    else
        warn "could not determine the preview tag from standard-version output"
        echo "Dry run only: tag push and GitHub release creation were skipped."
    fi

    exit 0
fi

require_cmd gh
gh auth status >/dev/null 2>&1 || die "GitHub CLI is not authenticated"

rm -f CHANGELOG.md
yarn run standard-version -r "$kind"

tag="$(node -pe "require('./package.json').version")"
git_tag="v$tag"

extract_release_notes "$tag" >"$note_file"
[ -s "$note_file" ] || die "generated release notes are empty"

cat >>"$note_file" <<EOF
### Docker images:

* CPU version: \`docker pull docker.jolibrain.com/deepdetect_cpu:v$tag\`
* GPU (CUDA only): \`docker pull docker.jolibrain.com/deepdetect_gpu:v$tag\`
* GPU (CUDA and TensorRT): \`docker pull docker.jolibrain.com/deepdetect_gpu_tensorrt:v$tag\`
* All images available from https://docker.jolibrain.com/, list images with \`curl -X GET https://docker.jolibrain.com/v2/_catalog\`
EOF

if ! git push origin "$git_tag"; then
    keep_note_file=1
    die "failed to push $git_tag to origin; release notes kept at $note_file"
fi

if ! gh release create "$git_tag" \
    --title "DeepDetect v$tag" \
    --notes-file "$note_file" \
    --draft \
    --verify-tag; then
    keep_note_file=1
    die "failed to create the GitHub release draft; release notes kept at $note_file"
fi

echo "Draft GitHub release created for $git_tag"
echo "Push the release commit with: git push origin master"
