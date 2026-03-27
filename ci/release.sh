#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 [major|minor|patch|premajor|preminor|prepatch|prerelease]" >&2
    exit 1
}

die() {
    echo "Error: $*" >&2
    exit 1
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
kind="${1:-minor}"

case "$kind" in
    major|minor|patch|premajor|preminor|prepatch|prerelease)
        ;;
    -h|--help)
        usage
        ;;
    *)
        die "unsupported release kind: $kind"
        ;;
esac

cd "$repo_root"

require_cmd git
require_cmd node
require_cmd yarn
require_cmd gh

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git repository"
[ -f package.json ] || die "package.json not found in $repo_root"
[ -f CHANGELOG.md ] || die "CHANGELOG.md not found in $repo_root"
git remote get-url origin >/dev/null 2>&1 || die "git remote 'origin' is not configured"
gh auth status >/dev/null 2>&1 || die "GitHub CLI is not authenticated"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$current_branch" = "master" ] || die "release must run from master, current branch is $current_branch"

if [ -n "$(git status --porcelain)" ]; then
    die "working tree must be clean before running a release"
fi

note_file="$(mktemp "${TMPDIR:-/tmp}/deepdetect-release-note.XXXXXX.md")"
keep_note_file=0

cleanup() {
    if [ "$keep_note_file" -eq 0 ]; then
        rm -f "$note_file"
    fi
}

trap cleanup EXIT

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
