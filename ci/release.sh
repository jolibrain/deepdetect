#!/bin/bash

set -e

kind=${1:-minor}
git rm CHANGELOG.md
yarn run standard-version -r $kind
tag=$(cat package.json | jq -r .version)

sed -ne "/^## \[$tag\]/,/^##.*202/p" CHANGELOG.md | sed -e '$d' -e '1d' > note.md

cat >> note.md <<EOF
### Docker images:

* CPU version: \`docker pull jolibrain/deepdetect_cpu:v$tag\`
* GPU (CUDA only): \`docker pull jolibrain/deepdetect_gpu:v$tag\`
* GPU (CUDA and Tensorrt) :\`docker pull jolibrain/deepdetect_cpu_tensorrt:v$tag\`
* GPU with torch backend: \`docker pull jolibrain/deepdetect_gpu_torch:v$tag\`
* All images available on https://hub.docker.com/u/jolibrain
EOF

trap "rm -f note.md" EXIT
gh release create --title "DeepDetect v$tag" -F note.md -d v$tag
