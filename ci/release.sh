#!/bin/bash

set -e

kind=${1:-minor}
git rm CHANGELOG.md
yarn run standard-version -r $kind
tag=$(cat package.json | jq -r .version)

sed -ne "/^## \[$tag\]/,/^##.*202/p" CHANGELOG.md | sed -e '$d' -e '1d' > note.md

cat >> note.md <<EOF
### Docker images:

* CPU version: \`docker pull docker.jolibrain.com/deepdetect_cpu:v$tag\`
* GPU (CUDA only): \`docker pull docker.jolibrain.com/deepdetect_gpu:v$tag\`
* GPU (CUDA and Tensorrt) :\`docker pull docker.jolibrain.com/deepdetect_cpu_tensorrt:v$tag\`
* GPU with torch backend: \`docker pull docker.jolibrain.com/deepdetect_gpu_torch:v$tag\`
* All images available from https://docker.jolibrain.com/, list images with `curl -X GET https://docker.jolibrain.com/v2/_catalog`
EOF

trap "rm -f note.md" EXIT
gh release create --title "DeepDetect v$tag" -F note.md -d v$tag
