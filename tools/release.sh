#!/bin/bash

kind=${1:-minor}
git rm CHANGELOG.md
yarn run standard-version -r $kind
tag=$(cat package.json | jq -r .version)

sed -ne "/^## $tag/,/^##.*202/p" CHANGELOG.md | sed -e '$d' -e '1d' > note.md
trap "rm -f note.md" EXIT
gh release create --title "DeepDetect v$tag" -F note.md -d v$tag
