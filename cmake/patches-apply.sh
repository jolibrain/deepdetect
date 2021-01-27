#!/bin/bash

set -e

for p in "$@"; do
    patch -p 1 --dry-run < $p
    patch -p 1 < $p
done
