#!/bin/bash
if ! patch -R -p1 --dry-run <$1; then
  patch -p1 <$1
fi
