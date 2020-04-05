#!/usr/bin/env bash
test_set_path=$1
mkdir "$1/alpha_copy"

for filename in "$1"/alpha/*.png; do
    echo $filename
    for ((i=0; i<=19; i++)); do
        cp "$filename" "$1/alpha_copy/$(basename "$filename" .png)_$i.png"
    done
done
