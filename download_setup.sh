#!/bin/bash

"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code is part of the 6D pose estimation for the Industrial Plentopic Dataset (IPD).
This script downloads the required Dataset for the project and extracts them to the specified directory. 
"""

# Installing aria2 and 7z
sudo apt update
sudo apt install -y aria2
sudo apt install -y p7zi-full

# defining the source and target files
SRC="https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main"
FILES=(
  ipd_base.zip
  ipd_models.zip
  ipd_val.zip
  ipd_test_all.zip
  ipd_test_all.z01
  ipd_train_pbr.zip
  ipd_train_pbr.z01
  ipd_train_pbr.z02
  ipd_train_pbr.z03
)

TARGET_DIR="./datasets"
mkdir -p "$TARGET_DIR"

# Download each file if it's not already exist
for file in "${FILES[@]}"; do
    if [ -f "$TARGET_DIR/$file" ]; then
        echo "Skipping $file (already exits)"
    else
        echo "Downloading $file..."
        aria2c -x16 -s16 -k1M -d "$TARGET_DIR" -o "$file" "$SRC/$file"
        if [ $? -ne 0 ]; then
            echo "Download failed for $file"
            exit 1
        fi
    fi
done

# Extract everything
cd "$TARGET_DIR"
echo "Extracting all files"
7z x ipd_base.zip
7z x ipd_models.zip -oipd
7z x ipd_val.zip     -oipd
7z x ipd_test_all.zip -oipd
7z x ipd_train_pbr.zip -oipd

echo "removing zip files"
rm -f \
  ipd_base.zip \
  ipd_models.zip \
  ipd_val.zip \
  ipd_test_all.zip ipd_test_all.z01 \
  ipd_train_pbr.zip ipd_train_pbr.z01 ipd_train_pbr.z02 ipd_train_pbr.z03

echo "Completed downloading and extracting files"