#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Install cmake for building the C++ extension
pip install cmake

# Build the C++ extension
echo "Building poker_cpp C++ extension..."
mkdir -p build
cd build
cmake ..
cmake --build .

# Copy the compiled .so object back to the root directory
cp poker_cpp*.so ../
echo "Build complete."
