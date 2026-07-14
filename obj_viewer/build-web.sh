#!/bin/bash
set -e

echo "Building OBJ Viewer for WebAssembly..."
echo ""

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build WASM
echo "Compiling to WebAssembly..."
wasm-pack build --target web --out-dir web/pkg

echo ""
echo "Build complete!"
echo ""
echo "To run locally:"
echo "  cd web"
echo "  python3 -m http.server 8080"
echo ""
echo "Then open http://localhost:8080 in Chrome, Edge, or Firefox."
