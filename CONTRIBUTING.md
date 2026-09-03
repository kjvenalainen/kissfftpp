# Contributing to KissFFT++

Thanks for helping improve KissFFT++. This project aims to keep the public
library small, portable, and predictable for production use.

## Development requirements

- A C++14-capable compiler (Clang, GCC, or MSVC)
- CMake 3.16 or newer for the self-contained build and smoke test
- Bazel 8 or newer to run the full reference-based test suite

## Before opening a pull request

1. Format changed C++ files with `clang-format`.
2. Build and run the portable smoke test:

   ```sh
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure
   ```

3. Run the full test suite when Bazel is available:

   ```sh
   bazel test //src:gtest --test_output=errors
   ```

4. Add or update tests for every observable behavior change. Do not add
   performance benchmarks to the normal test path; use `//src:fftBenchmark`.

Keep pull requests focused, explain API or performance trade-offs, and retain
the BSD 3-Clause copyright header in source files.
