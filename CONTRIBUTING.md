# Contributing to KissFFT++

Thanks for helping improve KissFFT++. This project aims to keep the public
library small, portable, and predictable for production use.

## Development requirements

- A C++14-capable compiler (Clang, GCC, or MSVC)
- Bazel 8 or newer

Clone with `git clone --recurse-submodules` so the bundled KissFFT reference
implementation is available to Bazel.

## Before opening a pull request

1. Format changed C++ files with `clang-format`.
2. Build the examples and benchmark:

   ```sh
   bazel build //src:main //src:fftBenchmark
   ```

3. Run the full test suite:

   ```sh
   bazel test //src:gtest --test_output=errors
   ```

4. Add or update tests for every observable behavior change. Do not add
   performance benchmarks to the normal test path; use `//src:fftBenchmark`.

Keep pull requests focused, explain API or performance trade-offs, and retain
the BSD 3-Clause copyright header in source files.
