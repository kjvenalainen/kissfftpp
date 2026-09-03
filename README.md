# KissFFT++

KissFFT++ is a zero-dependency, header-only complex and real FFT library for
C++14 and later. It is inspired by [KissFFT](https://github.com/mborgerding/kissfft)
and provides reusable transform plans with a small modern C++ API.

## Features

- Header-only library with no runtime dependencies
- Complex-to-complex and real-to-complex transforms
- Arbitrary positive complex transform lengths and even real transform lengths
- Reusable plans: transform calls make no heap allocations
- Configurable inverse scaling and optional contract checks
- Bazel build and test targets

## Install and integrate

If you just want an FFT, include `src/include/kissfftpp.h` in your project. This is the singular header for the entire implementation.

The public Bazel library target is `//src:kissfftpp`:

```starlark
cc_binary(
    name = "my_program",
    srcs = ["my_program.cpp"],
    deps = ["@kissfftpp//src:kissfftpp"],
)
```

## Usage

Create a plan once for a fixed transform length, pre-size both buffers, and
reuse the plan. The default scaling leaves the forward transform unscaled and
scales the inverse by `1 / N`, so an inverse of a forward transform recreates
the original signal.

```cpp
#include <cstddef>
#include <complex>
#include <vector>

#include <kissfftpp.h>

const std::size_t length = 1024;
kfft::FFT fft(length);
std::vector<std::complex<float>> input(length);
std::vector<std::complex<float>> spectrum(length);

fft.fft(input, spectrum);
fft.ifft(spectrum, input);  // `input` now contains the reconstructed signal.
```

Input and output can be the same vector. For real-valued input, use
`kfft::RealFFT`; its forward output contains `N / 2 + 1` bins from DC through
Nyquist:

```cpp
kfft::RealFFT fft(length);  // `length` must be even and at least 2.
std::vector<float> samples(length);
std::vector<std::complex<float>> spectrum(length / 2 + 1);

fft.fft(samples, spectrum);
fft.ifft(spectrum, samples);
```

To opt out of inverse normalization, pass `kfft::NoScaling` as the transform
template parameter: `fft.fft<kfft::NoScaling>(input, spectrum)`.

## Contracts and concurrency

By default, debug builds validate plan lengths and input/output sizes. Release
builds disable contract checks unless `KFFTPP_NO_CONTRACT_CHECKING=0` is
defined. With exceptions enabled, a failed contract throws `std::logic_error`;
without exceptions, it calls `std::terminate`. Define `KFFTPP_NO_EXCEPTIONS`
or `KFFTPP_NO_CONTRACT_CHECKING` explicitly to override automatic detection.

Plans keep reusable working buffers and therefore are not safe for concurrent
transform calls on the same instance. Create one plan per concurrently active
thread or protect shared access externally.

## Build, test, and benchmark

The Bazel suite compares results against the bundled KissFFT reference
implementation:

```sh
bazel build //src:main //src:fftBenchmark
bazel test //src:gtest --test_output=errors
bazel run //src:fftBenchmark -- --iterations 1000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow.

## License

KissFFT++ is licensed under the [BSD 3-Clause License](LICENSE).
