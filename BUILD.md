# Building mlmat

This document describes how to build the mlmat externals from source on
macOS and Windows. The build produces self-contained Max externals —
**no third-party runtime libraries need to be distributed with the package**:

- On **macOS** the externals link Apple's system-wide Accelerate framework
  (BLAS/LAPACK/vDSP), which ships with every Mac.
- On **Windows** a static OpenBLAS (BLAS) and clapack (LAPACK) are linked
  directly into each external, so nothing but the `.mxe` files is needed.

## Prerequisites

| | macOS | Windows |
|---|---|---|
| Compiler | Xcode 12 or later (with command line tools) | Visual Studio 2022 (Desktop development with C++) |
| CMake | 3.21 or later | 3.21 or later |
| Other | — | [vcpkg](https://vcpkg.io), with the `VCPKG_ROOT` environment variable pointing at your vcpkg checkout |

On Windows, vcpkg is used in *manifest mode*: the required libraries
(listed in [`vcpkg.json`](vcpkg.json)) are downloaded and built automatically
during the first CMake configure. GitHub Actions runners have vcpkg
preinstalled, so CI needs no extra setup.

## Building

Both platforms are a two-step build driven by
[CMake presets](CMakePresets.json):

### macOS

```bash
git clone --recursive <mlmat repo url>
cd mlmat
cmake --preset macos
cmake --build --preset macos-release
```

This produces universal (x86_64 + arm64) externals in the `externals/`
folder. To build a debug version use the `macos-debug` preset instead.
You can also open `build/mlmat.xcodeproj` in Xcode and build from the IDE.

### Windows

```bat
git clone --recursive <mlmat repo url>
cd mlmat
cmake --preset windows
cmake --build --preset windows-release
```

from a terminal where `VCPKG_ROOT` is set and the Visual Studio environment
is available (e.g. the "x64 Native Tools Command Prompt").

The first configure downloads and compiles OpenBLAS and clapack via vcpkg,
which can take a while (they are cached for subsequent builds). The
externals are written to the `externals\` folder as `.mxe` files.

## Installing

The repository *is* the Max package: after building, alias or copy the
`mlmat` folder into your Max `Packages` directory
(`~/Documents/Max 9/Packages` or the Max preference folder) and restart Max.
The externals will be available as `mlmat.*` objects.

## How the build works

Details are in [`source/cmake/mlmat_deps.cmake`](source/cmake/mlmat_deps.cmake),
which defines two dependency targets shared by all objects:

- **`mlmat::deps`** — used by the machine-learning objects. mlpack,
  ensmallen and cereal are consumed header-only directly from their git
  submodules. Armadillo is used in *no-wrapper* mode
  (`ARMA_DONT_USE_WRAPPER`): its headers call BLAS/LAPACK directly and no
  `libarmadillo` is built or shipped. On macOS BLAS/LAPACK come from the
  Accelerate framework; on Windows they are the vcpkg-provided static
  OpenBLAS/clapack.
- **`mlmat::fft`** — used by `mlmat.buffer_stft`, `mlmat.buffer_istft` and
  `mlmat.buffer_griffinlim`. This is [bqfft](https://hg.sr.ht/~breakfastquay/bqfft)
  (plus its bqvec dependency), a small permissively-licensed FFT library.
  On macOS bqfft is compiled with `HAVE_VDSP` so it uses vDSP from the
  Accelerate framework internally; on Windows it uses its built-in
  double-precision FFT. **Only power-of-two `fftsize` values are supported**
  by the spectral objects (16, 32, 64, … 65536); other sizes will report an
  error.

## Updating the dependency versions

The third-party dependencies are pinned git submodules. To update one,
check out the desired release tag inside the submodule and commit the
change from the package root:

```bash
cd source/mlpack
git fetch --tags
git checkout 4.8.0        # or the desired release
cd ..
git add source/mlpack
git commit -m "Update mlpack to 4.8.0"
```

Current pins: mlpack 4.8.0, armadillo 15.6.2 (note: the repository moved to
`gitlab.com/armadillo-lib/armadillo-code`), ensmallen 3.11.1, cereal (master —
1.3.2 predates a fix required by current clang), min-api (main branch, no
releases) and bqfft/bqvec (master). Armadillo, mlpack, ensmallen and cereal
are used header-only; the cereal and ensmallen directories do not need to be
built.

## Continuous integration

GitHub Actions workflows in `.github/workflows/` build both platforms and
create the release packages automatically whenever a `v*` tag is pushed:

```bash
git tag v0.6.0
git push origin v0.6.0
```

Both workflows stamp the tag's version into `package-info.json` and attach
the zipped packages (`mlmat.zip` for macOS, `mlmat_windows.zip` for Windows)
to a GitHub release of the same name. Manual `workflow_dispatch` runs build
the externals without publishing a release.

## Tests

A regression test for the matrix conversion layer lives in
[`tests/conversions/`](tests/conversions/). It compares the live
`matrix_conversions.hpp` bit-for-bit against a frozen copy of the original
implementation, using a fake of the Jitter API so it runs outside of Max.
It is built as part of the normal build; run it with:

```bash
ctest --test-dir build -C Release --output-on-failure
```

or run the `mlmat_conversions_test` binary directly (pass `bench` to also
print conversion timings). Any change to the conversion layer must keep
parity with the reference implementation — if a behavior change is
intended, update `matrix_conversions_reference.hpp` deliberately and note
why in the commit. To skip building the test, configure with
`-DMLMAT_BUILD_CONVERSION_TESTS=OFF`.

## Troubleshooting

- **Apple Silicon "cannot be loaded due to system security policy"** when
  loading freshly built externals in Max: ad-hoc codesign them, e.g.
  `codesign --force --deep -s - externals/mlmat.kmeans.mxo`. Externals built
  through the presets are already ad-hoc signed.
- **Windows: `Cannot find VCPKG_ROOT` / BLAS not found** — make sure
  `VCPKG_ROOT` points to a vcpkg checkout with the manifest baseline
  (`git clone https://github.com/microsoft/vcpkg` and run `bootstrap-vcpkg.bat`).
- **Windows: slow first configure** — vcpkg is compiling OpenBLAS. Later
  builds reuse the binary cache.
- **`fftsize is not valid`** — the spectral objects require power-of-two
  sizes.
- **Neural-network objects train differently after the ensmallen 3.x
  update** — ensmallen 3.10 changed SGD-family step-size semantics (the step
  size is now divided by the batch size). If you rely on old behavior,
  define `ENS_OLD_SEPARABLE_STEP_BEHAVIOR` before including ensmallen, or
  re-tune the `step_size` attribute.
