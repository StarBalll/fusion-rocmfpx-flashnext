# fusion build environment

Two supported ways to get a build environment.

## 1. Existing ROCm host / Strix Halo machine

- ROCm 6.x+ with HIP, gfx1151 target
- `hipcc` on PATH (`/opt/rocm/bin`)
- optional: `hipcub`/`rocprim` headers at `/opt/rocm-extras/opt/rocm-7.2.4/include`
  (auto-detected by `fusion/build.sh` if the system ROCm lacks them)

```bash
./fusion/build.sh --skip-patches   # or without flag to resync upstream first
```

## 2. Container (reproducible)

```bash
docker build -t fusion-rocmfpx -f fusion/env/Dockerfile .

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v "$PWD":/work -w /work \
  fusion-rocmfpx \
  ./fusion/build.sh --skip-patches
```

The image is based on `kyuz0/amd-strix-halo-toolboxes:latest` (ROCm 7.x userland for
Strix Halo, no kernel driver inside the container). The `build.sh` inside the container
installs to `/opt/llama-rocmfpx-next` inside the image; add `-v llm-opt:/opt/llama-rocmfpx-next`
to persist the install prefix.

## GPU runtime notes

- Benchmarks must run serially; never hold two 20GB+ models resident.
- Server logs go to files, never a TTY.
- `LD_LIBRARY_PATH=/opt/llama-rocmfpx-next/lib` is required for all binaries in the
  install prefix (append AFTER sourcing rocm.sh, which resets it).
