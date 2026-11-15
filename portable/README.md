# Ryzen 9 6900HX / Radeon 680M companion

## What this branch delivers

A real fork preserving NInfer upstream, plus a **separate llama.cpp GGUF build and launcher** for CPU and opt-in Vulkan. This is **not a native AMD port of NInfer**. The original root build remains CUDA-only. Nothing in `src/`, `kernel/`, or the upstream model executor has been replaced.

- Upstream base: `iamwavecut/ninfer-all` at `40d913db2e87e2211fce4089c0c7920480a5e7a0`.
- History branch: `build/ryzen-6900hx-680m-40d913d`. `master` retains the upstream snapshot.
- Portable backend: `ggml-org/llama.cpp` at `900f01ba09ac5735e16a634a2c37e000639c936a`, downloaded by immutable commit with SHA-256 verification. Apache-2.0 NInfer and MIT llama.cpp retain their own licenses.
- CPU profile: AVX2, FMA, F16C; native-host auto-tuning and AVX-512 disabled. No CUDA/ROCm dependency.
- 680M route: llama.cpp Vulkan, **experimental and not built or tested on a 680M here**. No claimed HIP support.
- Runtime defaults: 8 CPU threads, 2,048 context, batch 256 / microbatch 128, **zero GPU layers in both modes** until explicitly changed. These are safe starting points, not measured optimum settings.

## Build (Linux or Windows)

Requires Python 3.9+, CMake 3.22+, a C++17 compiler, and internet access for the pinned dependency. On Windows, use an x64 Visual Studio Developer terminal with C++ build tools; the scripts use the Release configuration. Windows has not been tested here.

```sh
git clone --branch build/ryzen-6900hx-680m-40d913d https://github.com/azjhf-dev/ninfer-all.git
cd ninfer-all
python portable/apu.py doctor
python portable/apu.py build --backend cpu --jobs 2
python -m unittest discover -s portable/tests -v
```

Use `python3` instead of `python` where needed. Two build jobs is deliberately conservative for memory; this does not set inference thread count. The build produces `llama-completion`, `llama-server`, and `llama-bench` under `portable/build-cpu/bin/` (or `bin/Release/` on multi-config generators). First compilation can take several minutes.

For a Debian/Ubuntu CPU setup, typical prerequisite packages are `build-essential cmake python3`. Confirm the CMake version meets the minimum. The helper never installs packages, changes BIOS settings, forces GPU memory reservations, or uses root.

## Run a local GGUF

Download a compatible GGUF separately. There is no model downloader or model upload in this helper. Use a small model to validate installation before trying a 27B model.

```sh
python portable/apu.py run --model /path/to/model.gguf --threads 8 --context 2048 --prompt "Hello"
python portable/apu.py serve --model /path/to/model.gguf --threads 8 --context 2048
python portable/apu.py bench --model /path/to/model.gguf --threads 8
```

Server binds to `127.0.0.1:8080`, one slot. It is not exposed to the LAN. Use `/health`, `/v1/models`, and `/v1/chat/completions` as supported by the pinned llama.cpp server. Remote HTTPS model downloads are disabled at build time. The server and upstream software still have their own features; this is not a claim that all upstream tools are network-isolated.

`bench` runs 256 prompt tokens / 64 generated tokens, 3 repetitions. Compare the same model, context, power profile and input on CPU and Vulkan. Test 4, 6, and 8 threads; do not assume all 16 logical threads is faster. `--dry-run` prints command arrays without executing them (a local GGUF and built executable must still exist for runtime commands).

## 680M Vulkan trial

Install a working Vulkan driver and SDK/toolchain first. On Linux this usually means the appropriate Mesa/RADV driver, Vulkan development headers/loader, `glslc`, and SPIR-V headers with CMake package files. Typical Debian/Ubuntu package names include `mesa-vulkan-drivers libvulkan-dev vulkan-tools glslc spirv-headers`, but versions and packaging vary. On Windows install an appropriate AMD driver plus the Vulkan SDK. Run `vulkaninfo --summary` and confirm the **Radeon 680M**, not a software renderer, is listed.

```sh
python portable/apu.py build --backend vulkan --jobs 2
python portable/apu.py devices --backend vulkan
python portable/apu.py run --backend vulkan --model /path/to/small-model.gguf --gpu-layers 1 --device Vulkan0
python portable/apu.py bench --backend vulkan --model /path/to/small-model.gguf --gpu-layers 1 --device Vulkan0
```

Replace `Vulkan0` above with the exact device ID shown for your **680M**. Vulkan offload requires explicit device selection so a different GPU is not selected automatically. Start with one layer, inspect the backend logs, then increase gradually if memory and stability permit. Device enumeration alone does not prove offload; check that logs place layers on the Vulkan GPU. No `HSA_OVERRIDE_GFX_VERSION` hack is applied. If Vulkan fails to configure, build, or run, use the separately built CPU profile instead. The script does not silently claim GPU acceleration after a failure.

The iGPU shares system RAM and bandwidth with the CPU. A larger BIOS framebuffer is not extra memory. A roughly 12.5 GB 27B quant needs additional memory for the OS, model buffers, context and GPU allocations. Total RAM, dual-channel bandwidth, drivers and power limits are unknown. No claim that the linked 27B model fits or is fast on this laptop, and no 75K context / 60 tok/s promise from the NVIDIA post transfers here. Vulkan can be slower than CPU on an APU: measure both.

## Formats and feature boundaries

| Feature | Portable companion |
| --- | --- |
| Compatible GGUF text inference | CPU smoke-tested with a tiny model |
| Linked Qwen3.8 condensed 27B GGUF | Not loaded/tested here; pinned loader has `qwen35` support, but that is not validation of this artifact |
| Native `.ninfer` artifacts | Not supported; launcher rejects them |
| NInfer CUDA/PTX kernels | Not ported |
| NInfer rk8v4 cache / MTP / DFlash2 / route profiles | Not inherited by this companion |
| Native NInfer CPU-only model execution | Not implemented |
| 680M Vulkan | Build option and launcher supplied; validation pending on suitable driver/toolchain/hardware |
| HIP/ROCm on gfx1035 | Not implemented or claimed |

The linked ASCII-condensed model prunes non-Latin vocabulary. Arabic and other removed scripts rely more heavily on byte fallback; translation quality is unverified. Prefer an ordinary multilingual GGUF for Arabic work. This branch does not further alter the vocabulary or weights.

## Why not flip CUDA to HIP?

The inspected snapshot has 427 `.cu` files under `src/`. The root CMake project declares CUDA mandatory and validates NVIDIA SM architectures. Device allocation, stream/graph scheduling and the model executor are coupled to CUDA; operation implementations use PTX, warp primitives, NVIDIA tensor instructions and cuBLAS routes. CPU expert-sharing code is partial work inside that GPU executor, not a complete CPU inference backend. A native port requires a backend boundary plus implementations and numerical tests for every supported operation, quant layout, cache policy and architecture. A source translation or fake success stub would not satisfy that requirement.

See [VALIDATION.md](VALIDATION.md) for exactly what was tested. Keep native AMD work separate until it passes real inference and numerical comparisons.

## Sources

- https://github.com/iamwavecut/ninfer-all (inspected source checkout, exact base above)
- https://github.com/ggml-org/llama.cpp (inspected pinned checkout, including Vulkan CMake and model architecture registry)
- https://huggingface.co/fakezeta/Qwen3.8-27B-UD-Q3_K_XL-Condensed-P2 (model author's format and vocabulary claims)
- https://huggingface.co/ggml-org/models (tiny test model repository)
