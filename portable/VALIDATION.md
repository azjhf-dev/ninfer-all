# Validation record

Date: 2026-11-15 (Singapore).

## Passed

Test host: Linux x86_64, GCC 11.4.0, CMake 3.31, Python 3.10. This is **not** the user's Ryzen 6900HX / Radeon 680M. No Ryzen or GPU performance inference is made from these tests.

1. Configured the CPU companion with CUDA, HIP and Vulkan OFF; native-host tuning OFF; AVX2/FMA/F16C ON; AVX-512 OFF.
2. Compiled `llama-completion`, `llama-server`, and `llama-bench` using the pinned llama.cpp revision. Compiled with an existing checkout of that exact revision to avoid duplicate downloads during iteration.
3. Separately configured a clean default FetchContent build successfully: immutable archive download and SHA-256 validation passed without the local-source override.
4. Five Python unit tests passed: positive argument checks; separate CPU/Vulkan build directories; native format rejection; explicit CPU offload rejection and literal prompt handling; loopback-only single-slot server configuration.
5. Loaded the real tiny GGUF `ggml-org/models`, `tinyllamas/stories260K.gguf` (SHA-256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`). CPU generation completed for `Once upon a time`.
6. Started the real CPU server on loopback port 18081 with that model: `GET /health` returned HTTP 200 and `{"status":"ok"}`; `POST /completion` generated 8 tokens with content `, there was a little girl`. Server was stopped after the test.
7. Ran the real benchmark tool with 256 prompt / 64 generation tokens, 3 repetitions, 2 CPU threads, zero offloaded layers. Both tests completed. Toy-model rates are deliberately omitted: they say nothing about 27B performance on the target APU.
8. Checked dynamic dependencies: no CUDA or ROCm library dependency in the CPU server.
9. CPU `--list-devices` reported no accelerators, as expected. CPU launcher rejected `--gpu-layers 1` with a clear error.

## Not passed / not attempted

- Vulkan configuration was attempted and failed at dependency detection: Vulkan loader library, headers and `glslc` are not installed on the test host. Thus the Vulkan target has **not been compiled**, and no shader correctness or performance claim is made.
- No access to a Radeon 680M or Ryzen 6900HX. Hardware validation, RAM sizing, thread-count tuning, partial/full GPU offload and comparison with CPU remain open.
- Windows build and driver setup not tested.
- The linked 27B condensed GGUF was not downloaded or loaded. Other GGUF families, quantizations, multilingual quality, long context, chat templates, vision and concurrent requests were not qualified.
- Native NInfer was not compiled: CUDA is not present, and it still requires NVIDIA hardware. No CPU-only native executor, AMD kernel port, `.ninfer` support or rk8v4 cache has been implemented by this branch.

## Repeat the local checks

```sh
python -m unittest discover -s portable/tests -v
python portable/apu.py build --backend cpu
python portable/apu.py run --model /path/to/tiny-model.gguf --threads 2 --context 512 --tokens 12 --prompt "Once upon a time"
python portable/apu.py bench --model /path/to/tiny-model.gguf --threads 2
python portable/apu.py serve --model /path/to/tiny-model.gguf --threads 2 --context 512
```

Use the same model, settings and power state for target-hardware comparisons. Preserve error logs and note driver versions before changing settings. Build and model caches are ignored by Git; no model weights or binaries are committed.
