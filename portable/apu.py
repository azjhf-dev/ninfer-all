#!/usr/bin/env python3
"""Portable GGUF companion for the NInfer fork. Not the native NInfer engine."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def build_dir(backend):
    return ROOT / ("build-" + backend)


def binary(backend, name):
    suffix = ".exe" if os.name == "nt" else ""
    for folder in (build_dir(backend) / "bin", build_dir(backend) / "bin" / "Release"):
        candidate = folder / (name + suffix)
        if candidate.is_file():
            return str(candidate)
    raise ValueError("Binary missing; first run: python portable/apu.py build --backend " + backend)


def command(args):
    if args.action == "build":
        return [["cmake", "-S", str(ROOT), "-B", str(build_dir(args.backend)),
                 "-DCMAKE_BUILD_TYPE=Release", "-DNINFER_PORTABLE_VULKAN=" +
                 ("ON" if args.backend == "vulkan" else "OFF")],
                ["cmake", "--build", str(build_dir(args.backend)), "--config", "Release",
                 "--target", "llama-completion", "llama-server", "llama-bench",
                 "--parallel", str(args.jobs)]]
    if args.action == "devices":
        return [[binary(args.backend, "llama-completion"), "--list-devices"]]
    model = Path(args.model).expanduser().resolve()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ValueError("Supply an existing local .gguf file. Native .ninfer files are not supported.")
    if args.backend == "cpu" and args.gpu_layers != 0:
        raise ValueError("CPU mode requires --gpu-layers 0")
    if args.backend == "vulkan" and args.gpu_layers and not args.device:
        raise ValueError("Select the 680M explicitly with --device using the devices command output")
    device = ["--device", args.device] if args.device else []
    ngl = str(args.gpu_layers)
    if args.action == "bench":
        return [[binary(args.backend, "llama-bench"), "-m", str(model), "-t", str(args.threads),
                 "-ngl", ngl, "-p", "256", "-n", "64", "-r", "3"] + device]
    name = "llama-server" if args.action == "serve" else "llama-completion"
    result = [binary(args.backend, name), "-m", str(model), "-t", str(args.threads),
              "-ngl", ngl, "-c", str(args.context), "-b", "256", "-ub", "128"]
    if args.action == "serve":
        result += ["--host", "127.0.0.1", "--port", str(args.port), "--parallel", "1"]
    else:
        result += ["-p", args.prompt, "-n", str(args.tokens)]
    return [result + device]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    doctor = sub.add_parser("doctor", help="Print local tool availability without changing the host")
    for action in ("build", "run", "serve", "bench", "devices"):
        p = sub.add_parser(action)
        p.add_argument("--backend", choices=("cpu", "vulkan"), default="cpu")
        p.add_argument("--dry-run", action="store_true")
        if action == "build":
            p.add_argument("--jobs", type=positive, default=2)
        elif action != "devices":
            p.add_argument("--model", required=True)
            p.add_argument("--threads", type=positive, default=8)
            p.add_argument("--device", help="Exact device name from devices; required for Vulkan offload")
            p.add_argument("--gpu-layers", type=nonnegative, default=0,
                           help="Default 0 in BOTH modes. Explicitly increase after listing devices.")
            if action != "bench":
                p.add_argument("--context", type=positive, default=2048)
            if action == "serve":
                p.add_argument("--port", type=positive, default=8080)
            elif action == "run":
                p.add_argument("--prompt", default="Explain what a language model is in one sentence.")
                p.add_argument("--tokens", type=positive, default=64)
    args = parser.parse_args(argv)
    if args.action == "doctor":
        print(json.dumps({"python": sys.version.split()[0], "cmake": shutil.which("cmake"),
                          "cxx": shutil.which("c++") or shutil.which("cl"),
                          "glslc": shutil.which("glslc"), "vulkaninfo": shutil.which("vulkaninfo"),
                          "note": "Tool discovery only, not proof of 680M support. No native NInfer AMD backend."}, indent=2))
        return 0
    try:
        for cmd in command(args):
            print(json.dumps(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
