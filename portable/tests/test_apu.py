import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("apu", Path(__file__).parents[1] / "apu.py")
apu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apu)


class Commands(unittest.TestCase):
    def test_positive(self):
        for bad in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                apu.positive(bad)
        self.assertEqual(apu.positive("8"), 8)

    def test_build_profiles_are_separate(self):
        for backend, enabled in (("cpu", "OFF"), ("vulkan", "ON")):
            args = argparse.Namespace(action="build", backend=backend, jobs=2)
            commands = apu.command(args)
            self.assertIn("-DNINFER_PORTABLE_VULKAN=" + enabled, commands[0])
            self.assertIn(str(apu.ROOT / ("build-" + backend)), commands[0])
            self.assertIn("llama-server", commands[1])

    def test_reject_native_format(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "weights.ninfer"
            model.touch()
            args = argparse.Namespace(action="run", backend="cpu", model=str(model))
            with self.assertRaisesRegex(ValueError, "existing local .gguf"):
                apu.command(args)

    def test_cpu_no_offload_and_literal_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model with spaces.gguf"
            model.touch()
            args = argparse.Namespace(action="run", backend="cpu", model=str(model),
                                      threads=8, gpu_layers=0, context=2048, device=None,
                                      prompt="literal $HOME ; echo bad", tokens=4)
            with patch.object(apu, "binary", return_value="llama-completion"):
                cmd = apu.command(args)[0]
                self.assertEqual(cmd[cmd.index("-p") + 1], args.prompt)
                self.assertEqual(cmd[cmd.index("-m") + 1], str(model))
                self.assertEqual(cmd[cmd.index("-ngl") + 1], "0")
                args.gpu_layers = 1
                with self.assertRaisesRegex(ValueError, "CPU mode"):
                    apu.command(args)

    def test_server_loopback_single_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.gguf"
            model.touch()
            args = argparse.Namespace(action="serve", backend="vulkan", model=str(model),
                                      threads=8, gpu_layers=4, context=2048, port=8080, device="Vulkan0")
            with patch.object(apu, "binary", return_value="llama-server"):
                cmd = apu.command(args)[0]
                self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
                self.assertEqual(cmd[cmd.index("--parallel") + 1], "1")
                self.assertEqual(cmd[cmd.index("-ngl") + 1], "4")
                self.assertEqual(cmd[cmd.index("--device") + 1], "Vulkan0")
                args.device = None
                with self.assertRaisesRegex(ValueError, "680M explicitly"):
                    apu.command(args)


if __name__ == "__main__":
    unittest.main()
