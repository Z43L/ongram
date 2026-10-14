from __future__ import annotations

import unittest

from onram_cli import COMMANDS, PROJECT_ROOT, ROOT, resolve


class UnifiedCliTests(unittest.TestCase):
    def test_every_registered_command_has_a_source_file(self):
        for name, (script, _cwd) in COMMANDS.items():
            with self.subTest(name=name):
                self.assertTrue(script.is_file(), script)

    def test_project_nram_tools_are_exposed_by_one_cli(self):
        for name in ("nram", "nram-eval", "nram-sweep", "nram-memory", "nram-adapt", "nram-prepare-data"):
            with self.subTest(name=name):
                self.assertIn(name, COMMANDS)
                script, cwd = COMMANDS[name]
                self.assertEqual(cwd, PROJECT_ROOT)
                self.assertTrue(script.is_file())

    def test_known_command_forwards_all_arguments(self):
        script, cwd, forwarded, action = resolve(["convert", "--model", "local", "--output", "x.nram"])
        self.assertEqual(script, ROOT / "nram_convert.py")
        self.assertEqual(cwd, ROOT)
        self.assertEqual(forwarded, ["--model", "local", "--output", "x.nram"])
        self.assertIsNone(action)

    def test_legacy_server_invocations_remain_available(self):
        script, cwd, forwarded, action = resolve(["--model", "m.gguf", "--port", "1234"])
        self.assertEqual(script, ROOT / "onram.py")
        self.assertEqual(cwd, ROOT)
        self.assertEqual(forwarded, ["--model", "m.gguf", "--port", "1234"])
        self.assertIsNone(action)
        self.assertEqual(resolve([])[0], ROOT / "onram.py")

    def test_help_and_version_are_frontend_commands(self):
        self.assertEqual(resolve(["--help"])[3], "help")
        self.assertEqual(resolve(["--version"])[3], "version")


if __name__ == "__main__":
    unittest.main()
