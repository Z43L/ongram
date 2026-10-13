import unittest
from pathlib import Path
from onram import Onram, ROOT


class OnramTests(unittest.TestCase):
    def test_completion_options_translate_ollama_options(self):
        options = Onram._completion_options("prompt", {
            "num_predict": 32,
            "temperature": 0.25,
            "top_k": 12,
            "top_p": 0.8,
            "repeat_penalty": 1.2,
            "stop": ["END"],
        })
        self.assertEqual(options["n_predict"], 32)
        self.assertEqual(options["temperature"], 0.25)
        self.assertEqual(options["top_k"], 12)
        self.assertEqual(options["top_p"], 0.8)
        self.assertEqual(options["repeat_penalty"], 1.2)
        self.assertEqual(options["stop"], ["END"])

    def test_completion_options_reject_context_overflow(self):
        with self.assertRaisesRegex(ValueError, "num_predict"):
            Onram._completion_options("prompt", {"num_predict": 513})

    def test_dataset_selection_cannot_escape_project_data(self):
        self.assertEqual(Onram._dataset("nra_qa_train.jsonl"), ROOT / "data" / "nra_qa_train.jsonl")
        with self.assertRaises(ValueError):
            Onram._dataset("../../etc/passwd")

    def test_name_tracks_the_active_gguf(self):
        app = Onram.__new__(Onram)
        app.model_path = Path("/models/small-Q4_K_M.gguf")
        self.assertEqual(app.model_name, "small:latest")

    def test_non_q4_gguf_name_is_preserved(self):
        app = Onram.__new__(Onram)
        app.model_path = Path("/models/custom-instruct-f16.gguf")
        self.assertEqual(app.model_name, "custom-instruct-f16:latest")


if __name__ == "__main__":
    unittest.main()
