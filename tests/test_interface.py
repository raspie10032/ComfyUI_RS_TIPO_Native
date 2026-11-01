import unittest
from unittest.mock import patch

import test_native  # Register the extension package without ComfyUI.
from rs_tipo_native.interface import validate_inputs, expand_strings, pipe_is_requested
from rs_tipo_native.nodes import RSTIPOPrompt


class InterfaceTests(unittest.TestCase):
    def test_count_excludes_stale_inputs_and_keeps_weighted_originals(self):
        self.assertEqual(validate_inputs("campus", 1, {"character1": "(blonde:1.2)", "character2": object()}), ("(blonde:1.2)",))
        for count in (-1, 11, True):
            with self.assertRaises(ValueError):
                validate_inputs("scene", count, {})
        with self.assertRaisesRegex(ValueError, "character 2"):
            validate_inputs("scene", 2, {"character1": "one"})
        self.assertEqual(validate_inputs("scene", 0, {}), ())

    def test_all_character_calls_use_original_scene(self):
        calls = []
        def run(common, person):
            calls.append((common, person))
            output = (common if person is None else person) + ", extra"
            return output, {"output": output}
        scene, characters, report = expand_strings("campus", ("blonde", "black hair"), run)
        self.assertEqual(calls, [("campus", None), ("campus", "blonde"), ("campus", "black hair")])
        self.assertEqual(scene, "campus, extra")
        self.assertEqual(characters, ("blonde, extra", "black hair, extra"))
        self.assertEqual([row["slot"] for row in report], ["common", 1, 2])
        self.assertEqual(expand_strings("campus", (), run, expand_common=False), ("campus", (), []))

    def test_pipe_error_precedes_model_load(self):
        graph = {"2": {"inputs": {"building_pipe": ["1", 0]}}}
        with patch("rs_tipo_native.nodes.acquire_model") as model:
            with self.assertRaisesRegex(ValueError, "at least one character"):
                RSTIPOPrompt().expand(character_count=0, common_prompt="scene", prompt=graph, unique_id="1")
            model.assert_not_called()
        self.assertTrue(pipe_is_requested(graph, 1))
        self.assertFalse(pipe_is_requested(graph, 2))

    def test_missing_adapter_allows_strings_but_rejects_pipe(self):
        with patch("rs_tipo_native.interface.pipe_adapter", return_value=None), patch("rs_tipo_native.nodes.acquire_model") as model:
            with self.assertRaisesRegex(ValueError, "Install it"):
                RSTIPOPrompt().expand(character_count=1, character1="blonde", prompt={"2": {"inputs": {"pipe": ["1", 0]}}}, unique_id="1")
            model.assert_not_called()

    def test_return_indexes_are_fixed(self):
        self.assertEqual(RSTIPOPrompt.RETURN_NAMES, ("building_pipe", "common_prompt", *(f"character{i}" for i in range(1, 11))))
        schema = RSTIPOPrompt.INPUT_TYPES()
        self.assertTrue(schema["optional"]["character10"][1]["forceInput"])
