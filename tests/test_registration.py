"""New workflows have one entry point; archived graphs retain their protocol."""
import unittest

import test_native  # Register the package without a running ComfyUI.
import rs_tipo_native as package


class RegistrationTests(unittest.TestCase):
    def test_one_non_deprecated_entry_point(self):
        visible = [name for name, cls in package.NODE_CLASS_MAPPINGS.items()
                   if not getattr(cls, 'DEPRECATED', False)]
        self.assertEqual(visible, ['RSTIPOPrompt'])

    def test_saved_legacy_class_ids_and_output_protocols_remain_available(self):
        expected = {
            'RSTIPONativeCompletion': ('generate', ('completion', 'info')),
            'RSTIPONativeExpand': ('expand', ('prompt', 'added_tags', 'info')),
            'RSTIPONativeControlledExpand': ('expand', ('prompt', 'added_tags', 'info')),
            'RSTIPONativePipeExpand': ('expand', ('building_pipe', 'common_prompt', 'info')),
        }
        for name, (function, outputs) in expected.items():
            with self.subTest(name=name):
                cls = package.NODE_CLASS_MAPPINGS[name]
                self.assertTrue(cls.DEPRECATED)
                self.assertEqual(cls.CATEGORY, 'RS/TIPO Native/Legacy')
                self.assertEqual(cls.FUNCTION, function)
                self.assertEqual(cls.RETURN_NAMES, outputs)
                self.assertTrue(callable(getattr(cls(), function)))
                self.assertIn('required', cls.INPUT_TYPES())
