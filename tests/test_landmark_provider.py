'''
import sys
import types
import unittest

import MrEmo


class LandmarkProviderTests(unittest.TestCase):
    def test_handles_missing_mediapipe_solutions(self):
        fake_mediapipe = types.SimpleNamespace()
        original = sys.modules.get("mediapipe")
        sys.modules["mediapipe"] = fake_mediapipe
        try:
            provider = MrEmo.LandmarkProvider()
        finally:
            if original is None:
                sys.modules.pop("mediapipe", None)
            else:
                sys.modules["mediapipe"] = original

        self.assertFalse(provider.available)


if __name__ == "__main__":
    unittest.main()
'''
# You can delete this file. Not needed but a safety net.