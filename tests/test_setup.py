import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.setup import ModelSetup


class TestModelSetupInfer(unittest.TestCase):
    def test_infer_with_none_load_exits_before_isfile(self):
        with patch('os.path.isfile') as mock_isfile, \
             patch('builtins.exit', side_effect=SystemExit(-1)) as mock_exit, \
             patch('builtins.print') as mock_print:
            with self.assertRaises(SystemExit) as cm:
                ModelSetup(infer=True, load=None)

        self.assertEqual(cm.exception.code, -1)
        mock_isfile.assert_not_called()
        mock_exit.assert_called_once_with(-1)
        mock_print.assert_called_once_with(\"no such file exists: None\")


if __name__ == '__main__':
    unittest.main()
