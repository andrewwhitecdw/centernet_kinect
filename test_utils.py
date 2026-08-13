from unittest.mock import patch

from pipeline import utils


def test_get_model_exits_when_no_checkpoint():
    """When no checkpoint is found, get_model should terminate with status -1."""
    calls = []

    def fake_exit(code):
        calls.append(code)
        raise SystemExit(code)

    with patch.object(utils, "glob", return_value=[]):
        with patch.object(utils.sys, "exit", fake_exit):
            try:
                utils.get_model()
            except SystemExit as exc:
                assert exc.code == -1
