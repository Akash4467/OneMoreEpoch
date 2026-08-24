"""Smoke tests: examples must run end-to-end without crashing.

Runs each example's main() in-process (fast, no subprocess) and checks
the key outcomes: training converges, and the debug showcase prints
every scenario in every mode without raising.
"""

import pytest

from onemoreepoch import config


@pytest.fixture(autouse=True)
def reset_mode():
    yield
    config.set_message_mode("classic")
    config.set_debug_checks(False)


def test_train_regression_runs_and_converges(capsys):
    from examples import train_regression

    train_regression.main()
    out = capsys.readouterr().out
    assert "Training complete" in out
    assert "model(3.0)" in out


def test_debug_showcase_covers_all_modes(capsys):
    from examples import debug_showcase

    debug_showcase.main()
    out = capsys.readouterr().out
    # Every mode section appears for the error scenarios.
    for mode in config.MESSAGE_MODES:
        assert f"--- {mode} ---" in out
    # Signature lines from each personality made it to the output.
    assert "Ye shaadi nahi ho sakti" in out  # hindi, JOURNEY.md verbatim
    assert "Inner dimensions must agree" in out  # classic
    assert "linear algebra" in out  # roast
    # Gradient explosion warning fired in fun modes too.
    assert "Bhai throttle maar" in out
    # NaN loss detection from the TrainingLogger section.
    assert "diverged" in out
