import pytest

from onemoreepoch import config


# Resets message mode and debug checks to defaults after each test
@pytest.fixture(autouse=True)
def reset_mode():
    yield
    config.set_message_mode("classic")
    config.set_debug_checks(False)


# Checks the regression example runs end to end and converges
def test_train_regression_runs_and_converges(capsys):
    from examples import train_regression

    train_regression.main()
    out = capsys.readouterr().out
    assert "Training complete" in out
    assert "model(3.0)" in out


# Checks the debug showcase runs every scenario in every message mode without raising
def test_debug_showcase_covers_all_modes(capsys):
    from examples import debug_showcase

    debug_showcase.main()
    out = capsys.readouterr().out
    for mode in config.MESSAGE_MODES:
        assert f"--- {mode} ---" in out
    assert "Ye shaadi nahi ho sakti" in out
    assert "Inner dimensions must agree" in out
    assert "linear algebra" in out
    assert "Bhai throttle maar" in out
    assert "diverged" in out
