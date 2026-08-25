import pytest

from onemoreepoch import config
from onemoreepoch.messages import hindi
from onemoreepoch.utils import TrainingLogger


# Resets the message mode to classic after each test
@pytest.fixture(autouse=True)
def reset_mode():
    yield
    config.set_message_mode("classic")


# Tests per-epoch log line formatting
class TestEpochLogging:
    # Checks the epoch line contains the epoch number and loss value
    def test_epoch_line_format(self, capsys):
        TrainingLogger().log_epoch(3, 0.123456)
        out = capsys.readouterr().out
        assert "Epoch    3" in out
        assert "0.123456" in out

    # Checks log_every skips epochs that aren't a multiple of the interval
    def test_log_every_skips_epochs(self, capsys):
        logger = TrainingLogger(log_every=10)
        logger.log_epoch(5, 0.5)
        assert capsys.readouterr().out == ""
        logger.log_epoch(10, 0.4)
        assert "Epoch   10" in capsys.readouterr().out

    # Checks classic mode never prints banter
    def test_classic_mode_has_no_banter(self, capsys):
        TrainingLogger().log_epoch(1, 0.5)
        out = capsys.readouterr().out
        assert "Model:" not in out

    # Checks fun-mode banter rotates deterministically through the banter list
    def test_fun_mode_banter_rotates_deterministically(self, capsys):
        config.set_message_mode("hindi")
        logger = TrainingLogger()
        logger.log_epoch(1, 0.5)
        logger.log_epoch(2, 0.4)
        out = capsys.readouterr().out
        assert hindi.EPOCH_BANTER[1] in out
        assert hindi.EPOCH_BANTER[2] in out


# Tests loss-health detection (NaN/Inf and sustained increases)
class TestLossHealth:
    # Checks a NaN loss is detected and reported as diverged
    def test_nan_loss_detected(self, capsys):
        TrainingLogger().log_epoch(1, float("nan"))
        assert "diverged" in capsys.readouterr().out

    # Checks an infinite loss is detected and reported as diverged
    def test_inf_loss_detected(self, capsys):
        TrainingLogger().log_epoch(1, float("inf"))
        assert "diverged" in capsys.readouterr().out

    # Checks the increasing-loss hint fires once after a sustained streak
    def test_loss_increasing_hint_after_streak(self, capsys):
        logger = TrainingLogger(log_every=1000)
        for epoch, loss in enumerate([0.1, 0.2, 0.3, 0.4], start=1):
            logger.log_epoch(epoch, loss)
        out = capsys.readouterr().out
        assert "increased" in out
        assert out.count("increased") == 1

    # Checks no hint fires when the loss is decreasing
    def test_no_hint_when_loss_decreases(self, capsys):
        logger = TrainingLogger(log_every=1000)
        for epoch, loss in enumerate([0.4, 0.3, 0.2, 0.1], start=1):
            logger.log_epoch(epoch, loss)
        assert "increased" not in capsys.readouterr().out


# Tests the end-of-training summary
class TestFinish:
    # Checks the summary reports epoch count and best loss
    def test_summary_reports_epochs_and_best(self, capsys):
        logger = TrainingLogger(log_every=1000)
        logger.log_epoch(1, 0.5)
        logger.log_epoch(2, 0.2)
        logger.log_epoch(3, 0.3)
        logger.finish()
        out = capsys.readouterr().out
        assert "3 epochs" in out
        assert "0.200000" in out
