"""Tests for the TrainingLogger."""

import pytest

from onemoreepoch import config
from onemoreepoch.messages import hindi
from onemoreepoch.utils import TrainingLogger


@pytest.fixture(autouse=True)
def reset_mode():
    yield
    config.set_message_mode("classic")


class TestEpochLogging:
    def test_epoch_line_format(self, capsys):
        TrainingLogger().log_epoch(3, 0.123456)
        out = capsys.readouterr().out
        assert "Epoch    3" in out
        assert "0.123456" in out

    def test_log_every_skips_epochs(self, capsys):
        logger = TrainingLogger(log_every=10)
        logger.log_epoch(5, 0.5)
        assert capsys.readouterr().out == ""
        logger.log_epoch(10, 0.4)
        assert "Epoch   10" in capsys.readouterr().out

    def test_classic_mode_has_no_banter(self, capsys):
        TrainingLogger().log_epoch(1, 0.5)
        out = capsys.readouterr().out
        assert "Model:" not in out

    def test_fun_mode_banter_rotates_deterministically(self, capsys):
        config.set_message_mode("hindi")
        logger = TrainingLogger()
        logger.log_epoch(1, 0.5)
        logger.log_epoch(2, 0.4)
        out = capsys.readouterr().out
        assert hindi.EPOCH_BANTER[1] in out
        assert hindi.EPOCH_BANTER[2] in out


class TestLossHealth:
    def test_nan_loss_detected(self, capsys):
        TrainingLogger().log_epoch(1, float("nan"))
        assert "diverged" in capsys.readouterr().out

    def test_inf_loss_detected(self, capsys):
        TrainingLogger().log_epoch(1, float("inf"))
        assert "diverged" in capsys.readouterr().out

    def test_loss_increasing_hint_after_streak(self, capsys):
        logger = TrainingLogger(log_every=1000)  # silence epoch lines
        for epoch, loss in enumerate([0.1, 0.2, 0.3, 0.4], start=1):
            logger.log_epoch(epoch, loss)
        out = capsys.readouterr().out
        assert "increased" in out
        assert out.count("increased") == 1  # hint fires once, not per epoch

    def test_no_hint_when_loss_decreases(self, capsys):
        logger = TrainingLogger(log_every=1000)
        for epoch, loss in enumerate([0.4, 0.3, 0.2, 0.1], start=1):
            logger.log_epoch(epoch, loss)
        assert "increased" not in capsys.readouterr().out


class TestFinish:
    def test_summary_reports_epochs_and_best(self, capsys):
        logger = TrainingLogger(log_every=1000)
        logger.log_epoch(1, 0.5)
        logger.log_epoch(2, 0.2)
        logger.log_epoch(3, 0.3)
        logger.finish()
        out = capsys.readouterr().out
        assert "3 epochs" in out
        assert "0.200000" in out
