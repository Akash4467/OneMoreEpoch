import math

from onemoreepoch import config
from onemoreepoch.messages import get_banter, get_message

_INCREASE_STREAK_HINT = 3


# Logs epoch progress and flags unhealthy loss curves
class TrainingLogger:
    # Initializes tracking state for best loss and increase streaks
    def __init__(self, *, log_every: int = 1) -> None:
        self.log_every = max(1, log_every)
        self._epochs_seen = 0
        self._best_loss = math.inf
        self._prev_loss: float | None = None
        self._increase_streak = 0
        self._increase_hint_shown = False

    # Records one epoch's loss and prints a progress line
    def log_epoch(self, epoch: int, loss: float) -> None:
        loss = float(loss)
        self._epochs_seen += 1

        if math.isnan(loss) or math.isinf(loss):
            print(get_message("nan_loss", value=loss, epoch=epoch))
            return

        self._best_loss = min(self._best_loss, loss)
        self._track_increase(loss)

        if epoch % self.log_every == 0:
            line = f"Epoch {epoch:>4} | Loss: {loss:.6f}"
            if config.get_message_mode() != "classic":
                line += f"  {get_banter(epoch)}"
            print(line)

    # Prints the end-of-training summary
    def finish(self) -> None:
        best = self._best_loss if self._best_loss != math.inf else float("nan")
        print(get_message("training_complete", epochs=self._epochs_seen, best=best))

    # Warns once if loss keeps climbing for several consecutive epochs
    def _track_increase(self, loss: float) -> None:
        if self._prev_loss is not None and loss > self._prev_loss:
            self._increase_streak += 1
        else:
            self._increase_streak = 0
            self._increase_hint_shown = False
        self._prev_loss = loss

        if (
            self._increase_streak >= _INCREASE_STREAK_HINT
            and not self._increase_hint_shown
        ):
            self._increase_hint_shown = True
            print(
                get_message("loss_increasing", count=self._increase_streak, value=loss)
            )


__all__ = ["TrainingLogger"]
