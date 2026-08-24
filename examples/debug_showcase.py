"""Debug showcase: every OneMoreEpoch failure scenario, in every mode.

Deliberately triggers each error/warning the framework knows how to
explain, and prints the resulting message in classic, hindi, and roast
modes side by side — a tour of the fun-mode system (JOURNEY.md,
ADR-010).

Run:
    python examples/debug_showcase.py
"""

import warnings

import numpy as np

from onemoreepoch import Tensor, config
from onemoreepoch.exceptions import GradientWarning
from onemoreepoch.nn import Linear
from onemoreepoch.optim import SGD
from onemoreepoch.utils import TrainingLogger


def scenario_matmul_mismatch() -> None:
    """(64, 128) @ (32, 10) — inner dims disagree."""
    Tensor.randn(64, 128) @ Tensor.randn(32, 10)


def scenario_broadcast_failure() -> None:
    """(3, 2) + (4, 5) — no broadcasting rule can save this."""
    Tensor.randn(3, 2) + Tensor.randn(4, 5)


def scenario_backward_no_grad() -> None:
    """backward() on a tensor that never asked for gradients."""
    Tensor(np.array(3.0)).backward()


def scenario_backward_non_scalar() -> None:
    """backward() on a matrix without a seed gradient."""
    Tensor.randn(4, 4, requires_grad=True).sum(axis=0).backward()


def scenario_bad_learning_rate() -> None:
    """A negative learning rate."""
    SGD(Linear(2, 2).parameters(), lr=-0.1)


def scenario_empty_parameters() -> None:
    """An optimizer with nothing to optimize."""
    SGD([], lr=0.01)


ERROR_SCENARIOS = [
    scenario_matmul_mismatch,
    scenario_broadcast_failure,
    scenario_backward_no_grad,
    scenario_backward_non_scalar,
    scenario_bad_learning_rate,
    scenario_empty_parameters,
]


def show_error_scenarios() -> None:
    for scenario in ERROR_SCENARIOS:
        print("=" * 72)
        print(f"# {scenario.__name__}: {scenario.__doc__}")
        for mode in config.MESSAGE_MODES:
            config.set_message_mode(mode)
            print(f"\n--- {mode} ---")
            try:
                scenario()
            except Exception as exc:  # noqa: BLE001 — showcase prints everything
                print(exc)
        print()


def show_gradient_explosion() -> None:
    """Exploding gradients: repeated squaring of a large value."""
    print("=" * 72)
    print(f"# gradient_explosion: {show_gradient_explosion.__doc__}")
    config.set_debug_checks(True)
    for mode in config.MESSAGE_MODES:
        config.set_message_mode(mode)
        print(f"\n--- {mode} ---")
        x = Tensor(np.array(50.0), requires_grad=True)
        loss = ((x * x) * x).sum()  # d/dx = 3x^2 = 7500 — boom
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", GradientWarning)
            loss.backward()
        for warning in caught:
            print(warning.message)
    config.set_debug_checks(False)
    print()


def show_nan_loss_and_banter() -> None:
    """NaN loss detection + epoch banter from the TrainingLogger."""
    print("=" * 72)
    print(f"# training_logger: {show_nan_loss_and_banter.__doc__}")
    for mode in config.MESSAGE_MODES:
        config.set_message_mode(mode)
        print(f"\n--- {mode} ---")
        logger = TrainingLogger()
        logger.log_epoch(1, 0.75)
        logger.log_epoch(2, 0.42)
        logger.log_epoch(3, float("nan"))
        logger.finish()
    print()


def main() -> None:
    show_error_scenarios()
    show_gradient_explosion()
    show_nan_loss_and_banter()
    config.set_message_mode("classic")


if __name__ == "__main__":
    main()
