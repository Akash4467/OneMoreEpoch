import warnings

import numpy as np

from onemoreepoch import Tensor, config
from onemoreepoch.exceptions import GradientWarning
from onemoreepoch.nn import Linear
from onemoreepoch.optim import SGD
from onemoreepoch.utils import TrainingLogger


# Triggers a matmul inner-dimension mismatch
def scenario_matmul_mismatch() -> None:
    Tensor.randn(64, 128) @ Tensor.randn(32, 10)


# Triggers a broadcast failure between incompatible shapes
def scenario_broadcast_failure() -> None:
    Tensor.randn(3, 2) + Tensor.randn(4, 5)


# Triggers backward() on a tensor that never asked for gradients
def scenario_backward_no_grad() -> None:
    Tensor(np.array(3.0)).backward()


# Triggers backward() on a non-scalar tensor without a seed gradient
def scenario_backward_non_scalar() -> None:
    Tensor.randn(4, 4, requires_grad=True).sum(axis=0).backward()


# Triggers optimizer construction with a negative learning rate
def scenario_bad_learning_rate() -> None:
    SGD(Linear(2, 2).parameters(), lr=-0.1)


# Triggers optimizer construction with an empty parameter list
def scenario_empty_parameters() -> None:
    SGD([], lr=0.01)


_SCENARIO_DESCRIPTIONS = {
    scenario_matmul_mismatch: "(64, 128) @ (32, 10) - inner dims disagree",
    scenario_broadcast_failure: "(3, 2) + (4, 5) - no broadcasting rule can save this",
    scenario_backward_no_grad: "backward() on a tensor that never asked for gradients",
    scenario_backward_non_scalar: "backward() on a matrix without a seed gradient",
    scenario_bad_learning_rate: "A negative learning rate",
    scenario_empty_parameters: "An optimizer with nothing to optimize",
}

ERROR_SCENARIOS = [
    scenario_matmul_mismatch,
    scenario_broadcast_failure,
    scenario_backward_no_grad,
    scenario_backward_non_scalar,
    scenario_bad_learning_rate,
    scenario_empty_parameters,
]


# Runs every error scenario in every message mode and prints the resulting message
def show_error_scenarios() -> None:
    for scenario in ERROR_SCENARIOS:
        print("=" * 72)
        print(f"# {scenario.__name__}: {_SCENARIO_DESCRIPTIONS[scenario]}")
        for mode in config.MESSAGE_MODES:
            config.set_message_mode(mode)
            print(f"\n--- {mode} ---")
            try:
                scenario()
            except Exception as exc:  # noqa: BLE001
                print(exc)
        print()


# Triggers and prints a gradient-explosion warning in every message mode
def show_gradient_explosion() -> None:
    print("=" * 72)
    print("# gradient_explosion: exploding gradients from repeated squaring")
    config.set_debug_checks(True)
    for mode in config.MESSAGE_MODES:
        config.set_message_mode(mode)
        print(f"\n--- {mode} ---")
        x = Tensor(np.array(50.0), requires_grad=True)
        loss = ((x * x) * x).sum()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", GradientWarning)
            loss.backward()
        for warning in caught:
            print(warning.message)
    config.set_debug_checks(False)
    print()


# Demonstrates NaN loss detection and epoch banter from the TrainingLogger
def show_nan_loss_and_banter() -> None:
    print("=" * 72)
    print("# training_logger: NaN loss detection + epoch banter")
    for mode in config.MESSAGE_MODES:
        config.set_message_mode(mode)
        print(f"\n--- {mode} ---")
        logger = TrainingLogger()
        logger.log_epoch(1, 0.75)
        logger.log_epoch(2, 0.42)
        logger.log_epoch(3, float("nan"))
        logger.finish()
    print()


# Runs every showcase section and resets the message mode to classic
def main() -> None:
    show_error_scenarios()
    show_gradient_explosion()
    show_nan_loss_and_banter()
    config.set_message_mode("classic")


if __name__ == "__main__":
    main()
