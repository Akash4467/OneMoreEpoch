import importlib

import pytest

from onemoreepoch import config
from onemoreepoch.messages import classic, get_banter, get_message, hindi, roast

ALL_MODULES = {"classic": classic, "hindi": hindi, "roast": roast}

FORMAT_KWARGS = {
    "shape_mismatch_matmul": {"left": (64, 128), "right": (32, 10)},
    "broadcast_failure": {"op": "Add", "shapes": ((3, 2), (4, 5))},
    "backward_no_grad": {},
    "backward_non_scalar": {"shape": (4, 4)},
    "gradient_explosion": {"value": 1e5},
    "vanishing_gradient": {"value": 1e-9},
    "nan_loss": {"value": float("nan"), "epoch": 7},
    "loss_increasing": {"count": 3, "value": 0.42},
    "lr_invalid": {"value": -0.1},
    "empty_params": {},
    "training_complete": {"epochs": 100, "best": 0.001},
    "unknown_backend": {"name": "tpu", "available": ["numpy"]},
    "rust_backend_unavailable": {},
    "state_dict_key_mismatch": {"missing": ["weight"], "unexpected": ["nope"]},
    "state_dict_shape_mismatch": {
        "name": "weight",
        "expected": (2, 2),
        "actual": (3, 3),
    },
    "optimizer_param_invalid": {
        "optimizer": "SGD",
        "param": "momentum",
        "value": 1.5,
        "constraint": "0.0 <= momentum < 1.0",
    },
    "module_param_invalid": {
        "module": "Dropout",
        "param": "p",
        "value": 1.5,
        "constraint": "0.0 <= p < 1.0",
    },
    "dataset_empty": {},
    "dataset_length_mismatch": {"lengths": [4, 5]},
    "dataloader_bad_batch_size": {"value": -1},
}


# Resets message mode and debug checks after each test
@pytest.fixture(autouse=True)
def reset_mode():
    yield
    config.set_message_mode("classic")
    config.set_debug_checks(False)


# Tests that all three personality catalogs stay consistent with each other
class TestCatalogConsistency:
    # Checks every mode defines exactly the same set of message keys
    def test_all_modes_define_same_keys(self):
        expected = set(classic.MESSAGES)
        for name, module in ALL_MODULES.items():
            assert set(module.MESSAGES) == expected, f"{name} keys differ"

    # Checks every template formats cleanly with its expected kwargs
    def test_all_templates_format_cleanly(self):
        for module in ALL_MODULES.values():
            for key, template in module.MESSAGES.items():
                assert key in FORMAT_KWARGS, f"missing test kwargs for {key}"
                assert template.format(**FORMAT_KWARGS[key])

    # Checks every mode has at least one banter line
    def test_every_mode_has_banter(self):
        for name, module in ALL_MODULES.items():
            assert module.EPOCH_BANTER, f"{name} has no banter"


# Tests message mode selection and environment-variable initialization
class TestModeSelection:
    # Checks classic is the default mode
    def test_default_mode_is_classic(self):
        assert config.get_message_mode() == "classic"

    # Checks changing the mode changes resolved message text
    def test_set_mode_changes_messages(self):
        config.set_message_mode("hindi")
        msg = get_message("shape_mismatch_matmul", left=(2, 3), right=(4, 5))
        assert "shaadi" in msg

        config.set_message_mode("roast")
        msg = get_message("shape_mismatch_matmul", left=(2, 3), right=(4, 5))
        assert msg == roast.MESSAGES["shape_mismatch_matmul"].format(
            left=(2, 3), right=(4, 5)
        )

    # Checks an invalid mode name is rejected
    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            config.set_message_mode("shakespeare")

    # Checks ONEMOREEPOCH_MESSAGES env var sets the initial mode
    def test_env_var_initialization(self, monkeypatch):
        monkeypatch.setenv("ONEMOREEPOCH_MESSAGES", "roast")
        importlib.reload(config)
        assert config.get_message_mode() == "roast"
        monkeypatch.delenv("ONEMOREEPOCH_MESSAGES")
        importlib.reload(config)

    # Checks EDUCATIONAL_MODE=1 maps to hindi mode
    def test_educational_mode_env_maps_to_hindi(self, monkeypatch):
        monkeypatch.setenv("EDUCATIONAL_MODE", "1")
        importlib.reload(config)
        assert config.get_message_mode() == "hindi"
        monkeypatch.delenv("EDUCATIONAL_MODE")
        importlib.reload(config)


# Tests epoch banter rotation
class TestBanter:
    # Checks banter rotates deterministically and wraps around
    def test_rotation_is_deterministic_and_wraps(self):
        config.set_message_mode("hindi")
        n = len(hindi.EPOCH_BANTER)
        assert get_banter(0) == get_banter(n)
        assert get_banter(1) == hindi.EPOCH_BANTER[1]
