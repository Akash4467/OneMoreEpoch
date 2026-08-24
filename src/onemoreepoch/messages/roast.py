"""Roast-style training messages — savage but safe-for-work.

Text templates only (ADR-010): this module never raises exceptions or
performs computation. Placeholders are filled via ``str.format`` by
``messages.get_message()``.
"""

MESSAGES: dict[str, str] = {
    "shape_mismatch_matmul": (
        "🔥 You tried to matmul {left} with {right}.\n"
        "Those inner dimensions don't match, and honestly, neither do "
        "you and linear algebra right now."
    ),
    "broadcast_failure": (
        "🔥 {op} with shapes {shapes}? Bold move.\n"
        "Broadcasting has rules. Reading them is free."
    ),
    "backward_no_grad": (
        "🔥 Calling backward() on a tensor with requires_grad=False.\n"
        "That tensor literally told you it doesn't do gradients. "
        "It's not you, it's the flag you forgot to set."
    ),
    "backward_non_scalar": (
        "🔥 backward() on shape {shape} without a seed gradient.\n"
        "Which of those numbers did you want the gradient OF? Exactly. "
        "Call .sum() or .mean() first."
    ),
    "gradient_explosion": (
        "🔥 Gradients hit {value:.3e}. That's not training, that's a "
        "fireworks show.\n"
        "Lower the learning rate before your weights reach the moon."
    ),
    "vanishing_gradient": (
        "🔥 Max |grad| = {value:.3e}. Your gradients are basically on "
        "vacation.\n"
        "The model isn't learning — it's meditating."
    ),
    "nan_loss": (
        "🔥 Loss is {value} at epoch {epoch}.\n"
        "Congratulations, you've trained a model to produce pure "
        "nonsense. Lower the learning rate."
    ),
    "loss_increasing": (
        "🔥 Loss has gone UP {count} epochs straight (now {value:.6f}).\n"
        "The model is unlearning. Impressive, in the wrong direction."
    ),
    "lr_invalid": (
        "🔥 A learning rate of {value}. Sure, and while we're at it, "
        "let's divide by zero too. Make it positive."
    ),
    "empty_params": (
        "🔥 An optimizer with zero parameters.\n"
        "What exactly were you hoping it would optimize? Vibes?"
    ),
    "training_complete": (
        "🏁 {epochs} epochs, best loss {best:.6f}.\n"
        "Not bad. The bar was on the floor, but not bad."
    ),
    "unknown_backend": (
        "🔥 Backend {name!r}? Never heard of it, and neither has this "
        "registry.\nAvailable backends: {available}."
    ),
    "rust_backend_unavailable": (
        "🔥 Asking for the Rust backend without building it first.\n"
        "Bold strategy. Run `maturin develop`, or settle for NumPy."
    ),
    "state_dict_key_mismatch": (
        "🔥 That state_dict doesn't match this module at all.\n"
        "Missing: {missing}\n"
        "Unexpected: {unexpected}"
    ),
    "state_dict_shape_mismatch": (
        "🔥 {name!r} shape mismatch: expected {expected}, got {actual}.\n"
        "Close, but close only counts in horseshoes."
    ),
    "optimizer_param_invalid": (
        "🔥 {optimizer}: {param}={value!r} is not it.\n"
        "Constraint: {constraint}. Try reading the docs next time."
    ),
    "module_param_invalid": (
        "🔥 {module}: {param}={value!r} is not it.\n"
        "Constraint: {constraint}. Try reading the docs next time."
    ),
    "dataset_empty": (
        "🔥 A dataset with zero samples. Bold choice to train on nothing."
    ),
    "dataset_length_mismatch": (
        "🔥 TensorDataset arrays with mismatched lengths: {lengths}.\n"
        "They're supposed to be parallel. As in, actually the same length."
    ),
    "dataloader_bad_batch_size": (
        "🔥 batch_size={value}? DataLoader wants a positive integer, not "
        "a personality trait."
    ),
}

EPOCH_BANTER: list[str] = [
    "Loss went down. Even a broken clock is right twice a day.",
    "Progress detected. Screenshot it before it regresses.",
    "The model is learning. Slowly. Like, geologically slowly.",
    "Another epoch survived. Both of you.",
    "Weights updated. Whether they improved is another question.",
    "Converging. Or circling the drain. Time will tell.",
]
