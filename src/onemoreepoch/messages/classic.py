"""Classic training messages — professional English, the default mode.

Text templates only (ADR-010): this module never raises exceptions or
performs computation. Placeholders are filled via ``str.format`` by
``messages.get_message()``.
"""

MESSAGES: dict[str, str] = {
    "shape_mismatch_matmul": (
        "Matrix multiplication shape mismatch.\n"
        "Left matrix : {left}\n"
        "Right matrix: {right}\n"
        "Inner dimensions must agree (left columns == right rows)."
    ),
    "broadcast_failure": (
        "Operands of {op} could not be broadcast together.\n"
        "Shapes: {shapes}\n"
        "Broadcasting requires each dimension pair to be equal or 1."
    ),
    "backward_no_grad": (
        "Cannot call backward() on a tensor with requires_grad=False.\n"
        "Create the tensor with requires_grad=True, or check that it is "
        "connected to at least one tensor that requires gradients."
    ),
    "backward_non_scalar": (
        "backward() on a non-scalar tensor (shape {shape}) requires an "
        "explicit gradient argument.\n"
        "Reduce to a scalar first (e.g. .sum() or .mean()), or pass a "
        "seed gradient of matching shape."
    ),
    "gradient_explosion": (
        "Gradient explosion detected (max |grad| = {value:.3e}).\n"
        "Consider lowering the learning rate or clipping gradients."
    ),
    "vanishing_gradient": (
        "Vanishing gradient detected (max |grad| = {value:.3e}).\n"
        "Gradients this small stall learning; consider a different "
        "activation, initialization, or architecture."
    ),
    "nan_loss": (
        "Loss is {value} at epoch {epoch}.\n"
        "Training has diverged; lower the learning rate and check for "
        "invalid inputs (log of zero, division by zero)."
    ),
    "loss_increasing": (
        "Loss has increased for {count} consecutive epochs "
        "(now {value:.6f}).\n"
        "The learning rate may be too high."
    ),
    "lr_invalid": ("Learning rate must be positive, got {value}."),
    "empty_params": (
        "Optimizer received an empty parameter list.\n"
        "Pass model.parameters() from a module that owns at least one "
        "Parameter."
    ),
    "training_complete": (
        "Training complete: {epochs} epochs, best loss {best:.6f}."
    ),
    "unknown_backend": (
        "Unknown backend: {name!r}.\nAvailable backends: {available}."
    ),
    "rust_backend_unavailable": (
        "The Rust backend is not available.\n"
        "Build the native extension first (e.g. `maturin develop`), or "
        "use the default NumPy backend."
    ),
    "state_dict_key_mismatch": (
        "state_dict keys do not match this module.\n"
        "Missing: {missing}\n"
        "Unexpected: {unexpected}"
    ),
    "state_dict_shape_mismatch": (
        "Shape mismatch for {name!r}: expected {expected}, got {actual}."
    ),
    "optimizer_param_invalid": (
        "{optimizer}: invalid {param}={value!r}. Constraint: {constraint}."
    ),
    "module_param_invalid": (
        "{module}: invalid {param}={value!r}. Constraint: {constraint}."
    ),
    "dataset_empty": ("Dataset has no samples (len() == 0)."),
    "dataset_length_mismatch": (
        "TensorDataset arrays have mismatched lengths: {lengths}."
    ),
    "dataloader_bad_batch_size": (
        "DataLoader batch_size must be a positive integer, got {value}."
    ),
}

EPOCH_BANTER: list[str] = [
    "Steady progress.",
    "Gradients flowing normally.",
    "Parameters updated.",
    "Optimization proceeding.",
    "Another step down the loss surface.",
    "Convergence in progress.",
]
