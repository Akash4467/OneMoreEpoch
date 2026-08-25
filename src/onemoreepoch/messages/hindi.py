MESSAGES: dict[str, str] = {
    "shape_mismatch_matmul": (
        "🚨 Arre bhai!\n"
        "Shape mismatch.\n"
        "Left Matrix : {left}\n"
        "Right Matrix: {right}\n"
        "Ye shaadi nahi ho sakti."
    ),
    "broadcast_failure": (
        "🚨 Arre {op} karne chale the, par shapes {shapes} ka jodi "
        "nahi bana.\n"
        "Broadcasting ke rishte mein har dimension barabar ya 1 honi "
        "chahiye."
    ),
    "backward_no_grad": (
        "🛑 Bhai, is tensor ne gradient ka form hi nahi bhara "
        "(requires_grad=False).\n"
        "Pehle requires_grad=True karo, phir backward() bulao."
    ),
    "backward_non_scalar": (
        "🛑 Non-scalar tensor (shape {shape}) pe seedha backward()? "
        "Aise nahi hota bhai.\n"
        "Pehle .sum() ya .mean() se scalar banao, ya seed gradient do."
    ),
    "gradient_explosion": (
        "⚠️ Gradient Explosion Detected!\n"
        "Bhai throttle maar... learning rate bahut zyada hai.\n"
        "(max |grad| = {value:.3e})"
    ),
    "vanishing_gradient": (
        "😴 Gradient so gaya (max |grad| = {value:.3e}).\n"
        "Itne chhote gradient se model kuch nahi seekhega — activation "
        "ya init badlo."
    ),
    "nan_loss": (
        "💀 Epoch {epoch} pe loss {value} ho gaya.\n"
        "Model ki vaat lag gayi — learning rate kam karo aur inputs "
        "check karo."
    ),
    "loss_increasing": (
        "📈 Loss {count} epochs se upar ja raha hai (abhi {value:.6f}).\n"
        "Bhai, learning rate thoda kam kar."
    ),
    "lr_invalid": ("🤨 Learning rate {value}? Ye kya mazaak hai — positive number do."),
    "empty_params": (
        "🫗 Optimizer ko khaali parameter list mili.\n"
        "Bina parameters ke kya optimize karein bhai? model.parameters() "
        "bhejo."
    ),
    "training_complete": (
        "🎉 Training khatam: {epochs} epochs, best loss {best:.6f}.\n"
        "Ek aur epoch? Naam hi OneMoreEpoch hai."
    ),
    "unknown_backend": (
        "🤷 Backend {name!r} ka pata nahi.\n" "Available backends: {available}."
    ),
    "rust_backend_unavailable": (
        "🛑 Rust backend abhi ready nahi hai.\n"
        "Pehle native extension build karo (`maturin develop`), ya "
        "default NumPy backend use karo."
    ),
    "state_dict_key_mismatch": (
        "🧩 state_dict ke keys match nahi kar rahe is module se.\n"
        "Missing: {missing}\n"
        "Unexpected: {unexpected}"
    ),
    "state_dict_shape_mismatch": (
        "📐 {name!r} ka shape match nahi: expected {expected}, mila {actual}."
    ),
    "optimizer_param_invalid": (
        "🤨 {optimizer}: {param}={value!r} galat hai. Chahiye: {constraint}."
    ),
    "module_param_invalid": (
        "🤨 {module}: {param}={value!r} galat hai. Chahiye: {constraint}."
    ),
    "dataset_empty": ("📭 Dataset khaali hai (len() == 0)."),
    "dataset_length_mismatch": (
        "📏 TensorDataset ke arrays ki lengths match nahi kar rahi: {lengths}."
    ),
    "dataloader_bad_batch_size": (
        "🤨 DataLoader ka batch_size positive integer hona chahiye, mila {value}."
    ),
}

EPOCH_BANTER: list[str] = [
    'Model: "Aaj kuch toofani karte hain."',
    'Model: "Gradient flow full speed pe hai."',
    'Model: "Loss neeche, confidence upar."',
    'Model: "Ek aur epoch, ek aur kamaal."',
    'Model: "Weights set, scene set."',
    'Model: "Seekh raha hoon bhai, tension mat le."',
]
