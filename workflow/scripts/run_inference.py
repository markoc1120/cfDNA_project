import numpy as np
import torch
import torch.nn.functional as F

from cfdna.models import get_model

if 'snakemake' in globals():
    matrix_path = snakemake.input.matrix
    score_path = snakemake.output.score
    checkpoint = snakemake.params.checkpoint
    model_type = snakemake.params.model_type

    matrix = np.load(matrix_path).astype(np.float32)
    x = torch.tensor(matrix).unsqueeze(0).unsqueeze(0)

    # Model instantiation depends on type
    if model_type == 'vae':
        model = get_model(
            model_type,
            input_height=matrix.shape[0],
            input_width=matrix.shape[1],
        )
    elif model_type == 'mlp':
        n_inputs = matrix.shape[0] + matrix.shape[1]
        model = get_model(model_type, n_inputs=n_inputs)
    else:
        model = get_model(model_type)

    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    with torch.no_grad():
        if model_type == 'vae':
            vae_output = model(x)
            score = F.mse_loss(vae_output.reconstruction, x).item()
        else:
            logit = model(x).item()
            score = torch.sigmoid(torch.tensor(logit)).item()

    with open(score_path, 'w') as f:
        f.write(f'{score}\n')
