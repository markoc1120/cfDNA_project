import numpy as np
import torch

from cfdna.models import get_model

if 'snakemake' in globals():
    matrix_path = snakemake.input.matrix
    score_path = snakemake.output.score
    checkpoint = snakemake.params.checkpoint
    model_type = snakemake.params.model_type

    matrix = np.load(matrix_path).astype(np.float32)
    x = torch.tensor(matrix).unsqueeze(0).unsqueeze(0)
    n_inputs = matrix.shape[0] + matrix.shape[1]

    model = get_model(model_type)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    with torch.no_grad():
        logit = model(x).item()
        score = torch.sigmoid(torch.tensor(logit)).item()

    with open(score_path, 'w') as f:
        f.write(f'{score}\n')
