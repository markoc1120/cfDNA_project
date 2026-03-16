import numpy as np
import torch
from models import get_model


if 'snakemake' in globals():
    matrix_path = snakemake.input.matrix
    cov_path = snakemake.input.cov
    min_cov_path = snakemake.input.min_cov
    score_path = snakemake.output.score
    checkpoint = snakemake.params.checkpoint
    model_type = snakemake.params.model_type

    with open(cov_path) as f:
        sample_cov = int(f.read().strip())
    with open(min_cov_path) as f:
        min_cov = int(f.read().strip())

    if sample_cov < min_cov:
        with open(score_path, 'w') as f:
            f.write('NaN\n')
    else:
        matrix = np.load(matrix_path).astype(np.float32)
        x = torch.tensor(matrix).unsqueeze(0).unsqueeze(0)
        n_inputs = matrix.shape[0] + matrix.shape[1]

        model = get_model('cnn_model', n_inputs)

        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model.eval()

        with torch.no_grad():
            logit = model(x).item()
            score = torch.sigmoid(torch.tensor(logit)).item()

        with open(score_path, 'w') as f:
            f.write(f'{score}\n')
