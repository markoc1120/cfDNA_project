import numpy as np
import torch

from cfdna.models import get_model
from cfdna.preprocessing.transforms import build_transform_pipeline

if 'snakemake' in globals():
    matrix_path = snakemake.input.matrix
    output_path = snakemake.output[0]
    checkpoint = snakemake.params.checkpoint
    model_type = snakemake.params.model_type

    transform_configs = snakemake.config.get('preprocessing', {}).get('transforms', [])
    transform_fn = build_transform_pipeline(transform_configs) if transform_configs else None

    # load standardization stats
    transform_kwargs = {}
    if any(tc['name'] == 'standardization' for tc in transform_configs):
        stats_path = checkpoint.replace('.pt', '.stats.pt')
        stats = torch.load(stats_path, weights_only=True)
        transform_kwargs['train_mean'] = stats['train_mean']
        transform_kwargs['train_std'] = stats['train_std']

    matrix = np.load(matrix_path).astype(np.float32)
    x = torch.from_numpy(matrix)
    if transform_fn is not None:
        x = transform_fn(x, **transform_kwargs)
    x = x.unsqueeze(0).unsqueeze(0)

    # Model instantiation depends on type
    h, w = x.shape[2], x.shape[3]
    if model_type == 'vae':
        model = get_model(
            model_type,
            input_height=h,
            input_width=w,
        )
    elif model_type == 'mlp':
        n_inputs = h + w
        model = get_model(model_type, n_inputs=n_inputs)
    else:
        model = get_model(model_type)

    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()

    with torch.no_grad():
        if model_type == 'vae':
            vae_output = model(x)
            np.savez(
                output_path,
                mu=vae_output.mu.squeeze(0).cpu().numpy(),
                logvar=vae_output.logvar.squeeze(0).cpu().numpy(),
            )
        else:
            logit = model(x).item()
            score = torch.sigmoid(torch.tensor(logit)).item()
            with open(output_path, 'w') as f:
                f.write(f'{score}\n')
