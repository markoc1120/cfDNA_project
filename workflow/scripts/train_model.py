import glob
import logging
import os

import torch
import torch.nn as nn
import torchmetrics

from cfdna.models import get_model
from cfdna.preprocessing.transforms import build_transform_pipeline
from cfdna.training.trainer import NegReconMSE, compute_best_roc_data, train
from cfdna.training.utils import SelfTargetLoader, get_dataloaders

logger = logging.getLogger(__name__)

if 'snakemake' in globals():
    config = snakemake.config

    model_cfg = config['model']
    training_cfg = config['training']
    preproc_cfg = config['preprocessing']
    data_cfg = config['data']

    seed = training_cfg.get('seed', 42)
    torch.manual_seed(seed)

    # device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f'using device: {device}')

    # build transform pipeline
    transform_configs = preproc_cfg.get('transforms', [])
    needs_standardization = any(t['name'] == 'standardization' for t in transform_configs)
    transform_fn = build_transform_pipeline(transform_configs) if transform_configs else None

    model_name = model_cfg['name']

    # apply_coverage_filter (downsample mode only)
    qc_kwargs: dict = {}
    if preproc_cfg.get('coverage_handling', 'downsample') == 'downsample':
        coverage_threshold = preproc_cfg.get('min_cov')
        if coverage_threshold is None:
            with open(data_cfg['training_min_coverage_file']) as f:
                coverage_threshold = float(f.read().strip())
        else:
            coverage_threshold = float(coverage_threshold)

        dhs_dir = data_cfg['training_dhs_dir']
        dhs_files = [
            os.path.basename(f).removesuffix('.bed')
            for f in glob.glob(f'{dhs_dir}*.bed')
            if '_wl' not in os.path.basename(f)
            and not os.path.basename(f).endswith('_negative.bed')
        ]

        qc_kwargs = dict(
            cov_dir=data_cfg['training_base_matrices'],
            dhs_files=dhs_files,
            coverage_threshold=coverage_threshold,
            max_sample_loss=preproc_cfg.get('max_sample_loss', 0.2),
        )

    # get dataloaders
    train_loader, valid_loader, test_loader = get_dataloaders(
        output_dir=data_cfg['training_output_dir'],
        transform_fn=transform_fn,
        needs_standardization=needs_standardization,
        train_size=training_cfg.get('train_size', 80),
        valid_size=training_cfg.get('valid_size', 10),
        batch_size=training_cfg.get('batch_size', 32),
        seed=seed,
        suffix=snakemake.params.input_type,
        only_positive=(model_name == 'vae'),
        **qc_kwargs,
    )
    train_mean = getattr(train_loader.dataset, 'train_mean', None)
    train_std = getattr(train_loader.dataset, 'train_std', None)

    # build model
    model_params = model_cfg.get('params', {})
    # determine n_inputs from a sample batch (MLP)
    if model_name == 'mlp':
        sample_x, _ = next(iter(train_loader))
        n_inputs = sample_x.shape[2] + sample_x.shape[3]
        model = get_model(model_name, n_inputs=n_inputs, **model_params)
    elif model_name == 'vae':
        sample_x, _ = next(iter(train_loader))
        model = get_model(
            model_name,
            input_height=sample_x.shape[2],
            input_width=sample_x.shape[3],
            **model_params,
        )
    else:
        model = get_model(model_name, **model_params)
    model = model.to(device)

    # optimizer
    lr = training_cfg.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss and metric - VAE uses reconstruction loss, supervised models use BCE
    is_vae = model_name == 'vae'
    if is_vae:
        from cfdna.models.vae import vae_loss

        beta = model_cfg.get('params', {}).get('beta', 1.0)

        def loss_fn(vae_output, target):
            return vae_loss(
                vae_output.reconstruction, target, vae_output.mu, vae_output.logvar, beta=beta
            )

        metric = NegReconMSE().to(device)
        train_loader = SelfTargetLoader(train_loader)
        valid_loader = SelfTargetLoader(valid_loader)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        metric = torchmetrics.classification.BinaryAUROC().to(device)

    # scheduler
    scheduler = None
    scheduler_name = training_cfg.get('scheduler', None)
    if scheduler_name == 'reduce_on_plateau':
        scheduler_params = training_cfg.get('scheduler_params', {})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

    # train
    checkpoint_path = model_cfg['checkpoint']
    history = train(
        model,
        optimizer,
        loss_fn,
        metric,
        train_loader,
        valid_loader,
        n_epochs=training_cfg.get('n_epochs', 20),
        patience=training_cfg.get('patience', 10),
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        device=device,
    )
    print(f'Training complete. Best checkpoint saved to: {checkpoint_path}')

    # ROC is only meaningful for supervised models (CNN/MLP), not VAE
    if not is_vae:
        roc_metric = torchmetrics.classification.BinaryROC().to(device)
        roc_data = compute_best_roc_data(
            model=model,
            valid_loader=valid_loader,
            roc_metric=roc_metric,
            device=device,
        )
        history = {**history, **roc_data}

    history_path = model_cfg['checkpoint'].replace('.pt', '.history.pt')
    torch.save(history, history_path)
    print(f'Training history saved to: {history_path}')

    if train_mean is not None and train_std is not None:
        stats_path = model_cfg['checkpoint'].replace('.pt', '.stats.pt')
        torch.save({'train_mean': train_mean, 'train_std': train_std}, stats_path)
        print(f'Standardization stats saved to: {stats_path}')
