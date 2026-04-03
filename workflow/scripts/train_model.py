import logging

import torch
import torch.nn as nn
import torchmetrics

from cfdna.models import get_model
from cfdna.preprocessing.transforms import build_transform_pipeline
from cfdna.training.trainer import train
from cfdna.training.utils import get_dataloaders

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
    )

    # build model
    model_params = model_cfg.get('params', {})
    # determine n_inputs from a sample batch (MLP)
    if model_cfg['name'] == 'mlp':
        sample_x, _ = next(iter(train_loader))
        n_inputs = sample_x.shape[2] + sample_x.shape[3]
        model = get_model(model_cfg['name'], n_inputs=n_inputs, **model_params)
    else:
        model = get_model(model_cfg['name'], **model_params)
    model = model.to(device)

    # optimizer
    lr = training_cfg.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss and metric
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
    logger.info(f'Training complete. Best checkpoint saved to: {checkpoint_path}')
