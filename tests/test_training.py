import os
import tempfile

import torch
import torch.nn as nn
import torchmetrics

from cfdna.models import get_model
from cfdna.training.trainer import train
from cfdna.training.utils import get_dataloaders

DEVICE = 'cpu'
N_EPOCHS = 2
BATCH_SIZE = 4
TRAIN_SIZE = 80
VALID_SIZE = 10
SEED = 42


def _run_training(matrix_dir, model_name, suffix, n_inputs=None):
    """Run a short training loop for one model and assert no errors."""
    train_loader, valid_loader, _ = get_dataloaders(
        output_dir=matrix_dir,
        train_size=TRAIN_SIZE,
        valid_size=VALID_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        suffix=suffix,
    )

    model = get_model(model_name, n_inputs=n_inputs) if n_inputs else get_model(model_name)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    metric = torchmetrics.classification.BinaryAUROC().to(DEVICE)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name

    try:
        history = train(
            model,
            optimizer,
            loss_fn,
            metric,
            train_loader,
            valid_loader,
            n_epochs=N_EPOCHS,
            patience=N_EPOCHS,
            checkpoint_path=checkpoint_path,
            device=DEVICE,
        )

        assert 'train_losses' in history
        assert len(history['train_losses']) == N_EPOCHS
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


# CNN models (use rebinned 46×2000 matrices)
def test_cnn_model(matrix_dir):
    _run_training(matrix_dir, 'cnn_model', suffix='rebinned')


# MLP models (use downsampled 300×2000 matrices)
def test_mlp_model(matrix_dir):
    _run_training(matrix_dir, 'mlp_model', suffix='downsampled', n_inputs=2000 + 300)
