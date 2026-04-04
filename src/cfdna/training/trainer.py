import time

import torch
import torchmetrics
import matplotlib.pyplot as plt


def evaluate_tm(model, data_loader, metric, device='cpu'):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()


# early stopping training + save best model + scheduler
def train(
    model,
    optimizer,
    loss_fn,
    metric,
    train_loader,
    valid_loader,
    n_epochs,
    patience=10,
    checkpoint_path=None,
    scheduler=None,
    device='cpu',
):
    checkpoint_path = checkpoint_path or 'my_checkpoint.pt'
    history: dict[str, list] = {'train_losses': [], 'train_metrics': [], 'valid_metrics': []}
    best_metric = float('-inf')
    patience_counter = 0
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()
        t0 = time.time()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)

        train_metric = metric.compute().item()
        valid_metric = evaluate_tm(model, valid_loader, metric, device).item()
        if valid_metric > best_metric:
            torch.save(model.state_dict(), checkpoint_path)
            best_metric = valid_metric
            best = ' (best)'
            patience_counter = 0
        else:
            patience_counter += 1
            best = ''

        t1 = time.time()
        history['train_losses'].append(total_loss / len(train_loader))
        history['train_metrics'].append(train_metric)
        history['valid_metrics'].append(valid_metric)
        print(
            f'Epoch {epoch + 1}/{n_epochs}, '
            f'train loss: {history["train_losses"][-1]:.4f}, '
            f'train metric: {history["train_metrics"][-1]:.4f}, '
            f'valid metric: {history["valid_metrics"][-1]:.4f}{best}'
            f' in {t1 - t0:.1f}s'
        )
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_metric)
            else:
                scheduler.step()
        if patience_counter >= patience:
            print('Early stopping!')
            break

    model.load_state_dict(torch.load(checkpoint_path))
    return history


def compute_best_roc_data(model, valid_loader, roc_metric, device='cpu'):
    roc_metric = torchmetrics.classification.BinaryROC().to(device)
    model.eval()
    roc_metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            roc_metric.update(y_pred, y_batch.to(torch.int))

    fpr, tpr, thr = roc_metric.compute()
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thr,
    }


def plot_training_progress(h):
    already_roc = False
    for plot in ('train_losses', 'valid_metrics', 'roc'):
        plt.figure(figsize=(8, 4))
        for history, opt_name in zip((h.values()), h.keys()):
            if plot == 'roc':
                plt.plot(history['fpr'], history['tpr'], label=opt_name, linewidth=1)
                if not already_roc:
                    plt.plot(
                        [0, 1], [0, 1], linestyle='--', linewidth=1, color='r', label='Random guess'
                    )
                    already_roc = True
            else:
                plt.plot(history[plot], label=opt_name, linewidth=1)

        plt.grid()
        plt.xlabel(
            {'train_losses': 'Epochs', 'valid_metrics': 'Epochs', 'roc': 'False positive rate'}[
                plot
            ]
        )
        plt.ylabel(
            {
                'train_losses': 'Training loss',
                'valid_metrics': 'Validation AUC',
                'roc': 'True positive rate',
            }[plot]
        )
        plt.legend(loc='best')
        plt.show()
