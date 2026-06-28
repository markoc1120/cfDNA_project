"""Offline JEPA / I-JEPA training, with an optional random hyperparameter search.

Mirrors workflow/scripts/train_model.py (same dataloaders / trainer) so trained models stay
consistent with the Snakemake pipeline. Works for any joint-embedding model whose forward returns
a JEPAOutput (the 'ijepa' model); the model is read from the config's model.name.

Two modes:
  * single run (default, --n-trials 0): train the model.params from the config for the full
    training.n_epochs and save to model.checkpoint.
  * random search (--n-trials N > 0): sample N configs (search space depends on the model),
    train each for --trial-epochs, rank by RankMe, then retrain the winner for --final-epochs
    and save it to model.checkpoint.

Why RankMe and not the SSL loss: for JEPA a *lower* validation prediction error often means
representational collapse. Training is also healthy-only, so there are no cancer labels to probe
against. RankMe (Garrido et al. 2023) -- the smooth effective rank of the validation embeddings --
is a label-free quality proxy that drops to ~1 under collapse.

Usage:
    uv run python train_ssl.py --config confs/jepa_position.yaml                # single run
    uv run python train_ssl.py --config confs/ijepa_within_cohort.yaml --n-trials 30
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from cfdna.models import get_model
from cfdna.models.ijepa import NegPredErr, jepa_loss
from cfdna.preprocessing.transforms import build_transform_pipeline
from cfdna.training.trainer import train
from cfdna.training.utils import get_dataloaders

logger = logging.getLogger('train_ssl')

# per-model search spaces. Values within each list are kept well-separated (no near-duplicates)
# so a small trial budget covers the grid. learning_rate is NOT searched (fixed via
# training.learning_rate, default 1e-4); pred_dim is derived (= embed_dim // 2, the I-JEPA narrow
# predictor); num_heads is fixed at the config model.params value.
# Grid = 3 x 3 x 3 x 3 x 3 x 2 = 486 configs.
SEARCH_SPACE = {
    'ijepa': {
        'embed_dim': [32, 128, 256],          # 4x+ spread (dropped 64)
        'depth': [2, 4, 8],                   # encoder depth; dropped 6 (close to 4/8)
        'pred_depth': [1, 2, 4],              # narrow-predictor depth; well-separated
        'num_target_blocks': [2, 4, 8],       # dropped 6
        'momentum': [0.99, 0.996, 0.999],     # very different EMA dynamics -> keep all
        'drop_rate': [0.0, 0.2],              # extremes (dropped 0.1)
    },
}


def sample_params(rng: random.Random, model_name: str) -> dict:
    space = SEARCH_SPACE.get(model_name)
    if space is None:
        raise ValueError(f'no search space for model {model_name!r}; use --n-trials 0')
    params = {k: rng.choice(v) for k, v in space.items()}
    # narrow predictor (I-JEPA bottleneck): half the encoder width, not searched independently.
    # embed_dim values are chosen divisible by the fixed num_heads (4), so pred_dim is too.
    params['pred_dim'] = params['embed_dim'] // 2
    return params


def tag_of(params: dict) -> str:
    parts = [f'{k}{params[k]:g}' if isinstance(params[k], (int, float)) else f'{k}{params[k]}'
             for k in sorted(params)]
    return '_'.join(parts)


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_coverage_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Mirror workflow/scripts/train_model.py: build coverage-filter kwargs."""
    preproc_cfg = config['preprocessing']
    data_cfg = config['data']

    if preproc_cfg.get('coverage_handling', 'downsample') != 'downsample':
        return {}

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

    return dict(
        cov_dir=data_cfg['training_base_matrices'],
        dhs_files=dhs_files,
        coverage_threshold=coverage_threshold,
        max_sample_loss=preproc_cfg.get('max_sample_loss', 0.2),
    )


@torch.no_grad()
def rankme(embeddings: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe (Garrido et al. 2023): smooth effective rank of the embedding matrix.

    Entropy of the normalized singular-value spectrum, exponentiated. Collapsed encoder -> ~1.
    """
    s = torch.linalg.svdvals(embeddings.float())
    p = s / (s.sum() + eps) + eps
    return float(torch.exp(-(p * torch.log(p)).sum()))


@torch.no_grad()
def collect_embeddings(model, loader, device: str) -> torch.Tensor:
    model.eval()
    embs = [model.embed(x.to(device)).cpu() for x, _ in loader]
    return torch.cat(embs, dim=0)


def build_model(model_name: str, config_params: dict, override: dict | None, device: str):
    """Model params from the config, with HP-search overrides merged in (override excludes lr)."""
    params = dict(config_params)
    if override:
        params.update(override)
    return get_model(model_name, **params).to(device)


def momentum_steps_for(
    model_name: str, config_params: dict, override: dict | None,
    momentum_epochs: int, steps_per_epoch: int,
) -> dict | None:
    """Set the EMA momentum-schedule length so the target encoder ramps momentum -> momentum_end
    and *freezes* after momentum_epochs epochs.

    This must be short enough to complete inside the early-stopping window (the JEPA loss tends to
    U-turn within a handful of epochs as the EMA target drifts), so it is scaled to momentum_epochs
    -- NOT the full n_epochs budget, which early-stopping never reaches.

    Only for ijepa; skipped if momentum_steps is set explicitly in the config or the override.
    """
    if model_name != 'ijepa':
        return override
    if 'momentum_steps' in config_params or (override and 'momentum_steps' in override):
        return override
    steps = max(1, momentum_epochs * steps_per_epoch)
    return {**(override or {}), 'momentum_steps': steps}


def train_once(
    model, train_loader, valid_loader, *, lr, n_epochs, patience,
    scheduler_name, scheduler_params, checkpoint_path, device,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metric = NegPredErr().to(device)  # checkpoint/early-stop on negative RMSE
    scheduler = None
    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **(scheduler_params or {})
        )
    return train(
        model, optimizer, jepa_loss, metric, train_loader, valid_loader,
        n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path,
        scheduler=scheduler, device=device,
    )


def write_dropped_dhs(checkpoint_path: str, qc_out: dict) -> None:
    """The benchmark rule depends on this file existing next to the checkpoint."""
    with open(checkpoint_path.replace('.pt', '.dropped_dhs.json'), 'w') as f:
        json.dump(
            {
                'dropped_dhs': qc_out.get('dropped_dhs', []),
                'dropped_sids': qc_out.get('dropped_sids', []),
                'coverage_threshold': qc_out.get('coverage_threshold'),
                'max_sample_loss': qc_out.get('max_sample_loss'),
            },
            f, indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--suffix', default='_rebinned')
    parser.add_argument('--n-trials', type=int, default=0, help='0 = single run; N>0 = random search')
    parser.add_argument('--trial-epochs', type=int, default=30, help='epochs per search trial')
    parser.add_argument('--final-epochs', type=int, default=None, help='final/single train epochs (default: training.n_epochs)')
    parser.add_argument('--momentum-epochs', type=int, default=10, help='epochs over which the EMA momentum ramps to momentum_end, then freezes (ijepa)')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--search-seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path('models/jepa_search/'))
    parser.add_argument('--device', default=None, help='cuda | mps | cpu (auto if omitted)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    with args.config.open() as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    model_name = model_cfg['name']
    training_cfg = config.get('training', {})
    preproc_cfg = config.get('preprocessing', {})
    data_cfg = config['data']

    seed = training_cfg.get('seed', 42)
    torch.manual_seed(seed)
    device = pick_device(args.device)
    config_params = model_cfg.get('params', {})
    scheduler_name = training_cfg.get('scheduler')
    scheduler_params = training_cfg.get('scheduler_params', {})
    final_epochs = args.final_epochs or training_cfg.get('n_epochs', 200)
    lr = training_cfg.get('learning_rate', 1e-4)  # fixed (not searched)
    checkpoint = model_cfg['checkpoint']
    os.makedirs(os.path.dirname(checkpoint) or '.', exist_ok=True)
    logger.info('config=%s model=%s device=%s', args.config, model_name, device)

    transform_configs = preproc_cfg.get('transforms', [])
    needs_standardization = any(t['name'] == 'standardization' for t in transform_configs)
    transform_fn = build_transform_pipeline(transform_configs) if transform_configs else None

    qc_kwargs = resolve_coverage_kwargs(config)
    qc_out: dict = {}

    # JEPA-family trains healthy-only on positive Lymphoid -> only_positive=True.
    train_loader, valid_loader, _ = get_dataloaders(
        output_dir=data_cfg['training_output_dir'],
        transform_fn=transform_fn,
        needs_standardization=needs_standardization,
        train_size=training_cfg.get('train_size', 80),
        valid_size=training_cfg.get('valid_size', 10),
        batch_size=training_cfg.get('batch_size', 32),
        seed=seed,
        suffix=args.suffix,
        only_positive=True,
        qc_out=qc_out,
        **qc_kwargs,
    )

    steps_per_epoch = len(train_loader)

    # ---- single run ----------------------------------------------------------------
    if args.n_trials <= 0:
        ovr = momentum_steps_for(model_name, config_params, None, args.momentum_epochs, steps_per_epoch)
        model = build_model(model_name, config_params, ovr, device)
        history = train_once(
            model, train_loader, valid_loader,
            lr=lr, n_epochs=final_epochs,
            patience=args.patience, scheduler_name=scheduler_name,
            scheduler_params=scheduler_params, checkpoint_path=checkpoint, device=device,
        )
        torch.save(history, checkpoint.replace('.pt', '.history.pt'))
        write_dropped_dhs(checkpoint, qc_out)
        rm = rankme(collect_embeddings(model, valid_loader, device))
        logger.info('single run done. RankMe=%.2f checkpoint=%s', rm, checkpoint)
        return

    # ---- random search -------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.search_seed)
    seen: set[str] = set()
    results: list[dict] = []

    for i in range(1, args.n_trials + 1):
        for _ in range(20):  # de-dupe
            params = sample_params(rng, model_name)
            key = tag_of(params)
            if key not in seen:
                seen.add(key)
                break

        ckpt = args.output_dir / f'trial{i:03d}_{tag_of(params)}.pt'
        logger.info('--- trial [%02d/%02d] %s ---', i, args.n_trials, tag_of(params))
        try:
            ovr = momentum_steps_for(
                model_name, config_params, params, args.momentum_epochs, steps_per_epoch)
            model = build_model(model_name, config_params, ovr, device)
            history = train_once(
                model, train_loader, valid_loader,
                lr=lr, n_epochs=args.trial_epochs, patience=args.patience,
                scheduler_name=scheduler_name, scheduler_params=scheduler_params,
                checkpoint_path=str(ckpt), device=device,
            )
            emb = collect_embeddings(model, valid_loader, device)
            run = {
                'params': params, 'tag': tag_of(params), 'checkpoint': str(ckpt),
                'rankme': rankme(emb), 'emb_std': float(emb.std(0).mean()),
                'best_neg_pred_err': float(max(history['valid_metrics']) if history['valid_metrics'] else float('-inf')),
                'epochs_run': len(history['valid_metrics']),
            }
        except Exception as e:  # noqa: BLE001 - record failures, keep searching
            logger.exception('trial failed: %s', tag_of(params))
            run = {'params': params, 'tag': tag_of(params), 'checkpoint': str(ckpt),
                   'rankme': float('-inf'), 'error': str(e)}
        run['trial'] = i
        results.append(run)
        logger.info('[%02d/%02d] %s -> RankMe=%.2f', i, args.n_trials, tag_of(params),
                    run.get('rankme', float('-inf')))

    # pick the winner purely by best validation metric (NegPredErr); RankMe is logged but not used.
    ranked = sorted(results, key=lambda r: r.get('best_neg_pred_err', float('-inf')), reverse=True)
    best = ranked[0]
    shutil.copyfile(best['checkpoint'], args.output_dir / 'best_jepa.pt')
    with (args.output_dir / 'jepa_search_results.json').open('w') as f:
        json.dump(
            {'config_path': str(args.config), 'model': model_name,
             'search_space': SEARCH_SPACE.get(model_name),
             'ranking_metric': 'max NegPredErr (valid metric); RankMe logged only',
             'best': best, 'ranked_runs': ranked},
            f, indent=2,
        )
    logger.info('=== winner === %s RankMe=%.2f negPredErr=%.4f',
                best['tag'], best.get('rankme', float('-inf')), best.get('best_neg_pred_err', float('-inf')))

    if final_epochs > 0:
        logger.info('retraining winner for %d epochs -> %s', final_epochs, checkpoint)
        ovr = momentum_steps_for(
            model_name, config_params, best['params'], args.momentum_epochs, steps_per_epoch)
        model = build_model(model_name, config_params, ovr, device)
        history = train_once(
            model, train_loader, valid_loader,
            lr=lr, n_epochs=final_epochs, patience=args.patience,
            scheduler_name=scheduler_name, scheduler_params=scheduler_params,
            checkpoint_path=checkpoint, device=device,
        )
        torch.save(history, checkpoint.replace('.pt', '.history.pt'))
        write_dropped_dhs(checkpoint, qc_out)
        final_rankme = rankme(collect_embeddings(model, valid_loader, device))
        logger.info('final model saved: %s (RankMe=%.2f; set model.params to %s in the config)',
                    checkpoint, final_rankme, best['params'])


if __name__ == '__main__':
    main()



# """Offline JEPA / I-JEPA training, with an optional random hyperparameter search.

# Mirrors workflow/scripts/train_model.py (same dataloaders / trainer) so trained models stay
# consistent with the Snakemake pipeline. Works for any joint-embedding model whose forward returns
# a JEPAOutput (the 'ijepa' model); the model is read from the config's model.name.

# Two modes:
#   * single run (default, --n-trials 0): train the model.params from the config for the full
#     training.n_epochs and save to model.checkpoint.
#   * random search (--n-trials N > 0): sample N configs (search space depends on the model),
#     train each for --trial-epochs, rank by RankMe, then retrain the winner for --final-epochs
#     and save it to model.checkpoint.

# Why RankMe and not the SSL loss: for JEPA a *lower* validation prediction error often means
# representational collapse. Training is also healthy-only, so there are no cancer labels to probe
# against. RankMe (Garrido et al. 2023) -- the smooth effective rank of the validation embeddings --
# is a label-free quality proxy that drops to ~1 under collapse.

# Usage:
#     uv run python train_ssl.py --config confs/jepa_position.yaml                # single run
#     uv run python train_ssl.py --config confs/ijepa_within_cohort.yaml --n-trials 30
# """

# from __future__ import annotations

# import argparse
# import glob
# import json
# import logging
# import os
# import random
# import shutil
# from pathlib import Path
# from typing import Any

# import torch
# import yaml

# from cfdna.models import get_model
# from cfdna.models.ijepa import NegPredErr, jepa_loss
# from cfdna.preprocessing.transforms import build_transform_pipeline
# from cfdna.training.trainer import train
# from cfdna.training.utils import get_dataloaders

# logger = logging.getLogger('train_ssl')

# # per-model search spaces. Values within each list are kept well-separated (no near-duplicates)
# # so a small trial budget covers the grid. learning_rate is NOT searched (fixed via
# # training.learning_rate, default 1e-4); pred_dim is derived (= embed_dim // 2, the I-JEPA narrow
# # predictor); num_heads is fixed at the config model.params value.
# # Grid = 3 x 3 x 3 x 3 x 3 x 2 = 486 configs.
# SEARCH_SPACE = {
#     'ijepa': {
#         'embed_dim': [32, 128, 256],          # 4x+ spread (dropped 64)
#         'depth': [2, 4, 8],                   # encoder depth; dropped 6 (close to 4/8)
#         'pred_depth': [1, 2, 4],              # narrow-predictor depth; well-separated
#         'num_target_blocks': [2, 4, 8],       # dropped 6
#         'momentum': [0.99, 0.996, 0.999],     # very different EMA dynamics -> keep all
#         'drop_rate': [0.0, 0.2],              # extremes (dropped 0.1)
#     },
# }


# def sample_params(rng: random.Random, model_name: str) -> dict:
#     space = SEARCH_SPACE.get(model_name)
#     if space is None:
#         raise ValueError(f'no search space for model {model_name!r}; use --n-trials 0')
#     params = {k: rng.choice(v) for k, v in space.items()}
#     # narrow predictor (I-JEPA bottleneck): half the encoder width, not searched independently.
#     # embed_dim values are chosen divisible by the fixed num_heads (4), so pred_dim is too.
#     params['pred_dim'] = params['embed_dim'] // 2
#     return params


# def tag_of(params: dict) -> str:
#     parts = [f'{k}{params[k]:g}' if isinstance(params[k], (int, float)) else f'{k}{params[k]}'
#              for k in sorted(params)]
#     return '_'.join(parts)


# def pick_device(explicit: str | None) -> str:
#     if explicit:
#         return explicit
#     if torch.cuda.is_available():
#         return 'cuda'
#     if torch.backends.mps.is_available():
#         return 'mps'
#     return 'cpu'


# def resolve_coverage_kwargs(config: dict[str, Any]) -> dict[str, Any]:
#     """Mirror workflow/scripts/train_model.py: build coverage-filter kwargs."""
#     preproc_cfg = config['preprocessing']
#     data_cfg = config['data']

#     if preproc_cfg.get('coverage_handling', 'downsample') != 'downsample':
#         return {}

#     coverage_threshold = preproc_cfg.get('min_cov')
#     if coverage_threshold is None:
#         with open(data_cfg['training_min_coverage_file']) as f:
#             coverage_threshold = float(f.read().strip())
#     else:
#         coverage_threshold = float(coverage_threshold)

#     dhs_dir = data_cfg['training_dhs_dir']
#     dhs_files = [
#         os.path.basename(f).removesuffix('.bed')
#         for f in glob.glob(f'{dhs_dir}*.bed')
#         if '_wl' not in os.path.basename(f)
#         and not os.path.basename(f).endswith('_negative.bed')
#     ]

#     return dict(
#         cov_dir=data_cfg['training_base_matrices'],
#         dhs_files=dhs_files,
#         coverage_threshold=coverage_threshold,
#         max_sample_loss=preproc_cfg.get('max_sample_loss', 0.2),
#     )


# @torch.no_grad()
# def rankme(embeddings: torch.Tensor, eps: float = 1e-7) -> float:
#     """RankMe (Garrido et al. 2023): smooth effective rank of the embedding matrix.

#     Entropy of the normalized singular-value spectrum, exponentiated. Collapsed encoder -> ~1.
#     """
#     s = torch.linalg.svdvals(embeddings.float())
#     p = s / (s.sum() + eps) + eps
#     return float(torch.exp(-(p * torch.log(p)).sum()))


# @torch.no_grad()
# def collect_embeddings(model, loader, device: str) -> torch.Tensor:
#     model.eval()
#     embs = [model.embed(x.to(device)).cpu() for x, _ in loader]
#     return torch.cat(embs, dim=0)


# def build_model(model_name: str, config_params: dict, override: dict | None, device: str):
#     """Model params from the config, with HP-search overrides merged in (override excludes lr)."""
#     params = dict(config_params)
#     if override:
#         params.update(override)
#     return get_model(model_name, **params).to(device)


# def momentum_steps_for(
#     model_name: str, config_params: dict, override: dict | None,
#     momentum_epochs: int, steps_per_epoch: int,
# ) -> dict | None:
#     """Set the EMA momentum-schedule length so the target encoder ramps momentum -> momentum_end
#     and *freezes* after momentum_epochs epochs.

#     This must be short enough to complete inside the early-stopping window (the JEPA loss tends to
#     U-turn within a handful of epochs as the EMA target drifts), so it is scaled to momentum_epochs
#     -- NOT the full n_epochs budget, which early-stopping never reaches.

#     Only for ijepa; skipped if momentum_steps is set explicitly in the config or the override.
#     """
#     if model_name != 'ijepa':
#         return override
#     if 'momentum_steps' in config_params or (override and 'momentum_steps' in override):
#         return override
#     steps = max(1, momentum_epochs * steps_per_epoch)
#     return {**(override or {}), 'momentum_steps': steps}


# def train_once(
#     model, train_loader, valid_loader, *, lr, n_epochs, patience,
#     scheduler_name, scheduler_params, checkpoint_path, device,
# ) -> dict:
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     metric = NegPredErr().to(device)  # checkpoint/early-stop on negative RMSE
#     scheduler = None
#     if scheduler_name == 'reduce_on_plateau':
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, **(scheduler_params or {})
#         )
#     return train(
#         model, optimizer, jepa_loss, metric, train_loader, valid_loader,
#         n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path,
#         scheduler=scheduler, device=device,
#     )


# def write_dropped_dhs(checkpoint_path: str, qc_out: dict) -> None:
#     """The benchmark rule depends on this file existing next to the checkpoint."""
#     with open(checkpoint_path.replace('.pt', '.dropped_dhs.json'), 'w') as f:
#         json.dump(
#             {
#                 'dropped_dhs': qc_out.get('dropped_dhs', []),
#                 'dropped_sids': qc_out.get('dropped_sids', []),
#                 'coverage_threshold': qc_out.get('coverage_threshold'),
#                 'max_sample_loss': qc_out.get('max_sample_loss'),
#             },
#             f, indent=2,
#         )


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('--config', required=True, type=Path)
#     parser.add_argument('--suffix', default='_rebinned')
#     parser.add_argument('--n-trials', type=int, default=0, help='0 = single run; N>0 = random search')
#     parser.add_argument('--trial-epochs', type=int, default=30, help='epochs per search trial')
#     parser.add_argument('--final-epochs', type=int, default=None, help='final/single train epochs (default: training.n_epochs)')
#     parser.add_argument('--momentum-epochs', type=int, default=10, help='epochs over which the EMA momentum ramps to momentum_end, then freezes (ijepa)')
#     parser.add_argument('--patience', type=int, default=10)
#     parser.add_argument('--search-seed', type=int, default=0)
#     parser.add_argument(
#         '--rankme-floor', type=float, default=2.5,
#         help='min RankMe a trial must reach to be eligible; among eligible, pick best (max) NegPredErr',
#     )
#     parser.add_argument('--output-dir', type=Path, default=Path('models/jepa_search/'))
#     parser.add_argument('--device', default=None, help='cuda | mps | cpu (auto if omitted)')
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

#     with args.config.open() as f:
#         config = yaml.safe_load(f)

#     model_cfg = config['model']
#     model_name = model_cfg['name']
#     training_cfg = config.get('training', {})
#     preproc_cfg = config.get('preprocessing', {})
#     data_cfg = config['data']

#     seed = training_cfg.get('seed', 42)
#     torch.manual_seed(seed)
#     device = pick_device(args.device)
#     config_params = model_cfg.get('params', {})
#     scheduler_name = training_cfg.get('scheduler')
#     scheduler_params = training_cfg.get('scheduler_params', {})
#     final_epochs = args.final_epochs or training_cfg.get('n_epochs', 200)
#     lr = training_cfg.get('learning_rate', 1e-4)  # fixed (not searched)
#     checkpoint = model_cfg['checkpoint']
#     os.makedirs(os.path.dirname(checkpoint) or '.', exist_ok=True)
#     logger.info('config=%s model=%s device=%s', args.config, model_name, device)

#     transform_configs = preproc_cfg.get('transforms', [])
#     needs_standardization = any(t['name'] == 'standardization' for t in transform_configs)
#     transform_fn = build_transform_pipeline(transform_configs) if transform_configs else None

#     qc_kwargs = resolve_coverage_kwargs(config)
#     qc_out: dict = {}

#     # JEPA-family trains healthy-only on positive Lymphoid -> only_positive=True.
#     train_loader, valid_loader, _ = get_dataloaders(
#         output_dir=data_cfg['training_output_dir'],
#         transform_fn=transform_fn,
#         needs_standardization=needs_standardization,
#         train_size=training_cfg.get('train_size', 80),
#         valid_size=training_cfg.get('valid_size', 10),
#         batch_size=training_cfg.get('batch_size', 32),
#         seed=seed,
#         suffix=args.suffix,
#         only_positive=True,
#         qc_out=qc_out,
#         **qc_kwargs,
#     )

#     steps_per_epoch = len(train_loader)

#     # ---- single run ----------------------------------------------------------------
#     if args.n_trials <= 0:
#         ovr = momentum_steps_for(model_name, config_params, None, args.momentum_epochs, steps_per_epoch)
#         model = build_model(model_name, config_params, ovr, device)
#         history = train_once(
#             model, train_loader, valid_loader,
#             lr=lr, n_epochs=final_epochs,
#             patience=args.patience, scheduler_name=scheduler_name,
#             scheduler_params=scheduler_params, checkpoint_path=checkpoint, device=device,
#         )
#         torch.save(history, checkpoint.replace('.pt', '.history.pt'))
#         write_dropped_dhs(checkpoint, qc_out)
#         rm = rankme(collect_embeddings(model, valid_loader, device))
#         logger.info('single run done. RankMe=%.2f checkpoint=%s', rm, checkpoint)
#         return

#     # ---- random search -------------------------------------------------------------
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     rng = random.Random(args.search_seed)
#     seen: set[str] = set()
#     results: list[dict] = []

#     for i in range(1, args.n_trials + 1):
#         for _ in range(20):  # de-dupe
#             params = sample_params(rng, model_name)
#             key = tag_of(params)
#             if key not in seen:
#                 seen.add(key)
#                 break

#         ckpt = args.output_dir / f'trial{i:03d}_{tag_of(params)}.pt'
#         logger.info('--- trial [%02d/%02d] %s ---', i, args.n_trials, tag_of(params))
#         try:
#             ovr = momentum_steps_for(
#                 model_name, config_params, params, args.momentum_epochs, steps_per_epoch)
#             model = build_model(model_name, config_params, ovr, device)
#             history = train_once(
#                 model, train_loader, valid_loader,
#                 lr=lr, n_epochs=args.trial_epochs, patience=args.patience,
#                 scheduler_name=scheduler_name, scheduler_params=scheduler_params,
#                 checkpoint_path=str(ckpt), device=device,
#             )
#             emb = collect_embeddings(model, valid_loader, device)
#             run = {
#                 'params': params, 'tag': tag_of(params), 'checkpoint': str(ckpt),
#                 'rankme': rankme(emb), 'emb_std': float(emb.std(0).mean()),
#                 'best_neg_pred_err': float(max(history['valid_metrics']) if history['valid_metrics'] else float('-inf')),
#                 'epochs_run': len(history['valid_metrics']),
#             }
#         except Exception as e:  # noqa: BLE001 - record failures, keep searching
#             logger.exception('trial failed: %s', tag_of(params))
#             run = {'params': params, 'tag': tag_of(params), 'checkpoint': str(ckpt),
#                    'rankme': float('-inf'), 'error': str(e)}
#         run['trial'] = i
#         results.append(run)
#         logger.info('[%02d/%02d] %s -> RankMe=%.2f', i, args.n_trials, tag_of(params),
#                     run.get('rankme', float('-inf')))

#     # collapse gate (RankMe >= floor), then pick the best predictor among survivors by NegPredErr.
#     # Falls back to RankMe ranking if no trial clears the floor.
#     floor = args.rankme_floor
#     passing = [r for r in results if r.get('rankme', float('-inf')) >= floor]
#     if passing:
#         passing_trials = {r['trial'] for r in passing}
#         ranked = sorted(passing, key=lambda r: r.get('best_neg_pred_err', float('-inf')), reverse=True)
#         ranked += sorted(
#             (r for r in results if r['trial'] not in passing_trials),
#             key=lambda r: r.get('rankme', float('-inf')), reverse=True,
#         )
#     else:
#         logger.warning('no trial reached RankMe floor %.1f; ranking by RankMe instead', floor)
#         ranked = sorted(results, key=lambda r: r.get('rankme', float('-inf')), reverse=True)
#     best = ranked[0]
#     shutil.copyfile(best['checkpoint'], args.output_dir / 'best_jepa.pt')
#     with (args.output_dir / 'jepa_search_results.json').open('w') as f:
#         json.dump(
#             {'config_path': str(args.config), 'model': model_name,
#              'search_space': SEARCH_SPACE.get(model_name),
#              'ranking_metric': f'max NegPredErr among RankMe>={floor} (else RankMe)',
#              'rankme_floor': floor,
#              'best': best, 'ranked_runs': ranked},
#             f, indent=2,
#         )
#     logger.info('=== winner === %s RankMe=%.2f negPredErr=%.4f',
#                 best['tag'], best.get('rankme', float('-inf')), best.get('best_neg_pred_err', float('-inf')))

#     if final_epochs > 0:
#         logger.info('retraining winner for %d epochs -> %s', final_epochs, checkpoint)
#         ovr = momentum_steps_for(
#             model_name, config_params, best['params'], args.momentum_epochs, steps_per_epoch)
#         model = build_model(model_name, config_params, ovr, device)
#         history = train_once(
#             model, train_loader, valid_loader,
#             lr=lr, n_epochs=final_epochs, patience=args.patience,
#             scheduler_name=scheduler_name, scheduler_params=scheduler_params,
#             checkpoint_path=checkpoint, device=device,
#         )
#         torch.save(history, checkpoint.replace('.pt', '.history.pt'))
#         write_dropped_dhs(checkpoint, qc_out)
#         final_rankme = rankme(collect_embeddings(model, valid_loader, device))
#         logger.info('final model saved: %s (RankMe=%.2f; set model.params to %s in the config)',
#                     checkpoint, final_rankme, best['params'])


# if __name__ == '__main__':
#     main()
