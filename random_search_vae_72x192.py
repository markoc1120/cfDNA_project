# """Random search for the VAE model on 32x192 inputs.

# Mirrors workflow/scripts/train_model.py so trained models are consistent with
# the Snakemake pipeline. Runs N trials, each sampling hyperparameters from a
# fixed search space, and records the best validation NegReconMSE per trial.

# Usage:
#     uv run python random_search_vae.py \
#         --config confs/thesis.yaml \
#         --output-dir models/vae_randomsearch/ \
#         --n-trials 60 \
#         --epochs 20

# For the healthy-only / Lymphoid + negative-Lymphoid run, prepare a separate
# config (different data paths, different training_dhs_dir, etc.) and pass:
#     --config confs/thesis_lymphoid.yaml --no-only-positive
# """

# from __future__ import annotations

# import argparse
# import glob
# import json
# import logging
# import math
# import os
# import random
# import shutil
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Any

# import torch
# import yaml

# from cfdna.models import get_model
# from cfdna.models.vae import vae_loss
# from cfdna.preprocessing.transforms import build_transform_pipeline
# from cfdna.training.trainer import NegReconMSE, train
# from cfdna.training.utils import SelfTargetLoader, get_dataloaders

# INPUT_HEIGHT = 72
# INPUT_WIDTH = 192

# LATENT_DIMS = [2, 4, 8, 16, 32, 64, 128, 256]
# BASE_CHANNELS = [16, 32, 64]
# BETAS = [0.01, 0.1, 1.0, 10.0]
# LR_RANGE = (1e-5, 1e-2)


# @dataclass(frozen=True)
# class RunConfig:
#     latent_dim: int
#     base_channels: int
#     beta: float
#     learning_rate: float

#     def tag(self) -> str:
#         return (
#             f'ld{self.latent_dim}_bc{self.base_channels}'
#             f'_b{self.beta:.4g}_lr{self.learning_rate:.2e}'
#         )


# def pick_device() -> str:
#     if torch.cuda.is_available():
#         return 'cuda'
#     if torch.backends.mps.is_available():
#         return 'mps'
#     return 'cpu'


# def sample_config(rng: random.Random) -> RunConfig:
#     return RunConfig(
#         latent_dim=rng.choice(LATENT_DIMS),
#         base_channels=rng.choice(BASE_CHANNELS),
#         beta=rng.choice(BETAS),
#         learning_rate=math.exp(rng.uniform(math.log(LR_RANGE[0]), math.log(LR_RANGE[1]))),
#     )


# def setup_logging(log_path: Path) -> logging.Logger:
#     logger = logging.getLogger('random_search_vae')
#     logger.setLevel(logging.INFO)
#     logger.handlers.clear()
#     fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#     fh = logging.FileHandler(log_path)
#     fh.setFormatter(fmt)
#     sh = logging.StreamHandler()
#     sh.setFormatter(fmt)
#     logger.addHandler(fh)
#     logger.addHandler(sh)
#     return logger


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


# def train_one(
#     cfg: RunConfig,
#     train_loader,
#     valid_loader,
#     *,
#     n_epochs: int,
#     patience: int,
#     scheduler_name: str | None,
#     scheduler_params: dict | None,
#     device: str,
#     checkpoint_path: Path,
#     seed: int,
# ) -> dict:
#     torch.manual_seed(seed)
#     model = get_model(
#         'vae',
#         latent_dim=cfg.latent_dim,
#         base_channels=cfg.base_channels,
#         input_height=INPUT_HEIGHT,
#         input_width=INPUT_WIDTH,
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
#     metric = NegReconMSE().to(device)

#     def loss_fn(out, target):
#         return vae_loss(out.reconstruction, target, out.mu, out.logvar, beta=cfg.beta)

#     scheduler = None
#     if scheduler_name == 'reduce_on_plateau':
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, **(scheduler_params or {})
#         )

#     history = train(
#         model,
#         optimizer,
#         loss_fn,
#         metric,
#         train_loader,
#         valid_loader,
#         n_epochs=n_epochs,
#         patience=patience,
#         checkpoint_path=str(checkpoint_path),
#         scheduler=scheduler,
#         device=device,
#     )
#     best_valid = max(history['valid_metrics']) if history['valid_metrics'] else float('-inf')
#     return {
#         'config': asdict(cfg),
#         'tag': cfg.tag(),
#         'checkpoint': str(checkpoint_path),
#         'best_valid_metric': float(best_valid),
#         'epochs_run': len(history['valid_metrics']),
#     }


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('--config', required=True, type=Path)
#     parser.add_argument(
#         '--only-positive',
#         dest='only_positive',
#         action=argparse.BooleanOptionalAction,
#         default=True,
#         help='Exclude _negative DHS matrices (default: True, matches VAE workflow)',
#     )
#     parser.add_argument('--output-dir', type=Path, default=Path('models/vae_randomsearch/'))
#     parser.add_argument('--suffix', default='_downsampled')
#     parser.add_argument('--n-trials', type=int, default=60)
#     parser.add_argument('--epochs', type=int, default=20)
#     parser.add_argument('--patience', type=int, default=5)
#     parser.add_argument('--search-seed', type=int, default=0)
#     parser.add_argument('--device', default=None, help='cuda | mps | cpu (auto if omitted)')
#     args = parser.parse_args()

#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     logger = setup_logging(args.output_dir / 'random_search.log')

#     with args.config.open() as f:
#         config = yaml.safe_load(f)

#     training_cfg = config.get('training', {})
#     preproc_cfg = config.get('preprocessing', {})
#     data_cfg = config['data']

#     seed = training_cfg.get('seed', 42)
#     torch.manual_seed(seed)
#     device = args.device or pick_device()

#     transform_configs = preproc_cfg.get('transforms', [])
#     needs_standardization = any(t['name'] == 'standardization' for t in transform_configs)
#     transform_fn = build_transform_pipeline(transform_configs) if transform_configs else None

#     qc_kwargs = resolve_coverage_kwargs(config)
#     qc_out: dict = {}

#     logger.info('config: %s', args.config)
#     logger.info('data dir: %s', data_cfg['training_output_dir'])
#     logger.info('only_positive: %s', args.only_positive)
#     logger.info('device: %s', device)
#     logger.info(
#         'n_trials=%d epochs=%d patience=%d search_seed=%d data_seed=%d',
#         args.n_trials, args.epochs, args.patience, args.search_seed, seed,
#     )
#     logger.info(
#         'search space: latent_dim=%s base_channels=%s beta=%s lr=log-uniform[%g,%g]',
#         LATENT_DIMS, BASE_CHANNELS, BETAS, LR_RANGE[0], LR_RANGE[1],
#     )

#     train_loader, valid_loader, _ = get_dataloaders(
#         output_dir=data_cfg['training_output_dir'],
#         transform_fn=transform_fn,
#         needs_standardization=needs_standardization,
#         train_size=training_cfg.get('train_size', 80),
#         valid_size=training_cfg.get('valid_size', 10),
#         batch_size=training_cfg.get('batch_size', 32),
#         seed=seed,
#         suffix=args.suffix,
#         only_positive=args.only_positive,
#         qc_out=qc_out,
#         **qc_kwargs,
#     )
#     train_loader = SelfTargetLoader(train_loader)
#     valid_loader = SelfTargetLoader(valid_loader)

#     if qc_out:
#         with (args.output_dir / 'qc.json').open('w') as f:
#             json.dump(
#                 {
#                     'dropped_dhs': qc_out.get('dropped_dhs', []),
#                     'dropped_sids': qc_out.get('dropped_sids', []),
#                     'coverage_threshold': qc_out.get('coverage_threshold'),
#                     'max_sample_loss': qc_out.get('max_sample_loss'),
#                 },
#                 f,
#                 indent=2,
#             )

#     scheduler_name = training_cfg.get('scheduler', None)
#     scheduler_params = training_cfg.get('scheduler_params', {})

#     rng = random.Random(args.search_seed)
#     seen: set[tuple] = set()
#     results: list[dict] = []

#     for i in range(1, args.n_trials + 1):
#         # de-dupe identical samples (rare but harmless)
#         for _ in range(20):
#             cfg = sample_config(rng)
#             key = (cfg.latent_dim, cfg.base_channels, cfg.beta, round(cfg.learning_rate, 8))
#             if key not in seen:
#                 seen.add(key)
#                 break

#         logger.info('--- trial [%02d/%02d] %s ---', i, args.n_trials, cfg.tag())
#         ckpt = args.output_dir / f'trial{i:03d}_{cfg.tag()}.pt'
#         try:
#             run = train_one(
#                 cfg,
#                 train_loader,
#                 valid_loader,
#                 n_epochs=args.epochs,
#                 patience=args.patience,
#                 scheduler_name=scheduler_name,
#                 scheduler_params=scheduler_params,
#                 device=device,
#                 checkpoint_path=ckpt,
#                 seed=seed,
#             )
#         except Exception as e:
#             logger.exception('trial failed: %s', cfg.tag())
#             run = {
#                 'config': asdict(cfg),
#                 'tag': cfg.tag(),
#                 'checkpoint': str(ckpt),
#                 'best_valid_metric': float('-inf'),
#                 'epochs_run': 0,
#                 'error': str(e),
#             }
#         run['trial'] = i
#         results.append(run)
#         logger.info(
#             '[%02d/%02d] latent_dim=%d base_channels=%d beta=%g lr=%.2e '
#             '-> best_valid=%.4f (%d epochs)',
#             i, args.n_trials, cfg.latent_dim, cfg.base_channels, cfg.beta,
#             cfg.learning_rate, run['best_valid_metric'], run['epochs_run'],
#         )

#     ranked = sorted(results, key=lambda r: r['best_valid_metric'], reverse=True)
#     best = ranked[0]
#     best_ckpt = args.output_dir / 'best_vae.pt'
#     if Path(best['checkpoint']).exists():
#         shutil.copyfile(best['checkpoint'], best_ckpt)

#     logger.info('=== winner ===')
#     logger.info(
#         'tag=%s best_valid=%.4f checkpoint=%s',
#         best['tag'], best['best_valid_metric'], best_ckpt,
#     )

#     summary = {
#         'input_shape': [INPUT_HEIGHT, INPUT_WIDTH],
#         'config_path': str(args.config),
#         'only_positive': args.only_positive,
#         'data_dir': data_cfg['training_output_dir'],
#         'n_trials': args.n_trials,
#         'epochs': args.epochs,
#         'patience': args.patience,
#         'search_seed': args.search_seed,
#         'data_seed': seed,
#         'search_space': {
#             'latent_dim': LATENT_DIMS,
#             'base_channels': BASE_CHANNELS,
#             'beta': BETAS,
#             'learning_rate_range': list(LR_RANGE),
#         },
#         'best': best,
#         'best_checkpoint': str(best_ckpt),
#         'ranked_runs': ranked,
#     }
#     with (args.output_dir / 'random_search_results.json').open('w') as f:
#         json.dump(summary, f, indent=2)
#     logger.info('results saved to %s', args.output_dir / 'random_search_results.json')


# if __name__ == '__main__':
#     main()


# """Offline VAE training, with an optional random hyperparameter search.

# Mirrors train_ssl.py's structure (and workflow/scripts/train_model.py's dataloaders) but for the
# VAE: reconstruction loss (vae_loss with beta), NegReconMSE validation metric, and SelfTargetLoader
# (the target is the input itself).

# Two modes:
#   * single run (default, --n-trials 0): train the model.params from the config for the full
#     training.n_epochs and save to model.checkpoint.
#   * random search (--n-trials N > 0): sample N configs over latent_dim / base_channels / beta,
#     train each for --trial-epochs, pick the best validation NegReconMSE, then retrain the winner
#     for --final-epochs and save it to model.checkpoint.

# learning_rate is NOT searched -- it is fixed via training.learning_rate (default 1e-3), matching
# train_ssl.py. This script is for 72x192 inputs only (it errors if the data has another shape).

# Usage:
#     uv run python random_search_vae.py --config confs/vae_run.yaml                 # single run
#     uv run python random_search_vae.py --config confs/vae_run.yaml --n-trials 30   # search + retrain
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
# from cfdna.models.vae import vae_loss
# from cfdna.preprocessing.transforms import build_transform_pipeline
# from cfdna.training.trainer import NegReconMSE, train
# from cfdna.training.utils import SelfTargetLoader, get_dataloaders

# logger = logging.getLogger('random_search_vae')

# INPUT_HEIGHT = 72
# INPUT_WIDTH = 192

# # Values within each list are kept well-separated (no near-duplicates) so a small trial budget
# # covers the grid. learning_rate is fixed via training.learning_rate (not searched).
# # Grid = 4 x 3 x 4 = 48 configs.
# SEARCH_SPACE = {
#     'latent_dim': [2, 8, 32, 128],
#     'base_channels': [16, 32, 64],
#     'beta': [0.01, 0.1, 1.0, 10.0],
# }


# def sample_params(rng: random.Random) -> dict:
#     return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


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


# def build_vae(config_params: dict, override: dict | None, input_hw: tuple[int, int], device: str):
#     """VAE from config params + search overrides. beta is a loss arg, not a constructor arg, so it
#     is stripped here; input_height/width come from the data."""
#     params = dict(config_params)
#     if override:
#         params.update(override)
#     params.pop('beta', None)
#     return get_model(
#         'vae', input_height=input_hw[0], input_width=input_hw[1], **params
#     ).to(device)


# def train_once(
#     model, beta, train_loader, valid_loader, *, lr, n_epochs, patience,
#     scheduler_name, scheduler_params, checkpoint_path, device,
# ) -> dict:
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     metric = NegReconMSE().to(device)  # checkpoint/early-stop on negative reconstruction RMSE

#     def loss_fn(out, target):
#         return vae_loss(out.reconstruction, target, out.mu, out.logvar, beta=beta)

#     scheduler = None
#     if scheduler_name == 'reduce_on_plateau':
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, **(scheduler_params or {})
#         )
#     return train(
#         model, optimizer, loss_fn, metric, train_loader, valid_loader,
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


# def beta_of(config_params: dict, override: dict | None) -> float:
#     if override and 'beta' in override:
#         return float(override['beta'])
#     return float(config_params.get('beta', 1.0))


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('--config', required=True, type=Path)
#     parser.add_argument('--suffix', default='_rebinned')
#     parser.add_argument('--n-trials', type=int, default=0, help='0 = single run; N>0 = random search')
#     parser.add_argument('--trial-epochs', type=int, default=30, help='epochs per search trial')
#     parser.add_argument('--final-epochs', type=int, default=None, help='final/single train epochs (default: training.n_epochs)')
#     parser.add_argument('--patience', type=int, default=10)
#     parser.add_argument('--search-seed', type=int, default=0)
#     parser.add_argument('--output-dir', type=Path, default=Path('models/vae_search/'))
#     parser.add_argument(
#         '--only-positive', dest='only_positive', action=argparse.BooleanOptionalAction,
#         default=True, help='Exclude _negative DHS matrices (default: True, matches VAE workflow)',
#     )
#     parser.add_argument('--device', default=None, help='cuda | mps | cpu (auto if omitted)')
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

#     with args.config.open() as f:
#         config = yaml.safe_load(f)

#     model_cfg = config['model']
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
#     lr = training_cfg.get('learning_rate', 1e-3)  # fixed (not searched)
#     checkpoint = model_cfg['checkpoint']
#     os.makedirs(os.path.dirname(checkpoint) or '.', exist_ok=True)
#     logger.info('config=%s device=%s only_positive=%s', args.config, device, args.only_positive)

#     transform_configs = preproc_cfg.get('transforms', [])
#     needs_standardization = any(t['name'] == 'standardization' for t in transform_configs)
#     transform_fn = build_transform_pipeline(transform_configs) if transform_configs else None

#     qc_kwargs = resolve_coverage_kwargs(config)
#     qc_out: dict = {}

#     train_loader, valid_loader, _ = get_dataloaders(
#         output_dir=data_cfg['training_output_dir'],
#         transform_fn=transform_fn,
#         needs_standardization=needs_standardization,
#         train_size=training_cfg.get('train_size', 80),
#         valid_size=training_cfg.get('valid_size', 10),
#         batch_size=training_cfg.get('batch_size', 32),
#         seed=seed,
#         suffix=args.suffix,
#         only_positive=args.only_positive,
#         qc_out=qc_out,
#         **qc_kwargs,
#     )

#     # this script is for 72x192 inputs only; verify the data matches, then wrap loaders -> (x, x)
#     input_hw = (INPUT_HEIGHT, INPUT_WIDTH)
#     sample_x, _ = next(iter(train_loader))
#     got = (sample_x.shape[2], sample_x.shape[3])
#     if got != input_hw:
#         raise ValueError(f'expected {input_hw} inputs but got {got}; this script is 72x192 only')
#     train_loader = SelfTargetLoader(train_loader)
#     valid_loader = SelfTargetLoader(valid_loader)

#     # ---- single run ----------------------------------------------------------------
#     if args.n_trials <= 0:
#         model = build_vae(config_params, None, input_hw, device)
#         history = train_once(
#             model, beta_of(config_params, None), train_loader, valid_loader,
#             lr=lr, n_epochs=final_epochs, patience=args.patience,
#             scheduler_name=scheduler_name, scheduler_params=scheduler_params,
#             checkpoint_path=checkpoint, device=device,
#         )
#         torch.save(history, checkpoint.replace('.pt', '.history.pt'))
#         write_dropped_dhs(checkpoint, qc_out)
#         best_valid = max(history['valid_metrics']) if history['valid_metrics'] else float('-inf')
#         logger.info('single run done. NegReconMSE=%.4f checkpoint=%s', best_valid, checkpoint)
#         return

#     # ---- random search -------------------------------------------------------------
#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     rng = random.Random(args.search_seed)
#     seen: set[str] = set()
#     results: list[dict] = []

#     for i in range(1, args.n_trials + 1):
#         for _ in range(20):  # de-dupe
#             params = sample_params(rng)
#             key = tag_of(params)
#             if key not in seen:
#                 seen.add(key)
#                 break

#         ckpt = args.output_dir / f'trial{i:03d}_{tag_of(params)}.pt'
#         logger.info('--- trial [%02d/%02d] %s ---', i, args.n_trials, tag_of(params))
#         try:
#             model = build_vae(config_params, params, input_hw, device)
#             history = train_once(
#                 model, beta_of(config_params, params), train_loader, valid_loader,
#                 lr=lr, n_epochs=args.trial_epochs, patience=args.patience,
#                 scheduler_name=scheduler_name, scheduler_params=scheduler_params,
#                 checkpoint_path=str(ckpt), device=device,
#             )
#             run = {
#                 'params': params, 'tag': tag_of(params), 'checkpoint': str(ckpt),
#                 'best_valid_metric': float(max(history['valid_metrics']) if history['valid_metrics'] else float('-inf')),
#                 'epochs_run': len(history['valid_metrics']),
#             }
#         except Exception as e:  # noqa: BLE001 - record failures, keep searching
#             logger.exception('trial failed: %s', tag_of(params))
#             run = {'params': params, 'tag': tag_of(params), 'checkpoint': str(ckpt),
#                    'best_valid_metric': float('-inf'), 'error': str(e)}
#         run['trial'] = i
#         results.append(run)
#         logger.info('[%02d/%02d] %s -> NegReconMSE=%.4f', i, args.n_trials, tag_of(params),
#                     run.get('best_valid_metric', float('-inf')))

#     # pick the winner by best validation metric (NegReconMSE)
#     ranked = sorted(results, key=lambda r: r.get('best_valid_metric', float('-inf')), reverse=True)
#     best = ranked[0]
#     shutil.copyfile(best['checkpoint'], args.output_dir / 'best_vae.pt')
#     with (args.output_dir / 'vae_search_results.json').open('w') as f:
#         json.dump(
#             {'config_path': str(args.config), 'input_shape': list(input_hw),
#              'search_space': SEARCH_SPACE, 'ranking_metric': 'max NegReconMSE (valid metric)',
#              'only_positive': args.only_positive, 'best': best, 'ranked_runs': ranked},
#             f, indent=2,
#         )
#     logger.info('=== winner === %s NegReconMSE=%.4f', best['tag'], best.get('best_valid_metric'))

#     # retrain the winner for the full schedule and save to model.checkpoint
#     if final_epochs > 0:
#         logger.info('retraining winner for %d epochs -> %s', final_epochs, checkpoint)
#         model = build_vae(config_params, best['params'], input_hw, device)
#         history = train_once(
#             model, beta_of(config_params, best['params']), train_loader, valid_loader,
#             lr=lr, n_epochs=final_epochs, patience=args.patience,
#             scheduler_name=scheduler_name, scheduler_params=scheduler_params,
#             checkpoint_path=checkpoint, device=device,
#         )
#         torch.save(history, checkpoint.replace('.pt', '.history.pt'))
#         write_dropped_dhs(checkpoint, qc_out)
#         logger.info('final model saved: %s (set model.params to %s in the config for inference)',
#                     checkpoint, best['params'])


# if __name__ == '__main__':
#     main()


"""Offline VAE training, with an optional random hyperparameter search.

Mirrors train_ssl.py's structure (and workflow/scripts/train_model.py's dataloaders) but for the
VAE: reconstruction loss (vae_loss with beta), NegReconMSE validation metric, and SelfTargetLoader
(the target is the input itself).

Two modes:
  * single run (default, --n-trials 0): train the model.params from the config for the full
    training.n_epochs and save to model.checkpoint.
  * random search (--n-trials N > 0): sample N configs over latent_dim / base_channels / beta,
    train each for --trial-epochs, pick the best validation NegReconMSE, then retrain the winner
    for --final-epochs and save it to model.checkpoint.

learning_rate IS searched (log-uniform); the config training.learning_rate is used only for the
single-run mode. This script is for 72x192 inputs only (it errors if the data has another shape).

Usage:
    uv run python random_search_vae.py --config confs/vae_run.yaml                 # single run
    uv run python random_search_vae.py --config confs/vae_run.yaml --n-trials 30   # search + retrain
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from cfdna.models import get_model
from cfdna.models.vae import vae_loss
from cfdna.preprocessing.transforms import build_transform_pipeline
from cfdna.training.trainer import NegReconMSE, train
from cfdna.training.utils import SelfTargetLoader, get_dataloaders

logger = logging.getLogger('random_search_vae')

INPUT_HEIGHT = 72
INPUT_WIDTH = 192

# Discrete dims are kept well-separated (no near-duplicates). learning_rate is sampled
# log-uniformly from LR_RANGE.
SEARCH_SPACE = {
    'latent_dim': [2, 8, 32, 128],
    'base_channels': [16, 32, 64],
    'beta': [0.01, 0.1, 1.0, 10.0],
}
LR_RANGE = (1e-4, 1e-2)  # log-uniform


def sample_params(rng: random.Random) -> dict:
    params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    params['learning_rate'] = math.exp(rng.uniform(math.log(LR_RANGE[0]), math.log(LR_RANGE[1])))
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


def build_vae(config_params: dict, override: dict | None, input_hw: tuple[int, int], device: str):
    """VAE from config params + search overrides. beta (loss arg) and learning_rate (optimizer arg)
    are not constructor args, so they are stripped here; input_height/width are fixed at 72x192."""
    params = dict(config_params)
    if override:
        params.update(override)
    params.pop('beta', None)
    params.pop('learning_rate', None)
    return get_model(
        'vae', input_height=input_hw[0], input_width=input_hw[1], **params
    ).to(device)


def train_once(
    model, beta, train_loader, valid_loader, *, lr, n_epochs, patience,
    scheduler_name, scheduler_params, checkpoint_path, device,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metric = NegReconMSE().to(device)  # checkpoint/early-stop on negative reconstruction RMSE

    def loss_fn(out, target):
        return vae_loss(out.reconstruction, target, out.mu, out.logvar, beta=beta)

    scheduler = None
    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **(scheduler_params or {})
        )
    return train(
        model, optimizer, loss_fn, metric, train_loader, valid_loader,
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


def beta_of(config_params: dict, override: dict | None) -> float:
    if override and 'beta' in override:
        return float(override['beta'])
    return float(config_params.get('beta', 1.0))


def lr_of(override: dict | None, config_lr: float) -> float:
    """Searched learning_rate (from override) if present, else the config's fixed lr."""
    if override and 'learning_rate' in override:
        return float(override['learning_rate'])
    return config_lr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--suffix', default='_rebinned')
    parser.add_argument('--n-trials', type=int, default=0, help='0 = single run; N>0 = random search')
    parser.add_argument('--trial-epochs', type=int, default=30, help='epochs per search trial')
    parser.add_argument('--final-epochs', type=int, default=None, help='final/single train epochs (default: training.n_epochs)')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--search-seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path('models/vae_search/'))
    parser.add_argument(
        '--only-positive', dest='only_positive', action=argparse.BooleanOptionalAction,
        default=True, help='Exclude _negative DHS matrices (default: True, matches VAE workflow)',
    )
    parser.add_argument('--device', default=None, help='cuda | mps | cpu (auto if omitted)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    with args.config.open() as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
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
    config_lr = training_cfg.get('learning_rate', 1e-3)  # used for single-run; trials sample lr
    checkpoint = model_cfg['checkpoint']
    os.makedirs(os.path.dirname(checkpoint) or '.', exist_ok=True)
    logger.info('config=%s device=%s only_positive=%s', args.config, device, args.only_positive)

    transform_configs = preproc_cfg.get('transforms', [])
    needs_standardization = any(t['name'] == 'standardization' for t in transform_configs)
    transform_fn = build_transform_pipeline(transform_configs) if transform_configs else None

    qc_kwargs = resolve_coverage_kwargs(config)
    qc_out: dict = {}

    train_loader, valid_loader, _ = get_dataloaders(
        output_dir=data_cfg['training_output_dir'],
        transform_fn=transform_fn,
        needs_standardization=needs_standardization,
        train_size=training_cfg.get('train_size', 80),
        valid_size=training_cfg.get('valid_size', 10),
        batch_size=training_cfg.get('batch_size', 32),
        seed=seed,
        suffix=args.suffix,
        only_positive=args.only_positive,
        qc_out=qc_out,
        **qc_kwargs,
    )

    # this script is for 72x192 inputs only; verify the data matches, then wrap loaders -> (x, x)
    input_hw = (INPUT_HEIGHT, INPUT_WIDTH)
    sample_x, _ = next(iter(train_loader))
    got = (sample_x.shape[2], sample_x.shape[3])
    if got != input_hw:
        raise ValueError(f'expected {input_hw} inputs but got {got}; this script is 72x192 only')
    train_loader = SelfTargetLoader(train_loader)
    valid_loader = SelfTargetLoader(valid_loader)

    # ---- single run ----------------------------------------------------------------
    if args.n_trials <= 0:
        model = build_vae(config_params, None, input_hw, device)
        history = train_once(
            model, beta_of(config_params, None), train_loader, valid_loader,
            lr=lr_of(None, config_lr), n_epochs=final_epochs, patience=args.patience,
            scheduler_name=scheduler_name, scheduler_params=scheduler_params,
            checkpoint_path=checkpoint, device=device,
        )
        torch.save(history, checkpoint.replace('.pt', '.history.pt'))
        write_dropped_dhs(checkpoint, qc_out)
        best_valid = max(history['valid_metrics']) if history['valid_metrics'] else float('-inf')
        logger.info('single run done. NegReconMSE=%.4f checkpoint=%s', best_valid, checkpoint)
        return

    # ---- random search -------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.search_seed)
    seen: set[str] = set()
    results: list[dict] = []

    for i in range(1, args.n_trials + 1):
        for _ in range(20):  # de-dupe
            params = sample_params(rng)
            key = tag_of(params)
            if key not in seen:
                seen.add(key)
                break

        ckpt = args.output_dir / f'trial{i:03d}_{tag_of(params)}.pt'
        logger.info('--- trial [%02d/%02d] %s ---', i, args.n_trials, tag_of(params))
        try:
            model = build_vae(config_params, params, input_hw, device)
            history = train_once(
                model, beta_of(config_params, params), train_loader, valid_loader,
                lr=lr_of(params, config_lr), n_epochs=args.trial_epochs, patience=args.patience,
                scheduler_name=scheduler_name, scheduler_params=scheduler_params,
                checkpoint_path=str(ckpt), device=device,
            )
            run = {
                'params': params, 'tag': tag_of(params), 'checkpoint': str(ckpt),
                'best_valid_metric': float(max(history['valid_metrics']) if history['valid_metrics'] else float('-inf')),
                'epochs_run': len(history['valid_metrics']),
            }
        except Exception as e:  # noqa: BLE001 - record failures, keep searching
            logger.exception('trial failed: %s', tag_of(params))
            run = {'params': params, 'tag': tag_of(params), 'checkpoint': str(ckpt),
                   'best_valid_metric': float('-inf'), 'error': str(e)}
        run['trial'] = i
        results.append(run)
        logger.info('[%02d/%02d] %s -> NegReconMSE=%.4f', i, args.n_trials, tag_of(params),
                    run.get('best_valid_metric', float('-inf')))

    # pick the winner by best validation metric (NegReconMSE)
    ranked = sorted(results, key=lambda r: r.get('best_valid_metric', float('-inf')), reverse=True)
    best = ranked[0]
    shutil.copyfile(best['checkpoint'], args.output_dir / 'best_vae.pt')
    with (args.output_dir / 'vae_search_results.json').open('w') as f:
        json.dump(
            {'config_path': str(args.config), 'input_shape': list(input_hw),
             'search_space': {**SEARCH_SPACE, 'learning_rate_range': list(LR_RANGE)},
             'ranking_metric': 'max NegReconMSE (valid metric)',
             'only_positive': args.only_positive, 'best': best, 'ranked_runs': ranked},
            f, indent=2,
        )
    logger.info('=== winner === %s NegReconMSE=%.4f', best['tag'], best.get('best_valid_metric'))

    # retrain the winner for the full schedule and save to model.checkpoint
    if final_epochs > 0:
        logger.info('retraining winner for %d epochs -> %s', final_epochs, checkpoint)
        model = build_vae(config_params, best['params'], input_hw, device)
        history = train_once(
            model, beta_of(config_params, best['params']), train_loader, valid_loader,
            lr=lr_of(best['params'], config_lr), n_epochs=final_epochs, patience=args.patience,
            scheduler_name=scheduler_name, scheduler_params=scheduler_params,
            checkpoint_path=checkpoint, device=device,
        )
        torch.save(history, checkpoint.replace('.pt', '.history.pt'))
        write_dropped_dhs(checkpoint, qc_out)
        logger.info('final model saved: %s (set model.params to %s in the config for inference)',
                    checkpoint, best['params'])


if __name__ == '__main__':
    main()
