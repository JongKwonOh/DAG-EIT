# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
# CC BY-NC-SA 4.0

import os
import re
import json
import click
import torch
import torch.utils.data as torchdata
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
import numpy as np
from scipy.io import loadmat

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')  # False warning printed by PyTorch 1.12.

# ----------------------------------------------------------------------------
# Random tensor dataset for fallback when --data is not provided.

class MatDataset(torchdata.Dataset):
    def __init__(self, path, use_labels=False, cache=True, resolution=None, **kwargs):
        super().__init__()
        self.path = path
        self.use_labels = bool(use_labels)
        self.cache = bool(cache)

        mat = loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        key = 'thresh_p_data' if 'thresh_p_data' in keys else keys[-1]
        arr = mat[key]

        if arr.ndim == 3:          # (B, H, W) -> (B,1,H,W)
            arr = arr.astype(np.float32)[:, None, :, :]
        elif arr.ndim == 4:        # (B, C, H, W)
            arr = arr.astype(np.float32)
        else:
            raise IOError(f"Expected 3D/4D array in '{key}', got {arr.shape}")

        imgs = torch.from_numpy(arr)
        self.images = imgs
        self.length, self.num_channels, self.H, self.W = imgs.shape
        self.resolution = int(self.H) if resolution is None else int(resolution)
        self.has_labels = False if not self.use_labels else True
        self.label_dim = 0 if not self.has_labels else 1
        self.name = f"MatDataset({os.path.basename(path)}:{key})"
        self._empty_label = torch.zeros(0, dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self._empty_label
        return x, y


class EITDataset(torchdata.Dataset):
    """
    EIT .pt format expected keys:
      - Vbd:   (B, 64, 64) or compatible
      - sigma: (B, 64, 64)
      - Ex:    (B, 16, 64, 64)
      - Ey:    (B, 16, 64, 64)

    Output:
      x: (34, H, W) = [Vbd(1), sigma(1), Ex(16), Ey(16)]
    """
    def __init__(self, split='train', data_path='.', resolution=None,
                 norm='none', norm_mean=None, norm_std=None, **kwargs):
        super().__init__()
        file_path = data_path

        print(f"[EITDataset] Loading ({split}): {file_path}")
        data = torch.load(file_path, map_location='cpu')

        Vbd = data['Vbd']
        sigma = data['sigma']
        Ex = data['Ex']
        Ey = data['Ey']

        sigma = sigma.unsqueeze(1)
        Vbd = Vbd.unsqueeze(1)

        self.images = torch.cat([Vbd, sigma, Ex, Ey], dim=1).to(torch.float32)

        self.length, self.num_channels, self.H, self.W = self.images.shape
        self.resolution = int(self.H) if resolution is None else int(resolution)

        self.has_labels = False
        self.label_dim = 0
        self.name = f"EITDataset({split})"
        self._empty_label = torch.zeros(0, dtype=torch.float32)

        self.norm = str(norm)
        self.norm_mean = None
        self.norm_std = None

        if self.norm == 'standardize':
            if norm_mean is None or norm_std is None:
                with torch.no_grad():
                    x = self.images
                    mean = x.mean(dim=(0,2,3), keepdim=True)
                    var  = x.var(dim=(0,2,3), keepdim=True, unbiased=False)
                    std  = torch.sqrt(var + 1e-8)
                self.norm_mean = mean
                self.norm_std = std
            else:
                m = torch.as_tensor(norm_mean, dtype=torch.float32)
                s = torch.as_tensor(norm_std, dtype=torch.float32)
                if m.ndim == 1:
                    m = m.view(1, -1, 1, 1)
                if s.ndim == 1:
                    s = s.view(1, -1, 1, 1)
                self.norm_mean = m
                self.norm_std = s

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.norm == 'standardize':
            x = (x - self.norm_mean[0]) / self.norm_std[0]
        y = self._empty_label
        return x, y

# ----------------------------------------------------------------------------
# Parse helpers

def parse_int_list(s):
    if isinstance(s, list):
        return s
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        p = p.strip()
        if p == "":
            continue
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

def parse_path_list(s):
    """Comma-separated path list -> [paths]."""
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s if str(x).strip() != ""]
    s = str(s).strip()
    if s == "":
        return []
    return [p.strip() for p in s.split(',') if p.strip() != ""]

# ----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path(s) to the dataset (EIT: comma-separated train pt files).', metavar='ZIP|DIR|PT|LIST', type=str, required=True)
@click.option('--valid-data',    help='(EIT) Path to validation pt file.', metavar='PT',            type=str, default=None, show_default=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--lr-min',        help='Minimum LR for warmup+cosine scheduler', metavar='FLOAT',    type=float, default=1e-6, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                               type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

@click.option('--report-channels', help='Extra per-channel loss stats to log, e.g. "0,1" or "0-3". Empty disables.',
              metavar='LIST', type=str, default='0,1', show_default=True)

@click.option('--norm', help='EIT/channel normalization: none|standardize',
              type=click.Choice(['none', 'standardize']), default='none', show_default=True)
@click.option('--sigma-data', help='EDM sigma_data. If <=0, auto-set based on --norm.',
              type=float, default=-1.0, show_default=True)

@click.option('--wd', help='AdamW weight decay', type=float, default=1e-4, show_default=True)
@click.option('--grad-clip', help='Global grad norm clip (0 disables)', type=float, default=1.0, show_default=True)
@click.option('--lr-warmup-kimg',
              help='Warmup length in kimg for warmup+cosine scheduler (1-A).',
              metavar='KIMG',
              type=click.IntRange(min=0),
              default=1000,
              show_default=True)

@click.option('--train-mix-ratio',
              help='Mix ratio for multiple EIT train datasets, e.g. "2,1,1,1". If omitted, defaults to all ones matching the number of datasets.',
              metavar='LIST',
              type=str,
              default=None,
              show_default=True)

def main(report_channels, train_mix_ratio, **kwargs):
    opts = dnnlib.EasyDict(kwargs)
    opts.report_channels = parse_int_list(report_channels)
    opts.train_mix_ratio = parse_int_list(train_mix_ratio)

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except Exception:
        pass

    dist.init()

    c = dnnlib.EasyDict()
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()

    c.optimizer_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.AdamW',
        lr=opts.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=float(opts.wd),
    )

    if opts.data is None or str(opts.data).strip() == "":
        raise click.ClickException('--data must be specified')

    data_str = str(opts.data).strip()

    if os.path.splitext(data_str.split(',')[0])[-1].lower() == ".mat":
        c.dataset_kwargs = dnnlib.EasyDict(
            class_name=f'{__name__}.MatDataset',
            path=data_str,
            use_labels=opts.cond,
            cache=opts.cache,
        )
        dataset_path_for_print = data_str
        use_labels_flag = bool(opts.cond)
        xflip_flag = False
        cache_flag = bool(opts.cache)

    elif "eit" in data_str.lower():
        train_paths = parse_path_list(data_str)
        if len(train_paths) == 0:
            raise click.ClickException('EIT mode: --data must contain at least one train .pt path')
        if opts.valid_data is None or str(opts.valid_data).strip() == "":
            raise click.ClickException('EIT mode: --valid-data must be specified')

        tmp_ds = EITDataset(split='train', data_path=train_paths[0], norm=opts.norm)

        if opts.norm == 'standardize':
            C = tmp_ds.num_channels
            if dist.get_rank() == 0:
                m = tmp_ds.norm_mean.view(-1).to(torch.float32)
                s = tmp_ds.norm_std.view(-1).to(torch.float32)
            else:
                m = torch.zeros([C], dtype=torch.float32)
                s = torch.ones([C], dtype=torch.float32)

            device = torch.device('cuda')
            m = m.to(device)
            s = s.to(device)
            torch.distributed.broadcast(m, src=0)
            torch.distributed.broadcast(s, src=0)

            norm_mean = m.detach().cpu().tolist()
            norm_std  = s.detach().cpu().tolist()
        else:
            norm_mean = None
            norm_std  = None

        c.train_dataset_kwargs_list = [
            dnnlib.EasyDict(
                class_name=f'{__name__}.EITDataset',
                split='train',
                data_path=p,
                norm=opts.norm,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
            for p in train_paths
        ]

        if len(opts.train_mix_ratio) == 0:
            c.train_dataset_mix_ratio = tuple([1] * len(train_paths))
        else:
            if len(opts.train_mix_ratio) != len(train_paths):
                raise click.ClickException(
                    f'--train-mix-ratio length mismatch: got {len(opts.train_mix_ratio)} values '
                    f'but --data contains {len(train_paths)} dataset(s)'
                )
            if sum(opts.train_mix_ratio) <= 0:
                raise click.ClickException('--train-mix-ratio must have positive sum')
            c.train_dataset_mix_ratio = tuple(int(x) for x in opts.train_mix_ratio)

        c.valid_dataset_kwargs = dnnlib.EasyDict(
            class_name=f'{__name__}.EITDataset',
            split='valid',
            data_path=str(opts.valid_data).strip(),
            norm=opts.norm,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        c.dataset_kwargs = c.train_dataset_kwargs_list[0]

        dataset_path_for_print = "TRAIN:\n  " + "\n  ".join(train_paths) + f"\nVALID:\n  {str(opts.valid_data).strip()}"
        use_labels_flag = False
        xflip_flag = False
        cache_flag = False

    else:
        c.dataset_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImageFolderDataset',
            path=data_str,
            use_labels=opts.cond,
            xflip=opts.xflip,
            cache=opts.cache
        )
        dataset_path_for_print = data_str
        use_labels_flag = bool(opts.cond)
        xflip_flag = bool(opts.xflip)
        cache_flag = bool(opts.cache)

    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = getattr(dataset_obj, 'resolution', 64)
        if use_labels_flag and not getattr(dataset_obj, 'has_labels', False):
            raise click.ClickException('--cond=True requires labels specified in dataset or random labels generated')
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1, 1], model_channels=128, channel_mult=[2, 2, 2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1, 3, 3, 1], model_channels=128, channel_mult=[2, 2, 2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1, 2, 3, 4])

    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

        if float(opts.sigma_data) > 0:
            c.network_kwargs.sigma_data = float(opts.sigma_data)
        else:
            c.network_kwargs.sigma_data = 1.0 if getattr(opts, 'norm', 'none') == 'standardize' else 0.5

    c.loss_kwargs.report_channels = tuple(int(x) for x in opts.report_channels)

    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    c.grad_clip_norm = float(opts.grad_clip)

    total_steps = int((c.total_kimg * 1000) // max(int(c.batch_size), 1))
    warmup_steps = int((int(opts.lr_warmup_kimg) * 1000) // max(int(c.batch_size), 1))

    c.scheduler_kwargs = dnnlib.EasyDict(
        class_name='training.training_loop.WarmupCosineLR',
        total_steps=max(total_steps, 1),
        warmup_steps=max(warmup_steps, 0),
        eta_min=float(opts.lr_min),
    )

    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    cond_str = 'cond' if use_labels_flag else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path(s):         {dataset_path_for_print}')
    dist.print0(f'Class-conditional:       {use_labels_flag}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0(f'xflip/cache:             {xflip_flag}/{cache_flag}')
    dist.print0(f'Extra loss stats channels: {opts.report_channels}')
    dist.print0(f'Norm:                    {getattr(opts,"norm","none")}')
    if opts.precond == 'edm':
        dist.print0(f'EDM sigma_data:          {c.network_kwargs.get("sigma_data", 0.5)}')
    dist.print0(f'AdamW weight decay:      {float(opts.wd):.3g}')
    dist.print0(f'Grad clip norm:          {float(opts.grad_clip):.3g}')
    dist.print0(f'LR scheduler:            WarmupCosineLR (total_steps={max(total_steps,1)}, warmup_steps={max(warmup_steps,0)}, eta_min={float(opts.lr_min)})')
    dist.print0(f'LR warmup (kimg):        {int(opts.lr_warmup_kimg)}')

    if "eit" in data_str.lower():
        dist.print0(f'EIT train mix ratio:     {getattr(c, "train_dataset_mix_ratio", None)}')
        dist.print0('BEST checkpointing:      based on VALIDATION loss (allch/ch01)')
    dist.print0()

    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    training_loop.training_loop(**c)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()