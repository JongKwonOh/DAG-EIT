# # Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# #
# # This work is licensed under a Creative Commons
# # Attribution-NonCommercial-ShareAlike 4.0 International License.
# # You should have received a copy of the license along with this
# # work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# """Main training loop."""

# import os
# import time
# import copy
# import json
# import pickle
# import psutil
# import numpy as np
# import torch
# import dnnlib
# from torch_utils import distributed as dist
# from torch_utils import training_stats
# from torch_utils import misc

# #----------------------------------------------------------------------------

# def training_loop(
#     run_dir             = '.',      # Output directory.
#     dataset_kwargs      = {},       # Options for training set.
#     data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
#     network_kwargs      = {},       # Options for model and preconditioning.
#     loss_kwargs         = {},       # Options for loss function.
#     optimizer_kwargs    = {},       # Options for optimizer.
#     augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
#     seed                = 0,        # Global random seed.
#     batch_size          = 512,      # Total batch size for one training iteration.
#     batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
#     total_kimg          = 200000,   # Training duration, measured in thousands of training images.
#     ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
#     ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
#     lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
#     loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
#     kimg_per_tick       = 50,       # Interval of progress prints.
#     snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
#     state_dump_ticks    = 500,      # How often to dump training state, None = disable.
#     resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
#     resume_state_dump   = None,     # Start from the given training state, None = reset training state.
#     resume_kimg         = 0,        # Start from the given training progress.
#     cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
#     device              = torch.device('cuda'),
# ):
#     # Initialize.
#     start_time = time.time()
#     np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
#     torch.manual_seed(np.random.randint(1 << 31))
#     torch.backends.cudnn.benchmark = cudnn_benchmark
#     torch.backends.cudnn.allow_tf32 = False
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

#     # Select batch size per GPU.
#     batch_gpu_total = batch_size // dist.get_world_size()
#     if batch_gpu is None or batch_gpu > batch_gpu_total:
#         batch_gpu = batch_gpu_total
#     num_accumulation_rounds = batch_gpu_total // batch_gpu
#     assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

#     # Load dataset.
#     dist.print0('Loading dataset...')
#     dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
#     dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
#     dataset_iterator = iter(torch.utils.data.DataLoader(
#         dataset=dataset_obj,
#         sampler=dataset_sampler,
#         batch_size=batch_gpu,
#         **data_loader_kwargs
#     ))

#     # Construct network.
#     dist.print0('Constructing network...')
#     interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
#     net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
#     net.train().requires_grad_(True).to(device)
#     if dist.get_rank() == 0:
#         with torch.no_grad():
#             images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
#             sigma = torch.ones([batch_gpu], device=device)
#             labels = torch.zeros([batch_gpu, net.label_dim], device=device)
#             misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

#     # Setup optimizer.
#     dist.print0('Setting up optimizer...')
#     loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
#     optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
#     augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
#     ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
#     ema = copy.deepcopy(net).eval().requires_grad_(False)

#     # -------------------------------
#     # [BEST MODEL] helpers / init
#     # -------------------------------
#     best_channels = (0, 1)

#     best_metric_all = float('inf')
#     best_kimg_all = 0.0

#     best_metric_01 = float('inf')
#     best_kimg_01 = 0.0

#     def _state_dict_to_cpu(sd):
#         out = {}
#         for k, v in sd.items():
#             if torch.is_tensor(v):
#                 out[k] = v.detach().cpu()
#             else:
#                 out[k] = v
#         return out

#     def _compute_all_metric(loss_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         loss_tensor expected: (B,C,H,W) or (B,C,...) or scalar-ish
#         returns: scalar tensor on current device
#         """
#         if not torch.is_tensor(loss_tensor):
#             return torch.zeros([], device=device)
#         return loss_tensor.mean()

#     def _compute_ch01_metric(loss_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         loss_tensor expected: (B,C,H,W) or (B,C,...)
#         returns: scalar tensor on current device
#         """
#         if not torch.is_tensor(loss_tensor):
#             return torch.zeros([], device=device)
#         if loss_tensor.ndim < 2:
#             return loss_tensor.mean()

#         C = loss_tensor.shape[1]
#         ch_list = [ch for ch in best_channels if 0 <= int(ch) < C]
#         if len(ch_list) == 0:
#             return loss_tensor.mean()
#         sel = loss_tensor[:, ch_list, ...]
#         return sel.mean()

#     def _save_best(prefix: str, cur_nimg: int, metric_value: float, channels_tag: str):
#         """
#         prefix:
#           - "best_model"      -> best_model.pkl / best_model.pt
#           - "best_model_01"   -> best_model_01.pkl / best_model_01.pt
#         channels_tag: "all" or "01" (메타 저장용)
#         """
#         # pkl: network snapshot style
#         data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
#         for key, value in data.items():
#             if isinstance(value, torch.nn.Module):
#                 value = copy.deepcopy(value).eval().requires_grad_(False)
#                 misc.check_ddp_consistency(value)
#                 data[key] = value.cpu()
#             del value

#         if dist.get_rank() == 0:
#             pkl_path = os.path.join(run_dir, f'{prefix}.pkl')
#             with open(pkl_path, 'wb') as f:
#                 pickle.dump(data, f)

#             pt_path = os.path.join(run_dir, f'{prefix}.pt')
#             torch.save(dict(
#                 net_state=_state_dict_to_cpu(net.state_dict()),
#                 ema_state=_state_dict_to_cpu(ema.state_dict()),
#                 optimizer_state=optimizer.state_dict(),
#                 cur_nimg=int(cur_nimg),
#                 cur_kimg=float(cur_nimg / 1000.0),
#                 best_metric=float(metric_value),
#                 best_type=str(channels_tag),
#                 best_channels=tuple(int(x) for x in best_channels),
#             ), pt_path)

#         del data

#     # Resume training from previous snapshot.
#     if resume_pkl is not None:
#         dist.print0(f'Loading network weights from "{resume_pkl}"...')
#         if dist.get_rank() != 0:
#             torch.distributed.barrier()
#         with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
#             data = pickle.load(f)
#         if dist.get_rank() == 0:
#             torch.distributed.barrier()
#         misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
#         misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
#         del data
#     if resume_state_dump:
#         dist.print0(f'Loading training state from "{resume_state_dump}"...')
#         data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
#         misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
#         optimizer.load_state_dict(data['optimizer_state'])
#         del data

#     # Train.
#     dist.print0(f'Training for {total_kimg} kimg...')
#     dist.print0()
#     cur_nimg = resume_kimg * 1000
#     cur_tick = 0
#     tick_start_nimg = cur_nimg
#     tick_start_time = time.time()
#     maintenance_time = tick_start_time - start_time
#     dist.update_progress(cur_nimg // 1000, total_kimg)
#     stats_jsonl = None

#     while True:
#         optimizer.zero_grad(set_to_none=True)

#         # -------------------------------
#         # per-iter metric accumulator (over grad-accum rounds)
#         # -------------------------------
#         iter_metric_all = torch.zeros([], device=device, dtype=torch.float32)
#         iter_metric_01  = torch.zeros([], device=device, dtype=torch.float32)

#         for round_idx in range(num_accumulation_rounds):
#             with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
#                 images, labels = next(dataset_iterator)
#                 images = images.to(device).to(torch.float32)
#                 labels = labels.to(device)

#                 loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)

#                 # 기존 로깅(전체 loss 텐서)
#                 training_stats.report('Loss/loss', loss)

#                 with torch.no_grad():
#                     ld = loss.detach()
#                     iter_metric_all += _compute_all_metric(ld)
#                     iter_metric_01  += _compute_ch01_metric(ld)

#                 loss.sum().mul(loss_scaling / batch_gpu_total).backward()

#         # Update weights.
#         for g in optimizer.param_groups:
#             g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
#         for param in net.parameters():
#             if param.grad is not None:
#                 torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
#         optimizer.step()

#         # Update EMA.
#         ema_halflife_nimg = ema_halflife_kimg * 1000
#         if ema_rampup_ratio is not None:
#             ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
#         ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
#         for p_ema, p_net in zip(ema.parameters(), net.parameters()):
#             p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

#         # Perform maintenance tasks once per tick.
#         cur_nimg += batch_size
#         done = (cur_nimg >= total_kimg * 1000)

#         with torch.no_grad():
#             iter_metric_all /= float(num_accumulation_rounds)
#             iter_metric_01  /= float(num_accumulation_rounds)

#             torch.distributed.all_reduce(iter_metric_all, op=torch.distributed.ReduceOp.SUM)
#             torch.distributed.all_reduce(iter_metric_01,  op=torch.distributed.ReduceOp.SUM)
#             iter_metric_all /= float(dist.get_world_size())
#             iter_metric_01  /= float(dist.get_world_size())

#             # 로그
#             training_stats.report('Loss/loss_allch_mean', iter_metric_all.detach())
#             training_stats.report('Loss/loss_ch01_mean', iter_metric_01.detach())

#             m_all = float(iter_metric_all.cpu().item())
#             m_01  = float(iter_metric_01.cpu().item())

#             # best (ALL channels) -> best_model.*
#             if (dist.get_rank() == 0) and (m_all < best_metric_all):
#                 best_metric_all = m_all
#                 best_kimg_all = cur_nimg / 1000.0
#                 dist.print0(f'[BEST-ALL] updated: Loss/loss_allch_mean={best_metric_all:.6f} at kimg={best_kimg_all:.1f}')
#                 _save_best(prefix="best_model", cur_nimg=cur_nimg, metric_value=best_metric_all, channels_tag="all")

#             # best (0,1 channels) -> best_model_01.*
#             if (dist.get_rank() == 0) and (m_01 < best_metric_01):
#                 best_metric_01 = m_01
#                 best_kimg_01 = cur_nimg / 1000.0
#                 dist.print0(f'[BEST-01] updated: Loss/loss_ch01_mean={best_metric_01:.6f} at kimg={best_kimg_01:.1f}')
#                 _save_best(prefix="best_model_01", cur_nimg=cur_nimg, metric_value=best_metric_01, channels_tag="01")

#         if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
#             continue

#         # Print status line.
#         tick_end_time = time.time()
#         fields = []
#         fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
#         fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
#         fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
#         fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
#         fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
#         fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
#         fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
#         fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
#         fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
#         torch.cuda.reset_peak_memory_stats()
#         dist.print0(' '.join(fields))

#         # Check for abort.
#         if (not done) and dist.should_stop():
#             done = True
#             dist.print0()
#             dist.print0('Aborting...')

#         # Save network snapshot.
#         if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
#             data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
#             for key, value in data.items():
#                 if isinstance(value, torch.nn.Module):
#                     value = copy.deepcopy(value).eval().requires_grad_(False)
#                     misc.check_ddp_consistency(value)
#                     data[key] = value.cpu()
#                 del value
#             if dist.get_rank() == 0:
#                 with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
#                     pickle.dump(data, f)
#             del data

#         # Save full dump of the training state.
#         if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
#             torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

#         # Update logs.
#         training_stats.default_collector.update()
#         if dist.get_rank() == 0:
#             if stats_jsonl is None:
#                 stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
#             stats_jsonl.write(json.dumps(dict(
#                 training_stats.default_collector.as_dict(),
#                 timestamp=time.time(),
#                 best_metric_all=float(best_metric_all),
#                 best_kimg_all=float(best_kimg_all),
#                 best_metric_01=float(best_metric_01),
#                 best_kimg_01=float(best_kimg_01),
#             )) + '\n')
#             stats_jsonl.flush()
#         dist.update_progress(cur_nimg // 1000, total_kimg)

#         # Update state.
#         cur_tick += 1
#         tick_start_nimg = cur_nimg
#         tick_start_time = time.time()
#         maintenance_time = tick_start_time - tick_end_time
#         if done:
#             break

#     # Done.
#     dist.print0()
#     dist.print0('Exiting...')

# #----------------------------------------------------------------------------

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import time
import copy
import json
import pickle
import math
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

#----------------------------------------------------------------------------
# 1)-A: One scheduler for warmup + cosine (no manual LR writes in training loop)

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup + Cosine decay scheduler.

    - During warmup (step < warmup_steps): lr increases linearly from 0 -> base_lr
    - After warmup: lr decays with cosine from base_lr -> eta_min
    """
    def __init__(self, optimizer, total_steps, warmup_steps=0, eta_min=0.0, last_epoch=-1):
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if self.total_steps <= 1:
            return [self.eta_min for _ in self.base_lrs]

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            scale = step / float(self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        denom = max(self.total_steps - self.warmup_steps, 1)
        t = (step - self.warmup_steps) / float(denom)
        t = min(max(t, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))

        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]

#----------------------------------------------------------------------------

def _normalize_mix_ratio(num_datasets: int, ratio):
    if num_datasets <= 0:
        raise ValueError("num_datasets must be >= 1")

    if ratio is None:
        return [1] * num_datasets

    ratio = [int(x) for x in ratio]

    if len(ratio) == 0:
        return [1] * num_datasets

    if len(ratio) != num_datasets:
        raise ValueError(
            f"train_dataset_mix_ratio length mismatch: got {len(ratio)}, "
            f"but number of train datasets is {num_datasets}"
        )

    if sum(ratio) <= 0:
        raise ValueError("train_dataset_mix_ratio must have positive sum")

    if any(x < 0 for x in ratio):
        raise ValueError("train_dataset_mix_ratio must be non-negative")

    return ratio

def _compute_mix_counts(batch_n: int, ratio):
    """Return per-source counts summing to batch_n, roughly matching ratio."""
    ratio = [int(x) for x in ratio]
    assert batch_n >= 1

    if len(ratio) == 0:
        return [batch_n]

    if sum(ratio) <= 0:
        raise ValueError("mix ratio must have positive sum")

    total = sum(ratio)
    base = batch_n // total
    counts = [r * base for r in ratio]
    rem = batch_n - sum(counts)

    order = sorted(range(len(ratio)), key=lambda i: (-ratio[i], i))
    for k in range(rem):
        counts[order[k % len(order)]] += 1

    assert sum(counts) == batch_n
    return counts

class _MixedBatchIterator:
    """Pull fixed-size mini-batches from multiple iterators and concat along batch dim."""
    def __init__(self, iters, counts):
        self.iters = iters
        self.counts = counts

    def __iter__(self):
        return self

    def __next__(self):
        xs = []
        ys = []
        for it, n in zip(self.iters, self.counts):
            if n <= 0:
                continue
            x, y = next(it)
            xs.append(x)
            ys.append(y)
        x = torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
        y = torch.cat(ys, dim=0) if len(ys) > 1 else ys[0]
        return x, y

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',
    dataset_kwargs      = {},
    data_loader_kwargs  = {},
    network_kwargs      = {},
    loss_kwargs         = {},
    optimizer_kwargs    = {},
    scheduler_kwargs    = None,
    augment_kwargs      = None,
    seed                = 0,
    batch_size          = 512,
    batch_gpu           = None,
    total_kimg          = 200000,
    ema_halflife_kimg   = 500,
    ema_rampup_ratio    = 0.05,
    lr_rampup_kimg      = 10000,
    loss_scaling        = 1,
    kimg_per_tick       = 50,
    snapshot_ticks      = 50,
    state_dump_ticks    = 500,
    resume_pkl          = None,
    resume_state_dump   = None,
    resume_kimg         = 0,
    cudnn_benchmark     = True,
    device              = torch.device('cuda'),

    train_dataset_kwargs_list = None,
    train_dataset_mix_ratio   = None,
    valid_dataset_kwargs      = None,

    grad_clip_norm      = 0.0,
):
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    dist.print0('Loading dataset...')

    if train_dataset_kwargs_list is not None and len(train_dataset_kwargs_list) > 0:
        dataset_obj_for_interface = dnnlib.util.construct_class_by_name(**train_dataset_kwargs_list[0])
    else:
        dataset_obj_for_interface = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    if train_dataset_kwargs_list is None:
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs
        )
        dataset_iterator = iter(train_loader)
    else:
        if len(train_dataset_kwargs_list) == 0:
            raise ValueError("train_dataset_kwargs_list is empty")

        train_datasets = [dnnlib.util.construct_class_by_name(**kw) for kw in train_dataset_kwargs_list]
        mix_ratio = _normalize_mix_ratio(len(train_datasets), train_dataset_mix_ratio)
        counts = _compute_mix_counts(batch_gpu, mix_ratio)

        iters = []
        used_counts = []
        for ds, n in zip(train_datasets, counts):
            if n <= 0:
                continue
            smp = misc.InfiniteSampler(dataset=ds, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
            loader = torch.utils.data.DataLoader(
                dataset=ds,
                sampler=smp,
                batch_size=n,
                **data_loader_kwargs
            )
            iters.append(iter(loader))
            used_counts.append(n)

        if len(iters) == 0:
            raise ValueError(
                f"All per-dataset batch counts became zero. "
                f"batch_gpu={batch_gpu}, mix_ratio={mix_ratio}"
            )

        assert sum(used_counts) == batch_gpu, f"mix counts mismatch: {used_counts} vs batch_gpu={batch_gpu}"
        dist.print0(f"Train dataset count:      {len(train_datasets)}")
        dist.print0(f"Train dataset mix ratio:  {tuple(mix_ratio)}")
        dist.print0(f"Per-GPU mixed batch cnts: {tuple(used_counts)}")
        dataset_iterator = _MixedBatchIterator(iters, used_counts)

    valid_loader = None
    valid_sampler = None
    if valid_dataset_kwargs is not None:
        dist.print0('Loading validation dataset...')
        valid_dataset = dnnlib.util.construct_class_by_name(**valid_dataset_kwargs)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            sampler=valid_sampler,
            batch_size=batch_gpu,
            drop_last=False,
            **data_loader_kwargs
        )

    dist.print0('Constructing network...')
    interface_kwargs = dict(
        img_resolution=dataset_obj_for_interface.resolution,
        img_channels=dataset_obj_for_interface.num_channels,
        label_dim=dataset_obj_for_interface.label_dim
    )
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)

    scheduler = None
    if scheduler_kwargs is not None:
        scheduler = dnnlib.util.construct_class_by_name(optimizer=optimizer, **scheduler_kwargs)

    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    best_channels = (0, 1)
    best_metric_all = float('inf')
    best_kimg_all = 0.0
    best_metric_01 = float('inf')
    best_kimg_01 = 0.0

    def _state_dict_to_cpu(sd):
        out = {}
        for k, v in sd.items():
            out[k] = v.detach().cpu() if torch.is_tensor(v) else v
        return out

    def _save_best(prefix: str, cur_nimg: int, metric_value: float, channels_tag: str):
        data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
        for key, value in data.items():
            if isinstance(value, torch.nn.Module):
                value = copy.deepcopy(value).eval().requires_grad_(False)
                misc.check_ddp_consistency(value)
                data[key] = value.cpu()
            del value

        if dist.get_rank() == 0:
            pkl_path = os.path.join(run_dir, f'{prefix}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)

            pt_path = os.path.join(run_dir, f'{prefix}.pt')
            torch.save(dict(
                net_state=_state_dict_to_cpu(net.state_dict()),
                ema_state=_state_dict_to_cpu(ema.state_dict()),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=(scheduler.state_dict() if scheduler is not None else None),
                cur_nimg=int(cur_nimg),
                cur_kimg=float(cur_nimg / 1000.0),
                best_metric=float(metric_value),
                best_type=str(channels_tag),
                best_channels=tuple(int(x) for x in best_channels),
            ), pt_path)

        del data

    def _eval_validation(cur_tick: int):
        if valid_loader is None:
            return None, None

        if valid_sampler is not None:
            valid_sampler.set_epoch(cur_tick)

        ema.eval()
        sum_all = torch.zeros([], device=device, dtype=torch.float64)
        den_all = torch.zeros([], device=device, dtype=torch.float64)
        sum_01  = torch.zeros([], device=device, dtype=torch.float64)
        den_01  = torch.zeros([], device=device, dtype=torch.float64)

        with torch.no_grad():
            for v_images, v_labels in valid_loader:
                v_images = v_images.to(device).to(torch.float32)
                v_labels = v_labels.to(device)

                v_loss = loss_fn(net=ema, images=v_images, labels=v_labels, augment_pipe=None)

                sum_all += v_loss.detach().sum().to(torch.float64)
                den_all += float(v_loss.numel())

                if torch.is_tensor(v_loss) and v_loss.ndim >= 2:
                    C = v_loss.shape[1]
                    ch_list = [ch for ch in best_channels if 0 <= int(ch) < C]
                    sel = v_loss[:, ch_list, ...] if len(ch_list) > 0 else v_loss
                else:
                    sel = v_loss

                sum_01 += sel.detach().sum().to(torch.float64)
                den_01 += float(sel.numel())

        torch.distributed.all_reduce(sum_all, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(den_all, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sum_01,  op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(den_01,  op=torch.distributed.ReduceOp.SUM)

        val_mean_all = (sum_all / torch.clamp(den_all, min=1.0)).to(torch.float32)
        val_mean_01  = (sum_01  / torch.clamp(den_01,  min=1.0)).to(torch.float32)

        training_stats.report('Loss/val_loss_allch_mean', val_mean_all)
        training_stats.report('Loss/val_loss_ch01_mean',  val_mean_01)

        return float(val_mean_all.detach().cpu().item()), float(val_mean_01.detach().cpu().item())

    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data

    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        if scheduler is not None and ('scheduler_state' in data) and (data['scheduler_state'] is not None):
            scheduler.load_state_dict(data['scheduler_state'])
        del data

    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    last_val_all = None
    last_val_01 = None

    while True:
        optimizer.zero_grad(set_to_none=True)

        iter_metric_all = torch.zeros([], device=device, dtype=torch.float32)
        iter_metric_01  = torch.zeros([], device=device, dtype=torch.float32)

        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32)
                labels = labels.to(device)

                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)

                with torch.no_grad():
                    ld = loss.detach()
                    iter_metric_all += ld.mean()
                    if torch.is_tensor(ld) and ld.ndim >= 2:
                        C = ld.shape[1]
                        ch_list = [ch for ch in best_channels if 0 <= int(ch) < C]
                        iter_metric_01 += ld[:, ch_list, ...].mean() if len(ch_list) > 0 else ld.mean()
                    else:
                        iter_metric_01 += ld.mean()

                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        if grad_clip_norm is not None and float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip_norm))

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)

        with torch.no_grad():
            iter_metric_all /= float(num_accumulation_rounds)
            iter_metric_01  /= float(num_accumulation_rounds)

            torch.distributed.all_reduce(iter_metric_all, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(iter_metric_01,  op=torch.distributed.ReduceOp.SUM)
            iter_metric_all /= float(dist.get_world_size())
            iter_metric_01  /= float(dist.get_world_size())

            training_stats.report('Loss/loss_allch_mean', iter_metric_all.detach())
            training_stats.report('Loss/loss_ch01_mean',  iter_metric_01.detach())

        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        v_all, v_01 = _eval_validation(cur_tick=cur_tick)
        last_val_all, last_val_01 = v_all, v_01

        if v_all is not None and dist.get_rank() == 0:
            if v_all < best_metric_all:
                best_metric_all = float(v_all)
                best_kimg_all = cur_nimg / 1000.0
                dist.print0(f'[BEST-ALL][VAL] updated: Loss/val_loss_allch_mean={best_metric_all:.6f} at kimg={best_kimg_all:.1f}')
                _save_best(prefix="best_model", cur_nimg=cur_nimg, metric_value=best_metric_all, channels_tag="all")

            if v_01 < best_metric_01:
                best_metric_01 = float(v_01)
                best_kimg_01 = cur_nimg / 1000.0
                dist.print0(f'[BEST-01][VAL] updated: Loss/val_loss_ch01_mean={best_metric_01:.6f} at kimg={best_kimg_01:.1f}')
                _save_best(prefix="best_model_01", cur_nimg=cur_nimg, metric_value=best_metric_01, channels_tag="01")

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data

        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(
                dict(
                    net=net,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=(scheduler.state_dict() if scheduler is not None else None),
                ),
                os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt')
            )

        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(
                training_stats.default_collector.as_dict(),
                timestamp=time.time(),
                best_metric_all=float(best_metric_all),
                best_kimg_all=float(best_kimg_all),
                best_metric_01=float(best_metric_01),
                best_kimg_01=float(best_kimg_01),
                val_loss_allch_mean=(None if last_val_all is None else float(last_val_all)),
                val_loss_ch01_mean=(None if last_val_01 is None else float(last_val_01)),
            )) + '\n')
            stats_jsonl.flush()

        dist.update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    dist.print0()
    dist.print0('Exiting...')