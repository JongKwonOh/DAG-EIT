import os, io, json, time, pickle, tqdm
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io
from datetime import datetime
import math
import random

import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Only check
x15_idx_lst = [10, 27, 33, 47, 58, 69, 72]
x2_idx_lst = [0, 12, 20, 22, 37, 57, 76]

ellipse_idx_lst = [8, 23, 29, 46, 50, 64, 73]

ELLIPSE_MAJOR_AXIS = 37.4439
ELLIPSE_MINOR_AXIS = 29.9551
ELLIPSE_SEMI_MAJOR_AXIS = ELLIPSE_MAJOR_AXIS / 2.0
ELLIPSE_SEMI_MINOR_AXIS = ELLIPSE_MINOR_AXIS / 2.0

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.8+
    torch.use_deterministic_algorithms(True, warn_only=True)


def _today_yyyymmdd():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    except Exception:
        return time.strftime("%Y%m%d", time.localtime())

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _save_config_snapshot(config_dict, folder):
    """
    Save config to config.yaml (prefer) else config.json in the given folder.
    """
    yaml_ok = False
    try:
        import yaml
        with open(os.path.join(folder, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True, sort_keys=False)
        yaml_ok = True
    except Exception:
        yaml_ok = False

    if not yaml_ok:
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    """randomly select k indices from a [grid_size, grid_size] grid."""
    np.random.seed(seed)
    indices = np.random.choice(grid_size**2, k, replace=False)
    ii, jj = np.unravel_index(indices, (grid_size, grid_size))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32, device=device)
    for y, x in zip(ii, jj):
        mask[y, x] = 1.0
    return mask

def get_eit_mask(grid_size, device=torch.device('cuda')):
    H, W = grid_size
    assert H == W == 64

    row_idx = torch.arange(0, 63, 4)  # 0,4,...,60
    col_idx = torch.arange(0, 63, 4)  # 0,4,...,60

    mask = torch.zeros((H, W), dtype=torch.uint8)

    for r in row_idx.tolist():
        excl = {(r - 4) % W, r % W, (r + 4) % W}  # (r-4, r, r+4)
        selected_cols = [int(c) for c in col_idx.tolist() if c not in excl]
        mask[r, selected_cols] = 1

    return mask.to(device)

def _ema_update(prev, x, beta: float):
    """Exponential moving average update. prev can be None. Works with floats."""
    if prev is None:
        return float(x)
    return float(beta) * float(prev) + (1.0 - float(beta)) * float(x)

def _plot_and_save_auto_scale(out_dir, y_values, title, filename):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y_values, dtype=np.float64)
    plt.figure()
    plt.plot(y)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)

    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()

def _momentum_toward(current, target, velocity, mu: float, lr: float):
    mu = float(mu)
    lr = float(lr)
    current = float(current)
    target = float(target)
    velocity = float(velocity)
    # print(velocity)
    velocity = mu * velocity + (1.0 - mu) * (target - current)
    current = current + lr * velocity
    return current, velocity

def _adaptive_zeta_update(
    *,
    zeta_init: float,
    use_EMA: bool,
    g_norm: float,
    gnorm_ema,
    gnorm_ema_beta: float,
    zeta_ema,
    zeta_ema_beta: float,
    step_target,
    zeta_min: float,
    zeta_max: float,
    use_zeta_momentum: bool,
    zeta_momentum_mu: float,
    zeta_momentum_lr: float,
    zeta_vel: float,
    sync_zeta_ema_with_momentum: bool,
):
    if not use_EMA:
        zeta_used = float(zeta_init)
        zeta_used = float(np.clip(zeta_used, zeta_min, zeta_max))
        return zeta_used, gnorm_ema, zeta_ema, zeta_vel, step_target

    # EMA on grad norm
    gnorm_ema = _ema_update(gnorm_ema, float(g_norm), float(gnorm_ema_beta))

    # step_target auto set (if None)
    if step_target is None:
        step_target = float(zeta_init) * float(gnorm_ema)
    # print("gnorm_ema : ", gnorm_ema)
    zeta_raw = float(step_target) / (float(gnorm_ema) + 1e-12)
    zeta_raw = float(np.clip(zeta_raw, zeta_min, zeta_max))

    # print(gnorm_ema)

    # EMA on zeta
    zeta_ema = _ema_update(zeta_ema, zeta_raw, float(zeta_ema_beta))
    zeta_ema = float(np.clip(float(zeta_ema), zeta_min, zeta_max))

    # momentum (optional)
    if use_zeta_momentum:
        zeta_mom, zeta_vel = _momentum_toward(
            current=float(zeta_ema),
            target=float(zeta_raw),
            velocity=float(zeta_vel),
            mu=float(zeta_momentum_mu),
            lr=float(zeta_momentum_lr),
        )
        zeta_mom = float(np.clip(zeta_mom, zeta_min, zeta_max))
        zeta_used = zeta_mom

        if sync_zeta_ema_with_momentum:
            zeta_ema = zeta_used
    else:
        zeta_used = float(zeta_ema)

    return float(zeta_used), gnorm_ema, zeta_ema, float(zeta_vel), step_target


# =========================
# Loss functions
# =========================
def polar_data_to_elliptical_xy_data(
    tensor,
    r_max=1.0,
    a=18.72195,    # semi-major axis = 37.4439 / 2
    b=14.97755,    # semi-minor axis = 29.9551 / 2
    mode="nearest",
    align_corners=True,
):
    assert tensor.ndim == 4, "input must be (B, C, H, W)"
    assert a > 0 and b > 0, "a and b must be positive"

    B, C, Hr, Wt = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    tensor = torch.cat([tensor, tensor[..., :1]], dim=-1)  # (B,C,H,W+1)

    # 출력 크기는 기존과 동일: (Hr, Wt)
    xs = torch.linspace(-a, a, Hr, device=device, dtype=dtype)  # X axis
    ys = torch.linspace(-b, b, Wt, device=device, dtype=dtype)  # Y axis
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")              # (H,W)

    # ellipse-normalized coordinates
    xx_n = xx / float(a)
    yy_n = yy / float(b)

    rr = torch.sqrt(xx_n * xx_n + yy_n * yy_n)
    theta = torch.atan2(yy_n, xx_n)
    theta = torch.remainder(theta, 2.0 * math.pi)

    u = (theta / (2.0 * math.pi)) * 2.0 - 1.0
    v = (rr / float(r_max)) * 2.0 - 1.0

    grid = torch.stack([u, v], dim=-1)          # (H,W,2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # (B,H,W,2)

    # ellipse mask
    mask = (rr <= float(r_max)).to(dtype)       # (H,W)
    mask = mask.unsqueeze(0).unsqueeze(0)       # (1,1,H,W)

    out = F.grid_sample(
        tensor,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )

    out = out * mask
    return out

def polar_data_to_xy_data(tensor, r_max=1.0, mode="nearest", align_corners=True):
    assert tensor.ndim == 4, "input must be (B, C, H, W)"

    B, C, Hr, Wt = tensor.shape
    device = tensor.device
    dtype = tensor.dtype
    
    tensor = torch.cat([tensor, tensor[..., :1]], dim=-1)  # (B,C,H,W+1)
    Wt_pad = Wt + 1

    xs = torch.linspace(-1, 1, Hr, device=device, dtype=dtype)  # X axis
    ys = torch.linspace(-1, 1, Wt, device=device, dtype=dtype)  # Y axis
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")                 # (H,W)

    rr = torch.sqrt(xx * xx + yy * yy)
    theta = torch.atan2(yy, xx)
    theta = torch.remainder(theta, 2.0 * math.pi)

    u = (theta / (2.0 * math.pi)) * 2.0 - 1.0      # theta ∈ [-1,1]
    v = (rr / float(r_max)) * 2.0 - 1.0            # r     ∈ [-1,1]
    
    grid = torch.stack([u, v], dim=-1)             # (H,W,2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)    # (B,H,W,2)

    # disk mask (outside r_max → 0)
    mask = (rr <= float(r_max)).to(dtype)          # (H,W)
    mask = mask.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)

    out = F.grid_sample(
        tensor,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )                                             

    out = out * mask
    return out

def get_pde_loss2(a, grad_u, device='cpu', offset=0):
    """
    grad_u = (data_Z, grad_u_x_N, grad_u_y_N)

    - circular data: polar -> xy
    - elliptical data: elliptical coord -> xy
    """
    gamma = a
    grad_u_x = grad_u[-2]
    grad_u_y = grad_u[-1]

    B, _, H, W = gamma.shape

    # ellipse case
    if offset in ellipse_idx_lst:
        R_x, R_y = 37.4439/2, 29.9551/2
        gamma = polar_data_to_elliptical_xy_data(
            gamma, a=R_x, b=R_y
        )
        grad_u_x = polar_data_to_elliptical_xy_data(
            grad_u_x, a=R_x, b=R_y
        )
        grad_u_y = polar_data_to_elliptical_xy_data(
            grad_u_y, a=R_x, b=R_y
        )
        
    else : 
        # circular case
        R_x, R_y = 1.0, 1.0
        if offset in x15_idx_lst:
            R_x, R_y = 1.5, 1.5
        elif offset in x2_idx_lst:
            R_x, R_y = 2, 2

        gamma = polar_data_to_xy_data(gamma, r_max=R_x)
        grad_u_x = polar_data_to_xy_data(grad_u_x, r_max=R_x)
        grad_u_y = polar_data_to_xy_data(grad_u_y, r_max=R_x)

    grad_u_xy = torch.stack([grad_u_x, grad_u_y], dim=-1)
    grem = gamma.unsqueeze(-1) * grad_u_xy

    gx = grem[..., 0]
    gy = grem[..., 1]

    div = torch.zeros_like(gx)
    
    dx = (2.0 * R_x) / (H - 1)
    dy = (2.0 * R_y) / (W - 1)

    div[..., :-1, :] += (gx[..., 1:, :] - gx[..., :-1, :]) * 2 / H
    div[..., :, :-1] += (gy[..., :, 1:] - gy[..., :, :-1]) * 2 / W
    neg_div = -div

    return neg_div

def get_eit_loss(dtn, a, dtn_GT, a_GT, grad_u, dtn_mask, is_PDE_loss, device=torch.device('cuda'), offset=0):
    """Return the PDE loss tensor and observation residual for dtn."""
    if is_PDE_loss:
        pde_loss = get_pde_loss2(a, grad_u, device=device, offset=offset)
    else :
        pde_loss = torch.zeros_like(a).to(device)

    observation_loss_dtn = (dtn - dtn_GT).squeeze()
    observation_loss_dtn = observation_loss_dtn * dtn_mask

    return pde_loss, observation_loss_dtn

def compute_grad_pde_eff(
    grad_obs,
    grad_pde,
    use_pde_ortho=True,
    ortho_eps=1e-12,
    preserve_pde_norm=True,
):
    """
    Compute the effective PDE guidance gradient using a fixed alpha = 1.0.

    No alpha scheduling is used. When the observation and PDE gradients
    conflict, the correction strength is always fixed to alpha = 1.
    """
    if not use_pde_ortho:
        return grad_pde

    # Fixed orthogonalization coefficient.
    alpha = 1.0

    dot = torch.sum(
        grad_obs.detach() * grad_pde.detach()
    ).item()

    if dot >= 0.0:
        return grad_pde

    pde_norm2 = torch.sum(
        grad_pde.detach() * grad_pde.detach()
    ).item()

    if pde_norm2 <= ortho_eps:
        return grad_pde

    coeff = dot / (pde_norm2 + ortho_eps)

    grad_pde_eff_raw = grad_obs - alpha * coeff * grad_pde

    # Preserve the norm of the original PDE gradient.
    if preserve_pde_norm:
        pde_norm = torch.sqrt(
            torch.sum(
                grad_pde.detach() * grad_pde.detach()
            ) + ortho_eps
        )
        raw_norm = torch.sqrt(
            torch.sum(
                grad_pde_eff_raw.detach()
                * grad_pde_eff_raw.detach()
            ) + ortho_eps
        )
        grad_pde_eff = (
            grad_pde_eff_raw / raw_norm * pde_norm
        )
    else:
        grad_pde_eff = grad_pde_eff_raw

    return grad_pde_eff

# =========================
# Core generation for ONE sample (offset)
# =========================
def _generate_eit_one_sample(net, data, offset, config, known_index_dtn, device):
    """
    Run EDM sampling for a single offset, returns:
      dtn_pred (1,1,64,64), a_pred (1,1,64,64),
      grad_u_x_pred (1,16,64,64), grad_u_y_pred (1,16,64,64),
      rel_err_dtn (float), rel_err_a (float),
      losses_pde (list[float]), losses_obs_dtn (list[float]),
      losses_obs_a (list[float]),
      zetas_obs (list[float]), zetas_pde (list[float])
    """
    batch_size = config['generate']['batch_size']
    assert batch_size == 1, "whole_dataset 모드에서는 batch_size=1을 권장/가정합니다."

    # --- Load GT for this offset ---
    data_Z = torch.tensor(data['dataZ'][offset:offset+1, :, :], dtype=torch.float64, device=device).unsqueeze(1)
    dtn_GT = torch.tensor(data['Vbd'][offset:offset+1, :, :], dtype=torch.float64, device=device).unsqueeze(1)
    a_GT   = torch.tensor(data['sigma'][offset:offset+1, :, :], dtype=torch.float64, device=device).unsqueeze(1)

    _ = torch.tensor(data['Ex'][offset:offset+1, :, :, :], dtype=torch.float64, device=device)
    _ = torch.tensor(data['Ey'][offset:offset+1, :, :, :], dtype=torch.float64, device=device)

    # # Normalization
    # data_Z = (data_Z - data_Z.mean(dim=(0,2,3)))/(data_Z.std(dim=(0,2,3))+1e-12)
    # dtn_GT = (dtn_GT - dtn_GT.mean(dim=(0,2,3)))/(dtn_GT.std(dim=(0,2,3))+1e-12)
    # a_GT = (a_GT - a_GT.mean(dim=(0,2,3)))/(a_GT.std(dim=(0,2,3))+1e-12)

    # --- EDM settings ---
    sigma_min = max(config['generate']['sigma_min'], net.sigma_min)
    sigma_max = min(config['generate']['sigma_max'], net.sigma_max)
    num_steps = int(config['test']['iterations'])
    rho       = config['generate']['rho']

    # --- Latents ---
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if getattr(net, "label_dim", 0):
        class_labels = torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, size=[batch_size], device=device)
        ]

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])  # t_N = 0

    x_next = latents.to(torch.float64) * sigma_t_steps[0]

    is_PDE_loss = bool(config['generate'].get('is_PDE_loss', False))

    use_EMA = bool(config.get('generate', {}).get('use_EMA', False))

    # --- NEW: separate zeta init ---
    zeta_obs_init = float(config['generate'].get('zeta_obs', 1))
    zeta_pde_init = float(config['generate'].get('zeta_pde', 1))

    # --- NEW: separate EMA params ---
    gnorm_ema_beta_obs = float(config['generate'].get('gnorm_ema_beta_obs', 0.9))
    zeta_ema_beta_obs  = float(config['generate'].get('zeta_ema_beta_obs', 0.9))

    gnorm_ema_beta_pde = float(config['generate'].get('gnorm_ema_beta_pde', 0.9))
    zeta_ema_beta_pde  = float(config['generate'].get('zeta_ema_beta_pde', 0.9))

    # --- NEW: zeta momentum 옵션 (obs/pde 각각) ---
    use_zeta_momentum_obs = bool(config['generate'].get('use_zeta_momentum_obs', True))
    zeta_momentum_mu_obs  = float(config['generate'].get('zeta_momentum_mu_obs', 0.9))
    zeta_momentum_lr_obs  = float(config['generate'].get('zeta_momentum_lr_obs', 1.0))
    sync_zeta_ema_with_momentum_obs = bool(config['generate'].get('sync_zeta_ema_with_momentum_obs', True))

    use_zeta_momentum_pde = bool(config['generate'].get('use_zeta_momentum_pde', True))
    zeta_momentum_mu_pde  = float(config['generate'].get('zeta_momentum_mu_pde', 0.9))
    zeta_momentum_lr_pde  = float(config['generate'].get('zeta_momentum_lr_pde',1.0))
    sync_zeta_ema_with_momentum_pde = bool(config['generate'].get('sync_zeta_ema_with_momentum_pde', True))

    # velocity state for zeta momentum (per-sample)
    zeta_vel_obs = 0.0
    zeta_vel_pde = 0.0

    zeta_obs_min = float(config['generate'].get('zeta_obs_min', 0.0))
    zeta_obs_max = float(config['generate'].get('zeta_obs_max', 100.0))
    zeta_pde_min = float(config['generate'].get('zeta_pde_min', 0.0))
    zeta_pde_max = float(config['generate'].get('zeta_pde_max', 10.0))

    step_target_total = config['generate'].get('step_target_total', None)
    step_target_total = None if step_target_total is None else float(step_target_total)

    step_target_obs = config['generate'].get('step_target_obs', step_target_total)
    step_target_obs = None if step_target_obs is None else float(step_target_obs)

    step_target_pde = config['generate'].get('step_target_pde', step_target_total)
    step_target_pde = None if step_target_pde is None else float(step_target_pde)

    # EMA states
    gnorm_ema_obs = None
    zeta_ema_obs  = None

    gnorm_ema_pde = None
    zeta_ema_pde  = None

    # --- angle/ortho eps ---
    ortho_eps = float(config['generate'].get('ortho_eps', 1e-12))

    use_pde_ortho = bool(config['generate'].get('use_pde_ortho', True))

    # --- Trackers ---
    losses_pde = []
    losses_obs_dtn = []
    losses_obs_a = []
    zetas_obs = []
    zetas_pde = []

    pde_eff_norms = []

    # --- Sampling loop ---
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(
        list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))),
        unit='step',
        leave=False,
        desc=f"offset={offset}"
    ):
        x_cur = x_next.detach().clone().requires_grad_(True)
        sigma_t = net.round_sigma(sigma_t_cur)

        # Euler
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

        # Heun
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

        # channels unpack
        dtn_N      = x_N[:, 0, :, :].unsqueeze(1)         # (B,1,64,64)
        a_N        = x_N[:, 1, :, :].unsqueeze(1)         # (B,1,64,64)
        grad_u_x_N = x_N[:, 2:18, :, :]                   # (B,16,64,64)
        grad_u_y_N = x_N[:, 18:, :, :]                    # (B,16,64,64)

        # pde + dtn observation residual
        pde_loss, observation_loss_dtn = get_eit_loss(
            dtn_N, a_N, dtn_GT, a_GT, (data_Z, grad_u_x_N, grad_u_y_N), known_index_dtn, is_PDE_loss=is_PDE_loss, device=device, offset=offset
        )

        L_pde     = torch.norm(pde_loss, 2)#**2
        L_obs_dtn = torch.norm(observation_loss_dtn, 2)#**2

        losses_pde.append(float(L_pde.detach().item()))
        losses_obs_dtn.append(float(L_obs_dtn.detach().item()))

        observation_loss_a = (a_N - a_GT).squeeze()
        L_obs_a = torch.norm(observation_loss_a, 2)
        losses_obs_a.append(float(L_obs_a.detach().item()))

        # =========================
        # grads (SEPARATE)
        # =========================
        grad_obs = torch.autograd.grad(outputs=L_obs_dtn, inputs=x_cur, retain_graph=True)[0]  # (B,34,H,W)
        # print(L_obs_dtn.item(), torch.norm(grad_obs, 2).item())
        if is_PDE_loss:
            grad_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur)[0]  # (B,34,H,W)
        else:
            grad_pde = torch.zeros_like(x_cur)

        if (not is_PDE_loss) or (not use_pde_ortho):
            grad_pde_eff = grad_pde
        else:
            grad_pde_eff = compute_grad_pde_eff(
                grad_obs,
                grad_pde,
                use_pde_ortho=use_pde_ortho,
                ortho_eps=ortho_eps,
            )

        pde_eff_norms.append(float(torch.norm(grad_pde_eff.detach(), p=2).item()))

        if not use_EMA:
            zeta_obs_used = float(zeta_obs_init)
            zeta_pde_used = float(zeta_pde_init)

            zeta_obs_used = float(np.clip(zeta_obs_used, zeta_obs_min, zeta_obs_max))
            zeta_pde_used = float(np.clip(zeta_pde_used, zeta_pde_min, zeta_pde_max))

        else:
            g_norm_obs = float(torch.norm(grad_obs.detach(), p=2).item())
            zeta_obs_used, gnorm_ema_obs, zeta_ema_obs, zeta_vel_obs, step_target_obs = _adaptive_zeta_update(
                zeta_init=zeta_obs_init,
                use_EMA=True,
                g_norm=g_norm_obs,
                gnorm_ema=gnorm_ema_obs,
                gnorm_ema_beta=gnorm_ema_beta_obs,
                zeta_ema=zeta_ema_obs,
                zeta_ema_beta=zeta_ema_beta_obs,
                step_target=step_target_obs,
                zeta_min=zeta_obs_min,
                zeta_max=zeta_obs_max,
                use_zeta_momentum=use_zeta_momentum_obs,
                zeta_momentum_mu=zeta_momentum_mu_obs,
                zeta_momentum_lr=zeta_momentum_lr_obs,
                zeta_vel=zeta_vel_obs,
                sync_zeta_ema_with_momentum=sync_zeta_ema_with_momentum_obs,
            )

            g_norm_pde = float(torch.norm(grad_pde_eff.detach(), p=2).item())
            zeta_pde_used, gnorm_ema_pde, zeta_ema_pde, zeta_vel_pde, step_target_pde = _adaptive_zeta_update(
                zeta_init=zeta_pde_init,
                use_EMA=True,
                g_norm=g_norm_pde,
                gnorm_ema=gnorm_ema_pde,
                gnorm_ema_beta=gnorm_ema_beta_pde,
                zeta_ema=zeta_ema_pde,
                zeta_ema_beta=zeta_ema_beta_pde,
                step_target=step_target_pde,
                zeta_min=zeta_pde_min,
                zeta_max=zeta_pde_max,
                use_zeta_momentum=use_zeta_momentum_pde,
                zeta_momentum_mu=zeta_momentum_mu_pde,
                zeta_momentum_lr=zeta_momentum_lr_pde,
                zeta_vel=zeta_vel_pde,
                sync_zeta_ema_with_momentum=sync_zeta_ema_with_momentum_pde,
            )

        zetas_obs.append(float(zeta_obs_used))
        zetas_pde.append(float(zeta_pde_used))

        # =========================
        # update rule (CHANNEL-WISE)
        # x_next shape = (B,34,H,W)
        #   - channel 1: use grad_obs + grad_pde_eff
        #   - channel 0, 2~: use grad_obs only
        # =========================
        update = zeta_obs_used * grad_obs                      # all channels get obs update
        if is_PDE_loss:
            update[:, 1, :, :] = update[:, 1, :, :] + zeta_pde_used * grad_pde_eff[:, 1, :, :]
        # if PDE loss off, skip

        x_next = x_next - update

    # --- Final predictions ---
    x_final = x_next
    dtn_pred = x_final[:, 0, :, :].unsqueeze(1)    # (1,1,64,64)
    a_pred   = x_final[:, 1, :, :].unsqueeze(1)    # (1,1,64,64)

    grad_u_x_pred = x_final[:, 2:18, :, :]         # (1,16,64,64)
    grad_u_y_pred = x_final[:, 18:,  :, :]         # (1,16,64,64)

    rel_err_dtn = (torch.norm(dtn_pred - dtn_GT, 2) / torch.norm(dtn_GT, 2)).item()
    rel_err_a   = (torch.norm(a_pred   - a_GT,   2) / torch.norm(a_GT,   2)).item()

    return (
        dtn_pred, a_pred,
        grad_u_x_pred, grad_u_y_pred,
        rel_err_dtn, rel_err_a,
        losses_pde, losses_obs_dtn, losses_obs_a,
        zetas_obs, zetas_pde, 
        pde_eff_norms
    )


# =========================
# Main generate function
# =========================
def generate_eit(config):
    datapath      = config['data']['datapath']
    offset        = int(config['data'].get('offset', 0))
    whole_dataset = bool(config['data'].get('whole_dataset', False))

    device = torch.device(config['generate']['device'])
    seed   = int(config['generate']['seed'])
    # torch.manual_seed(seed)
    set_seed(seed)

    # ------------------ Load data ------------------
    data = torch.load(datapath)

    if 'sigma' not in data:
        raise KeyError("data dict에 'sigma' 키가 없습니다. (N,64,64) 형태의 sigma가 필요합니다.)")
    N_total = int(data['sigma'].shape[0])

    # ------------------ Load network ------------------
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    # ------------------ Masks ------------------
    H = net.img_resolution
    W = net.img_resolution
    
    is_mask = bool(config['generate']['is_mask'])

    if is_mask:
        known_index_dtn = get_eit_mask(grid_size=(H, W), device=device)
        print('known_index_dtn : 16*13')
    else:
        known_index_dtn = torch.ones((H, W)).to(device)
        print('known_index_dtn : 64*64')

    # ------------------ Output dirs ------------------
    ts = _today_yyyymmdd()
    equation_type = "EIT"
    mode_tag = "whole" if whole_dataset else f"offset_{offset:05d}"

    save_root = config.get('generate', {}).get('save_root', None)
    if save_root is None or (isinstance(save_root, str) and save_root.strip() == ""):
        out_dir = os.path.join("./results", equation_type, ts, mode_tag)
    else:
        out_dir = os.path.join("./results", equation_type, ts, str(save_root), mode_tag)

    _ensure_dir(out_dir)
    _save_config_snapshot(config, out_dir)

    net.eval()
    def _sync_if_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()
    # ------------------ Run ------------------
    if not whole_dataset:
        print(f"[Mode] single offset = {offset} / {N_total-1}")
        _sync_if_cuda()
        runtime_start = time.perf_counter()

        (
            dtn_pred, a_pred,
            grad_u_x_pred, grad_u_y_pred,
            rel_dtn, rel_a,
            losses_pde, losses_obs_dtn, losses_obs_a,
            zetas_obs, zetas_pde,
            pde_eff_norms
        ) = _generate_eit_one_sample(
            net=net,
            data=data,
            offset=offset,
            config=config,
            known_index_dtn=known_index_dtn,
            device=device,
        )

        _sync_if_cuda()
        runtime_sec = time.perf_counter() - runtime_start

        print(f'Relative error of dtn: {rel_dtn:.6f}')
        print(f'Relative error of a:   {rel_a:.6f}')

        # save mat
        out_mat = os.path.join(out_dir, "EIT_results.mat")
        scipy.io.savemat(out_mat, {
            'dtn': dtn_pred.detach().cpu().numpy(),
            'a':   a_pred.detach().cpu().numpy(),
            'grad_u_x': grad_u_x_pred.detach().cpu().numpy(),
            'grad_u_y': grad_u_y_pred.detach().cpu().numpy(),
            'offset': np.array([offset], dtype=np.int32),
        })

        # save metrics
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "mode": "single",
                "offset": int(offset),
                "N_total": int(N_total),
                "relative_error_dtn (%)": float(rel_dtn)*100,
                "relative_error_a (%)": float(rel_a)*100,
                "runtime_sec": float(runtime_sec),
                "timestamp": ts,
            }, f, indent=2)

        # save losses
        torch.save(torch.tensor(losses_pde, dtype=torch.float64),     os.path.join(out_dir, "loss_pde.pt"))
        torch.save(torch.tensor(losses_obs_dtn, dtype=torch.float64), os.path.join(out_dir, "loss_obs_dtn.pt"))
        torch.save(torch.tensor(losses_obs_a, dtype=torch.float64),   os.path.join(out_dir, "loss_obs_a.pt"))

        # save zeta curves (USED)
        torch.save(torch.tensor(zetas_obs, dtype=torch.float64), os.path.join(out_dir, "zeta_obs.pt"))
        torch.save(torch.tensor(zetas_pde, dtype=torch.float64), os.path.join(out_dir, "zeta_pde.pt"))
        torch.save(torch.tensor(pde_eff_norms, dtype=torch.float64),
            os.path.join(out_dir, "pde_eff_norm.pt"))

        torch.save({
            "steps": torch.arange(len(losses_pde), dtype=torch.int32),
            "pde": torch.tensor(losses_pde, dtype=torch.float64),
            "obs_dtn": torch.tensor(losses_obs_dtn, dtype=torch.float64),
            "obs_a": torch.tensor(losses_obs_a, dtype=torch.float64),
            "zeta_obs": torch.tensor(zetas_obs, dtype=torch.float64),
            "zeta_pde": torch.tensor(zetas_pde, dtype=torch.float64),
            "pde_eff_norm": torch.tensor(pde_eff_norms, dtype=torch.float64),
            "offset": int(offset),
            "timestamp": ts,
        }, os.path.join(out_dir, "losses_all.pt"))

        # optional plotting (single)
        if bool(config['generate'].get('is_plot', False)):
            _plot_and_save_auto_scale(out_dir, losses_pde,     "PDE loss over steps", "loss_pde.png")
            _plot_and_save_auto_scale(out_dir, losses_obs_dtn, "Observation loss (Dtn) over steps", "loss_obs_dtn.png")
            _plot_and_save_auto_scale(out_dir, losses_obs_a,   "Loss (a) over steps", "loss_obs_a.png")
            _plot_and_save_auto_scale(out_dir, zetas_obs,      "zeta_obs (USED) over steps", "zeta_obs.png")
            _plot_and_save_auto_scale(out_dir, zetas_pde,      "zeta_pde (USED) over steps", "zeta_pde.png")
            _plot_and_save_auto_scale(out_dir, pde_eff_norms,      "pde_eff_norms (USED) over steps", "pde_eff_norms.png")

        print(f"Saved results to: {out_mat}")
        print(f"Saved losses/zeta (.pt) and plots (if enabled) to: {out_dir}")
        print("Done.")
        return

    # ------------------ Whole dataset mode ------------------
    print(f"[Mode] whole_dataset = True, N_total = {N_total}")

    # global obesity_1_idx_lst, obesity_2_idx_lst

    dtn_preds = []
    a_preds   = []
    grad_u_x_preds = []
    grad_u_y_preds = []
    rels_dtn  = []
    rels_a    = []

    num_steps = int(config['test']['iterations'])
    sum_loss_pde = np.zeros(num_steps, dtype=np.float64)
    sum_loss_obs = np.zeros(num_steps, dtype=np.float64)
    sum_loss_obs_a = np.zeros(num_steps, dtype=np.float64)

    # mean zeta over samples
    sum_zeta_obs = np.zeros(num_steps, dtype=np.float64)
    sum_zeta_pde = np.zeros(num_steps, dtype=np.float64)

    sum_pde_eff_norm = np.zeros(num_steps, dtype=np.float64)

    _sync_if_cuda()
    runtime_start = time.perf_counter()

    for off in tqdm.tqdm(range(N_total), desc="Whole dataset", unit="sample"):
        set_seed(seed + off)
        (
            dtn_pred, a_pred,
            grad_u_x_pred, grad_u_y_pred,
            rel_dtn, rel_a,
            losses_pde, losses_obs_dtn, losses_obs_a,
            zetas_obs, zetas_pde,
            pde_eff_norms
        ) = _generate_eit_one_sample(
            net=net,
            data=data,
            offset=off,
            config=config,
            known_index_dtn=known_index_dtn,
            device=device,
        )

        dtn_preds.append(dtn_pred.detach().cpu().numpy())
        a_preds.append(a_pred.detach().cpu().numpy())
        grad_u_x_preds.append(grad_u_x_pred.detach().cpu().numpy())
        grad_u_y_preds.append(grad_u_y_pred.detach().cpu().numpy())
        rels_dtn.append(rel_dtn)
        rels_a.append(rel_a)

        if len(losses_pde) == num_steps:
            sum_loss_pde += np.asarray(losses_pde, dtype=np.float64)
        if len(losses_obs_dtn) == num_steps:
            sum_loss_obs += np.asarray(losses_obs_dtn, dtype=np.float64)
        if len(losses_obs_a) == num_steps:
            sum_loss_obs_a += np.asarray(losses_obs_a, dtype=np.float64)

        if len(zetas_obs) == num_steps:
            sum_zeta_obs += np.asarray(zetas_obs, dtype=np.float64)
        if len(zetas_pde) == num_steps:
            sum_zeta_pde += np.asarray(zetas_pde, dtype=np.float64)

        if len(pde_eff_norms) == num_steps:
            sum_pde_eff_norm += np.asarray(pde_eff_norms, dtype=np.float64)

    _sync_if_cuda()
    runtime_sec = time.perf_counter() - runtime_start

    dtn_preds = np.concatenate(dtn_preds, axis=0)             # (N,1,64,64)
    a_preds   = np.concatenate(a_preds,   axis=0)             # (N,1,64,64)
    grad_u_x_preds = np.concatenate(grad_u_x_preds, axis=0)   # (N,16,64,64)
    grad_u_y_preds = np.concatenate(grad_u_y_preds, axis=0)   # (N,16,64,64)
    rels_dtn  = np.asarray(rels_dtn, dtype=np.float64)
    rels_a    = np.asarray(rels_a,   dtype=np.float64)

    mean_rel_dtn   = float(rels_dtn.mean())
    median_rel_dtn = float(np.median(rels_dtn))
    mean_rel_a     = float(rels_a.mean())
    median_rel_a   = float(np.median(rels_a))

    print(f"[EIT WHOLE] #samples = {N_total}")
    print(f"[EIT WHOLE] mean relative L2 error (dtn)   = {mean_rel_dtn:.6f}")
    print(f"[EIT WHOLE] median relative L2 error (dtn) = {median_rel_dtn:.6f}")
    print(f"[EIT WHOLE] mean relative L2 error (a)     = {mean_rel_a:.6f}")
    print(f"[EIT WHOLE] median relative L2 error (a)   = {median_rel_a:.6f}")

    out_mat = os.path.join(out_dir, "EIT_results_whole.mat")
    scipy.io.savemat(out_mat, {
        'dtn': dtn_preds,
        'a':   a_preds,
        'grad_u_x': grad_u_x_preds,
        'grad_u_y': grad_u_y_preds,
        'rel_dtn': rels_dtn,
        'rel_a':   rels_a,
    })

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "mode": "whole_dataset",
            "N_total": int(N_total),
            "mean_relative_error_dtn (%)": mean_rel_dtn*100,
            "median_relative_error_dtn (%)": median_rel_dtn*100,
            "mean_relative_error_a (%)": mean_rel_a*100,
            "median_relative_error_a (%)": median_rel_a*100,
            "runtime_sec": float(runtime_sec),
            "timestamp": ts,
        }, f, indent=2)

    mean_loss_pde = (sum_loss_pde / max(N_total, 1)).astype(np.float64)
    mean_loss_obs = (sum_loss_obs / max(N_total, 1)).astype(np.float64)
    mean_loss_a   = (sum_loss_obs_a / max(N_total, 1)).astype(np.float64)

    torch.save(torch.tensor(mean_loss_pde, dtype=torch.float64), os.path.join(out_dir, "loss_pde_mean.pt"))
    torch.save(torch.tensor(mean_loss_obs, dtype=torch.float64), os.path.join(out_dir, "loss_obs_dtn_mean.pt"))
    torch.save(torch.tensor(mean_loss_a,   dtype=torch.float64), os.path.join(out_dir, "loss_obs_a_mean.pt"))

    # mean zeta
    mean_zeta_obs = (sum_zeta_obs / max(N_total, 1)).astype(np.float64)
    mean_zeta_pde = (sum_zeta_pde / max(N_total, 1)).astype(np.float64)
    torch.save(torch.tensor(mean_zeta_obs, dtype=torch.float64), os.path.join(out_dir, "zeta_obs_mean.pt"))
    torch.save(torch.tensor(mean_zeta_pde, dtype=torch.float64), os.path.join(out_dir, "zeta_pde_mean.pt"))

    mean_pde_eff_norm = (sum_pde_eff_norm / max(N_total, 1)).astype(np.float64)
    torch.save(torch.tensor(mean_pde_eff_norm, dtype=torch.float64),
               os.path.join(out_dir, "pde_eff_norm_mean.pt"))

    torch.save({
        "steps": torch.arange(num_steps, dtype=torch.int32),
        "pde_mean": torch.tensor(mean_loss_pde, dtype=torch.float64),
        "obs_dtn_mean": torch.tensor(mean_loss_obs, dtype=torch.float64),
        "obs_a_mean": torch.tensor(mean_loss_a, dtype=torch.float64),
        "zeta_obs_mean": torch.tensor(mean_zeta_obs, dtype=torch.float64),
        "zeta_pde_mean": torch.tensor(mean_zeta_pde, dtype=torch.float64),
        "pde_eff_norm_mean": torch.tensor(mean_pde_eff_norm, dtype=torch.float64),
        "timestamp": ts,
    }, os.path.join(out_dir, "losses_all_mean.pt"))

    if bool(config['generate'].get('is_plot', False)):
        _plot_and_save_auto_scale(out_dir, mean_loss_pde, "MEAN PDE loss over steps (whole dataset)", "loss_pde_mean.png")
        _plot_and_save_auto_scale(out_dir, mean_loss_obs, "MEAN Observation loss (Dtn) over steps (whole dataset)", "loss_obs_dtn_mean.png")
        _plot_and_save_auto_scale(out_dir, mean_loss_a,   "MEAN loss (a) over steps (whole dataset)", "loss_obs_a_mean.png")
        _plot_and_save_auto_scale(out_dir, mean_zeta_obs, "MEAN zeta_obs (USED) over steps (whole dataset)", "zeta_obs_mean.png")
        _plot_and_save_auto_scale(out_dir, mean_zeta_pde, "MEAN zeta_pde (USED) over steps (whole dataset)", "zeta_pde_mean.png")
        _plot_and_save_auto_scale(out_dir, mean_pde_eff_norm, "MEAN mean_pde_eff_norm (USED) over steps (whole dataset)", "mean_pde_eff_norm.png")

    print(f"Saved WHOLE results to: {out_mat}")
    print(f"Saved metrics/losses/zeta/plots (if enabled) to: {out_dir}")
    print("Done.")


# =========================
# Runner
# =========================
if __name__ == "__main__":
    import sys
    cfg = None
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        try:
            import yaml
            cfg = yaml.safe_load(txt)
        except Exception:
            cfg = json.loads(txt)
    else:
        raise SystemExit("Please provide a config file path (yaml/json).")

    generate_eit(cfg)

###############