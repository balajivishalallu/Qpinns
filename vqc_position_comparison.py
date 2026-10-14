"""
vqc_position_comparison.py
===========================
WISER Summer Program 2026 — VQC Positioning Experiment

Experiment:
    Train THREE QAPINN configurations with VQC placed at different
    structural positions within the hybrid network:

    Position 1  (pos='first')   : Input (x,t) -> [VQC] -> Dense(40) -> Dense(40) -> u
    Position 2  (pos='middle')  : Input (x,t) -> Dense(40) -> [VQC] -> Dense(40) -> u
    Position 3  (pos='final')   : Input (x,t) -> Dense(40) -> Dense(40) -> [VQC] -> u

    Each model is trained for EPOCHS epochs on the 1D viscous Burgers' equation.
    After training, generate a 3-panel comparative spatiotemporal plot.
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*requires_grad=True.*")

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.domain import BurgersDomain
from src.qapinn import HybridQAPINN, compute_qapinn_pde_residual, count_parameters


# =============================================================================
# CONFIGURATION
# =============================================================================

EPOCHS      = 5000        # training epochs per position (set 500 for quick test)
N_QUBITS    = 4
N_LAYERS    = 2
LR          = 1e-3
NU          = 0.01        # viscosity coefficient
N_F         = 500         # PDE collocation points
N_I         = 100         # initial condition points
N_B         = 100         # boundary condition points
SEED        = 42
LOG_EVERY   = 100         # print log every N epochs
DEVICE_NAME = 'default.qubit'

SAVE_PATH   = 'evaluation/plots/vqc_position_comparison.png'
DATA_DIR    = 'evaluation'
os.makedirs('evaluation/plots', exist_ok=True)

# Three configurations to test
CONFIGS = [
    {
        'position'   : 'first',
        'label'      : 'Position 1',
        'subtitle'   : 'VQC at Input Boundary',
        'arch'       : '(x,t) -> [VQC] -> Dense(40) -> Dense(40) -> u',
        'color'      : '#388bfd',   # blue
    },
    {
        'position'   : 'middle',
        'label'      : 'Position 2',
        'subtitle'   : 'VQC Between Hidden Layers',
        'arch'       : '(x,t) -> Dense(40) -> [VQC] -> Dense(40) -> u',
        'color'      : '#3fb950',   # green
    },
    {
        'position'   : 'final',
        'label'      : 'Position 3',
        'subtitle'   : 'VQC Before Output Node',
        'arch'       : '(x,t) -> Dense(40) -> Dense(40) -> [VQC] -> u',
        'color'      : '#e3b341',   # yellow
    },
]


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_one_config(cfg: dict,
                     domain: BurgersDomain,
                     u_exact: np.ndarray,
                     X_eval: torch.Tensor,
                     epochs: int,
                     device_name: str = 'default.qubit') -> dict:
    pos   = cfg['position']
    label = cfg['label']

    print(f"\n{'='*62}", flush=True)
    print(f"  Training  {label}  — {cfg['subtitle']}", flush=True)
    print(f"  Architecture: {cfg['arch']}", flush=True)
    print(f"{'='*62}", flush=True)

    X_f, x_f, t_f, X_i, u_i, X_b, u_b = domain.sample_training_data(
        N_f=N_F, N_i=N_I, N_b=N_B)

    torch.manual_seed(SEED)
    model = HybridQAPINN(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        position=pos,
        measurement_mode='expectation',
        encoding='angle',
        entanglement='linear',
        hidden_features=40,
        device_name=device_name
    )
    param_info = count_parameters(model)
    print(f"  Params — Total: {param_info['total']:,}  "
          f"| Quantum: {param_info['quantum']}  "
          f"| Classical: {param_info['classical']:,}  "
          f"| vs c-PINN reduction: {param_info['reduction_pct']:.1f}%", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    loss_weights = (1.0, 10.0, 10.0)
    history      = []
    t0           = time.time()
    print_freq   = 10 if epochs <= 100 else LOG_EVERY

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        residual, _ = compute_qapinn_pde_residual(model, x_f, t_f, nu=NU)
        loss_f = torch.mean(residual ** 2)

        loss_i = torch.mean((model(X_i) - u_i) ** 2)
        loss_b = torch.mean((model(X_b) - u_b) ** 2)

        total_loss = (loss_weights[0] * loss_f
                      + loss_weights[1] * loss_i
                      + loss_weights[2] * loss_b)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(total_loss.item())

        if epoch % print_freq == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                u_pred_flat = model(X_eval).cpu().numpy().flatten()
                u_pred_2d   = u_pred_flat.reshape(u_exact.shape)
                rel_l2      = (np.linalg.norm(u_pred_2d - u_exact)
                               / np.linalg.norm(u_exact))
            elapsed = time.time() - t0
            print(f"    Epoch {epoch:5d}/{epochs} | "
                  f"Loss: {total_loss.item():.4e} | "
                  f"Rel L2: {rel_l2:.4e} | "
                  f"Time: {elapsed:.1f}s", flush=True)

    elapsed_total = time.time() - t0

    model.eval()
    with torch.no_grad():
        u_final = model(X_eval).cpu().numpy().flatten().reshape(u_exact.shape)
    rel_l2_final = (np.linalg.norm(u_final - u_exact)
                    / np.linalg.norm(u_exact))

    print(f"\n  {label} DONE in {elapsed_total:.1f}s  "
          f"| Final Rel L2: {rel_l2_final:.4e}", flush=True)

    return {
        'u_pred'       : u_final,
        'history'      : history,
        'rel_l2_final' : rel_l2_final,
        'param_info'   : param_info,
        'elapsed'      : elapsed_total,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(results: list,
                    x_grid: np.ndarray,
                    t_grid: np.ndarray,
                    u_exact: np.ndarray,
                    epochs: int):

    BG       = '#0d1117'
    AX_BG    = '#161b22'
    GRID_C   = '#30363d'
    WIRE_C   = '#c9d1d9'
    TITLE_C  = '#58a6ff'
    PARAM_C  = '#e3b341'
    RED_C    = '#f85149'

    n_configs = len(CONFIGS)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        3, n_configs + 1,
        figure=fig,
        height_ratios=[1.4, 1.0, 0.8],
        hspace=0.45, wspace=0.32
    )

    vmin, vmax = u_exact.min(), u_exact.max()
    cmap_sol   = 'RdBu_r'
    if vmin < 0.0 < vmax:
        norm_sol = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm_sol = Normalize(vmin=vmin, vmax=vmax)

    # Exact
    ax_ex = fig.add_subplot(gs[0, 0])
    ax_ex.set_facecolor(AX_BG)
    im = ax_ex.pcolormesh(t_grid, x_grid, u_exact,
                           cmap=cmap_sol, norm=norm_sol, shading='auto')
    cb = fig.colorbar(im, ax=ax_ex, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
    cb.outline.set_edgecolor(GRID_C)
    ax_ex.set_title('Exact Solution\nu(x,t)', color=TITLE_C,
                     fontsize=11, fontweight='bold', pad=8)
    ax_ex.set_xlabel('Time  t', color=WIRE_C, fontsize=9)
    ax_ex.set_ylabel('Space  x', color=WIRE_C, fontsize=9)
    ax_ex.tick_params(colors=WIRE_C)
    ax_ex.spines[:].set_color(GRID_C)

    # Predictions
    for i, (cfg, res) in enumerate(zip(CONFIGS, results)):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.set_facecolor(AX_BG)
        im2 = ax.pcolormesh(t_grid, x_grid, res['u_pred'],
                             cmap=cmap_sol, norm=norm_sol, shading='auto')
        cb2 = fig.colorbar(im2, ax=ax, pad=0.02)
        cb2.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
        cb2.outline.set_edgecolor(GRID_C)

        rel_l2 = res['rel_l2_final']
        n_p    = res['param_info']['total']
        title  = (f"{cfg['label']}  —  {cfg['subtitle']}\n"
                  f"Rel L2: {rel_l2:.4f}  |  Params: {n_p:,}")
        ax.set_title(title, color=cfg['color'],
                     fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel('Time  t', color=WIRE_C, fontsize=9)
        ax.set_ylabel('Space  x', color=WIRE_C, fontsize=9)
        ax.tick_params(colors=WIRE_C)
        ax.spines[:].set_color(GRID_C)

    # Pointwise Error |pred - exact|
    cmap_err = 'hot'
    for i, (cfg, res) in enumerate(zip(CONFIGS, results)):
        ax_err = fig.add_subplot(gs[1, i + 1])
        ax_err.set_facecolor(AX_BG)
        err = np.abs(res['u_pred'] - u_exact)
        im_e = ax_err.pcolormesh(t_grid, x_grid, err,
                                  cmap=cmap_err, shading='auto',
                                  vmin=0, vmax=max(1e-5, err.max()))
        cb_e = fig.colorbar(im_e, ax=ax_err, pad=0.02)
        cb_e.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
        cb_e.outline.set_edgecolor(GRID_C)
        ax_err.set_title(f'|Error|  {cfg["label"]}', color=RED_C,
                          fontsize=9, fontweight='bold')
        ax_err.set_xlabel('Time  t', color=WIRE_C, fontsize=9)
        ax_err.set_ylabel('Space  x', color=WIRE_C, fontsize=9)
        ax_err.tick_params(colors=WIRE_C)
        ax_err.spines[:].set_color(GRID_C)

    # Exact error placeholder
    ax_ez = fig.add_subplot(gs[1, 0])
    ax_ez.set_facecolor(AX_BG)
    ax_ez.pcolormesh(t_grid, x_grid, np.zeros_like(u_exact),
                     cmap='hot', shading='auto', vmin=0, vmax=1)
    ax_ez.set_title('Exact  (ref)', color=WIRE_C, fontsize=9)
    ax_ez.set_xlabel('Time  t', color=WIRE_C, fontsize=9)
    ax_ez.set_ylabel('Space  x', color=WIRE_C, fontsize=9)
    ax_ez.tick_params(colors=WIRE_C)
    ax_ez.spines[:].set_color(GRID_C)
    ax_ez.text(0.5, 0.5, 'Zero Error\n(Reference)',
               transform=ax_ez.transAxes, ha='center', va='center',
               color=WIRE_C, fontsize=9, alpha=0.6)

    # Training Loss Curves
    ax_loss = fig.add_subplot(gs[2, :])
    ax_loss.set_facecolor(AX_BG)
    ax_loss.set_yscale('log')

    for cfg, res in zip(CONFIGS, results):
        hist    = res['history']
        ep_axis = list(range(1, len(hist) + 1))
        ax_loss.plot(ep_axis, hist,
                     color=cfg['color'], lw=1.6, alpha=0.85,
                     label=(f"{cfg['label']}  |  "
                            f"Final L2={res['rel_l2_final']:.4f}  |  "
                            f"{res['elapsed']:.1f}s"))

    ax_loss.set_xlabel('Training Epoch', color=WIRE_C, fontsize=10)
    ax_loss.set_ylabel('Total Loss  [log scale]', color=WIRE_C, fontsize=10)
    ax_loss.set_title('Training Loss Convergence — All Three Positions',
                       color=TITLE_C, fontsize=11, fontweight='bold')
    ax_loss.tick_params(colors=WIRE_C)
    ax_loss.spines[:].set_color(GRID_C)
    ax_loss.grid(True, color=GRID_C, lw=0.6, alpha=0.7, which='both')
    ax_loss.legend(fontsize=9, facecolor=BG, edgecolor=GRID_C,
                   labelcolor=WIRE_C, loc='upper right')

    fig.suptitle(
        'VQC Layer Position Comparison  —  1D Viscous Burgers Equation\n'
        f'4 Qubits  |  2 Variational Layers  |  {epochs} Epochs',
        color=TITLE_C, fontsize=14, fontweight='bold', y=0.98
    )

    arch_lines = []
    for cfg, res in zip(CONFIGS, results):
        arch_lines.append(
            f"{cfg['label']}: {cfg['arch']}  "
            f"[{res['param_info']['total']:,} params]"
        )
    fig.text(0.5, 0.005,
             '     |     '.join(arch_lines),
             ha='center', va='bottom',
             fontsize=7.5, color=PARAM_C)

    plt.savefig(SAVE_PATH, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Plot saved to: {SAVE_PATH}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main(epochs: int):
    print()
    print("=" * 62, flush=True)
    print("  VQC LAYER POSITION COMPARISON EXPERIMENT", flush=True)
    print(f"  Qubit count  : {N_QUBITS}", flush=True)
    print(f"  Epochs each  : {epochs}", flush=True)
    print(f"  Backend      : {DEVICE_NAME}", flush=True)
    print(f"  Configurations: {len(CONFIGS)}  (Position 1, 2, 3)", flush=True)
    print(f"  Equation     : 1D Viscous Burgers  (nu={NU})", flush=True)
    print("=" * 62, flush=True)

    domain  = BurgersDomain(seed=SEED)
    exact_path = os.path.join(DATA_DIR, 'exact_solution.npy')
    if os.path.exists(exact_path):
        u_exact = np.load(exact_path)
    else:
        # Fallback exact calculation
        x_val = np.linspace(-1.0, 1.0, 256)
        t_val = np.linspace(0.0, 1.0, 100)
        t_grid, x_grid = np.meshgrid(t_val, x_val)
        u_exact = -np.sin(np.pi * x_grid)

    _, _, X_eval, x_flat, t_flat = domain.generate_eval_grid(Nx=256, Nt=100)

    Nx, Nt = u_exact.shape
    x_vals  = np.linspace(-1.0, 1.0, Nx)
    t_vals  = np.linspace(0.0,  1.0, Nt)
    t_grid, x_grid = np.meshgrid(t_vals, x_vals)

    results = []
    total_t0 = time.time()

    for cfg in CONFIGS:
        res = train_one_config(cfg, domain, u_exact, X_eval, epochs,
                               device_name=DEVICE_NAME)
        results.append(res)

        save_name = f"pos_{cfg['position']}"
        np.save(os.path.join(DATA_DIR, f'{save_name}_pred.npy'),     res['u_pred'])
        np.save(os.path.join(DATA_DIR, f'{save_name}_history.npy'), res['history'])

    total_elapsed = time.time() - total_t0

    print()
    print("=" * 72, flush=True)
    print(f"  ALL CONFIGURATIONS TRAINED IN {total_elapsed:.1f} seconds", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'Config':>12}  {'Params':>8}  "
          f"{'Rel L2 Error':>14}  {'Time (s)':>10}  {'Architecture'}", flush=True)
    print("  " + "-" * 68, flush=True)
    for cfg, res in zip(CONFIGS, results):
        pi = res['param_info']
        print(f"  {cfg['label']:>12}  {pi['total']:>8,}  "
              f"{res['rel_l2_final']:>14.6f}  "
              f"{res['elapsed']:>10.1f}  "
              f"{cfg['arch']}", flush=True)
    print("=" * 72, flush=True)

    plot_comparison(results, x_grid, t_grid, u_exact, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VQC Position Comparison Experiment — WISER 2026')
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Training epochs per config (default: {EPOCHS})')
    args = parser.parse_args()

    main(args.epochs)
