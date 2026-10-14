"""
train_viscosity_sensitivity.py
==============================
Experiment: Viscosity Scale Sensitivity (\nu = 0.1 vs 0.01 vs 0.001)

Objective:
    Test the physical resilience and architectural breakdown limit of the QAPINN
    as the kinematic viscosity coefficient \nu approaches a true mathematical discontinuity:
        1. \nu = 0.1   (Smooth diffusion-dominated regime)
        2. \nu = 0.01  (Moderate steepening / intermediate regime)
        3. \nu = 0.001 (Sharp shock front formation / quasi-discontinuity)

    Trains Hybrid QAPINN across all three scenarios for 5000 epochs each,
    calculating exact solutions and pointwise absolute error maps |u_exact - u_pred|.
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.domain import BurgersDomain
from src.qapinn import HybridQAPINN, compute_qapinn_pde_residual
from src.exact_solution import compute_exact_grid

def train_viscosity_scenario(nu=0.01, epochs=5000, lr=1e-3, seed=42, device='cpu'):
    print(f"\n==================================================================", flush=True)
    print(f"  Training Viscosity Sensitivity Scenario: [nu = {nu}]", flush=True)
    print(f"==================================================================", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), 'evaluation', 'viscosity_sensitivity')
    os.makedirs(out_dir, exist_ok=True)

    domain = BurgersDomain(seed=seed)
    X_f, x_f, t_f, X_i, u_i, X_b, u_b = domain.sample_training_data(N_f=500, N_i=100, N_b=100, device=device)
    X_mesh, T_mesh, X_eval, x_flat, t_flat = domain.generate_eval_grid(Nx=256, Nt=100, device=device)

    print(f"  Computing Cole-Hopf exact reference solution for nu = {nu}...", flush=True)
    u_exact = compute_exact_grid(x_flat, t_flat, nu=nu)

    torch.manual_seed(seed)
    model = HybridQAPINN(
        n_qubits=4,
        n_layers=2,
        position='first',
        measurement_mode='expectation',
        encoding='angle',
        entanglement='linear',
        hidden_features=40,
        device_name='default.qubit'
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    loss_weights = (1.0, 10.0, 10.0)
    history = {'epoch': [], 'loss_total': [], 'relative_l2': []}
    print_freq = 500 if epochs >= 500 else 10
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        residual, _ = compute_qapinn_pde_residual(model, x_f, t_f, nu=nu)
        loss_f = torch.mean(residual ** 2)
        loss_i = torch.mean((model(X_i) - u_i) ** 2)
        loss_b = torch.mean((model(X_b) - u_b) ** 2)

        total_loss = (loss_weights[0] * loss_f
                      + loss_weights[1] * loss_i
                      + loss_weights[2] * loss_b)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % print_freq == 0 or epoch == epochs or epoch == 1:
            model.eval()
            with torch.no_grad():
                u_pred_flat = model(X_eval).cpu().numpy().flatten()
                u_pred_2d = u_pred_flat.reshape(u_exact.shape)
                rel_l2 = np.linalg.norm(u_pred_2d - u_exact) / np.linalg.norm(u_exact)

            history['epoch'].append(epoch)
            history['loss_total'].append(total_loss.item())
            history['relative_l2'].append(rel_l2)

            elapsed = time.time() - t0
            print(f"    [nu = {nu:6.3f}] Epoch {epoch:4d}/{epochs} | "
                  f"Loss: {total_loss.item():.4e} | Rel L2: {rel_l2:.4e} | "
                  f"Elapsed: {elapsed:.1f}s", flush=True)

    elapsed_total = time.time() - t0

    model.eval()
    with torch.no_grad():
        u_final = model(X_eval).cpu().numpy().flatten().reshape(u_exact.shape)

    err_map = np.abs(u_exact - u_final)

    nu_str = str(nu).replace('.', '_')
    np.save(os.path.join(out_dir, f'u_exact_nu_{nu_str}.npy'), u_exact)
    np.save(os.path.join(out_dir, f'u_pred_nu_{nu_str}.npy'), u_final)
    np.save(os.path.join(out_dir, f'err_map_nu_{nu_str}.npy'), err_map)
    np.save(os.path.join(out_dir, f'history_nu_{nu_str}.npy'), history)

    print(f"  [nu = {nu}] Complete in {elapsed_total:.1f}s | Final Rel L2: {history['relative_l2'][-1]:.4e} | Max Pointwise Error: {err_map.max():.4f}", flush=True)
    return u_exact, u_final, err_map, history

def main(epochs=5000):
    print("==================================================================", flush=True)
    print("  VISCOSITY SCALE SENSITIVITY EXPERIMENT (nu = 0.1 vs 0.01 vs 0.001)", flush=True)
    print("==================================================================", flush=True)

    nu_values = [0.1, 0.01, 0.001]
    for nu in nu_values:
        train_viscosity_scenario(nu=nu, epochs=epochs)

    print("\n==================================================================", flush=True)
    print("  ALL VISCOSITY SCENARIOS COMPLETED!", flush=True)
    print("==================================================================", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Viscosity Scale Sensitivity Experiment")
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs per scenario (default: 5000)')
    args = parser.parse_args()

    main(epochs=args.epochs)
