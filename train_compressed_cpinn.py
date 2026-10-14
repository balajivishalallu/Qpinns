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
from src.cpinn import CompressedPINN, compute_composite_loss
from src.exact_solution import compute_exact_grid

def train_compressed_cpinn(epochs=5000, lr=1e-3, nu=0.01, seed=42, device='cpu'):
    print("==================================================================", flush=True)
    print("Starting 1D Burgers Equation Compressed c-PINN Ablation Training", flush=True)
    print("==================================================================", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), 'evaluation')
    os.makedirs(out_dir, exist_ok=True)

    domain = BurgersDomain(seed=seed)
    X_f, x_f, t_f, X_i, u_i, X_b, u_b = domain.sample_training_data(N_f=2000, N_i=200, N_b=200, device=device)
    X_mesh, T_mesh, X_eval, x_flat, t_flat = domain.generate_eval_grid(Nx=256, Nt=100, device=device)

    exact_path = os.path.join(out_dir, 'exact_solution.npy')
    if os.path.exists(exact_path):
        u_exact = np.load(exact_path)
    else:
        u_exact = compute_exact_grid(x_flat, t_flat, nu=nu)
        np.save(exact_path, u_exact)

    model = CompressedPINN().to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Compressed c-PINN Initialized | Architecture: 2 -> 189 -> 7 -> 1", flush=True)
    print(f"Total Trainable Parameters: {num_params} (Matches QAPINN baseline 1,905 params exactly!)", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        'epoch': [],
        'loss_total': [],
        'loss_f': [],
        'loss_i': [],
        'loss_b': [],
        'relative_l2': []
    }

    start_time = time.time()
    weights = (1.0, 10.0, 10.0)
    print_freq = 10 if epochs <= 100 else 100

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss, loss_f, loss_i, loss_b = compute_composite_loss(
            model, X_f, x_f, t_f, X_i, u_i, X_b, u_b, nu=nu, weights=weights
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % print_freq == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                u_pred = model(X_eval).cpu().numpy().reshape(u_exact.shape)
                l2_err = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

            history['epoch'].append(epoch)
            history['loss_total'].append(loss.item())
            history['loss_f'].append(loss_f.item())
            history['loss_i'].append(loss_i.item())
            history['loss_b'].append(loss_b.item())
            history['relative_l2'].append(l2_err)

            if epoch % 500 == 0 or epoch == epochs:
                elapsed = time.time() - start_time
                print(f"[COMPRESSED c-PINN] Epoch {epoch:4d}/{epochs} ({epoch/epochs*100:.1f}%) | "
                      f"Loss: {loss.item():.6e} | Rel L2: {l2_err:.4e} | Elapsed: {elapsed:.1f}s", flush=True)

    elapsed_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        u_final_pred = model(X_eval).cpu().numpy().reshape(u_exact.shape)

    torch.save(model.state_dict(), os.path.join(out_dir, 'compressed_cpinn_model.pt'))
    np.save(os.path.join(out_dir, 'compressed_cpinn_history.npy'), history)
    np.save(os.path.join(out_dir, 'compressed_cpinn_pred.npy'), u_final_pred)

    print("==================================================================", flush=True)
    print(f"Compressed c-PINN Training Complete in {elapsed_time:.2f} seconds!", flush=True)
    print(f"Final Relative L2 Validation Error Floor: {history['relative_l2'][-1]:.4e}", flush=True)
    print("==================================================================", flush=True)

    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Compressed c-PINN (1905 params) for 1D Burgers Equation")
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs (default: 5000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity coefficient nu')
    args = parser.parse_args()

    train_compressed_cpinn(epochs=args.epochs, lr=args.lr, nu=args.nu)
