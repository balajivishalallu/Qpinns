"""
train_entanglement_topology.py
==============================
Experiment 5: Entanglement Topology Matrix (None vs. Linear vs. Circular)

Objective:
    Quantify the link between quantum entanglement entropy S(rho_A) and physics loss stability.
    Tests three CNOT entanglement topologies in the VQC layer of Hybrid QAPINN:
        1. None     (Zero entanglement, 0 CNOT gates)
        2. Linear   (CNOT chain q0 -> q1 -> q2 -> q3)
        3. Circular (CNOT closed loop q0 -> q1 -> q2 -> q3 -> q0)

    During a 3000-epoch training cycle on 1D Viscous Burgers' equation, state vectors
    are extracted every 100 epochs to calculate reduced density matrix rho_A and Von Neumann Entropy.
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
from src.vqc_circuit import create_vqc_node
from src.exact_solution import compute_exact_grid

def compute_von_neumann_entropy(model, X_eval_sample):
    model.eval()
    n_qubits = model.n_qubits
    n_layers = model.n_layers
    entanglement = getattr(model, 'entanglement', 'linear')

    state_qnode = create_vqc_node(
        n_qubits=n_qubits,
        n_layers=n_layers,
        encoding=model.encoding,
        entanglement=entanglement,
        measurement_mode='state'
    )

    weights = model.qlayer.weights.detach()
    entropies = []
    num_samples = min(10, X_eval_sample.shape[0])

    for i in range(num_samples):
        x_single = X_eval_sample[i:i+1]
        with torch.no_grad():
            if model.position == 'middle':
                h_in = model.act(model.fc_in(x_single))
                x_single = model.fc_pre_vqc(h_in)
            elif model.position == 'final':
                h1 = model.act(model.fc1(x_single))
                h2 = model.act(model.fc2(h1))
                x_single = model.fc_pre_vqc(h2)

        state = state_qnode(x_single.squeeze(0), weights)
        psi = state.detach().cpu().numpy()

        n_A = n_qubits // 2
        dim_A = 2**n_A
        dim_B = 2**(n_qubits - n_A)
        psi_mat = psi.reshape((dim_A, dim_B))

        rho_A = psi_mat @ psi_mat.conj().T
        eigenvals = np.linalg.eigvalsh(rho_A)
        eigenvals = eigenvals[eigenvals > 1e-12]

        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        entropies.append(entropy)

    return float(np.mean(entropies))

def train_topology(entanglement='linear', epochs=3000, lr=1e-3, nu=0.01, seed=42, device='cpu'):
    print(f"\n==================================================================", flush=True)
    print(f"  Training Entanglement Topology: [{entanglement.upper()}]", flush=True)
    print(f"==================================================================", flush=True)

    domain = BurgersDomain(seed=seed)
    X_f, x_f, t_f, X_i, u_i, X_b, u_b = domain.sample_training_data(N_f=500, N_i=100, N_b=100, device=device)
    X_mesh, T_mesh, X_eval, x_flat, t_flat = domain.generate_eval_grid(Nx=256, Nt=100, device=device)

    out_dir = os.path.join(os.path.dirname(__file__), 'evaluation', 'entanglement_topology')
    os.makedirs(out_dir, exist_ok=True)

    exact_path = os.path.join(os.path.dirname(__file__), 'evaluation', 'exact_solution.npy')
    if os.path.exists(exact_path):
        u_exact = np.load(exact_path)
    else:
        u_exact = compute_exact_grid(x_flat, t_flat, nu=nu)
        np.save(exact_path, u_exact)

    torch.manual_seed(seed)
    model = HybridQAPINN(
        n_qubits=4,
        n_layers=2,
        position='first',
        measurement_mode='expectation',
        encoding='angle',
        entanglement=entanglement,
        hidden_features=40,
        device_name='default.qubit'
    ).to(device)
    setattr(model, 'entanglement', entanglement)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        'epoch': [],
        'loss_total': [],
        'loss_f': [],
        'loss_i': [],
        'loss_b': [],
        'relative_l2': [],
        'von_neumann_entropy': [],
        'loss_std_window': []
    }

    recent_losses = []
    loss_weights = (1.0, 10.0, 10.0)
    print_freq = 100 if epochs >= 500 else 10
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

        recent_losses.append(total_loss.item())
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        if epoch % print_freq == 0 or epoch == epochs or epoch == 1:
            model.eval()
            with torch.no_grad():
                u_pred_flat = model(X_eval).cpu().numpy().flatten()
                u_pred_2d = u_pred_flat.reshape(u_exact.shape)
                rel_l2 = np.linalg.norm(u_pred_2d - u_exact) / np.linalg.norm(u_exact)

            entropy = compute_von_neumann_entropy(model, X_eval[:10])
            loss_std = float(np.std(recent_losses))

            history['epoch'].append(epoch)
            history['loss_total'].append(total_loss.item())
            history['loss_f'].append(loss_f.item())
            history['loss_i'].append(loss_i.item())
            history['loss_b'].append(loss_b.item())
            history['relative_l2'].append(rel_l2)
            history['von_neumann_entropy'].append(entropy)
            history['loss_std_window'].append(loss_std)

            elapsed = time.time() - t0
            print(f"    [{entanglement.upper():8s}] Epoch {epoch:4d}/{epochs} | "
                  f"Loss: {total_loss.item():.4e} | Rel L2: {rel_l2:.4e} | "
                  f"Entropy: {entropy:.4f} bits | Elapsed: {elapsed:.1f}s", flush=True)

    elapsed_total = time.time() - t0

    model.eval()
    with torch.no_grad():
        u_final = model(X_eval).cpu().numpy().flatten().reshape(u_exact.shape)

    np.save(os.path.join(out_dir, f'history_{entanglement}.npy'), history)
    np.save(os.path.join(out_dir, f'pred_{entanglement}.npy'), u_final)

    print(f"  [{entanglement.upper()}] Training Complete in {elapsed_total:.1f}s | Final Rel L2: {history['relative_l2'][-1]:.4e}", flush=True)
    return history, u_final

def main(epochs=3000):
    print("==================================================================", flush=True)
    print("  EXPERIMENT 5: ENTANGLEMENT TOPOLOGY MATRIX (NONE vs LINEAR vs CIRCULAR)", flush=True)
    print("==================================================================", flush=True)

    topologies = ['none', 'linear', 'circular']
    results = {}

    for topo in topologies:
        hist, pred = train_topology(entanglement=topo, epochs=epochs)
        results[topo] = (hist, pred)

    print("\n==================================================================", flush=True)
    print("  ALL ENTANGLEMENT TOPOLOGY EXPERIMENTS COMPLETED!", flush=True)
    print("==================================================================", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment 5: Entanglement Topology Matrix")
    parser.add_argument('--epochs', type=int, default=3000, help='Training epochs per topology (default: 3000)')
    args = parser.parse_args()

    main(epochs=args.epochs)
