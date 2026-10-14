import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*requires_grad=True.*")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

from src.vqc_circuit import create_vqc_node, get_weight_shape

class HybridQAPINN(nn.Module):
    """
    Hybrid Quantum-Assisted Physics-Informed Neural Network (QAPINN).
    Links a PennyLane Variational Quantum Circuit (VQC) with classical dense layers.
    
    Supports:
    - Layer positioning: 'first' (Position 1), 'middle' (Position 2), 'final' (Position 3)
    - Measurement mode: 'expectation' (returns N features) or 'probability' (returns 2^N features)
    - Qubit scaling: N in {3, 4, 5}
    - Encoding: 'angle' or 'amplitude'
    """
    def __init__(self, n_qubits=4, n_layers=2, position='first', measurement_mode='expectation', 
                 encoding='angle', entanglement='linear', hidden_features=40,
                 device_name='default.qubit'):
        super(HybridQAPINN, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.position = position
        self.measurement_mode = measurement_mode
        self.encoding = encoding
        self.hidden_features = hidden_features
        
        if measurement_mode == 'expectation':
            self.vqc_out_dim = n_qubits
        elif measurement_mode == 'probability':
            self.vqc_out_dim = 2**n_qubits
        else:
            raise ValueError(f"Unsupported measurement_mode for TorchLayer: {measurement_mode}")

        self.qnode = create_vqc_node(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding=encoding,
            entanglement=entanglement,
            measurement_mode=measurement_mode,
            device_name=device_name
        )
        weight_shapes = get_weight_shape(n_qubits, n_layers)
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        if position == 'first':
            self.fc1 = nn.Linear(self.vqc_out_dim, hidden_features)
            self.fc2 = nn.Linear(hidden_features, hidden_features)
            self.out = nn.Linear(hidden_features, 1)
            self.act = nn.Tanh()

        elif position == 'middle':
            self.fc_in = nn.Linear(2, hidden_features)
            self.fc_pre_vqc = nn.Linear(hidden_features, 2)
            self.fc_post_vqc = nn.Linear(self.vqc_out_dim, hidden_features)
            self.out = nn.Linear(hidden_features, 1)
            self.act = nn.Tanh()

        elif position == 'final':
            self.fc1 = nn.Linear(2, hidden_features)
            self.fc2 = nn.Linear(hidden_features, hidden_features)
            self.fc_pre_vqc = nn.Linear(hidden_features, 2)
            self.out = nn.Linear(self.vqc_out_dim, 1)
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown position: {position}")

        self._init_classical_weights()

    def _init_classical_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_t):
        if self.position == 'first':
            q_out = self.qlayer(x_t)
            h1 = self.act(self.fc1(q_out))
            h2 = self.act(self.fc2(h1))
            u_hat = self.out(h2)

        elif self.position == 'middle':
            h_in = self.act(self.fc_in(x_t))
            pre_vqc = self.fc_pre_vqc(h_in)
            q_out = self.qlayer(pre_vqc)
            h_post = self.act(self.fc_post_vqc(q_out))
            u_hat = self.out(h_post)

        elif self.position == 'final':
            h1 = self.act(self.fc1(x_t))
            h2 = self.act(self.fc2(h1))
            pre_vqc = self.fc_pre_vqc(h2)
            q_out = self.qlayer(pre_vqc)
            u_hat = self.out(q_out)

        return u_hat

def compute_qapinn_pde_residual(model, x_f, t_f, nu=0.01):
    X_f = torch.cat([x_f, t_f], dim=1)
    u = model(X_f)

    grads = torch.autograd.grad(
        outputs=u,
        inputs=[t_f, x_f],
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )
    u_t = grads[0]
    u_x = grads[1]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = u_t + u * u_x - nu * u_xx
    return residual, u

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_params = sum(p.numel() for name, p in model.named_parameters() if 'qlayer' in name and p.requires_grad)
    classical_params = total_params - quantum_params
    classical_control_params = 3441
    reduction_pct = ((classical_control_params - total_params) / classical_control_params) * 100.0

    return {
        'total': total_params,
        'quantum': quantum_params,
        'classical': classical_params,
        'reduction_pct': reduction_pct
    }

if __name__ == '__main__':
    print("Testing Hybrid QAPINN Framework...")
    model_pos1 = HybridQAPINN(n_qubits=4, position='first', measurement_mode='expectation')
    params_pos1 = count_parameters(model_pos1)
    print(f"Position 1 (Expectation 4 qubits): Total Params = {params_pos1['total']}, "
          f"Quantum = {params_pos1['quantum']}, Classical = {params_pos1['classical']}, "
          f"Reduction vs c-PINN = {params_pos1['reduction_pct']:.2f}%")
