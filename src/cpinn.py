import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*requires_grad=True.*")

class ClassicalPINN(nn.Module):
    """
    Classical Physics-Informed Neural Network (c-PINN) for 1D Viscous Burgers' Equation.
    Architecture: 2 inputs (x, t) -> 3 hidden layers (40 neurons each, Tanh) -> 1 output u_hat.
    Total trainable parameters = 3,441.
    """
    def __init__(self, in_features=2, hidden_features=40, out_features=1, num_layers=3):
        super(ClassicalPINN, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_t):
        return self.net(x_t)

class CompressedPINN(nn.Module):
    """
    Compressed Classical c-PINN for 1D Viscous Burgers' Equation Ablation Control.
    Hidden layer dimensions: Input 2 -> Dense(189) -> Dense(7) -> Output(1).
    Total trainable parameters = 567 + 1330 + 8 = 1,905 (EXACT MATCH to QAPINN baseline!).
    """
    def __init__(self, in_features=2, h1=189, h2=7, out_features=1):
        super(CompressedPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, out_features)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_t):
        return self.net(x_t)

def compute_pde_residual(model, x_f, t_f, nu=0.01):
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

def compute_composite_loss(model, X_f, x_f, t_f, X_i, u_i, X_b, u_b, nu=0.01, weights=(1.0, 10.0, 10.0)):
    w_f, w_i, w_b = weights
    
    residual, _ = compute_pde_residual(model, x_f, t_f, nu=nu)
    loss_f = torch.mean(residual ** 2)

    u_i_pred = model(X_i)
    loss_i = torch.mean((u_i_pred - u_i) ** 2)

    u_b_pred = model(X_b)
    loss_b = torch.mean((u_b_pred - u_b) ** 2)

    total_loss = w_f * loss_f + w_i * loss_i + w_b * loss_b
    return total_loss, loss_f, loss_i, loss_b

if __name__ == '__main__':
    model = ClassicalPINN()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Classical PINN: {num_params} params")

    cmodel = CompressedPINN()
    cnum_params = sum(p.numel() for p in cmodel.parameters() if p.requires_grad)
    print(f"Compressed PINN: {cnum_params} params (Matches QAPINN baseline 1905 exactly!)")
