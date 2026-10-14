import numpy as np
import scipy.integrate as integrate
import os

def cole_hopf_exact_point(x, t, nu=0.01):
    """
    Computes exact analytical solution of 1D Viscous Burgers' Equation:
    u_t + u * u_x = nu * u_xx
    Initial Condition: u(x, 0) = sin(pi * x) for x in [0, 1] (or x in [-1, 1])
    Boundary Condition: u(x_min, t) = u(x_max, t) = 0
    Using numerically stabilized Cole-Hopf Transformation with quadrature.
    """
    if t == 0.0:
        return np.sin(np.pi * x)
    
    if np.isclose(x, 0.0) or np.isclose(x, 1.0) or np.isclose(x, -1.0):
        return 0.0

    sigma = np.sqrt(2.0 * nu * t)
    eta_min = max(-2.0, x - 10.0 * sigma)
    eta_max = min(3.0, x + 10.0 * sigma)

    # 100-point numerical grid for log-stabilized quad evaluation
    etas = np.linspace(eta_min, eta_max, 500)
    psi = (1.0 - np.cos(np.pi * etas)) / np.pi
    exponents = -((x - etas) ** 2) / (4.0 * nu * t) - psi / (2.0 * nu)
    max_exp = np.max(exponents)

    def f_num(eta):
        p = (1.0 - np.cos(np.pi * eta)) / np.pi
        exp_val = -((x - eta) ** 2) / (4.0 * nu * t) - p / (2.0 * nu) - max_exp
        return ((x - eta) / t) * np.exp(exp_val)

    def f_den(eta):
        p = (1.0 - np.cos(np.pi * eta)) / np.pi
        exp_val = -((x - eta) ** 2) / (4.0 * nu * t) - p / (2.0 * nu) - max_exp
        return np.exp(exp_val)

    num, _ = integrate.quad(f_num, eta_min, eta_max, limit=100)
    den, _ = integrate.quad(f_den, eta_min, eta_max, limit=100)

    if abs(den) < 1e-15:
        return 0.0

    return num / den

def compute_exact_grid(x_flat, t_flat, nu=0.01):
    Nx = len(x_flat)
    Nt = len(t_flat)
    u_exact = np.zeros((Nt, Nx))

    for i, t in enumerate(t_flat):
        for j, x in enumerate(x_flat):
            u_exact[i, j] = cole_hopf_exact_point(x, t, nu=nu)

    return u_exact

if __name__ == '__main__':
    x_flat = np.linspace(0.0, 1.0, 256)
    t_flat = np.linspace(0.0, 1.0, 100)
    print(f"Calculating Cole-Hopf exact reference solution on grid {len(x_flat)}x{len(t_flat)}...")
    u_exact = compute_exact_grid(x_flat, t_flat, nu=0.01)
    
    os.makedirs('evaluation', exist_ok=True)
    np.save('evaluation/exact_solution.npy', u_exact)
    np.save('evaluation/x_grid.npy', x_flat)
    np.save('evaluation/t_grid.npy', t_flat)

    print("Successfully saved Cole-Hopf exact solution to evaluation/exact_solution.npy")
    print(f"Shape: {u_exact.shape}, Min: {u_exact.min():.5f}, Max: {u_exact.max():.5f}")
