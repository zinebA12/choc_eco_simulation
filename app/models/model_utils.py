import os
import time
from collections import OrderedDict

import numpy as np
import numpy.core.multiarray
import torch
import torch.nn as nn

from app.models import physics

torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

MLP_PATH = os.path.join(os.path.dirname(__file__), "mlp_reference.pth")
PINN_PATH = os.path.join(os.path.dirname(__file__), "pinn_reference.pth")


class ModelUnavailableError(Exception):
    """Levée quand un fichier de poids .pth est absent du serveur."""
    


class MLP(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PINN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x))


_mlp_cache = None
_pinn_cache = None


def load_mlp():
    """Charge le MLP pré-entraîné une seule fois, gardé en mémoire ensuite."""
    global _mlp_cache
    if _mlp_cache is None:
        if not os.path.exists(MLP_PATH):
            raise ModelUnavailableError(
                "Modèle MLP introuvable sur le serveur (mlp_reference.pth manquant)."
            )
        # weights_only=True : on ne charge QUE des tenseurs, jamais du code
        # exécutable. torch.load() sans cette option peut exécuter du code
        # arbitraire via pickle si le fichier est corrompu/malveillant.
        # On ne charge ici que des fichiers déposés nous-mêmes sur le
        # serveur -- JAMAIS un fichier .pth envoyé par un utilisateur.
        ckpt = torch.load(MLP_PATH, map_location="cpu", weights_only=True)
        model = MLP(hidden_dim=ckpt["hidden_dim"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _mlp_cache = (model, ckpt)
    return _mlp_cache


def load_pinn():
    global _pinn_cache
    if _pinn_cache is None:
        if not os.path.exists(PINN_PATH):
            raise ModelUnavailableError(
                "Modèle PINN introuvable sur le serveur (pinn_reference.pth manquant)."
            )
        ckpt = torch.load(PINN_PATH, map_location="cpu", weights_only=True)
        model = PINN(hidden_dim=ckpt["hidden_dim"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _pinn_cache = (model, ckpt)
    return _pinn_cache


def predict_mlp(x_grid, t_grid):
    """Retourne U(t, x) prédit par le MLP sur la grille donnée."""
    model, ckpt = load_mlp()
    X_mesh, T_mesh = np.meshgrid(x_grid, t_grid)
    x_flat, t_flat = X_mesh.ravel(), T_mesh.ravel()

    x_n = (x_flat - ckpt["x_mean"]) / ckpt["x_std"]
    t_n = (t_flat - ckpt["t_mean"]) / ckpt["t_std"]
    Xg = torch.tensor(np.column_stack([x_n, t_n]), dtype=torch.float32)

    with torch.no_grad():
        out = model(Xg).numpy().flatten()

    u_log = out * ckpt["u_std"] + ckpt["u_mean"]
    u = np.exp(u_log) - ckpt["eps"]
    return u.reshape(X_mesh.shape)


def predict_pinn(x_grid, t_grid):
    """Retourne (U prédit, D_entrainement, alpha_entrainement).
    Le PINN ignore le D/alpha du formulaire : il ne connaît que celui
    utilisé lors de son entraînement (stocké dans le checkpoint)."""
    model, ckpt = load_pinn()
    X_mesh, T_mesh = np.meshgrid(x_grid, t_grid)
    Xg = torch.tensor(
        np.column_stack([X_mesh.ravel(), T_mesh.ravel()]), dtype=torch.float32
    )
    with torch.no_grad():
        out = model(Xg).numpy().flatten()
    return out.reshape(X_mesh.shape), ckpt["D"], ckpt["alpha"]

_pinn_scenario_cache: "OrderedDict[tuple, PINN]" = OrderedDict()
_PINN_CACHE_MAX = 10  # nb max de PINN "à la volée" gardés en mémoire (borne la RAM)


def _pinn_residual(model, x, t, D, alpha):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u_pred = model(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - D * u_xx + alpha * u_pred


def train_pinn_live(A, x0, sigma, D, alpha, L, T,
                     max_epochs=5000, time_budget_s=30,
                     adam_time_fraction=0.6,   # 60% du budget pour Adam, 40% pour L-BFGS
                     lr=1e-3, lambda_pde=8.0, N_ic=300, N_bc=300, N_r=1000):
    """
    Entraîne un PINN pour le scénario demandé : phase Adam (exploration)
    suivie d'un fine-tuning L-BFGS (raffinement), comme train_pinn_scenario
    dans le notebook (cellules 87-88 + 69). Le budget temps est réparti
    explicitement entre les deux phases pour garantir que L-BFGS
    s'exécute réellement, plutôt que de dépendre du temps restant après Adam.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    model = PINN(hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N_ic_focus = int(0.5 * N_ic)
    N_ic_unif = N_ic - N_ic_focus
    x_ic_unif = np.random.uniform(-L, L, N_ic_unif)
    x_ic_focus = np.clip(np.random.normal(loc=x0, scale=sigma, size=N_ic_focus), -L, L)
    x_ic = np.concatenate([x_ic_unif, x_ic_focus])
    t_ic = np.zeros(N_ic)
    u_ic = physics.u_analytique(x_ic, t_ic, A, x0, sigma, D, alpha)

    t_bc = np.random.uniform(0, T, N_bc)
    x_bc = np.where(np.arange(N_bc) < N_bc // 2, -L, L).astype(float)
    u_bc = physics.u_analytique(x_bc, t_bc, A, x0, sigma, D, alpha)

    x_data = torch.tensor(np.concatenate([x_ic, x_bc]), dtype=torch.float32).view(-1, 1)
    t_data = torch.tensor(np.concatenate([t_ic, t_bc]), dtype=torch.float32).view(-1, 1)
    u_data = torch.tensor(np.concatenate([u_ic, u_bc]), dtype=torch.float32).view(-1, 1)
    X_data = torch.cat([x_data, t_data], dim=1)

    adam_deadline = time_budget_s * adam_time_fraction
    total_start = time.monotonic()

    # --- Phase 1 : Adam (exploration), bornée à adam_deadline -----------
    for _ in range(max_epochs):
        if time.monotonic() - total_start > adam_deadline:
            break

        optimizer.zero_grad()
        x_r = torch.tensor(np.random.uniform(-L, L, N_r), dtype=torch.float32).view(-1, 1)
        t_r = torch.tensor(np.random.uniform(0, T, N_r), dtype=torch.float32).view(-1, 1)

        loss_data = nn.MSELoss()(model(X_data), u_data)
        residual = _pinn_residual(model, x_r, t_r, D, alpha)
        loss = loss_data + lambda_pde * torch.mean(residual ** 2)
        loss.backward()
        optimizer.step()

    # --- Phase 2 : L-BFGS (raffinement), avec le budget restant garanti --
    remaining = time_budget_s - (time.monotonic() - total_start)
    if remaining > 0.5:
        x_r_fixed = torch.tensor(np.random.uniform(-L, L, N_r), dtype=torch.float32).view(-1, 1)
        t_r_fixed = torch.tensor(np.random.uniform(0, T, N_r), dtype=torch.float32).view(-1, 1)

        # max_iter dimensionné large (500, comme le notebook) : la vraie
        # limite ici est le temps global, pas ce compteur.
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=500,
            history_size=50, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            loss_data = nn.MSELoss()(model(X_data), u_data)
            residual = _pinn_residual(model, x_r_fixed, t_r_fixed, D, alpha)
            loss = loss_data + lambda_pde * torch.mean(residual ** 2)
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure)

    model.eval()
    return model


def predict_pinn_live(x_grid, t_grid, A, x0, sigma, D, alpha, L, T):
    """PINN entraîné pour CE scénario, avec cache borné (LRU) pour éviter
    de ré-entraîner à chaque requête si les mêmes paramètres reviennent."""
    key = (round(A, 6), round(x0, 6), round(sigma, 6), round(D, 6),
           round(alpha, 6), round(L, 6), round(T, 6))

    if key in _pinn_scenario_cache:
        _pinn_scenario_cache.move_to_end(key)
        model = _pinn_scenario_cache[key]
    else:
        model = train_pinn_live(A, x0, sigma, D, alpha, L, T)
        _pinn_scenario_cache[key] = model
        if len(_pinn_scenario_cache) > _PINN_CACHE_MAX:
            _pinn_scenario_cache.popitem(last=False)  # évince le plus ancien scénario

    X_mesh, T_mesh = np.meshgrid(x_grid, t_grid)
    Xg = torch.tensor(np.column_stack([X_mesh.ravel(), T_mesh.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        out = model(Xg).numpy().flatten()
    return out.reshape(X_mesh.shape)