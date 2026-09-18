import numpy as np
from flask import Blueprint, render_template, request

from app.models import model_utils, physics, plotting
from app.models.validation import (
    BOUNDS,
    ValidationError,
    validate_method,
    validate_params,
)

simulation_bp = Blueprint(
    "simulation", __name__,
    template_folder="../views/templates",
    static_folder="../views/static",
)

DEFAULT_PARAMS = {name: bounds[2] for name, bounds in BOUNDS.items()}


@simulation_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html", params=DEFAULT_PARAMS, error=None, plot_url=None)


@simulation_bp.route("/simulate", methods=["POST"])
def simulate():
    submitted = {name: request.form.get(name, default) for name, (_, _, default) in BOUNDS.items()}
    submitted["method"] = request.form.get("method", "vf")

    try:
        method = validate_method(submitted["method"])
        p = validate_params(request.form)
    except ValidationError as e:
        return render_template("index.html", params=submitted, error=str(e), plot_url=None), 400

    if method == "vf":
        try:
            x_grid, t_grid, U_pred = physics.u_volume_finie(
                A=p["A"], x0=p["x0"], sigma=p["sigma"], D=p["D"], alpha=p["alpha"],
                L=p["L"], T=p["T"], Nx=p["Nx"], clf=p["clf"],
            )
        except ValueError as e:
            return render_template("index.html", params=submitted, error=str(e), plot_url=None), 400
        label = "Volumes finis"
        U_ana = physics.u_analytique(x_grid[None, :], t_grid[:, None],
                                      p["A"], p["x0"], p["sigma"], p["D"], p["alpha"])
        plot_url = plotting.make_plot(x_grid, t_grid, U_pred, U_ana, label)
        return render_template("index.html", params=p, error=None, plot_url=plot_url, selected_method=method)

    elif method == "mlp":
        try:
            x_grid = np.linspace(-p["L"], p["L"], p["Nx"])
            t_grid = np.linspace(0, p["T"], 100)
            U_pred = model_utils.predict_mlp(x_grid, t_grid)
        except model_utils.ModelUnavailableError as e:
            return render_template("index.html", params=submitted, error=str(e), plot_url=None), 400
        label = "MLP (pré-entraîné)"
        U_ana = physics.u_analytique(x_grid[None, :], t_grid[:, None],
                                      p["A"], p["x0"], p["sigma"], p["D"], p["alpha"])
        plot_url = plotting.make_plot(x_grid, t_grid, U_pred, U_ana, label)
        return render_template("index.html", params=p, error=None, plot_url=plot_url, selected_method=method)

    elif method == "pinn":
        x_grid = np.linspace(-p["L"], p["L"], p["Nx"])
        t_grid = np.linspace(0, p["T"], 100)
        U_pred = _resolve_pinn(p, x_grid, t_grid)
        label = "PINN"
        U_ana = physics.u_analytique(x_grid[None, :], t_grid[:, None],
                                      p["A"], p["x0"], p["sigma"], p["D"], p["alpha"])
        plot_url = plotting.make_plot(x_grid, t_grid, U_pred, U_ana, label)
        return render_template("index.html", params=p, error=None, plot_url=plot_url, selected_method=method)

    else:  # method == "compare"
        try:
            x_grid, t_vf, U_vf_full = physics.u_volume_finie(
                A=p["A"], x0=p["x0"], sigma=p["sigma"], D=p["D"], alpha=p["alpha"],
                L=p["L"], T=p["T"], Nx=p["Nx"], clf=p["clf"],
            )
        except ValueError as e:
            return render_template("index.html", params=submitted, error=str(e), plot_url=None), 400

        fractions = [0, 0.04, 0.2, 0.4, 1.0]
        indices = sorted({min(int(f * (t_vf.size - 1)), t_vf.size - 1) for f in fractions})
        t_snap = t_vf[indices]
        U_vf_snap = U_vf_full[indices]

        mlp_available = True
        try:
            U_mlp_snap = model_utils.predict_mlp(x_grid, t_snap)
        except model_utils.ModelUnavailableError:
            mlp_available = False
            U_mlp_snap = None

        U_pinn_snap = _resolve_pinn(p, x_grid, t_snap)
        U_ana_snap = physics.u_analytique(x_grid[None, :], t_snap[:, None],
                                           p["A"], p["x0"], p["sigma"], p["D"], p["alpha"])

        plot_url = plotting.make_comparison_plot(x_grid, t_snap, U_ana_snap, U_vf_snap,
                                                   U_mlp_snap, U_pinn_snap, p, mlp_available)
        warning = None if mlp_available else "Modèle MLP indisponible : comparaison affichée sans le MLP."
        return render_template("index.html", params=p, error=warning, plot_url=plot_url, selected_method=method)


def _resolve_pinn(p, x_grid, t_grid):
    """Retourne les prédictions PINN : modèle pré-entraîné si le scénario
    correspond à son régime, sinon PINN entraîné à la volée (mis en cache)."""
    try:
        _, ckpt = model_utils.load_pinn()
        D_ref, alpha_ref = ckpt["D"], ckpt["alpha"]
    except model_utils.ModelUnavailableError:
        D_ref = alpha_ref = None

    same_regime = (
        D_ref is not None
        and abs(D_ref - p["D"]) < 1e-9
        and abs(alpha_ref - p["alpha"]) < 1e-9
    )
    if same_regime:
        U_pred, _, _ = model_utils.predict_pinn(x_grid, t_grid)
    else:
        U_pred = model_utils.predict_pinn_live(
            x_grid, t_grid, p["A"], p["x0"], p["sigma"], p["D"], p["alpha"], p["L"], p["T"]
        )
    return U_pred