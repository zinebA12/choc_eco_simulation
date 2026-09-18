import base64
import io

import matplotlib

matplotlib.use("Agg")  # backend non interactif, indispensable côté serveur
import matplotlib.pyplot as plt


def make_plot(x_grid, t_grid, U_pred, U_ana, label):
    """Trace la comparaison prédiction vs solution analytique,
    et retourne l'image encodée en base64 (rien n'est écrit sur disque)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fractions = [0, 1/3, 2/3, 1]
    indices = sorted({int(f * (t_grid.size - 1)) for f in fractions})
    for idx in indices:
        ax.plot(x_grid, U_pred[idx], label=f"{label}, t={t_grid[idx]:.2f}")
        ax.plot(x_grid, U_ana[idx], "--", color="black", alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"{label} vs solution analytique (pointillés)")
    ax.legend(fontsize=8)

    return _fig_to_base64(fig)


def make_comparison_plot(x_grid, t_snap, U_ana, U_vf, U_mlp, U_pinn, p, mlp_available):
    """Trace les 4 courbes (analytique, VF, MLP, PINN) superposées,
    un panneau par instant représentatif. Styles de trait distincts
    pour rester lisibles même quand les courbes coïncident presque."""
    n = len(t_snap)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, range(n)):
        ax.plot(x_grid, U_vf[idx], color="red", linestyle=(0, (6, 3)),
                 linewidth=3, alpha=0.9, label="Volumes finis", zorder=1)
        ax.plot(x_grid, U_ana[idx], color="black", linestyle="-",
                 linewidth=1.6, alpha=0.9, label="Analytique", zorder=2)
        if mlp_available:
            ax.plot(x_grid, U_mlp[idx], color="royalblue", linestyle=(0, (6, 3)),
                     linewidth=2.4, alpha=0.9, label="MLP", zorder=3)
        ax.plot(x_grid, U_pinn[idx], color="seagreen", linestyle=(0, (6, 3)),
                 linewidth=2, alpha=0.9, label="PINN", zorder=4)

        ax.set_title(f"t = {t_snap[idx]:.2f}", fontsize=13)
        ax.set_xlabel("x", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("u(x,t)", fontsize=11)
    axes[0].legend(fontsize=10, loc="upper right")

    fig.suptitle(f"Comparaison des méthodes — D={p['D']}, α={p['alpha']}, σ={p['sigma']}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return _fig_to_base64(fig, dpi=130)


def _fig_to_base64(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)  # libère la mémoire à chaque requête
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")