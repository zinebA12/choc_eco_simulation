import numpy as np


def u_analytique (x , t , A =0.2 , x0 =0.0 , sigma =1.0 , D =1.0 , alpha =0.1) :
# Solution analytique avec condition initiale gaussienne . 

  denom = np.sqrt (1 + (4* D * t / sigma **2))
  exp_part = np . exp ( -( x - x0 ) **2 / ( sigma **2 + 4* D * t ) )
  return A / denom * exp_part * np . exp ( - alpha * t )


def u_volume_finie(A =0.2 , x0 =0.0 , sigma =1.0 , D =1.0 , alpha =0.1 , L=10, T=5, Nx=200, clf=0.45):
    """
    Résout l'équation de diffusion-amortissement par volumes finis.
    Retourne x_VF, t_VF, U_num pour pouvoir tracer ou comparer le résultat.
    """

    # --- Maillage spatial : centres des cellules -------------------------

    dx = 2 * L / Nx
    x_VF = -L + (np.arange(Nx) + 0.5) * dx

    # --- Pas de temps fixé par la condition de stabilité CFL --------------
    # dt = clf * dx² / D, avec clf ≤ 0.5 pour garantir r ≤ 1/2 (schéma stable)
 
    dt = clf * dx**2 / D
    Nt = int(T / dt) + 1

    max_Nt = 20000
    if Nt > max_Nt:
        raise ValueError(
            f"Nt={Nt} dépasse la limite autorisée ({max_Nt}). "
            "Augmente D, réduis Nx, ou réduis T/clf."
        )
    U_num = np.zeros((Nt, Nx))
    t_VF = np.linspace(0, T, Nt)

     # --- Condition initiale : choc gaussien centré en x0 -------------------
    U_num[0, :] = A * np.exp(-(x_VF - x0)**2 / sigma**2)

    # --- Nombre de Courant r = D*dt/dx² : pilote la stabilité du schéma ----
    r = D * dt / dx**2

    for i in range(Nt - 1):
        for j in range(1, Nx - 1):
            U_num[i+1, j] = U_num[i, j] + r * (U_num[i, j+1] - 2*U_num[i, j] + U_num[i, j-1]) - alpha * dt * U_num[i, j]
        # Conditions aux limites : on impose la solution exacte aux deux bords
        # (domaine supposé assez grand pour que u soit quasi nul en x = ±L)
        U_num[i+1, 0] = u_analytique(-L, t_VF[i+1], A, x0, sigma, D, alpha)
        U_num[i+1, -1] = u_analytique(L, t_VF[i+1], A, x0, sigma, D, alpha)

    return x_VF, t_VF, U_num
