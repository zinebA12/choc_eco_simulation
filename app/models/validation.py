class ValidationError(Exception):
    """Erreur levée quand un paramètre utilisateur est invalide."""



# Bornes : (min, max, valeur_par_défaut)
BOUNDS = {
    "A":     (-10.0, 10.0, 0.2),
    "x0":    (-50.0, 50.0, 0.0),
    "sigma": (1e-3, 50.0, 1.0),     # > 0 strictement, jamais 0
    "D":     (1e-3, 100.0, 1.0),    # > 0 strictement, jamais 0
    "alpha": (0.0, 20.0, 0.1),
    "L":     (0.5, 100.0, 10.0),
    "T":     (0.01, 50.0, 5.0),
    "Nx":    (10, 2000, 200),       # entier
    "clf":   (0.01, 0.5, 0.45),     # 0.5 = limite de stabilité CFL
}


def _parse_float(raw, name, lo, hi):
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValidationError(f"'{name}' doit être un nombre (reçu: {raw!r}).")
    if not (lo <= val <= hi):
        raise ValidationError(f"'{name}' doit être compris entre {lo} et {hi} (reçu: {val}).")
    return val


def _parse_int(raw, name, lo, hi):
    try:
        val = int(float(raw))  # tolère "200" ou "200.0"
    except (TypeError, ValueError):
        raise ValidationError(f"'{name}' doit être un entier (reçu: {raw!r}).")
    if not (lo <= val <= hi):
        raise ValidationError(f"'{name}' doit être compris entre {lo} et {hi} (reçu: {val}).")
    return val


def validate_params(form):
    """
    form : dict-like (ex. request.form) contenant des chaînes de caractères.
    Retourne un dict de floats/int propres, prêts pour physics.py.
    Lève ValidationError si un champ est absent, non numérique, ou hors bornes.
    """
    params = {}
    for name, (lo, hi, default) in BOUNDS.items():
        raw = form.get(name, default)
        if name == "Nx":
            params[name] = _parse_int(raw, name, lo, hi)
        else:
            params[name] = _parse_float(raw, name, lo, hi)
    return params

VALID_METHODS = {"vf", "mlp", "pinn", "compare"}


def validate_method(raw):
    if raw not in VALID_METHODS:
        raise ValidationError(f"Méthode invalide (reçu: {raw!r}).")
    return raw