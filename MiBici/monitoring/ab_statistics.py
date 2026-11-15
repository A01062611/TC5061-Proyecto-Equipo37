"""Pruebas estadísticas para la comparación A/B de P2 MiBici."""

import numpy as np
from scipy.stats import chi2


def f2_desde_matriz(matriz):
    """F2 macro de las clases 1 y 2 desde matrices 3x3."""
    matriz = np.asarray(matriz, dtype=float)

    tp = np.diagonal(matriz, axis1=-2, axis2=-1)
    fn = matriz.sum(axis=-1) - tp
    fp = matriz.sum(axis=-2) - tp

    numerador = 5.0 * tp
    denominador = numerador + 4.0 * fn + fp

    f2 = np.divide(
        numerador,
        denominador,
        out=np.zeros_like(numerador),
        where=denominador > 0,
    )

    return f2[..., 1:3].mean(axis=-1)


def bootstrap_pareado(
    y,
    pred_champion,
    pred_challenger,
    estaciones,
    n_bootstrap=2000,
    seed=42,
):
    """IC percentil 95% de delta F2, remuestreando estaciones."""
    y = np.asarray(y, dtype=int)
    pred_champion = np.asarray(pred_champion, dtype=int)
    pred_challenger = np.asarray(pred_challenger, dtype=int)
    estaciones = np.asarray(estaciones)

    n = len(y)

    if not (
        n == len(pred_champion)
        == len(pred_challenger)
        == len(estaciones)
    ):
        raise ValueError("Los vectores deben tener igual longitud")

    if n == 0 or n_bootstrap < 2:
        raise ValueError("Datos o réplicas insuficientes")

    for vector in (y, pred_champion, pred_challenger):
        if not np.isin(vector, [0, 1, 2]).all():
            raise ValueError("Las clases deben ser 0, 1 o 2")

    ids, indices = np.unique(estaciones, return_inverse=True)
    n_estaciones = len(ids)

    conteos = np.zeros(
        (n_estaciones, 2, 3, 3),
        dtype=np.int64,
    )

    np.add.at(
        conteos,
        (indices, 0, y, pred_champion),
        1,
    )
    np.add.at(
        conteos,
        (indices, 1, y, pred_challenger),
        1,
    )

    totales = conteos.sum(axis=0)
    f2_original = f2_desde_matriz(totales)

    delta_original = float(
        f2_original[1] - f2_original[0]
    )

    rng = np.random.default_rng(seed)
    diferencias = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        seleccion = rng.integers(
            0, n_estaciones, size=n_estaciones
        )

        matrices = conteos[seleccion].sum(axis=0)
        valores = f2_desde_matriz(matrices)

        diferencias[i] = valores[1] - valores[0]

    ic_inf, ic_sup = np.quantile(
        diferencias, [0.025, 0.975]
    )

    return {
        "n_estaciones": int(n_estaciones),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "f2_champion": float(f2_original[0]),
        "f2_challenger": float(f2_original[1]),
        "delta_f2": delta_original,
        "ic95_inferior": float(ic_inf),
        "ic95_superior": float(ic_sup),
    }


def prueba_mcnemar(y, pred_champion, pred_challenger):
    """McNemar asintótica con corrección de continuidad."""
    y = np.asarray(y)
    a = np.asarray(pred_champion)
    b = np.asarray(pred_challenger)

    if len(y) == 0 or len(y) != len(a) or len(y) != len(b):
        raise ValueError("Vectores vacíos o de distinta longitud")

    correcto_a = a == y
    correcto_b = b == y

    solo_champion = int(np.sum(correcto_a & ~correcto_b))
    solo_challenger = int(np.sum(~correcto_a & correcto_b))
    discordantes = solo_champion + solo_challenger

    if discordantes == 0:
        estadistico = 0.0
        p_valor = 1.0
    else:
        estadistico = (
            (abs(solo_champion - solo_challenger) - 1) ** 2
            / discordantes
        )
        p_valor = float(chi2.sf(estadistico, df=1))

    return {
        "solo_champion_acierta": solo_champion,
        "solo_challenger_acierta": solo_challenger,
        "discordantes": discordantes,
        "estadistico": float(estadistico),
        "p_valor": p_valor,
        "metodo": "McNemar con correccion de continuidad",
    }
