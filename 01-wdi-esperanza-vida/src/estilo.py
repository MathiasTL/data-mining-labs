"""Estilo visual compartido por todas las figuras del informe.

Aplica las reglas de la guía de visualización: marcas finas, reja recesiva,
sin bordes superfluos, tipografía legible y texto siempre en tinta neutra
(nunca del color de la serie).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib as mpl
import matplotlib.pyplot as plt

from config import COLORES, DIR_FIG, DPI, PALETA_REGIONES


def aplicar_estilo() -> None:
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.facecolor": COLORES["superficie"],
        "axes.facecolor": COLORES["superficie"],
        "savefig.facecolor": COLORES["superficie"],
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlepad": 12,
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": COLORES["tinta_sec"],
        "axes.edgecolor": COLORES["reja"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORES["reja"],
        "grid.linewidth": 0.7,
        "text.color": COLORES["tinta"],
        "xtick.color": COLORES["tinta_sec"],
        "ytick.color": COLORES["tinta_sec"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.prop_cycle": mpl.cycler(color=PALETA_REGIONES),
    })


def colores_por_region(regiones) -> dict:
    """Asigna un color fijo a cada región por orden alfabético.

    La asignación depende de la entidad, no del subconjunto graficado: filtrar
    regiones nunca repinta a las que quedan.
    """
    return {r: PALETA_REGIONES[i % len(PALETA_REGIONES)]
            for i, r in enumerate(sorted(regiones))}


def titular(ax, titulo: str, subtitulo: str | None = None) -> None:
    """Título en dos niveles: la afirmación arriba, el detalle en tinta suave."""
    if subtitulo:
        ax.set_title(f"{titulo}\n", loc="left")
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9.5,
                color=COLORES["tinta_sec"], va="bottom")
    else:
        ax.set_title(titulo, loc="left")


def pie_fuente(fig, texto: str = "Fuente: World Development Indicators, Banco Mundial.") -> None:
    fig.text(0.005, -0.02, texto, fontsize=8, color=COLORES["tinta_tenue"], ha="left")


def guardar_fig(fig, nombre: str) -> Path:
    ruta = DIR_FIG / f"{nombre}.png"
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor=COLORES["superficie"])
    plt.close(fig)
    kb = ruta.stat().st_size / 1024
    print(f"  → {ruta.name} ({kb:.0f} KB)")
    return ruta


def etiquetas_finales(ax, puntos, sep_min_frac: float = 0.035, dx: float = 0.3) -> None:
    """Escribe etiquetas al final de cada línea evitando que se solapen.

    `puntos` es una lista de (x, y, texto, negrita). Las etiquetas se separan
    verticalmente lo mínimo necesario para que ninguna pise a otra, respetando
    el orden de los valores reales.
    """
    if not puntos:
        return
    y0, y1 = ax.get_ylim()
    sep = (y1 - y0) * sep_min_frac
    puntos = sorted(puntos, key=lambda p: p[1])
    ys = [p[1] for p in puntos]
    # Empuja hacia arriba cualquier etiqueta demasiado cerca de la anterior
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < sep:
            ys[i] = ys[i - 1] + sep
    # Si el empuje se salió del eje, corrige bajando todo el bloque
    desborde = ys[-1] - (y1 - sep * 0.3)
    if desborde > 0:
        ys = [y - desborde for y in ys]
    for (x, _, texto, negrita), y in zip(puntos, ys):
        ax.text(x + dx, y, texto, fontsize=8.5, va="center",
                color=COLORES["tinta_sec"],
                fontweight="semibold" if negrita else "normal")
