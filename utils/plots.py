import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.palette import BG, ACCENT, CMAP, INK as _INK

CHARTS = ["Line plot", "Scatter plot", "Bar chart", "Area chart", "Histogram", "Box plot",
          "Violin plot", "Density plot", "Pie chart", "Count plot", "Heatmap (correlation)"]

NEEDS_XY = {"Line plot", "Scatter plot", "Bar chart", "Area chart"}
NEEDS_ONE = {"Histogram", "Box plot", "Violin plot", "Density plot", "Pie chart", "Count plot"}


PALETTES = {
   
    "teal": ["#176B78", "#243B53", "#2A9D8F", "#D97706", "#6D4C9A", "#C44536"],
    "deep": ["#176B78", "#243B53", "#2A9D8F", "#D97706", "#6D4C9A", "#C44536"],
}
INK = _INK
CHART_BG = BG   # chart background = page colour


def apply_limits(ax, xlim, ylim):
    if xlim[0] is not None or xlim[1] is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim[0] is not None or ylim[1] is not None:
        ax.set_ylim(ylim[0], ylim[1])


def make_chart(df, kind, x=None, ys=None, col=None, heat_cols=None, title="", xlim=(None, None),
               ylim=(None, None), grid=True, palette="deep", size=(10, 5.5)):
    sns.set_theme(style="whitegrid" if grid else "white", palette=PALETTES.get(palette, palette),
                  rc={"figure.facecolor": "#F8FBFC", "axes.facecolor": "#F8FBFC", "savefig.facecolor": "#F8FBFC",
                      "axes.edgecolor": "#243B53", "grid.color": "#C5D2D9", "text.color": "#20353C",
                      "axes.labelcolor": "#20353C", "xtick.color": "#20353C", "ytick.color": "#20353C"})
    fig, ax = plt.subplots(figsize=size)
    ys = ys or []

    if kind == "Line plot":
        for y in ys:
            sns.lineplot(data=df, x=x, y=y, ax=ax, label=y)
    elif kind == "Scatter plot":
        for y in ys:
            sns.scatterplot(data=df, x=x, y=y, ax=ax, label=y)
    elif kind == "Bar chart":
        if len(df) > 60:
            raise ValueError("Bar chart supports up to 60 rows - narrow the row range.")
        long = df.melt(id_vars=x, value_vars=ys, var_name="series", value_name="value")
        sns.barplot(data=long, x=x, y="value", hue="series" if len(ys) > 1 else None, ax=ax)
        ax.set_ylabel(", ".join(ys))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    elif kind == "Area chart":
        d = df.sort_values(x)
        ax.stackplot(d[x], *[d[y] for y in ys], labels=ys, alpha=.75)
        ax.set_xlabel(x)
    elif kind == "Histogram":
        sns.histplot(data=df, x=col, kde=True, ax=ax, color="#176B78", edgecolor="#FFFFFF", alpha=.78, line_kws={"color": "#243B53", "linewidth": 2.5})
    elif kind == "Box plot":
        sns.boxplot(data=df, y=col, ax=ax, color="#2A9D8F", saturation=.85)
    elif kind == "Violin plot":
        sns.violinplot(data=df, y=col, ax=ax, color="#2A9D8F", saturation=.85)
    elif kind == "Density plot":
        sns.kdeplot(data=df, x=col, fill=True, ax=ax, color="#176B78", alpha=.35, linewidth=2.5)
    elif kind == "Pie chart":
        counts = df[col].astype(str).value_counts().head(10)
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    elif kind == "Count plot":
        order = df[col].value_counts().head(30).index
        sns.countplot(data=df, x=col, order=order, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    elif kind == "Heatmap (correlation)":
        sns.heatmap(df[heat_cols].corr(), annot=True, fmt=".2f", cmap=CMAP, vmin=-1, vmax=1, ax=ax)
    else:
        raise ValueError("Unknown chart type.")

    if kind not in {"Pie chart", "Heatmap (correlation)"}:
        apply_limits(ax, xlim, ylim)
    if len(ys) > 1 and kind in NEEDS_XY:
        ax.legend()
    elif ax.get_legend() is not None and len(ys) <= 1:
        ax.get_legend().remove()
    if title.strip():
        ax.set_title(title.strip(), fontsize=14, weight="bold", color=INK)
    fig.tight_layout()
    return fig


def fig_bytes(fig, fmt="png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    return buf.getvalue()


MIME = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}


def density_figure(values, x, normal, kde, column, method, mu, sigma, title=""):
    
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": "#F8FBFC", "axes.facecolor": "#F8FBFC",
        "savefig.facecolor": "#F8FBFC", "axes.edgecolor": "#243B53",
        "grid.color": "#C5D2D9", "text.color": "#20353C",
        "axes.labelcolor": "#20353C", "xtick.color": "#20353C", "ytick.color": "#20353C"
    })
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if method in ("Kernel density estimate", "Both") and np.isfinite(kde).any():
        ax.plot(x, kde, lw=2.8, label="Kernel density estimate", color="#168A8A", zorder=4)
        ax.fill_between(x, kde, alpha=.20, color="#168A8A")
    if method in ("Normal distribution PDF", "Both") and np.isfinite(normal).any():
        ax.plot(x, normal, lw=3.0, ls="--", color="#243B53", label=f"Normal (μ={mu:.3g}, σ={sigma:.3g})", zorder=5)
    ax.set_xlabel(column, fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(title.strip() or f"Probability density: {column}", fontsize=14, weight="bold", color="#20353C", pad=12)
    ax.grid(True, alpha=.35, linewidth=.8)
    ax.tick_params(colors="#20353C")
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#C5D2D9")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
