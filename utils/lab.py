"""Probability distributions and regression models.

Everything is calculated from the columns of the data the user has loaded.
"""
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats, optimize

from utils.palette import BG, ACCENT, INK, CMAP


C = {"blue": "#176B78", "pink": "#243B53", "teal": "#168A8A", "amber": "#D97706", "red": "#C44536",
     "slate": "#526777", "violet": "#2A7F8E"}


# ---- Small helpers shared by the sections below ----
def _style(ax, title="", xl="", yl=""):
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel(xl, fontsize=10, color=INK); ax.set_ylabel(yl, fontsize=10, color=INK)
    ax.tick_params(labelsize=9, colors=INK)
    ax.grid(alpha=.35, color="#B8C8D0", linewidth=.8); ax.set_facecolor("#F8FBFC")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontsize(9)


def _show(fig, width=760):
    fig.patch.set_facecolor("#F8FBFC")
    fig.tight_layout(pad=1.1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    st.image(buf.getvalue(), width=width)


def _formula(text):
    st.markdown(f'<div class="formula-card">{text}</div>', unsafe_allow_html=True)


def _metrics(items, per_row=5):
    items = list(items.items())
    for i in range(0, len(items), per_row):
        chunk = items[i:i + per_row]
        for c, (n, v) in zip(st.columns(len(chunk)), chunk):
            c.metric(n, v if isinstance(v, str) else ("-" if pd.isna(v) else f"{v:.4g}"))


def _col(sel, num_cols, k, tag, label="Column"):
    c = st.selectbox(label, num_cols, index=None, placeholder="Choose a column", key=k(tag))
    if not c:
        st.info("Choose a column from your data.")
        return None
    s = pd.to_numeric(sel[c], errors="coerce").dropna().astype(float)
    if len(s) < 3:
        st.warning("At least 3 numeric values are needed in this column.")
        return None
    return c, s


def _section(title):
    st.subheader(title)


def err_metrics(y, yhat, n_features=1):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    res = y - yhat
    sse, sst = float((res ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    n = len(y)
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return {"MSE": sse / n, "RMSE": np.sqrt(sse / n), "MAE": float(np.abs(res).mean()), "R²": r2}


def _split(n, frac=0.7, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    cut = max(2, int(n * frac))
    return idx[:cut], idx[cut:]


def _pair(sel, num_cols, k, tag):
    a, b = st.columns(2)
    x = a.selectbox("X (independent) column", num_cols, index=None,
                    placeholder="Choose X", key=k(f"{tag}_x"))
    y = b.selectbox("Y (dependent) column", num_cols, index=None,
                    placeholder="Choose Y", key=k(f"{tag}_y"))
    if not x or not y:
        st.info("Choose X and Y columns from your data.")
        return None
    if x == y:
        st.warning("X and Y must be different columns.")
        return None
    d = sel[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 5:
        st.warning("At least 5 complete rows are needed.")
        return None
    return x, y, d


# ---- Normal distribution ----
def normal_section(sel, num_cols, k):
    if not num_cols:
        st.warning("The selected data has no numeric columns."); return
    got = _col(sel, num_cols, k, "nd_col")
    if not got: return
    name, s = got
    x = s.to_numpy(); mu, sd = x.mean(), x.std(ddof=1)
    if sd == 0:
        st.warning("All values are identical (σ = 0)."); return
    _metrics({"Count": f"{len(x):,}", "Mean (μ)": mu, "Std deviation (σ)": sd}, 3)
    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    ax.hist(x, bins="auto", density=True, color="#2A7F8E", alpha=.72, edgecolor="#FFFFFF", linewidth=1.0, zorder=2)
    g = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
    ax.plot(g, stats.norm.pdf(g, mu, sd), color="#173B57", lw=3.0, label="Fitted normal curve", zorder=4)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#C5D2D9")
    _style(ax, f"Normal curve for {name}", name, "density")
    _show(fig)
    st.markdown("**Probability between two values**")
    a, b = st.columns(2)
    lo = a.number_input("Lower value a", value=None, placeholder="type a value", key=k("nd_a"))
    hi = b.number_input("Upper value b", value=None, placeholder="type a value", key=k("nd_b"))
    if lo is not None and hi is not None:
        if lo >= hi:
            st.warning("Lower value must be smaller than upper value.")
        else:
            _metrics({"Probability (normal curve)": stats.norm.cdf(hi, mu, sd) - stats.norm.cdf(lo, mu, sd),
                      "Probability (actual data)": float(np.mean((x >= lo) & (x <= hi)))}, 2)


# ---- Probability ----
def probability_section(sel, num_cols, k):
    if not num_cols:
        st.warning("The selected data has no numeric columns."); return
    a, b = st.columns(2)
    c1 = a.selectbox("Column for event A", num_cols, index=None, placeholder="Choose column", key=k("pr_c1"))
    o1 = b.selectbox("Condition for A", [">", "≥", "<", "≤", "="], index=None,
                     placeholder="Choose condition", key=k("pr_o1"))
    v1 = st.number_input("Value for A", value=None, placeholder="type a value", key=k("pr_v1"))
    if not (c1 and o1 and v1 is not None):
        st.info("Choose a column, a condition and a value for event A."); return
    ops = {">": np.greater, "≥": np.greater_equal, "<": np.less, "≤": np.less_equal, "=": np.equal}
    x1 = pd.to_numeric(sel[c1], errors="coerce")
    A = ops[o1](x1, v1) & x1.notna()
    _metrics({"Rows": f"{len(sel):,}", "Rows with A": f"{int(A.sum()):,}", "P(A)": float(A.mean())}, 3)
    st.markdown("**Optional - a second event B**")
    a, b = st.columns(2)
    c2 = a.selectbox("Column for event B", [c for c in num_cols if c != c1], index=None,
                     placeholder="Choose column", key=k("pr_c2"))
    o2 = b.selectbox("Condition for B", [">", "≥", "<", "≤", "="], index=None,
                     placeholder="Choose condition", key=k("pr_o2"))
    v2 = st.number_input("Value for B", value=None, placeholder="type a value", key=k("pr_v2"))
    if c2 and o2 and v2 is not None:
        x2 = pd.to_numeric(sel[c2], errors="coerce")
        B = ops[o2](x2, v2) & x2.notna()
        pa, pb, pab = A.mean(), B.mean(), (A & B).mean()
        _metrics({"P(B)": pb, "P(A and B)": pab,
                  "P(A given B)": pab / pb if pb else np.nan}, 4)


# ---- Binomial distribution ----
def binomial_section(sel, num_cols, k):
    if not num_cols:
        st.warning("The selected data has no numeric columns."); return
    a, b, c = st.columns(3)
    col = a.selectbox("Column", num_cols, index=None, placeholder="Choose column", key=k("bn_c"))
    op = b.selectbox("Success when value is", [">", "≥", "<", "≤", "="], index=None,
                     placeholder="Choose", key=k("bn_o"))
    val = c.number_input("Success value", value=None, placeholder="type a value", key=k("bn_v"))
    if not (col and op and val is not None):
        st.info("Choose a column, a condition and a value that defines a success."); return
    ops = {">": np.greater, "≥": np.greater_equal, "<": np.less, "≤": np.less_equal, "=": np.equal}
    x = pd.to_numeric(sel[col], errors="coerce").dropna()
    if x.empty:
        st.warning("No numeric values in this column."); return
    p = float(ops[op](x, val).mean())
    d, e = st.columns(2)
    n = d.number_input("Number of trials n", min_value=1, step=1, value=None,
                       placeholder="type n", key=k("bn_n"))
    kk = e.number_input("Number of successes k", min_value=0, step=1, value=None,
                        placeholder="type k", key=k("bn_k"))
    st.caption(f"Estimated p = **{p:.4f}** ({int(ops[op](x, val).sum()):,} successes out of {len(x):,} rows)")
    if n is None:
        return
    n = int(n); xs = np.arange(n + 1); pm = stats.binom.pmf(xs, n, p)
    kk = None if kk is None else int(min(kk, n))
    fig, ax = plt.subplots(figsize=(6.9, 2.8))
    ax.bar(xs, pm, color=[C["pink"] if v == kk else C["blue"] for v in xs], alpha=.85)
    _style(ax, f"Binomial(n={n}, p={p:.3f})", "k", "P(X = k)")
    _show(fig)
    out = {"Mean": n * p, "Variance": n * p * (1 - p)}
    if kk is not None:
        out.update({"P(X = k)": stats.binom.pmf(kk, n, p), "P(X ≤ k)": stats.binom.cdf(kk, n, p),
                    "P(X ≥ k)": stats.binom.sf(kk - 1, n, p)})
    _metrics(out, 3)


# ---- Poisson distribution ----
def poisson_section(sel, num_cols, k):
    counts = [c for c in num_cols
              if (lambda s: len(s) > 2 and (s >= 0).all() and np.allclose(s, np.round(s)))
              (pd.to_numeric(sel[c], errors="coerce").dropna())]
    if not counts:
        st.warning("Needs a numeric column of non-negative whole-number counts."); return
    col = st.selectbox("Count column", counts, index=None, placeholder="Choose a count column",
                       key=k("po_c"))
    if not col:
        st.info("Choose a count column."); return
    x = pd.to_numeric(sel[col], errors="coerce").dropna().astype(int)
    lam = x.mean()
    kmax = int(max(x.max(), lam * 2 + 5))
    ks = np.arange(kmax + 1)
    obs = np.array([(x == v).mean() for v in ks])
    fig, ax = plt.subplots(figsize=(6.9, 2.8))
    ax.bar(ks - .2, obs, width=.4, color=C["amber"], label="observed")
    ax.bar(ks + .2, stats.poisson.pmf(ks, lam), width=.4, color=C["teal"], label="Poisson(λ)")
    ax.legend(frameon=False)
    _style(ax, f"{col}: observed vs Poisson", "count k", "probability")
    _show(fig)
    _metrics({"λ (average count)": lam}, 3)
    kk = st.number_input("Number of events k", min_value=0, step=1, value=None,
                         placeholder="type k", key=k("po_k"))
    if kk is not None:
        kk = int(kk)
        _metrics({"P(X = k)": stats.poisson.pmf(kk, lam), "P(X ≤ k)": stats.poisson.cdf(kk, lam),
                  "P(X ≥ k)": stats.poisson.sf(kk - 1, lam)}, 3)


# ---- Central limit theorem ----
def clt_section(sel, num_cols, k):
    if not num_cols:
        st.warning("The selected data has no numeric columns."); return
    got = _col(sel, num_cols, k, "clt_col")
    if not got: return
    name, s = got
    a, b = st.columns(2)
    n = a.number_input("Sample size n", min_value=1, step=1, value=None, placeholder="e.g. 30", key=k("clt_n"))
    reps = b.number_input("Number of samples", min_value=10, max_value=20000, step=100, value=None,
                          placeholder="e.g. 1000", key=k("clt_r"))
    if n is None or reps is None:
        st.info("Enter a sample size and the number of samples."); return
    x = s.to_numpy(); n, reps = int(n), int(reps)
    rng = np.random.default_rng(1)
    means = rng.choice(x, size=(reps, n), replace=True).mean(axis=1)
    mu, sigma = x.mean(), x.std(ddof=0); se = sigma / np.sqrt(n)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.1))
    axs[0].hist(x, bins="auto", color=C["amber"], alpha=.85, density=True, edgecolor="white")
    _style(axs[0], f"Your column: {name}", name, "density")
    axs[1].hist(means, bins=40, color=C["violet"], alpha=.8, density=True, edgecolor="white")
    g = np.linspace(means.min(), means.max(), 300)
    if se > 0:
        axs[1].plot(g, stats.norm.pdf(g, mu, se), color=C["pink"], lw=2.4, label="Normal(μ, σ/√n)")
        axs[1].legend(frameon=False)
    _style(axs[1], f"{reps:,} sample means (n={n})", "sample mean", "density")
    _show(fig, 860)
    _metrics({"Population mean μ": mu, "Mean of sample means": means.mean(),
              "Standard error (σ/√n)": se}, 4)


# ---- One/two-tailed tests, type I and II error ----
def tails_section(sel, num_cols, k):
    st.markdown("#### One-tailed / two-tailed test, Type I & Type II error")
    if not num_cols:
        st.warning("The selected data has no numeric columns."); return
    got = _col(sel, num_cols, k, "tl_col")
    if not got: return
    name, s = got
    a, b, c = st.columns(3)
    mu0 = a.number_input("Hypothesised mean μ₀", value=None, placeholder="type μ₀", key=k("tl_mu0"))
    tail = b.selectbox("Test type", ["Two-tailed (H₁: μ ≠ μ₀)", "Right-tailed (H₁: μ > μ₀)",
                                     "Left-tailed (H₁: μ < μ₀)"], index=None,
                       placeholder="Choose", key=k("tl_tail"))
    alpha = c.selectbox("α (Type I error)", [0.01, 0.05, 0.10], index=None,
                        placeholder="Choose", key=k("tl_alpha"))
    mu1 = st.number_input("True mean μ₁ for β / power (optional)", value=None,
                          placeholder="blank = sample mean", key=k("tl_mu1"))
    if mu0 is None or not tail or not alpha:
        st.info("Enter μ₀, choose the test type and α."); return
    x = s.to_numpy(); n = len(x); xbar, sd = x.mean(), x.std(ddof=1)
    if sd == 0:
        st.warning("All values are identical."); return
    alt = "two-sided" if tail.startswith("Two") else "greater" if tail.startswith("Right") else "less"
    res = stats.ttest_1samp(x, mu0, alternative=alt)
    df = n - 1; se = sd / np.sqrt(n)
    if alt == "two-sided":
        tc = stats.t.ppf(1 - alpha / 2, df); rej = lambda t: abs(t) > tc
    elif alt == "greater":
        tc = stats.t.ppf(1 - alpha, df); rej = lambda t: t > tc
    else:
        tc = stats.t.ppf(alpha, df); rej = lambda t: t < tc
    delta = ((xbar if mu1 is None else mu1) - mu0) / se
    if alt == "two-sided":
        power = stats.nct.sf(tc, df, delta) + stats.nct.cdf(-tc, df, delta)
    elif alt == "greater":
        power = stats.nct.sf(tc, df, delta)
    else:
        power = stats.nct.cdf(tc, df, delta)
    reject = res.pvalue < alpha
    _metrics({"Sample mean x̄": xbar, "t statistic": res.statistic, "p-value": res.pvalue,
              "Decision": "Reject H₀" if reject else "Do not reject H₀"}, 4)
    _metrics({"Type I error (α)": alpha, "Type II error (β)": 1 - power, "Power (1 − β)": power}, 3)
    g = np.linspace(-4.5, 4.5, 600); pdf = stats.t.pdf(g, df)
    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    ax.plot(g, pdf, color=C["blue"], lw=2)
    mask = np.array([rej(v) for v in g])
    ax.fill_between(g, pdf, where=mask, color=C["red"], alpha=.5, label="rejection region (α)")
    ax.axvline(res.statistic, color=C["pink"], lw=2.2, label=f"your t = {res.statistic:.3f}")
    ax.legend(frameon=False)
    _style(ax, f"t distribution ({df} d.f.)", "t", "density")
    _show(fig)



# ---- Simple linear regression ----
def regression_section(sel, num_cols, k):
    st.markdown("#### Simple regression (with MSE, MAE, RMSE, R²)")
    if len(num_cols) < 2:
        st.warning("At least two numeric columns are required."); return
    got = _pair(sel, num_cols, k, "reg")
    if not got: return
    x, y, d = got
    res = stats.linregress(d[x], d[y])
    yhat = res.intercept + res.slope * d[x]
    xs = np.linspace(d[x].min(), d[x].max(), 100)
    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    ax.scatter(d[x], d[y], color=C["blue"], alpha=.65, edgecolor="white")
    ax.plot(xs, res.intercept + res.slope * xs, color=C["pink"], lw=2.5)
    _style(ax, f"{y} vs {x}", x, y)
    _show(fig)
    _metrics({"Correlation r": res.rvalue, "Slope b₁": res.slope, "Intercept b₀": res.intercept})
    _metrics(err_metrics(d[y], yhat, 1))


# ---- Multiple linear regression ----
def mlr_section(sel, num_cols, k):
    if len(num_cols) < 3:
        st.warning("At least three numeric columns (one target, two predictors) are needed."); return
    y = st.selectbox("Target (Y)", num_cols, index=None, placeholder="Choose Y", key=k("mlr_y"))
    xs = st.multiselect("Predictors (X) - 2 or more", [c for c in num_cols if c != y],
                        placeholder="Choose predictors", key=k("mlr_x"))
    if not y or len(xs) < 2:
        st.info("Choose a target and at least two predictors."); return
    d = sel[[y] + xs].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) <= len(xs) + 2:
        st.warning("Not enough complete rows for this many predictors."); return
    X = np.column_stack([np.ones(len(d)), d[xs].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)
    yhat = X @ beta; n, p = len(d), len(xs)
    st.dataframe(pd.DataFrame({"Coefficient": beta}, index=["Intercept"] + xs)
                 .style.format(precision=4), use_container_width=True)
    _metrics(err_metrics(d[y], yhat, p))
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.scatter(d[y], yhat, color=C["violet"], alpha=.65, edgecolor="white")
    lim = [min(d[y].min(), yhat.min()), max(d[y].max(), yhat.max())]
    ax.plot(lim, lim, color=C["pink"], ls="--", lw=2)
    _style(ax, "Actual vs predicted", f"actual {y}", "predicted")
    _show(fig, 620)


# ---- Polynomial regression and overfitting ----
def poly_section(sel, num_cols, k):
    if len(num_cols) < 2:
        st.warning("At least two numeric columns are required."); return
    got = _pair(sel, num_cols, k, "poly")
    if not got: return
    x, y, d = got
    if len(d) < 10:
        st.warning("At least 10 complete rows are needed."); return
    maxdeg = max(2, min(12, len(d) // 2 - 1))
    deg = st.number_input("Polynomial degree", min_value=1, max_value=maxdeg, step=1, value=None,
                          placeholder=f"1 to {maxdeg}", key=k("poly_deg"))
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    xm, xsd = xv.mean(), (xv.std() or 1.0); xz = (xv - xm) / xsd
    tr, te = _split(len(d))

    def fit(dg):
        with np.errstate(all="ignore"):
            return np.polyfit(xz[tr], yv[tr], dg)

    degs = list(range(1, maxdeg + 1)); rt, rv = [], []
    for dg in degs:
        c = fit(dg)
        rt.append(err_metrics(yv[tr], np.polyval(c, xz[tr]), dg)["RMSE"])
        rv.append(err_metrics(yv[te], np.polyval(c, xz[te]), dg)["RMSE"])
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    ax.plot(degs, rt, "o-", color=C["blue"], label="train RMSE")
    ax.plot(degs, rv, "o-", color=C["red"], label="test RMSE")
    best = degs[int(np.nanargmin(rv))]
    ax.axvline(best, color=C["teal"], ls="--", label=f"best test degree = {best}")
    ax.legend(frameon=False)
    _style(ax, "Model complexity vs error", "polynomial degree", "RMSE")
    _show(fig)
    if deg is None:
        st.info("Enter a degree to see the fitted curve and its train / test metrics."); return
    deg = int(deg); co = fit(deg)
    grid = np.linspace(xz.min(), xz.max(), 300)
    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    ax.scatter(xv[tr], yv[tr], color=C["blue"], alpha=.75, label="train", edgecolor="white")
    ax.scatter(xv[te], yv[te], color=C["amber"], alpha=.9, label="test", edgecolor="white", marker="s")
    ax.plot(grid * xsd + xm, np.polyval(co, grid), color=C["pink"], lw=2.5, label=f"degree {deg}")
    span = yv.max() - yv.min()
    ax.set_ylim(yv.min() - .3 * span, yv.max() + .3 * span)
    ax.legend(frameon=False)
    _style(ax, "Polynomial fit", x, y)
    _show(fig)
    m_tr = err_metrics(yv[tr], np.polyval(co, xz[tr]), deg)
    m_te = err_metrics(yv[te], np.polyval(co, xz[te]), deg)
    l, r = st.columns(2)
    l.markdown("**Training set**"); r.markdown("**Test set**")
    l.dataframe(pd.DataFrame(m_tr, index=["value"]).T.style.format(precision=4))
    r.dataframe(pd.DataFrame(m_te, index=["value"]).T.style.format(precision=4))
    if m_te["RMSE"] > 2 * m_tr["RMSE"] > 0:
        st.warning("Test RMSE is much larger than training RMSE - this degree is **overfitting**.")
    else:
        st.success("Training and test errors are reasonably close at this degree.")


# ---- Logistic regression ----
def logistic_section(sel, num_cols, k):
    binary = [c for c in sel.columns if sel[c].dropna().nunique() == 2]
    if not binary or not num_cols:
        st.warning("Needs a target column with exactly two distinct values and a numeric predictor."); return
    y = st.selectbox("Target (two classes)", binary, index=None, placeholder="Choose target", key=k("lg_y"))
    xs = st.multiselect("Predictors (numeric)", [c for c in num_cols if c != y],
                        placeholder="Choose predictors", key=k("lg_x"))
    if not y or not xs:
        st.info("Choose a two-class target and at least one predictor."); return
    d = sel[[y] + xs].dropna()
    classes = sorted(d[y].unique(), key=str)
    yb = (d[y] == classes[1]).to_numpy(float)
    Xr = d[xs].apply(pd.to_numeric, errors="coerce")
    ok = Xr.notna().all(axis=1).to_numpy()
    Xr, yb = Xr[ok].to_numpy(float), yb[ok]
    if len(yb) < 10 or yb.min() == yb.max():
        st.warning("Need at least 10 rows containing both classes."); return
    st.caption(f"Class **{classes[0]}** → 0, class **{classes[1]}** → 1")
    mu, sd = Xr.mean(0), Xr.std(0); sd[sd == 0] = 1
    A = np.column_stack([np.ones(len(Xr)), (Xr - mu) / sd])

    def loss(w):
        z = A @ w
        return np.sum(np.logaddexp(0, z) - yb * z) + 0.5e-2 * np.sum(w[1:] ** 2)

    def grad(w):
        g = A.T @ (1 / (1 + np.exp(-(A @ w))) - yb)
        g[1:] += 1e-2 * w[1:]
        return g

    w = optimize.minimize(loss, np.zeros(A.shape[1]), jac=grad, method="L-BFGS-B").x
    prob = 1 / (1 + np.exp(-(A @ w)))
    pred = (prob >= 0.5).astype(float)
    tp, tn = int(((pred == 1) & (yb == 1)).sum()), int(((pred == 0) & (yb == 0)).sum())
    fp, fn = int(((pred == 1) & (yb == 0)).sum()), int(((pred == 0) & (yb == 1)).sum())
    _metrics({"Accuracy": (tp + tn) / len(yb)}, 3)
    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(3.6, 3.1))
        cm = np.array([[tn, fp], [fn, tp]]); ax.imshow(cm, cmap=CMAP)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["actual 0", "actual 1"])
        ax.set_title("Confusion matrix", fontweight="bold", color=INK)
        _show(fig, 340)
    with r:
        st.dataframe(pd.DataFrame({"Coefficient": w[1:]}, index=xs).style.format(precision=4),
                     use_container_width=True)


# ---- ANOVA (book: "Statistics for Data Insights" ----

def _p_message(p, alpha):
    if p < alpha:
        return "→ Significant effect. Reject the Null Hypothesis."
    return "→ No significant effect. Accept the Null Hypothesis."


def oneway_anova_section(sel, num_cols, k):
    st.markdown("**One-way ANOVA** - compares the means of several groups (one independent variable)")
    st.caption("Each selected column is one group (for example the performance scores of Mumbai, "
               "Chicago and London).")
    if len(num_cols) < 3:
        st.warning("At least three numeric columns are needed to compare three groups."); return
    cols = st.multiselect("Group columns (3 or more)", num_cols, placeholder="Choose the group columns",
                          key=k("an1_cols"))
    alpha = st.selectbox("Significance level (α)", [0.01, 0.05, 0.10], index=None,
                         placeholder="Choose α", key=k("an1_alpha"))
    if len(cols) < 3 or not alpha:
        st.info("Choose at least three group columns and α."); return
    from scipy.stats import f_oneway
    groups = [pd.to_numeric(sel[c], errors="coerce").dropna().astype(float).to_numpy() for c in cols]
    if any(len(g) < 2 for g in groups):
        st.warning("Every group needs at least 2 numeric values."); return
    st.markdown("**H₀:** There is no difference between the mean of the groups  \n"
                "**H₁:** There is a difference between the mean of the groups")
    # Compare results using Oneway ANOVA
    stat, p = f_oneway(*groups)
    _metrics({"ANOVA F statistic": stat, "p-value": p}, 2)
    if p < alpha:
        st.success("Null Hypothesis Rejected")
    else:
        st.info("Fail to reject null hypothesis (or Null Hypothesis Accepted)")
    st.dataframe(pd.DataFrame({"n": [len(g) for g in groups], "mean": [g.mean() for g in groups],
                               "std": [g.std(ddof=1) for g in groups]}, index=cols)
                 .style.format(precision=4), use_container_width=True)
    st.session_state.last_hypothesis = {
        "test": "One-way ANOVA", "h0": "no difference between the group means",
        "ha": "at least one group mean differs", "statistic": float(stat), "p_value": float(p),
        "alpha": alpha, "decision": "Reject H₀" if p < alpha else "Do not reject H₀"}


def multiway_anova_section(sel, k, ways):
    """Two-way (ways == 2) and N-way (ways == 3 or more) ANOVA, as in the book."""
    name = "Two-way ANOVA" if ways == 2 else "N-way ANOVA"
    tag = "an2" if ways == 2 else "anN"
    if ways == 2:
        st.markdown("**Two-way ANOVA** - compares groups using two independent variables (factors)")
    else:
        st.markdown("**N-way ANOVA** - compares groups using N (3 or more) independent variables (factors)")
    num_cols = [c for c in sel.columns if pd.to_numeric(sel[c], errors="coerce").notna().sum() > 2]
    if len(sel.columns) < ways + 1 or not num_cols:
        st.warning(f"At least {ways + 1} columns are needed (one response and {ways} factors)."); return
    y = st.selectbox("Response column (numeric, e.g. Productivity)", num_cols, index=None,
                     placeholder="Choose the response", key=k(f"{tag}_y"))
    others = [c for c in sel.columns if c != y]
    if ways == 2:
        fs = st.multiselect("Factor columns (exactly 2, e.g. Working_Hours, Project_Complexity)", others,
                            max_selections=2, placeholder="Choose 2 factors", key=k(f"{tag}_f"))
    else:
        fs = st.multiselect("Factor columns (3 or more)", others, max_selections=5,
                            placeholder="Choose 3 or more factors", key=k(f"{tag}_f"))
    alpha = st.selectbox("Significance level (α)", [0.01, 0.05, 0.10], index=None,
                         placeholder="Choose α", key=k(f"{tag}_alpha"))
    if not y or not alpha or len(fs) < ways or (ways == 2 and len(fs) != 2):
        st.info(f"Choose the response, {'exactly 2' if ways == 2 else 'at least 3'} factors and α."); return
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        st.error("This test needs the 'statsmodels' package:  pip install statsmodels"); return
    # Dataset (internal safe names F1, F2, ... so any column name works in the formula)
    safe = {f"F{i + 1}": c for i, c in enumerate(fs)}
    df = pd.DataFrame({"Response": pd.to_numeric(sel[y], errors="coerce")})
    for fk, c in safe.items():
        df[fk] = sel[c].astype(str).where(sel[c].notna())
    df = df.dropna()
    if (df[list(safe)].nunique() < 2).any():
        st.warning("Every factor needs at least 2 different values."); return
    if (df[list(safe)].nunique() > 8).any():
        st.warning("A factor has more than 8 different values - choose a categorical column "
                   "(for example 0/1 or low/high)."); return
    formula = "Response ~ " + " * ".join(f"C({fk})" for fk in safe)
    try:
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
    except Exception as e:
        st.error(f"Could not run the ANOVA: {e}"); return
    if anova_table.loc["Residual", "df"] <= 0:
        st.warning("Not enough observations for this many factors: there are no residual degrees of "
                   "freedom. Use more rows or fewer factors."); return

    def nice(label):
        for fk, c in safe.items():
            label = label.replace(f"C({fk})", f"C({c})")
        return label
    st.markdown("**Model:** `" + nice(formula).replace("Response", y) + "`")
    shown = anova_table.copy(); shown.index = [nice(i) for i in shown.index]
    st.markdown(f"**{name.upper()} RESULTS**")
    st.dataframe(shown.style.format(precision=6, na_rep="NaN"), use_container_width=True)
    # Hypothesis test message
    for factor in anova_table.index:
        if factor == "Residual":
            continue
        p = anova_table.loc[factor, "PR(>F)"]
        st.markdown(f"**P-value for {nice(factor)}: {p:.4f}**  \n{_p_message(p, alpha)}")
    st.caption("Factors are treated as categories. H₀ for each row: that factor (or interaction) has no "
               "effect on the response.")


def anova_section(sel, num_cols, k):
    st.markdown("#### ANOVA (Analysis of Variance)")
    st.caption("A t-test only compares two groups; ANOVA compares more than two groups at the same time "
               "by analysing the variance between and within the groups.")
    kind = st.radio("ANOVA type", ["One-way ANOVA", "Two-way ANOVA", "N-way ANOVA"], index=None,
                    horizontal=True, key=k("anova_kind"))
    if kind is None:
        st.info("Choose an ANOVA type."); return
    if kind == "One-way ANOVA":
        oneway_anova_section(sel, num_cols, k)
    elif kind == "Two-way ANOVA":
        multiway_anova_section(sel, k, 2)
    else:
        multiway_anova_section(sel, k, 3)


# ---- Entry points called from app.py ----
def distributions_section(sel, num_cols, k):
    with st.container(border=True):
        st.subheader("12 · Probability distributions")
        topic = st.radio(
            "Topic", ["Normal distribution", "Probability", "Binomial distribution",
                      "Poisson distribution", "Central Limit Theorem"],
            index=None, horizontal=True, key=k("dist_topic"))
        if not num_cols:
            st.warning("The selected data has no numeric columns."); return
        if topic is None:
            st.info("Choose a topic.")
        else:
            {"Normal distribution": normal_section, "Probability": probability_section,
             "Binomial distribution": binomial_section, "Poisson distribution": poisson_section,
             "Central Limit Theorem": clt_section}[topic](sel, num_cols, k)


def regression_models_section(sel, num_cols, k):
    """Section 10: every regression topic in one place."""
    with st.container(border=True):
        st.subheader("10 · Regression")
        topic = st.radio(
            "Regression topic",
            ["Simple linear regression (MSE, MAE, RMSE, R²)", "Multiple linear regression",
             "Polynomial regression & overfitting", "Logistic regression"],
            index=None, horizontal=True, key=k("model_topic"))
        if topic is None:
            st.info("Choose a regression topic.")
        else:
            {"Simple linear regression (MSE, MAE, RMSE, R²)": regression_section,
             "Multiple linear regression": mlr_section,
             "Polynomial regression & overfitting": poly_section,
             "Logistic regression": logistic_section}[topic](sel, num_cols, k)
