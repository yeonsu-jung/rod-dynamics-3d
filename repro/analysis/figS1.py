#!/usr/bin/env python3
r"""Fig S1 E-G: average crossing number of a two-rod pair vs separation
d_ij, crossing angle theta, and skewness a_i.

ACN_ij = (1/4pi) \iint |(t_i x t_j) . (r_i - r_j)| / |r_i - r_j|^3 ds dt
(Eq. S22), evaluated by direct quadrature, with the symmetric closed form
Eq. S25 overlaid on panel E.

Usage: python3 -m repro.analysis.figS1
"""
import numpy as np

from .common import FIGS, paper_style

L = 1.0


def acn_numeric(d, theta, a1=0.0, a2=0.0, n=400):
    """Rod i along x at offset a1, rod j direction (cos t, sin t, 0)
    rotated by theta, at height d, offset a2."""
    ti = np.array([1.0, 0.0, 0.0])
    tj = np.array([np.cos(theta), np.sin(theta), 0.0])
    s = (np.linspace(-0.5, 0.5, n) + a1) * L
    t = (np.linspace(-0.5, 0.5, n) + a2) * L
    ds = L / (n - 1)
    S, T = np.meshgrid(s, t, indexing="ij")
    rx = S * ti[0] - T * tj[0]
    ry = S * ti[1] - T * tj[1]
    rz = np.full_like(rx, -d)
    cross = np.cross(ti, tj)
    num = np.abs(cross[0] * rx + cross[1] * ry + cross[2] * rz)
    den = (rx * rx + ry * ry + rz * rz) ** 1.5
    return float((num / den).sum() * ds * ds / (4 * np.pi))


def acn_s25(d):
    return np.arctan(1.0 / (d * np.sqrt(8 + 16 * d * d))) / np.pi


def main():
    plt = paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.4))

    ds = np.linspace(0.1, 1.0, 25)
    axes[0].plot(ds, [acn_numeric(d, np.pi / 2) for d in ds], "gs-", ms=3,
                 lw=0.8, label="numerical")
    axes[0].plot(ds, acn_s25(ds), "r--", lw=1, label="Eq. S25")
    axes[0].set_xlabel(r"$d_{ij}$")
    axes[0].set_ylabel("ACN")
    axes[0].legend(frameon=False)

    thetas = np.linspace(0.01, np.pi / 2, 25)
    axes[1].plot(thetas, [acn_numeric(0.5, th) for th in thetas], "gs-",
                 ms=3, lw=0.8)
    axes[1].set_xlabel(r"$\theta$")

    offs = np.linspace(-0.4, 0.4, 25)
    axes[2].plot(offs, [acn_numeric(0.5, np.pi / 2, a1=a) for a in offs],
                 "gs-", ms=3, lw=0.8)
    axes[2].set_xlabel(r"$a_i$")

    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "figS1.pdf")
    fig.savefig(FIGS / "figS1.png")
    print(f"wrote {FIGS/'figS1.pdf'}")


if __name__ == "__main__":
    main()
