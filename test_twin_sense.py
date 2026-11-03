#!/usr/bin/env python3
"""
Diagnostic: is the Schmid factor of the inherited {114}B2 twin system (twin table of crystlib, used by
ebsdlib.reconstruct_parent_orientation_directed) the lattice-correspondence image of the Schmid factor of the
(20-1)B19' twin that produced it?

For every lattice correspondence variant i of `system`:
  1. martensite twin: K1 = (2 0 -1)_M, eta1 = +-[1 0 2]_M (the only direction in K1 normal to b = [010]_M, the
     rotation axis of the twin). The physical sense of eta1 is found from the lattice: the simple shear
     F = I + s eta1 x n1 must map the martensite lattice onto its twin lattice, F L = Q L P with Q = 180 deg about
     n1 and P an integer matrix; this holds for one sense only (scan of s).
  2. B2 image: n_A = CIp_i^-1 (2 0 -1) (planes; CIp maps austenite planes to martensite planes),
               eta_A = CId_i^-1 eta1 (directions; CId maps austenite directions to martensite directions).
  3. the {114} entry j of twinSys[system]['114'] with n1_a[j] || +-n_A; with the sign that makes n1_a[j].n_A > 0,
     a1_a[j] is compared with eta_A: cos = +1 (parallel) means SF_B2 and SF_B19' have the same sign convention.
  4. sign agreement of SF_B19' = (n.l)(eta.l) in the martensite lattice and SF_B2 = (n1.l)(a1.l) in the austenite
     for random load axes l (l_M = L_M CId_i l_A): (n.l) keeps its sign under the correspondence exactly, (eta.l) can
     change sign for axes nearly normal to eta, so the agreement is ~100 % but not exactly.

Run (any checkpoint of the pipeline; only its recipe is used to rebuild the crystallography):
    python test_twin_sense.py ~/Documents/papers/mypapers/micromod/figs/def50/grain_0000.h5
Output: table on screen and twin_sense_check.csv in the working directory.
"""
import os, sys, csv
import numpy as np

ETA_M = np.array([1.0, 0.0, 2.0])        # [102]_M, sign fixed by physical_sense()
K1_M = np.array([2.0, 0.0, -1.0])        # (20-1)_M


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot180(n):
    n = unit(n)
    return 2 * np.outer(n, n) - np.eye(3)


def physical_sense(L_M, h=K1_M, uvw=ETA_M, s_max=1.5, ns=30001):
    """sigma (+1/-1) such that the shear along sigma*eta reproduces the lattice as the twin lattice, the shear s and
    the integer residual of both senses. L_M: direct lattice matrix (columns a, b, c, Cartesian)."""
    n = unit(np.linalg.inv(L_M).T @ h)
    e = unit(L_M @ uvw)
    Q = rot180(n)
    A = np.linalg.inv(L_M) @ Q.T                     # P(s) = A (I + s sigma e n^T) L_M
    P0, P1 = A @ L_M, A @ np.outer(e, n) @ L_M
    out = {}
    for sigma in (+1, -1):
        s = np.linspace(1e-3, s_max, ns)
        P = P0[None] + (sigma * s)[:, None, None] * P1[None]
        r = np.abs(P - np.round(P)).max(axis=(1, 2))
        k = int(np.argmin(r))
        out[sigma] = (float(s[k]), float(r[k]))
    sigma = min(out, key=lambda x: out[x][1])
    return sigma, out


def check(CId, CIp, n1_a, a1_a, L_A, L_M, n_axes=20000, seed=0, verbose=True):
    sigma, sense = physical_sense(L_M)
    eta_M = sigma * ETA_M
    if verbose:
        print(f"physical sense of eta1: {'+' if sigma > 0 else '-'}[1 0 2]_M  (shear s = {sense[sigma][0]:.4f}, "
              f"integer residual {sense[sigma][1]:.1e}; other sense: residual {sense[-sigma][1]:.1e})")
    n1 = np.array([unit(v) for v in n1_a]); a1 = np.array([unit(v) for v in a1_a])
    rng = np.random.default_rng(seed)
    lA = rng.normal(size=(n_axes, 3)); lA /= np.linalg.norm(lA, axis=1, keepdims=True)   # Cartesian B2 = Miller (cubic)
    rows = []
    nM_c = unit(np.linalg.inv(L_M).T @ K1_M); eM_c = unit(L_M @ eta_M)
    for i in range(CId.shape[2]):
        Ci, Pi = CId[:, :, i], CIp[:, :, i]
        hA = np.linalg.solve(Pi, K1_M)                  # plane indices in B2
        uA = np.linalg.solve(Ci, eta_M)                 # direction indices in B2
        nA = unit(np.linalg.inv(L_A).T @ hA); eA = unit(L_A @ uA)
        c = n1 @ nA; j = int(np.argmax(np.abs(c))); sgn = np.sign(c[j])
        n1j, a1j = sgn * n1[j], sgn * a1[j]
        cos_eta = float(a1j @ eA)
        # sign agreement of the two Schmid factors for random axes
        lM = (L_M @ (Ci @ lA.T)).T; lM /= np.linalg.norm(lM, axis=1, keepdims=True)
        sfM = (lM @ nM_c) * (lM @ eM_c)
        sfA = (lA @ n1j) * (lA @ a1j)
        agree = float(np.mean(np.sign(sfM) == np.sign(sfA)))
        big = np.abs(sfA) > 0.1
        agree_big = float(np.mean(np.sign(sfM[big]) == np.sign(sfA[big])))
        rows.append(dict(variant=i + 1, n_A=np.round(hA / np.abs(hA).max() * 4, 3).tolist(),
                         eta_A=np.round(uA / np.abs(uA).max() * 2, 3).tolist(), table_j=j + 1,
                         plane_dev_deg=round(float(np.degrees(np.arccos(min(1, abs(c[j]))))), 3),
                         eta_dev_deg=round(float(np.degrees(np.arccos(min(1, abs(cos_eta))))), 3),
                         cos_eta=round(cos_eta, 4), sense='parallel' if cos_eta > 0 else 'ANTIPARALLEL',
                         sf_sign_agreement=round(agree, 4), sf_sign_agreement_absSF_gt_0p1=round(agree_big, 4)))
    if verbose:
        print(f"{'var':>3} {'n_A (B2)':>24} {'eta_A (B2)':>24} {'tab':>3} {'pl.dev':>6} {'eta.dev':>7} {'cos':>7} "
              f"{'sense':>12} {'SF sign agree':>13} {'|SF|>0.1':>8}")
        for r in rows:
            print(f"{r['variant']:3d} {str(r['n_A']):>24} {str(r['eta_A']):>24} {r['table_j']:3d} {r['plane_dev_deg']:6.2f} "
                  f"{r['eta_dev_deg']:7.2f} {r['cos_eta']:+7.3f} {r['sense']:>12} {100 * r['sf_sign_agreement']:12.1f}% "
                  f"{100 * r['sf_sign_agreement_absSF_gt_0p1']:7.1f}%")
        n_par = sum(r['sense'] == 'parallel' for r in rows)
        print(f"\n{n_par}/{len(rows)} variants parallel: "
              + ('B2 {114} Schmid factor = image of the (20-1) Schmid factor for ALL variants'
                 if n_par == len(rows) else
                 'reversed for ALL variants: flip the sign of the B2 {114} Schmid factor' if n_par == 0 else
                 'MIXED: the sign relation depends on the correspondence variant'))
    return sigma, sense, rows


def main(path, system='NiTi', mode='114', phase_A='A', phase_M='M'):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import ebsdlib
    st = ebsdlib.load_checkpoint(path)
    d = st['dataEBSD']
    if not hasattr(d, 'twinSys') or system not in getattr(d, 'twinSys', {}):
        d.getTwinSys()
    CId = np.asarray(d.OR[system]['CId'], float); CIp = np.asarray(d.OR[system]['CIp'], float)
    tw = d.twinSys[system][mode]
    L_A = np.asarray(d.phases[phase_A]['L'], float); L_M = np.asarray(d.phases[phase_M]['L'], float)
    print(f"{path}\nCId {CId.shape}, CIp {CIp.shape}, twinSys[{system!r}][{mode!r}]: n1_a {np.shape(tw['n1_a'])}, "
          f"a1_a {np.shape(tw['a1_a'])}\nL_M =\n{np.round(L_M, 4)}")
    sigma, sense, rows = check(CId, CIp, tw['n1_a'], tw['a1_a'], L_A, L_M)
    with open('twin_sense_check.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('written', os.path.abspath('twin_sense_check.csv'))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(os.path.expanduser(sys.argv[1]), *sys.argv[2:])
