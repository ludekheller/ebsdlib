"""
Synthetic EBSD maps with known ground truth, for testing the grain-resolved analysis of ebsdlib
================================================================================================

Two classes:

* :class:`SyntheticEBSD` -- builds maps of B2 austenite whose former grains contain the domain architecture inherited
  from B19' plasticity (matrix, +-{114} twins, kinks +-beta*, +-2beta*, {113}/{115} twins of kinked hosts, {112} twins,
  off-axis {114} twins), all rotated about the common <110> axis d_s of the grain, perturbed by noise, orientation
  gradients, mis-indexed / non-indexed points and retained B19' points, and writes them as EDAX .ang files (header copied
  from a measured map, so phases, grid and frame convention are those of the real data). The ground truth of every
  point (grain, domain, class, rotation phi about d_s, phase, flags) is kept in the object and saved next to the .ang.

* :class:`SyntheticValidator` -- scores the outputs of the pipeline against that ground truth:
  former grains (ARI, completeness, purity), cluster -> domain matching, domain classes (confusion matrix, area
  accuracy), phi about d_s, twin / kink relationships of cluster pairs, and CSL (Sigma) boundaries of neighbour pairs.

Conventions (as in ebsdlib / pyebsd)
------------------------------------
* g : orientation matrix sample -> crystal (pyebsd ``M``); in the .ang file the Bunge Euler angles of R = g^T
  (pyebsd ``euler_angles_to_rotation_matrix`` gives R, crystal -> sample).
* Rotation of a domain c by phi about d_s (sample frame) relative to the matrix:  g_c = g_M R(d_s, phi), so that
  Q_Mc = g_M^T g_c = R(d_s, phi)  (Eq. Q of the paper with S = I) and the pipeline's phi* returns +phi.
* Map coordinates (X, Y) [um] of the .ang (origin top left, Y down) relate to the sample frame of the orientations by
  the swap used in ebsdlib (CrystalRef2Spatial = [[0,1,0],[1,0,0],[0,0,1]]):  s = (Y, X, 0).  z = section normal = wire
  axis (loading axis [0, 0, 1]).  -> to be confirmed on a measured map (see SyntheticEBSD.check_frame).
* Hexagonal grid: row j has NCOLS_ODD points starting at x = 0 (j even) or NCOLS_EVEN points starting at dx/2 (j odd);
  y = j * dx * sqrt(3)/2.

Only numpy is required (pandas optional for the summary tables). Usage: see matchar/notebooks/01_synthetic_validation.ipynb.
"""
import os
import json
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# rotations
# ----------------------------------------------------------------------------------------------------------------------
DEG = np.pi / 180.0


def rotation(axis, angle_deg):
    """Rodrigues: proper rotation matrix about `axis` (normalised here) by angle_deg."""
    d = np.asarray(axis, float); d = d / np.linalg.norm(d)
    a = angle_deg * DEG
    K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    return np.cos(a) * np.eye(3) + (1 - np.cos(a)) * np.outer(d, d) + np.sin(a) * K


def rotations(axes, angles_deg):
    """Vectorised Rodrigues: axes (N,3) unit, angles (N,) -> (N,3,3)."""
    d = np.asarray(axes, float)
    a = np.asarray(angles_deg, float) * DEG
    c, s = np.cos(a)[:, None, None], np.sin(a)[:, None, None]
    K = np.zeros((len(d), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -d[:, 2], d[:, 1], -d[:, 0]
    K = K - K.transpose(0, 2, 1)
    return c * np.eye(3) + (1 - c) * np.einsum('ni,nj->nij', d, d) + s * K


def rot_angle(Q):
    """Rotation angle [deg] of (...,3,3)."""
    tr = np.trace(Q, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def phi_about_axis(Q, d):
    """Rotation phi* [deg] about the unit axis d that comes closest to Q, and the residual r [deg] (Eqs. phistar,
    residual of the paper). Q (...,3,3), d (3,) or (...,3)."""
    d = np.broadcast_to(np.asarray(d, float), Q.shape[:-1])
    q = np.stack([Q[..., 2, 1] - Q[..., 1, 2], Q[..., 0, 2] - Q[..., 2, 0], Q[..., 1, 0] - Q[..., 0, 1]], -1)
    dQd = np.einsum('...i,...ij,...j->...', d, Q, d)
    phi = np.degrees(np.arctan2(np.einsum('...i,...i->...', d, q), np.trace(Q, axis1=-2, axis2=-1) - dQd))
    flat = d.reshape(-1, 3); R = rotations(flat, phi.reshape(-1)).reshape(Q.shape)
    r = rot_angle(np.einsum('...ij,...kj->...ik', Q, R))
    return phi, r


def euler_to_R(phi1, Phi, phi2):
    """Bunge zxz Euler [rad] -> R (crystal -> sample), identical to pyebsd.euler_angles_to_rotation_matrix."""
    c1, s1, c, s, c2, s2 = np.cos(phi1), np.sin(phi1), np.cos(Phi), np.sin(Phi), np.cos(phi2), np.sin(phi2)
    R = np.empty(np.shape(phi1) + (3, 3))
    R[..., 0, 0] = c1 * c2 - s1 * c * s2; R[..., 0, 1] = -c1 * s2 - s1 * c * c2; R[..., 0, 2] = s1 * s
    R[..., 1, 0] = s1 * c2 + c1 * c * s2; R[..., 1, 1] = -s1 * s2 + c1 * c * c2; R[..., 1, 2] = -c1 * s
    R[..., 2, 0] = s * s2; R[..., 2, 1] = s * c2; R[..., 2, 2] = c
    return R


def R_to_euler(R):
    """Inverse of euler_to_R (angles in [0, 2pi), [0, pi], [0, 2pi))."""
    R = np.asarray(R, float)
    Phi = np.arccos(np.clip(R[..., 2, 2], -1, 1))
    gen = np.abs(np.sin(Phi)) > 1e-8
    phi1 = np.where(gen, np.arctan2(R[..., 0, 2], -R[..., 1, 2]), np.arctan2(R[..., 1, 0], R[..., 0, 0]))
    phi2 = np.where(gen, np.arctan2(R[..., 2, 0], R[..., 2, 1]), 0.0)
    return np.mod(phi1, 2 * np.pi), Phi, np.mod(phi2, 2 * np.pi)


def cubic_symops():
    """The 24 proper rotations of the cubic point group (crystal frame)."""
    ops = []
    for perm in ([0, 1, 2], [1, 2, 0], [2, 0, 1], [1, 0, 2], [0, 2, 1], [2, 1, 0]):
        for signs in np.ndindex(2, 2, 2):
            S = np.zeros((3, 3))
            for i, p in enumerate(perm):
                S[i, p] = (-1) ** signs[i]
            if np.isclose(np.linalg.det(S), 1):
                ops.append(S)
    return np.array(ops)


SYM = cubic_symops()
FAM110 = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]], float) / np.sqrt(2)


def random_rotations(n, rng):
    """Uniformly distributed rotation matrices (n,3,3) from random unit quaternions."""
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def disorientation(ga, gb):
    """Disorientation angle [deg] between orientation stacks ga, gb (...,3,3), proper cubic symmetry."""
    D = np.einsum('...ij,...kj->...ik', gb, ga)                     # gb ga^T (crystal frame)
    ang = rot_angle(np.einsum('sij,...jk->...sik', SYM, D))
    return ang.min(axis=-1)


# ----------------------------------------------------------------------------------------------------------------------
# domain classes and CSL
# ----------------------------------------------------------------------------------------------------------------------
PHI_N = {n: 2 * np.degrees(np.arctan(np.sqrt(2) / n)) for n in (2, 3, 4, 5)}   # {11n} twin rotation about <110>


def class_table(beta_star=6.8):
    """Ideal rotation phi [deg] about d_s relative to the matrix and the host of every domain class (Table 'classes'
    of the paper). Hosts: 'M' matrix, '+K2'/'-K2' kinks by +-2beta*."""
    p, b = PHI_N[4], beta_star
    return {
        'M':    dict(phi=0.0, host=None, kind='matrix'),
        '+114': dict(phi=+p, host='M', kind='twin'),           # (20-1) twin of V1
        '-114': dict(phi=-p, host='M', kind='twin'),           # (20-1) twin of V2
        '+K2':  dict(phi=+2 * b, host='M', kind='kink'),       # V1', V2''
        '-K2':  dict(phi=-2 * b, host='M', kind='kink'),       # V2', V1''
        '+K1':  dict(phi=+b, host='M', kind='kink'),           # observed +-beta* kinks
        '-K1':  dict(phi=-b, host='M', kind='kink'),
        '+113': dict(phi=+(p + 2 * b), host='+K2', kind='twin'),   # (20-1) twin of V1'  in K+
        '-115': dict(phi=-(p - 2 * b), host='+K2', kind='twin'),   # (20-1) twin of V2'' in K+
        '-113': dict(phi=-(p + 2 * b), host='-K2', kind='twin'),   # (20-1) twin of V2'  in K-
        '+115': dict(phi=+(p - 2 * b), host='-K2', kind='twin'),   # (20-1) twin of V1'' in K-
        '+112': dict(phi=+PHI_N[2], host='M', kind='twin'),        # {112} twin of the matrix (Sigma3), test case
        'OA114': dict(phi=np.nan, host='M', kind='offaxis'),       # {114} twin about another <110> of the matrix
    }


# CSL rotations (axis in crystal frame, angle) of the Sigmas evaluated by ebsdlib.csl_boundary_fractions
CSL = {'3': ([1, 1, 1], 60.0), '9': ([1, 1, 0], 38.94), '11': ([1, 1, 0], 50.48), '19a': ([1, 1, 0], 26.53),
       '27a': ([1, 1, 0], 31.59), '27b': ([2, 1, 0], 35.43)}


def csl_of(ga, gb, sigmas=('3', '9', '11', '27a', '27b'), brandon_k=15.0):
    """CSL type of each pair (ga[i], gb[i]) by the Brandon criterion (smallest dev/tolerance wins); '' if none.
    Deviation = min over the symmetry-equivalent CSL rotations S1 C S2 of the angle between them and D = gb ga^T,
    evaluated as min over the conjugates S C S^T (and their inverses) and a one-sided S acting on D."""
    D = np.einsum('nij,nkj->nik', gb, ga)                             # crystal-frame misorientation
    best = np.full(len(D), np.inf); out = np.array([''] * len(D), dtype=object)
    SD = np.einsum('sij,njk->nsik', SYM, D)                           # (n, 24, 3, 3)
    for s in sigmas:
        ax, ang = CSL[s]
        C = rotation(ax, ang)
        Cv = np.einsum('sij,jk,slk->sil', SYM, C, SYM)                # conjugates S C S^T
        Cv = np.concatenate([Cv, Cv.transpose(0, 2, 1)])
        Cv = Cv[np.unique(np.round(Cv.reshape(len(Cv), -1), 6), axis=0, return_index=True)[1]]
        tol = brandon_k / np.sqrt(float(''.join(ch for ch in s if ch.isdigit())))
        dev = np.full(len(D), np.inf)
        for Ck in Cv:
            dev = np.minimum(dev, rot_angle(np.einsum('ji,nsjk->nsik', Ck, SD)).min(axis=1))
        score = dev / tol
        hit = (dev <= tol) & (score < best)
        out[hit] = s; best[hit] = score[hit]
    return out


def adjusted_rand_index(a, b):
    """ARI of two integer labelings (same length)."""
    a = np.asarray(a); b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True); _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1)); np.add.at(C, (ai, bi), 1)
    comb2 = lambda x: x * (x - 1) / 2
    sij = comb2(C).sum(); sa = comb2(C.sum(1)).sum(); sb = comb2(C.sum(0)).sum(); n = comb2(len(a))
    exp = sa * sb / n
    return float((sij - exp) / (0.5 * (sa + sb) - exp)) if 0.5 * (sa + sb) != exp else 1.0


# ----------------------------------------------------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------------------------------------------------
class SyntheticEBSD:
    """
    Synthetic EBSD map with known ground truth.

    Parameters
    ----------
    template_ang : str
        A measured .ang file; its header (phases, calibration, column description) is copied, only the grid lines are
        replaced. Phase ids for B2 / B19' are taken from `phase_A`, `phase_M`.
    size_um : (float, float)   map size (X, Y) [um]
    step_um : float            hexagonal step dx [um]
    phase_A, phase_M : int     phase ids of B2 and B19' in the template header
    beta_star : float          beta* = beta_M - 90 deg [deg] (6.8 as in the paper)
    map_to_sample : (3,3)      matrix mapping map vectors (X, Y, 0) to the sample frame (default: X/Y swap of ebsdlib)
    seed : int                 random seed (everything is reproducible)

    Workflow
    --------
    >>> syn = SyntheticEBSD(template, size_um=(30, 25), step_um=0.1, seed=1)
    >>> syn.add_grain(g_M=None, ds_index=None, layout='random', fractions={...})   # or several grains, see add_grain
    >>> syn.build()                                   # ideal orientations + truth
    >>> syn.perturb(noise_deg=0.5, gradient_deg_per_um=0.3, f_mis=0.01, f_ni=0.01, f_mart=0.3, w_mart_um=0.15)
    >>> syn.write('synthetic_S1.ang')                 # + synthetic_S1_truth.npz / _truth.json
    """

    def __init__(self, template_ang, size_um=(30.0, 25.0), step_um=0.1, phase_A=2, phase_M=3, beta_star=6.8,
                 map_to_sample=((0, 1, 0), (1, 0, 0), (0, 0, 1)), seed=0):
        self.template_ang = template_ang
        self.size_um = tuple(size_um); self.step = float(step_um)
        self.phase_A, self.phase_M = int(phase_A), int(phase_M)
        self.beta_star = float(beta_star)
        self.classes = class_table(beta_star)
        self.class_names = list(self.classes)
        self.map_to_sample = np.asarray(map_to_sample, float)
        self.rng = np.random.default_rng(seed); self.seed = seed
        self.grains = []
        self._grid()

    # ---------------------------------------------------------------- grid
    def _grid(self):
        dx = self.step; dy = dx * np.sqrt(3) / 2
        nx = int(np.floor(self.size_um[0] / dx)) + 1
        ny = int(np.floor(self.size_um[1] / dy)) + 1
        xs, ys, rows = [], [], []
        for j in range(ny):
            n = nx if j % 2 == 0 else nx - 1
            x0 = 0.0 if j % 2 == 0 else dx / 2
            xs.append(x0 + dx * np.arange(n)); ys.append(np.full(n, j * dy)); rows.append(np.full(n, j))
        self.X = np.concatenate(xs); self.Y = np.concatenate(ys); self.row = np.concatenate(rows)
        self.ncols_odd, self.ncols_even, self.nrows = nx, nx - 1, ny
        self.dy = dy
        self.N = len(self.X)
        self.S = (np.stack([self.X, self.Y, np.zeros(self.N)], 1)) @ self.map_to_sample.T   # sample-frame positions

    # ---------------------------------------------------------------- grains
    def fiber_orientation(self, fiber=(1, 1, 1), spread_deg=10.0):
        """Random orientation with the crystal direction `fiber` close to the sample z axis (wire texture)."""
        f = np.asarray(fiber, float); f /= np.linalg.norm(f)
        z = np.array([0, 0, 1.0])
        # g maps sample -> crystal; want g z ~ f: g = R_spin(f) * R(z -> f)
        v = np.cross(z, f); ang = np.degrees(np.arccos(np.clip(z @ f, -1, 1)))
        G0 = rotation(v, ang) if np.linalg.norm(v) > 1e-9 else np.eye(3)
        spin = rotation(f, self.rng.uniform(0, 360))
        tilt_ax = self.rng.normal(size=3); tilt_ax -= tilt_ax @ f * f
        tilt = rotation(tilt_ax, abs(self.rng.normal(0, spread_deg)))
        return tilt @ spin @ G0

    def add_grain(self, g_M=None, ds_index=None, seed_xy=None, layout='random', fractions=None,
                  mean_width_um=0.8, width_sigma=0.5, alpha_deg=None, sub_alpha_deg=None, sub_fraction=0.35,
                  offaxis=None):
        """
        Add a former grain.

        g_M        : (3,3) matrix orientation (sample -> crystal); default: <111> fibre (fiber_orientation)
        ds_index   : which of the six <110> of g_M is d_s (0..5); default random
        seed_xy    : (X, Y) [um] of the Voronoi seed; default random (single grain: map centre)
        layout     : 'random' -> lamellae drawn from `fractions`; or an explicit list of (class, width_um) along the
                     lamella normal, kinks may carry sub-lamellae: ('+K2', 1.2, [('+113', 0.2), ('+K2', 0.4), ...])
        fractions  : dict class -> area fraction for layout='random' (classes of class_table without hosts '+113' etc.,
                     which are put into the kinks with `sub_fraction`); default: paper-like 30 % + 500 C
        mean_width_um, width_sigma : log-normal lamella widths
        alpha_deg  : orientation of the lamella normal n_L about d_s (n_L _|_ d_s); default random
        sub_alpha_deg : normal of the sub-lamellae in kinks; default random
        offaxis    : None or dict(width_um=..., position_um=...) -> one {114} twin lamella about another <110>
        """
        if g_M is None:
            g_M = self.fiber_orientation()
        g_M = np.asarray(g_M, float)
        if ds_index is None:
            ds_index = int(self.rng.integers(6))
        ds = g_M.T @ FAM110[ds_index]; ds /= np.linalg.norm(ds)
        if seed_xy is None:
            seed_xy = (self.size_um[0] / 2, self.size_um[1] / 2) if not self.grains else \
                (self.rng.uniform(0, self.size_um[0]), self.rng.uniform(0, self.size_um[1]))
        e1 = np.cross(ds, [0, 0, 1.0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else \
            np.cross(ds, [1.0, 0, 0]) / np.linalg.norm(np.cross(ds, [1.0, 0, 0]))
        e2 = np.cross(ds, e1)
        a = self.rng.uniform(0, 180) if alpha_deg is None else alpha_deg
        sa = self.rng.uniform(0, 180) if sub_alpha_deg is None else sub_alpha_deg
        nL = np.cos(a * DEG) * e1 + np.sin(a * DEG) * e2
        nS = np.cos(sa * DEG) * e1 + np.sin(sa * DEG) * e2
        if layout == 'random':
            layout = self._random_layout(fractions, mean_width_um, width_sigma, sub_fraction)
        self.grains.append(dict(g_M=g_M, ds_index=ds_index, ds=ds, seed_xy=np.asarray(seed_xy, float), layout=layout,
                                nL=nL, nS=nS, alpha=a, sub_alpha=sa, offaxis=offaxis))
        return len(self.grains) - 1

    def _random_layout(self, fractions, mean_w, sig, sub_fraction, total_um=None):
        if fractions is None:
            fractions = {'M': 0.43, '+114': 0.30, '-114': 0.02, '+K2': 0.08, '-K2': 0.05, '+K1': 0.06, '-K1': 0.06}
        names = list(fractions); p = np.array([fractions[k] for k in names], float); p /= p.sum()
        total_um = total_um or 2.5 * np.hypot(*self.size_um)
        lay, t = [], 0.0
        while t < total_um:
            c = names[self.rng.choice(len(names), p=p)]
            w = float(mean_w * np.exp(self.rng.normal(0, sig)))
            if c in ('+K2', '-K2') and sub_fraction > 0:          # twins of the kinked host
                tw = ['+113', '-115'] if c == '+K2' else ['-113', '+115']
                sub, s = [], 0.0
                while s < w:
                    ws = float(0.3 * mean_w * np.exp(self.rng.normal(0, sig)))
                    sub.append((tw[self.rng.integers(2)] if self.rng.random() < sub_fraction else c, ws)); s += ws
                lay.append((c, w, sub))
            else:
                lay.append((c, w))
            t += w
        return lay

    # ---------------------------------------------------------------- build
    def build(self):
        """Assign every point to a grain (Voronoi of the seeds) and a domain; ideal orientations and truth arrays."""
        assert self.grains, 'add_grain first'
        seeds = np.array([g['seed_xy'] for g in self.grains])
        d2 = (self.X[:, None] - seeds[None, :, 0]) ** 2 + (self.Y[:, None] - seeds[None, :, 1]) ** 2
        self.grain = d2.argmin(1).astype(np.int32)
        self.cls = np.zeros(self.N, np.int16)              # index into class_names
        self.domain = np.full(self.N, -1, np.int32)        # unique domain id (lamella / sub-lamella)
        self.phi = np.zeros(self.N)                        # ideal rotation about d_s [deg] (NaN: off-axis)
        self.g_ideal = np.zeros((self.N, 3, 3))
        self.domains = []                                  # list of dicts (id, grain, class, phi, host domain)
        for gi, G in enumerate(self.grains):
            m = self.grain == gi
            s0 = np.array([*G['seed_xy'], 0.0]) @ self.map_to_sample.T
            t = (self.S[m] - s0) @ G['nL']                 # coordinate along the lamella normal
            tsub = (self.S[m] - s0) @ G['nS']
            idx = np.where(m)[0]
            cum = np.cumsum([0.0] + [L[1] for L in G['layout']])
            tt = t - t.min() + 1e-9 if len(t) else t
            tt = np.mod(tt, cum[-1])
            k = np.searchsorted(cum, tt, side='right') - 1
            dom_id = np.full(len(idx), -1); cname = np.empty(len(idx), dtype=object)
            for li, L in enumerate(G['layout']):
                sel = k == li
                if not sel.any():
                    continue
                if len(L) == 3:                            # kink with sub-lamellae
                    cs = np.cumsum([0.0] + [w for _, w in L[2]])
                    u = np.mod(tsub[sel] - tsub[sel].min(), cs[-1]); kk = np.searchsorted(cs, u, side='right') - 1
                    host = self._new_domain(gi, L[0], None)
                    for si, (sc, _) in enumerate(L[2]):
                        ss = np.where(sel)[0][kk == si]
                        did = host if sc == L[0] else self._new_domain(gi, sc, host)
                        dom_id[ss] = did; cname[ss] = sc
                else:
                    did = self._new_domain(gi, L[0], None)
                    dom_id[sel] = did; cname[sel] = L[0]
            # off-axis twin lamella (overrides)
            if G['offaxis']:
                oa = G['offaxis']; nOA = G['nS']
                u = (self.S[m] - s0) @ np.cross(G['ds'], nOA)
                sel = np.abs(u - oa.get('position_um', 0.0)) < oa.get('width_um', 1.0) / 2
                did = self._new_domain(gi, 'OA114', None)
                dom_id[sel] = did; cname[sel] = 'OA114'
            self.domain[idx] = dom_id
            self.cls[idx] = [self.class_names.index(c) for c in cname]
            # orientations
            phis = np.array([self.classes[c]['phi'] for c in cname], float)
            self.phi[idx] = phis
            on = np.isfinite(phis)
            Rs = rotations(np.repeat(G['ds'][None], on.sum(), 0), phis[on])
            self.g_ideal[idx[on]] = np.einsum('ij,njk->nik', G['g_M'], Rs)
            if (~on).any():                                # {114} twin about another <110> of the matrix
                u = FAM110[(G['ds_index'] + 2) % 6]; a_oa = G['g_M'].T @ u
                self.g_ideal[idx[~on]] = G['g_M'] @ rotation(a_oa, PHI_N[4])
        self.g = self.g_ideal.copy()
        self.phase = np.full(self.N, self.phase_A, np.int16)
        self.ci = np.full(self.N, 0.9); self.iq = np.full(self.N, 40000.0)
        self.flag = np.zeros(self.N, np.int8)              # 0 ok, 1 mis-indexed, 2 non-indexed, 3 B19'
        return self

    def _new_domain(self, grain, cname, host):
        did = len(self.domains)
        self.domains.append(dict(id=did, grain=grain, cls=cname, phi=self.classes[cname]['phi'], host=host))
        return did

    # ---------------------------------------------------------------- perturbations
    def perturb(self, noise_deg=0.0, gradient_deg_per_um=0.0, f_mis=0.0, f_ni=0.0, f_mart=0.0, w_mart_um=0.15,
                mart_pairs=None):
        """
        noise_deg           : std of the random rotation vector per point [deg] (Eq. synth_noise)
        gradient_deg_per_um : orientation gradient within each grain about a random axis, along a random direction
        f_mis, f_ni         : fractions of mis-indexed (random orientation, low CI) and non-indexed points (CI = -1)
        f_mart, w_mart_um   : fraction of the domain boundaries carrying a band of retained B19' of width w_mart_um
        mart_pairs          : restrict B19' bands to boundaries between these class pairs, e.g. [('M', '+114')]
        """
        rng = self.rng
        g = self.g_ideal.copy()
        if gradient_deg_per_um > 0:
            for gi in range(len(self.grains)):
                m = np.where(self.grain == gi)[0]
                ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
                e = rng.normal(size=3); e[2] = 0; e /= np.linalg.norm(e)
                s0 = self.S[m].mean(0)
                ang = gradient_deg_per_um * ((self.S[m] - s0) @ e)
                g[m] = np.einsum('nij,njk->nik', g[m], rotations(np.repeat(ax[None], len(m), 0), ang))
        if noise_deg > 0:
            w = rng.normal(0, noise_deg, size=(self.N, 3))
            ang = np.linalg.norm(w, axis=1); ax = w / np.maximum(ang[:, None], 1e-12)
            g = np.einsum('nij,njk->nik', g, rotations(ax, ang))
        self.g = g
        self.flag[:] = 0; self.phase[:] = self.phase_A; self.ci[:] = 0.9
        if f_mart > 0:
            nb = self._neighbour_pairs(1)
            dom = self.domain
            bnd = dom[nb[:, 0]] != dom[nb[:, 1]]
            pairs = nb[bnd]
            keys = np.unique(np.sort(np.stack([dom[pairs[:, 0]], dom[pairs[:, 1]]], 1), 1), axis=0)
            if mart_pairs:
                cn = np.array([d['cls'] for d in self.domains], dtype=object)
                ok = [tuple(sorted((cn[a], cn[b]))) in {tuple(sorted(p)) for p in mart_pairs} for a, b in keys]
                keys = keys[np.array(ok, bool)] if len(keys) else keys
            chosen = keys[rng.random(len(keys)) < f_mart]
            if len(chosen):
                cset = {(int(a), int(b)) for a, b in chosen}
                pts = [p for p, q in pairs if (min(dom[p], dom[q]), max(dom[p], dom[q])) in cset]
                pts = np.unique(pts) if pts else np.array([], int)
                if len(pts):
                    # widen to w_mart_um around the boundary points
                    from_pts = np.stack([self.X[pts], self.Y[pts]], 1)
                    mark = np.zeros(self.N, bool)
                    r2 = (w_mart_um / 2) ** 2
                    for x, y in from_pts:
                        near = (np.abs(self.X - x) <= w_mart_um / 2) & (np.abs(self.Y - y) <= w_mart_um / 2)
                        mark |= near & ((self.X - x) ** 2 + (self.Y - y) ** 2 <= r2)
                    self.phase[mark] = self.phase_M; self.flag[mark] = 3
                    self.g[mark] = random_rotations(mark.sum(), rng)
                    self.ci[mark] = 0.5
        ok = self.flag == 0
        if f_mis > 0:
            sel = ok & (rng.random(self.N) < f_mis)
            self.g[sel] = random_rotations(sel.sum(), rng); self.ci[sel] = 0.05; self.flag[sel] = 1
        if f_ni > 0:
            sel = (self.flag == 0) & (rng.random(self.N) < f_ni)
            self.ci[sel] = -1.0; self.flag[sel] = 2
        self.perturbation = dict(noise_deg=noise_deg, gradient_deg_per_um=gradient_deg_per_um, f_mis=f_mis,
                                 f_ni=f_ni, f_mart=f_mart, w_mart_um=w_mart_um, mart_pairs=mart_pairs)
        return self

    def _neighbour_pairs(self, shell=1):
        """First-shell neighbour pairs (i, j), i < j, on the hexagonal grid."""
        tol = 1.05 * self.step * shell
        order = np.lexsort((self.X, self.row))
        idx_of = {}
        for k in order:
            idx_of[(self.row[k], int(round(self.X[k] / (self.step / 2))))] = k
        pairs = []
        for k in range(self.N):
            r, c = self.row[k], int(round(self.X[k] / (self.step / 2)))
            for dr, dc in ((0, 2), (1, -1), (1, 1)):
                j = idx_of.get((r + dr, c + dc))
                if j is not None:
                    pairs.append((k, j))
        return np.array(pairs, int)

    # ---------------------------------------------------------------- output
    def _header(self):
        with open(self.template_ang) as f:
            head = []
            for line in f:
                if not line.startswith('#'):
                    break
                head.append(line)
        rep = {'# XSTEP:': f'# XSTEP: {self.step:.6f}\n', '# YSTEP:': f'# YSTEP: {self.dy:.6f}\n',
               '# NCOLS_ODD:': f'# NCOLS_ODD: {self.ncols_odd}\n', '# NCOLS_EVEN:': f'# NCOLS_EVEN: {self.ncols_even}\n',
               '# NROWS:': f'# NROWS: {self.nrows}\n', '# GRID:': '# GRID: HexGrid\n'}
        out = []
        for line in head:
            key = next((k for k in rep if line.startswith(k)), None)
            out.append(rep[key] if key else line)
        return ''.join(out)

    def write(self, fname):
        """Write the .ang file and the ground truth (<name>_truth.npz, <name>_truth.json)."""
        R = self.g.transpose(0, 2, 1)
        p1, P, p2 = R_to_euler(R)
        ni = self.flag == 2
        p1, P, p2 = [np.where(ni, 4 * np.pi, a) for a in (p1, P, p2)]
        with open(fname, 'w') as f:
            f.write(self._header())
            for k in range(self.N):
                f.write(f'{p1[k]:9.5f} {P[k]:9.5f} {p2[k]:9.5f} {self.X[k]:12.5f} {self.Y[k]:12.5f} '
                        f'{self.iq[k]:7.1f} {self.ci[k]:6.3f} {self.phase[k]:2d} {0:6d} {0.0:7.3f}\n')
        base = os.path.splitext(fname)[0]
        np.savez_compressed(base + '_truth.npz', X=self.X, Y=self.Y, grain=self.grain, domain=self.domain,
                            cls=self.cls, phi=self.phi, phase=self.phase, flag=self.flag, g_ideal=self.g_ideal,
                            g=self.g, class_names=np.array(self.class_names))
        meta = dict(template_ang=self.template_ang, size_um=self.size_um, step_um=self.step, seed=self.seed,
                    beta_star=self.beta_star, phase_A=self.phase_A, phase_M=self.phase_M,
                    map_to_sample=self.map_to_sample.tolist(), perturbation=getattr(self, 'perturbation', None),
                    grains=[dict(g_M=G['g_M'].tolist(), ds_index=G['ds_index'], ds=G['ds'].tolist(),
                                 seed_xy=G['seed_xy'].tolist(), alpha=G['alpha'], sub_alpha=G['sub_alpha'],
                                 offaxis=G['offaxis']) for G in self.grains],
                    domains=[{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()}
                             for d in self.domains])
        with open(base + '_truth.json', 'w') as f:
            json.dump(meta, f, indent=1)
        return fname

    # ---------------------------------------------------------------- self-checks
    def check(self, verbose=True):
        """Diagnostics on the IDEAL orientations: phi about d_s recovered by phi_about_axis for every on-axis point
        (error must be ~0), residual ~0, and the .ang Euler round trip. Returns a dict."""
        res = {}
        on = np.isfinite(self.phi)
        dsv = np.array([self.grains[i]['ds'] for i in self.grain])
        Q = np.einsum('nji,njk->nik', np.array([self.grains[i]['g_M'] for i in self.grain]), self.g_ideal)
        phi, r = phi_about_axis(Q[on], dsv[on])
        res['phi_err_max'] = float(np.abs(phi - self.phi[on]).max()) if on.any() else 0.0
        res['residual_max'] = float(r.max()) if on.any() else 0.0
        R = self.g.transpose(0, 2, 1); e = R_to_euler(R); R2 = euler_to_R(*e)
        res['euler_roundtrip_max_deg'] = float(rot_angle(np.einsum('nij,nkj->nik', R, R2)).max())
        res['n_points'] = int(self.N); res['n_domains'] = len(self.domains)
        res['class_area'] = {c: float((self.cls == i).mean()) for i, c in enumerate(self.class_names) if (self.cls == i).any()}
        if verbose:
            for k, v in res.items():
                print(f'{k:>24s}: {v}')
        return res

    def truth_csl_pairs(self, sigmas=('3', '9', '11', '27a', '27b'), brandon_k=15.0, ideal=True):
        """Neighbour pairs of B2 points in different domains with their CSL type from the ideal (or perturbed)
        orientations. Returns (pairs (n,2), sigma (n,) str)."""
        nb = self._neighbour_pairs(1)
        A = (self.phase[nb[:, 0]] == self.phase_A) & (self.phase[nb[:, 1]] == self.phase_A)
        nb = nb[A & (self.domain[nb[:, 0]] != self.domain[nb[:, 1]])]
        nb = nb[rot_angle(np.einsum('nij,nkj->nik', self.g_ideal[nb[:, 0]], self.g_ideal[nb[:, 1]])) > 1e-3]  # real boundaries
        G = self.g_ideal if ideal else self.g
        return nb, csl_of(G[nb[:, 0]], G[nb[:, 1]], sigmas, brandon_k)


# ----------------------------------------------------------------------------------------------------------------------
# validator
# ----------------------------------------------------------------------------------------------------------------------
class SyntheticValidator:
    """
    Compare pipeline results with the ground truth of a SyntheticEBSD map (object or *_truth.npz / .json).

    All methods take per-point arrays in the order of the .ang file (as read by pyebsd / ebsdlib) or per-cluster
    dicts, so the validator does not depend on ebsdlib internals.
    """

    def __init__(self, truth):
        if isinstance(truth, SyntheticEBSD):
            t = truth; self.X, self.Y = t.X, t.Y
            self.grain, self.domain, self.cls, self.phi = t.grain, t.domain, t.cls, t.phi
            self.phase, self.flag, self.g_ideal = t.phase, t.flag, t.g_ideal
            self.class_names = t.class_names; self.domains = t.domains; self.phase_A = t.phase_A
            self.grains = [dict(ds=G['ds'], g_M=G['g_M']) for G in t.grains]
        else:
            base = os.path.splitext(truth)[0].replace('_truth', '')
            z = np.load(base + '_truth.npz', allow_pickle=True)
            meta = json.load(open(base + '_truth.json'))
            for k in ('X', 'Y', 'grain', 'domain', 'cls', 'phi', 'phase', 'flag', 'g_ideal'):
                setattr(self, k, z[k])
            self.class_names = list(z['class_names']); self.domains = meta['domains']; self.phase_A = meta['phase_A']
            self.grains = [dict(ds=np.array(G['ds']), g_M=np.array(G['g_M'])) for G in meta['grains']]
        self.valid = (self.phase == self.phase_A) & (self.flag == 0)

    # ---------------------------------------------------------------- grains
    def score_grains(self, pred_grain, use=None):
        """pred_grain: per-point grain label (-1 / 0 = unassigned is counted as its own label).
        Returns ARI and per-true-grain completeness / per-predicted-grain purity (Eq. scores)."""
        use = self.valid if use is None else use
        t, p = self.grain[use], np.asarray(pred_grain)[use]
        out = dict(ARI=adjusted_rand_index(t, p), completeness={}, purity={})
        for g in np.unique(t):
            m = t == g; v, c = np.unique(p[m], return_counts=True); out['completeness'][int(g)] = float(c.max() / m.sum())
        for g in np.unique(p):
            m = p == g; v, c = np.unique(t[m], return_counts=True); out['purity'][int(g)] = float(c.max() / m.sum())
        return out

    # ---------------------------------------------------------------- clusters -> domains
    def match_clusters(self, labels):
        """labels: per-point cluster label (0 = none). Returns {cluster: dict(domain, cls, purity, n)} by majority."""
        labels = np.asarray(labels); out = {}
        for c in np.unique(labels[self.valid]):
            if c <= 0:
                continue
            m = (labels == c) & self.valid
            d, n = np.unique(self.domain[m], return_counts=True)
            k = n.argmax(); dom = int(d[k])
            out[int(c)] = dict(domain=dom, cls=self.domains[dom]['cls'], purity=float(n[k] / m.sum()), n=int(m.sum()))
        return out

    # ---------------------------------------------------------------- classes
    def score_classes(self, pred_class, mapping=None):
        """pred_class: per-point class name (str, '' = unclassified). `mapping` translates pipeline names to the
        generator names (e.g. {'+{114}': '+114'}). Sign convention: as in the pipeline, the sense covering most of the
        rotated area of each grain is positive, so the truth is sign-normalised per grain before comparison.
        Returns dict(accuracy, confusion (pandas DataFrame if available, else dict))."""
        pred = np.array([mapping.get(x, x) if mapping else x for x in pred_class], dtype=object)
        truth = self.normalised_truth_classes()
        m = self.valid
        acc = float(np.mean(pred[m] == truth[m]))
        labs = sorted(set(truth[m]) | set(pred[m]))
        conf = {a: {b: int(np.sum((truth[m] == a) & (pred[m] == b))) for b in labs} for a in labs}
        try:
            import pandas as pd
            conf = pd.DataFrame(conf).T.loc[labs, labs]; conf.index.name = 'truth \\ pred'
        except ImportError:
            pass
        return dict(accuracy=acc, confusion=conf)

    def sense_flip(self):
        """Per grain: -1 if the pipeline convention (majority of rotated area positive) flips the generator sign."""
        flip = {}
        for g in np.unique(self.grain):
            m = (self.grain == g) & self.valid & np.isfinite(self.phi) & (np.abs(self.phi) > 5)
            flip[int(g)] = -1 if (m.any() and (self.phi[m] < 0).sum() > (self.phi[m] > 0).sum()) else 1
        return flip

    def normalised_truth_classes(self):
        flip = self.sense_flip(); names = np.array(self.class_names, dtype=object)[self.cls]
        out = names.copy()
        sw = lambda s: ('-' + s[1:]) if s.startswith('+') else (('+' + s[1:]) if s.startswith('-') else s)
        for g, f in flip.items():
            if f < 0:
                m = self.grain == g
                out[m] = [sw(s) for s in names[m]]
        return out

    def score_phi(self, pred_phi):
        """pred_phi: per-point rotation about d_s [deg] (NaN = not evaluated). Error after sign normalisation."""
        flip = self.sense_flip(); f = np.array([flip[int(g)] for g in self.grain])
        t = self.phi * f; p = np.asarray(pred_phi, float)
        m = self.valid & np.isfinite(t) & np.isfinite(p)
        e = p[m] - t[m]
        return dict(n=int(m.sum()), mean=float(e.mean()) if m.any() else np.nan, std=float(e.std()) if m.any() else np.nan,
                    abs95=float(np.percentile(np.abs(e), 95)) if m.any() else np.nan)

    # ---------------------------------------------------------------- relationships
    def truth_relation(self, dom_a, dom_b, beta_star=6.8, tol=2.0):
        """Ideal relation between two domains from their rotations about d_s: 'same', 'twin{11n}', 'kink', 'other'."""
        a, b = self.domains[dom_a], self.domains[dom_b]
        if a['grain'] != b['grain']:
            return 'different grain'
        if a['phi'] is None or b['phi'] is None or (isinstance(a['phi'], float) and np.isnan(a['phi'])) \
                or (isinstance(b['phi'], float) and np.isnan(b['phi'])):
            return 'twin{114} off-axis' if 'M' in (a['cls'], b['cls']) else 'other'
        d = abs(a['phi'] - b['phi']); d = min(d % 180, 180 - d % 180)
        if d < tol:
            return 'same'
        for n, p in PHI_N.items():
            if abs(d - p) < tol:
                return f'twin{{11{n}}}'
        return 'kink' if d < 25 else 'other'

    def score_relations(self, pred_pairs, labels):
        """pred_pairs: iterable of (cluster_a, cluster_b, relation) from the pipeline, relation in
        {'twin{112}', ..., 'kink', 'none'}; labels: per-point cluster labels. Every pair of matched clusters is
        compared with truth_relation. Returns dict(confusion, accuracy, n)."""
        mc = self.match_clusters(labels)
        rows = []
        for a, b, rel in pred_pairs:
            if a in mc and b in mc:
                rows.append((self.truth_relation(mc[a]['domain'], mc[b]['domain']), rel))
        if not rows:
            return dict(n=0)
        t, p = map(np.array, zip(*rows))
        labs = sorted(set(t) | set(p))
        conf = {x: {y: int(np.sum((t == x) & (p == y))) for y in labs} for x in labs}
        return dict(n=len(rows), accuracy=float(np.mean(t == p)), confusion=conf)

    def score_csl(self, pred_pairs, pred_sigma, truth_pairs, truth_sigma):
        """CSL classification of neighbour pairs: pipeline (pairs, sigma) vs SyntheticEBSD.truth_csl_pairs()."""
        tk = {(min(i, j), max(i, j)): s for (i, j), s in zip(truth_pairs, truth_sigma)}
        pk = {(min(i, j), max(i, j)): s for (i, j), s in zip(pred_pairs, pred_sigma)}
        common = set(tk) & set(pk)
        labs = sorted({tk[k] for k in common} | {pk[k] for k in common})
        conf = {x: {y: sum(1 for k in common if tk[k] == x and pk[k] == y) for y in labs} for x in labs}
        frac_t = {s: float(np.mean([v == s for v in tk.values()])) for s in set(tk.values())}
        frac_p = {s: float(np.mean([v == s for v in pk.values()])) for s in set(pk.values())}
        return dict(n_common=len(common), accuracy=float(np.mean([tk[k] == pk[k] for k in common])) if common else np.nan,
                    confusion=conf, fractions_truth=frac_t, fractions_pred=frac_p)
