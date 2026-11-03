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


def ipf_colors(g, direction=(0, 0, 1)):
    """Standard cubic IPF colours (red <001>, green <101>, blue <111>) of the crystal directions parallel to the
    sample direction `direction`; g (N,3,3) sample -> crystal, or a per-point direction array (N,3) in the sample
    frame. Returns RGB (N,3) in [0, 1]."""
    g = np.asarray(g, float); d = np.asarray(direction, float)
    v = np.einsum('nij,nj->ni', g, d) if d.ndim == 2 else g @ (d / np.linalg.norm(d))
    v = np.sort(np.abs(v / np.linalg.norm(v, axis=1, keepdims=True)), axis=1)      # x <= y <= z
    x, y, z = v.T
    rgb = np.stack([z - y, y - x, x * np.sqrt(3)], 1)
    rgb = rgb / np.maximum(rgb.max(1, keepdims=True), 1e-12)
    return np.sqrt(np.clip(rgb, 0, 1))


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


# B2 compound twinning systems as in crystlibs.crystlib.niti_twinning (K1 plane, eta1 direction of the positive shear);
# the symmetry variants are generated by the proper cubic rotations (the pair (K1, eta1) is defined up to a joint sign)
TWIN_SYSTEMS = {'112': ([-2, 1, 1], [1, 1, 1]), '113': ([-3, 1, 1], [2, 3, 3]), '114': ([-4, 1, 1], [1, 2, 2]),
                '115': ([-5, 1, 1], [2, 5, 5])}


def twin_shear_direction(n_c, mode='114'):
    """eta1 (crystal, unit) of the positive twinning shear on the K1 plane n_c (crystal, any sign) of `mode`,
    signed consistently with n_c, from the variants of TWIN_SYSTEMS[mode]."""
    K, E = (np.asarray(v, float) for v in TWIN_SYSTEMS[mode])
    K /= np.linalg.norm(K); E /= np.linalg.norm(E)
    n_c = np.asarray(n_c, float) / np.linalg.norm(n_c)
    out = []
    for S in SYM:
        c = (S @ K) @ n_c
        if abs(abs(c) - 1) < 1e-6:
            out.append(np.sign(c) * (S @ E))
    eta = out[0]
    assert all(np.allclose(e, eta) for e in out), 'eta1 not unique for this K1'
    return eta


def twin_schmid(g_host, n_c, load_axis=(0, 0, 1), mode='114'):
    """Schmid factor (n.l)(eta1.l) of the twinning system on K1 = n_c (crystal of the host) for the loading direction
    `load_axis` (sample frame), as sf_parent of ebsdlib.check_twin_relationship (the twin side has -SF)."""
    l = np.asarray(load_axis, float); l /= np.linalg.norm(l)
    eta = twin_shear_direction(n_c, mode)
    N = g_host.T @ (np.asarray(n_c, float) / np.linalg.norm(n_c)); Et = g_host.T @ eta
    return float((N @ l) * (Et @ l))


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

    def add_grain(self, g_M=None, ds_index=None, seed_xy=None, layout='scheme', fractions=None,
                  mean_width_um=0.8, width_sigma=0.5, alpha_deg=None, sub_alpha_deg=None, sub_fraction=0.35,
                  offaxis=None, scheme=None):
        """
        Add a former grain.

        g_M        : (3,3) matrix orientation (sample -> crystal); default: <111> fibre (fiber_orientation)
        ds_index   : which of the six <110> of g_M is d_s (0..5); default: for layout='scheme' the <110> whose {114}
                     twins have the largest Schmid factor for tension along the wire axis, otherwise random
        seed_xy    : (X, Y) [um] of the Voronoi seed; default random (single grain: map centre)
        layout     : 'scheme' (default) -> crystallographic morphology of the variant scheme (Fig. 13 of the
                     functional paper), see _make_scheme: kink bands bounded by (100)_M = <100>_B2 planes _|_ d_s, and
                     (20-1)_M twins on the {114}_B2 plane of their own host (the two senses on the two mirror {114}
                     planes containing d_s -> V-shaped +/- pairs meeting at Sigma3), twins ending at the boundaries of
                     their host (Sigma11 / Sigma19a terminations at matrix|kink interfaces); `scheme` = dict of options;
                     'random' -> parallel lamellae drawn from `fractions` (one normal, sub-lamellae in kinks);
                     or an explicit list of (class, width_um) along the
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
            if layout == 'scheme' and (scheme or {}).get('select_by_load', True):
                # the <110> whose (20-1)_M = {114} twins are most favoured by the load becomes d_s
                la = (scheme or {}).get('load_axis', (0, 0, 1))
                sf = [max(twin_schmid(g_M, n, la) for n in self.twin_planes(FAM110[i]).values()) for i in range(6)]
                ds_index = int(np.argmax(sf))
            else:
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
        sch = None
        if layout == 'scheme':
            sch = self._make_scheme(g_M, ds_index, ds, mean_width_um, width_sigma, **(scheme or {}))
            layout = [('M', 1.0)]                          # unused for 'scheme' grains
        elif layout == 'random':
            layout = self._random_layout(fractions, mean_width_um, width_sigma, sub_fraction)
        self.grains.append(dict(g_M=g_M, ds_index=ds_index, ds=ds, seed_xy=np.asarray(seed_xy, float), layout=layout,
                                nL=nL, nS=nS, alpha=a, sub_alpha=sa, offaxis=offaxis, scheme=sch))
        return len(self.grains) - 1

    # ---------------------------------------------------------------- crystallographic ('scheme') morphology
    TWIN_CLASS = {('M', 1): '+114', ('M', -1): '-114', ('+K2', 1): '+113', ('+K2', -1): '-115',
                  ('-K2', 1): '+115', ('-K2', -1): '-113'}

    @staticmethod
    def twin_planes(u):
        """{114} plane normals (crystal) containing the <110> axis u, one per sense: n[sigma] such that the rotation
        R(u, sigma*phi114) maps n onto a {114} normal again, i.e. the K1 plane shared by host and twin."""
        u = np.asarray(u, float); u = u / np.linalg.norm(u)
        cand = []
        for perm in ([1, 1, 4], [1, 4, 1], [4, 1, 1]):
            for sg in np.ndindex(2, 2, 2):
                v = np.array(perm, float) * (-1) ** np.array(sg)
                if abs(v @ u) < 1e-9 and not any(np.allclose(v, -c) or np.allclose(v, c) for c in cand):
                    cand.append(v)
        cand = [c / np.linalg.norm(c) for c in cand]
        assert len(cand) == 2, cand
        out = {}
        for sigma in (1, -1):
            R = rotation(u, sigma * PHI_N[4])
            for n in cand:
                m = R @ n
                if any(abs(abs(m @ c) - 1) < 1e-6 for c in cand) and not np.allclose(abs(m @ n), 1):
                    out[sigma] = n
        assert set(out) == {1, -1} and not np.allclose(abs(out[1] @ out[-1]), 1), out
        return out

    @staticmethod
    def k1_planes(u, n=4):
        """{11n} plane normals (crystal) containing the <110> axis u, per sense sigma: the plane mapped onto a {11n}
        normal again by R(u, sigma*phi_n) (K1 shared by host and twin). n = 4 reproduces twin_planes."""
        u = np.asarray(u, float); u = u / np.linalg.norm(u)
        cand = []
        for perm in ([1, 1, n], [1, n, 1], [n, 1, 1]):
            for sg in np.ndindex(2, 2, 2):
                v = np.array(perm, float) * (-1) ** np.array(sg)
                if abs(v @ u) < 1e-9 and not any(np.allclose(v, -c) or np.allclose(v, c) for c in cand):
                    cand.append(v)
        cand = [c / np.linalg.norm(c) for c in cand]
        out = {}
        for sigma in (1, -1):
            R = rotation(u, sigma * PHI_N[n])
            for c in cand:
                m = R @ c
                if any(abs(abs(m @ d) - 1) < 1e-6 for d in cand) and not np.allclose(abs(m @ c), 1):
                    out[sigma] = c
        return out

    def link_diagnostics(self, grain=0, load_axis=(0, 0, 1), tol_deg=5.0):
        """Which relations of the matrix to the other classes present in a grain the pipeline will take as twin
        links (rotation about d_s within tol_deg of a {11n} twin rotation) and the Schmid factor of that system in
        the matrix: a negative value means the link points INTO the matrix, so the matrix is no source of the
        directed network and cannot be chosen as root. Returns a list of dicts."""
        G = self.grains[grain]; u = FAM110[G['ds_index']]
        present = sorted({self.class_names[i] for i in np.unique(self.cls[self.grain == grain])})
        out = []
        for c in present:
            phi = self.classes[c]['phi']
            if c == 'M' or not np.isfinite(phi):
                continue
            for n in (2, 3, 4, 5):
                for sigma in (1, -1):
                    d = abs(phi - sigma * PHI_N[n])
                    if d <= tol_deg:
                        k1 = self.k1_planes(u, n)[sigma]
                        out.append(dict(cls=c, phi=phi, mode=f'11{n}', dev=d, sf_matrix=twin_schmid(G['g_M'], k1, load_axis,
                                                                                                  mode=f'11{n}')))
        return out

    def _strips(self, R, frac, mean_w, sig):
        """Random strips [x0, x1) along a coordinate in [-R, R] covering the fraction `frac`."""
        out, x = [], -R
        if frac <= 0:
            return out
        gap_mean = mean_w * (1 - frac) / frac
        x += self.rng.uniform(0, gap_mean + mean_w)
        while x < R:
            w = float(mean_w * np.exp(self.rng.normal(0, sig)))
            out.append((x, x + w)); x += w + float(gap_mean * np.exp(self.rng.normal(0, sig)))
        return out

    @staticmethod
    def _flip(name):
        return ('-' + name[1:]) if name.startswith('+') else (('+' + name[1:]) if name.startswith('-') else name)

    def _make_scheme(self, g_M, ds_index, ds, mean_w, sig, kink_fraction=0.25, kink_width_um=None,
                     kink_classes=None, twin_fractions=None, v_pairs=None, extent_um=None, load_axis=(0, 0, 1),
                     select_by_load=True, kink_scatter_deg=0.0):
        """Random realisation of the scheme morphology of one grain, in the plane _|_ d_s (everything is invariant
        along d_s). Coordinates relative to the grain seed: a = s.e_a (e_a = the <100>_B2 _|_ d_s, normal of the
        (100)_M kink-band boundaries), b = s.e_b (e_b = d_s x e_a).
        kink_classes  : weights of the kink classes, default {'+K2': .35, '-K2': .25, '+K1': .2, '-K1': .2}
        twin_fractions: {host: {sigma: area fraction of the host}}, default M: +0.35/-0.03, K+: +0.20/-0.15, K-: +0.15/-0.20
        v_pairs       : {host: number of V-shaped +/- pairs (both lamellae start at a common apex and open to +e_b)}
        kink_scatter_deg: std of a random deviation of the rotation of every kink band from its ideal value
                        (+-beta*, +-2beta*), as measured (Table 4 of the functional paper: 1.5-2.6 deg); the twins of a
                        kinked host keep their exact twin relation to the host (rotation kappa + dphi +- phi114).
                        0 (default): ideal kinks, maps identical to before.
        select_by_load: the fractions are given for the parent variant V1 (+ sense); the sense of V1 is selected by the
                        loading: the sense whose (20-1)_M = {114} twins have the larger Schmid factor in the matrix
                        (tension along load_axis) is V1 -- otherwise the whole scheme is mirrored (all signs
                        reversed). In every host, a twin family is kept only if its Schmid factor in that host is
                        positive (the dominant family of the matrix always), as the pipeline directs every twin link
                        from the side with the positive Schmid factor (the twin side has -SF)."""
        u = FAM110[ds_index]
        c100 = np.eye(3)[[abs(u @ e) < 1e-9 for e in np.eye(3)].index(True)]
        e_a = g_M.T @ c100; e_a /= np.linalg.norm(e_a); e_b = np.cross(ds, e_a)
        n_sig = self.twin_planes(u)
        R = extent_um or 1.2 * np.hypot(*self.size_um)
        sf_M = {sg: twin_schmid(g_M, n_sig[sg], load_axis) for sg in (1, -1)}
        sense = (1 if sf_M[1] >= sf_M[-1] else -1) if select_by_load else 1
        kink_classes = kink_classes or {'+K2': 0.35, '-K2': 0.25, '+K1': 0.2, '-K1': 0.2}
        twin_fractions = twin_fractions or {'M': {1: 0.35, -1: 0.03}, '+K2': {1: 0.20, -1: 0.15},
                                            '-K2': {1: 0.15, -1: 0.20}}
        v_pairs = {'M': 2, '+K2': 1, '-K2': 1} if v_pairs is None else v_pairs
        if sense < 0:                                        # V1 is the type-2 variant: mirror the whole scheme
            kink_classes = {self._flip(k): v for k, v in kink_classes.items()}
            twin_fractions = {self._flip(h): {-sg: f for sg, f in d.items()} for h, d in twin_fractions.items()}
            v_pairs = {self._flip(h): n for h, n in v_pairs.items()}
        kw = kink_width_um or 2.0 * mean_w
        r_apex = 0.3 * min(self.size_um)                  # V apices within the central part of the map
        names = list(kink_classes); p = np.array([kink_classes[k] for k in names], float); p /= p.sum()
        kstrips = self._strips(R, kink_fraction, kw, sig)
        kc = [names[self.rng.choice(len(names), p=p)] for _ in kstrips]
        near = np.argsort([abs(a0 + a1) for a0, a1 in kstrips])[:len(names)]      # the kinks closest to the seed get
        for i, c in zip(near, self.rng.permutation(names)):                     # every kink class once
            kc[i] = c
        kinks = [(a0, a1, c) for (a0, a1), c in zip(kstrips, kc)]
        kink_dphi = (self.rng.normal(0, kink_scatter_deg, len(kinks)).tolist() if kink_scatter_deg > 0
                     else [0.0] * len(kinks))
        hosts = {}
        for h in ['M'] + sorted({k[2] for k in kinks}):
            kappa = self.classes[h]['phi']
            g_h = g_M @ rotation(ds, kappa)
            fam = {}
            sf_h = {sg: twin_schmid(g_h, n_sig[sg], load_axis) for sg in (1, -1)}
            dominant = max(twin_fractions.get(h, {1: 0}), key=lambda sg: twin_fractions.get(h, {}).get(sg, 0))
            for sigma in (1, -1):
                N = g_h.T @ n_sig[sigma]; N /= np.linalg.norm(N); T = np.cross(ds, N)
                frac = twin_fractions.get(h, {}).get(sigma, 0.0) if (h, sigma) in self.TWIN_CLASS else 0.0
                if select_by_load and sf_h[sigma] <= 0 and not (h == 'M' and sigma == dominant):
                    frac = 0.0                              # twins only where their Schmid factor in the host is > 0
                strips = [dict(t0=t0, t1=t1, start=None) for t0, t1 in self._strips(R, frac, mean_w, sig)]
                fam[sigma] = dict(N=N, T=T, strips=strips)
            # V-shaped pairs: both senses start at a common apex inside the host and open towards +e_b
            nv = v_pairs.get(h, 0) if (h, 1) in self.TWIN_CLASS else 0
            if select_by_load and min(sf_h.values()) <= 0:
                nv = 0                                       # a V pair needs both senses favoured
            for _ in range(nv):
                if h == 'M':
                    for _try in range(50):
                        a = self.rng.uniform(-r_apex, r_apex)
                        if not any(a0 <= a < a1 for a0, a1, _ in kinks):
                            break
                else:
                    ks = [k for k in kinks if k[2] == h]; a0, a1, _ = ks[self.rng.integers(len(ks))]
                    a = self.rng.uniform(a0, a1)
                apex = a * e_a + self.rng.uniform(-r_apex, r_apex) * e_b
                w = float(mean_w * np.exp(self.rng.normal(0, sig)))
                for sigma in (1, -1):
                    F = fam[sigma]; tc = apex @ F['N']
                    d = 1.0 if F['T'] @ e_b >= 0 else -1.0
                    F['strips'].append(dict(t0=tc - w / 2, t1=tc + w / 2, start=(apex @ F['T'], d)))
            fam['sf'] = sf_h
            hosts[h] = fam
        return dict(e_a=e_a, e_b=e_b, n_sig={k: v.tolist() for k, v in n_sig.items()}, kinks=kinks, hosts=hosts,
                    kink_dphi=kink_dphi, kink_scatter_deg=kink_scatter_deg,
                    extent_um=R, sense=sense, sf_matrix=sf_M, load_axis=list(load_axis))

    def _scheme_assign(self, G, S, return_dphi=False):
        """Class names and domain keys of points S (n,3, sample frame, absolute) of a 'scheme' grain (and the
        deviation of the rotation of their kink host from its ideal value, return_dphi=True)."""
        sch = G['scheme']; s0 = np.array([*G['seed_xy'], 0.0]) @ self.map_to_sample.T
        D = S - s0; a = D @ sch['e_a']
        host = np.array(['M'] * len(S), dtype=object); hkey = np.array(['M'] * len(S), dtype=object)
        dphi = np.zeros(len(S))
        kd = sch.get('kink_dphi') or [0.0] * len(sch['kinks'])
        for i, (a0, a1, c) in enumerate(sch['kinks']):
            sel = (a >= a0) & (a < a1); host[sel] = c; hkey[sel] = f'K{i}'; dphi[sel] = kd[i]
        cname = host.copy(); key = hkey.copy()
        for h, fam in sch['hosts'].items():
            mh = host == h
            if not mh.any():
                continue
            done = np.zeros(len(S), bool)
            for sigma in (1, -1):                           # + sense first (priority at the V apex)
                if sigma not in fam:
                    continue
                F = fam[sigma]; t = D[mh] @ F['N']; tt = D[mh] @ F['T']
                for j, st in enumerate(F['strips']):
                    c = (t >= st['t0']) & (t < st['t1'])
                    if st['start'] is not None:
                        c &= (tt - st['start'][0]) * st['start'][1] >= 0
                    idx = np.where(mh)[0][c & ~done[mh]]
                    cname[idx] = self.TWIN_CLASS[(h, sigma)]
                    key[idx] = [f'{k}/T{"+" if sigma > 0 else "-"}{j}' for k in hkey[idx]]
                    done[idx] = True
        return (cname, key, dphi) if return_dphi else (cname, key)

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
            if G.get('scheme') is not None:                # crystallographic morphology
                cname, key, dphi = self._scheme_assign(G, self.S[m], return_dphi=True)
                ids = {}
                for k in sorted(set(key), key=lambda x: ('/' in x, x)):     # hosts before their twins
                    c0 = cname[np.where(key == k)[0][0]]
                    ids[k] = self._new_domain(gi, c0, ids.get(k.split('/')[0]) if '/' in k else None)
                dom_id = np.array([ids[k] for k in key])
                G['t0'], G['sub0'] = 0.0, {}
            else:
                dom_id, cname = self._stack_assign(gi, G, t, tsub, idx)
                dphi = np.zeros(len(idx))
            # off-axis twin lamella (overrides)
            if G['offaxis']:
                sel = self._offaxis_mask(G, self.S[m] - s0)
                did = self._new_domain(gi, 'OA114', None)
                dom_id[sel] = did; cname[sel] = 'OA114'
            self.domain[idx] = dom_id
            self.cls[idx] = [self.class_names.index(c) for c in cname]
            # orientations
            phis = np.array([self.classes[c]['phi'] for c in cname], float) + dphi      # kink scatter (scheme)
            for d in np.unique(dom_id):                     # actual rotation of every domain
                self.domains[d]['phi'] = float(phis[np.where(dom_id == d)[0][0]])
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

    def _stack_assign(self, gi, G, t, tsub, idx):
        """Parallel-lamella layout ('random' or explicit list)."""
        cum = np.cumsum([0.0] + [L[1] for L in G['layout']])
        tt = t - t.min() + 1e-9 if len(t) else t
        G['t0'] = float(t.min()) - 1e-9 if len(t) else 0.0; G['sub0'] = {}
        tt = np.mod(tt, cum[-1])
        k = np.searchsorted(cum, tt, side='right') - 1
        dom_id = np.full(len(idx), -1); cname = np.empty(len(idx), dtype=object)
        for li, L in enumerate(G['layout']):
            sel = k == li
            if not sel.any():
                continue
            if len(L) == 3:                            # kink with sub-lamellae
                cs = np.cumsum([0.0] + [w for _, w in L[2]])
                G['sub0'][li] = float(tsub[sel].min())
                u = np.mod(tsub[sel] - tsub[sel].min(), cs[-1]); kk = np.searchsorted(cs, u, side='right') - 1
                host = self._new_domain(gi, L[0], None)
                for si, (sc, _) in enumerate(L[2]):
                    ss = np.where(sel)[0][kk == si]
                    did = host if sc == L[0] else self._new_domain(gi, sc, host)
                    dom_id[ss] = did; cname[ss] = sc
            else:
                did = self._new_domain(gi, L[0], None)
                dom_id[sel] = did; cname[sel] = L[0]
        return dom_id, cname

    def _offaxis_mask(self, G, D):
        """Points (sample-frame offsets D from the grain seed) inside the off-axis {114} twin lamella: a lamella on the
        {114} K1 plane containing the other <110> axis u_oa of the matrix about which the twin is rotated (the same
        u_oa as for its orientation), width_um, centred at position_um along the plane normal."""
        oa = G['offaxis']
        u_oa = FAM110[(G['ds_index'] + 2) % 6]
        N = G['g_M'].T @ self.k1_planes(u_oa, 4)[1]; N /= np.linalg.norm(N)
        return np.abs(D @ N - oa.get('position_um', 0.0)) < oa.get('width_um', 1.0) / 2

    def _new_domain(self, grain, cname, host):
        did = len(self.domains)
        self.domains.append(dict(id=did, grain=grain, cls=cname, phi=self.classes[cname]['phi'], host=host))
        return did

    # ---------------------------------------------------------------- 3D geometry (sections other than the map)
    def classes_at(self, S, grain=None):
        """Domain class of arbitrary points S (n,3) in the SAMPLE frame [um] from the same 3D lamella geometry as the
        map (lamellae are planes containing d_s, so they extend into the depth of the wire). Grains are taken as
        columnar along z (Voronoi of the seeds in the section plane, a visualisation assumption). Returns
        (grain (n,), class names (n,) object, phi (n,))."""
        S = np.asarray(S, float)
        if grain is None:
            seeds = np.array([np.array([*g['seed_xy'], 0.0]) @ self.map_to_sample.T for g in self.grains])
            grain = ((S[:, None, :2] - seeds[None, :, :2]) ** 2).sum(-1).argmin(1)
        cname = np.empty(len(S), dtype=object)
        for gi, G in enumerate(self.grains):
            m = np.where(grain == gi)[0]
            if not len(m):
                continue
            s0 = np.array([*G['seed_xy'], 0.0]) @ self.map_to_sample.T
            if G.get('scheme') is not None:
                out, _ = self._scheme_assign(G, S[m])
                if G['offaxis']:
                    out[self._offaxis_mask(G, S[m] - s0)] = 'OA114'
                cname[m] = out
                continue
            t = (S[m] - s0) @ G['nL']; tsub = (S[m] - s0) @ G['nS']
            cum = np.cumsum([0.0] + [L[1] for L in G['layout']])
            k = np.searchsorted(cum, np.mod(t - G['t0'], cum[-1]), side='right') - 1
            out = np.empty(len(m), dtype=object)
            for li, L in enumerate(G['layout']):
                sel = k == li
                if not sel.any():
                    continue
                if len(L) == 3:
                    cs = np.cumsum([0.0] + [w for _, w in L[2]])
                    u = np.mod(tsub[sel] - G['sub0'].get(li, 0.0), cs[-1]); kk = np.searchsorted(cs, u, side='right') - 1
                    out[np.where(sel)[0]] = [L[2][q][0] for q in kk]
                else:
                    out[sel] = L[0]
            if G['offaxis']:
                out[self._offaxis_mask(G, S[m] - s0)] = 'OA114'
            cname[m] = out
        phi = np.array([self.classes[c]['phi'] for c in cname], float)
        return grain, cname, phi

    def projection_view(self, grain=0, half_width_um=None, n=400):
        """Morphology of a grain seen along d_s (the view of the variant scheme): grid in the plane _|_ d_s through
        the grain seed, horizontal e_a = <100>_B2 _|_ d_s (normal of the kink-band boundaries), vertical e_b.
        Returns dict(a, b, cls (n,n), phi (n,n))."""
        G = self.grains[grain]; assert G.get('scheme') is not None, 'projection_view needs a scheme grain'
        sch = G['scheme']; L = half_width_um or 0.5 * max(self.size_um)
        a = np.linspace(-L, L, n); b = np.linspace(-L, L, n); A, B = np.meshgrid(a, b)
        s0 = np.array([*G['seed_xy'], 0.0]) @ self.map_to_sample.T
        S = s0 + A.ravel()[:, None] * sch['e_a'] + B.ravel()[:, None] * sch['e_b']
        cname, _ = self._scheme_assign(G, S)
        phi = np.array([self.classes[c]['phi'] for c in cname], float)
        return dict(a=a, b=b, cls=cname.reshape(A.shape), phi=phi.reshape(A.shape))

    def longitudinal_section(self, angle_deg=0.0, through_xy=None, height_um=None, step_um=None):
        """Points of a section containing the wire axis z (loading axis vertical), through the map point through_xy
        (default: map centre) along the map direction at angle_deg from the map X axis. Returns dict with the in-plane
        coordinate u [um] (horizontal), z [um] (vertical, z = 0 is the map plane), grain, class names, phi."""
        x0, y0 = through_xy if through_xy is not None else (self.size_um[0] / 2, self.size_um[1] / 2)
        e = np.array([np.cos(angle_deg * DEG), np.sin(angle_deg * DEG), 0.0])
        L = abs(self.size_um[0] * e[0]) + abs(self.size_um[1] * e[1])
        h = height_um or max(self.size_um); st = step_um or max(self.step, L / 400)
        u = np.arange(-L / 2, L / 2 + st / 2, st); z = np.arange(-h / 2, h / 2 + st / 2, st)
        U, Z = np.meshgrid(u, z)
        P = np.stack([x0 + U.ravel() * e[0], y0 + U.ravel() * e[1], np.zeros(U.size)], 1)   # map coordinates
        S = P @ self.map_to_sample.T; S[:, 2] = Z.ravel()                                       # sample frame, z = wire axis
        inside = (P[:, 0] >= 0) & (P[:, 0] <= self.size_um[0]) & (P[:, 1] >= 0) & (P[:, 1] <= self.size_um[1])
        grain, cname, phi = self.classes_at(S)
        cname[~inside] = None; phi[~inside] = np.nan
        return dict(u=u, z=z, shape=U.shape, grain=grain.reshape(U.shape), cls=cname.reshape(U.shape),
                    phi=phi.reshape(U.shape), angle_deg=angle_deg, through_xy=(x0, y0))

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
                                 offaxis=G['offaxis'],
                                 scheme=None if G.get('scheme') is None else dict(
                                     e_a=G['scheme']['e_a'].tolist(), e_b=G['scheme']['e_b'].tolist(),
                                     kinks=G['scheme']['kinks'], n_sig=G['scheme']['n_sig'],
                                     kink_dphi=G['scheme'].get('kink_dphi'),
                                     sense=G['scheme']['sense'], sf_matrix=G['scheme']['sf_matrix'],
                                     sf_hosts={h: f['sf'] for h, f in G['scheme']['hosts'].items()}))
                            for G in self.grains],
                    domains=[{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in d.items()}
                             for d in self.domains])
        with open(base + '_truth.json', 'w') as f:
            json.dump(meta, f, indent=1)
        return fname

    def grain_polygons(self, margin=None):
        """Polygons (X, Y) [um] of the true grains: Voronoi cells of the seeds clipped to the map rectangle (enlarged
        by `margin`, default half a step, so that the edge points are inside). Returns a list in grain order."""
        m = 0.5 * self.step if margin is None else margin
        x1, y1 = self.X.max() + m, self.Y.max() + m
        rect = np.array([[-m, -m], [x1, -m], [x1, y1], [-m, y1]])
        seeds = np.array([g['seed_xy'] for g in self.grains])
        polys = []
        for i, p in enumerate(seeds):
            poly = rect.copy()
            for j, q in enumerate(seeds):
                if i == j:
                    continue
                n = q - p; c = n @ (p + q) / 2               # keep x with n.x <= c
                v = poly @ n - c; out = []
                for k in range(len(poly)):
                    a, b, va, vb = poly[k], poly[(k + 1) % len(poly)], v[k], v[(k + 1) % len(poly)]
                    if va <= 0:
                        out.append(a)
                    if va * vb < 0:
                        out.append(a + (b - a) * va / (va - vb))
                poly = np.array(out)
                if len(poly) == 0:
                    break
            polys.append(poly)
        return polys

    def write_truth_rois(self, ang_fname, check=True):
        """Write the true grains as ROI polygons next to the .ang, in the format of the hand-drawn ROIs of the
        pipeline (<ang>_my_rois.h5: attrs format='rois-only', group 'rois' with attr n_rois, datasets roi_<k> (n,2)
        X, Y in um; roi_<k> = grain k). With check=True, every point must fall into the polygon of its own grain."""
        import h5py
        polys = self.grain_polygons()
        fn = os.path.splitext(ang_fname)[0] + '_my_rois.h5'
        with h5py.File(fn, 'w') as f:
            f.attrs['format'] = 'rois-only'
            g = f.create_group('rois'); g.attrs['n_rois'] = np.int64(len(polys))
            for k, P in enumerate(polys):
                g.create_dataset(f'roi_{k}', data=np.asarray(P, float))
        if check:
            from matplotlib.path import Path
            pts = np.stack([self.X, self.Y], 1); bad = 0
            for k, P in enumerate(polys):
                inside = Path(P).contains_points(pts, radius=1e-9)
                bad += int(np.sum(inside != (self.grain == k)))
            print(f'{os.path.basename(fn)}: {len(polys)} grain ROIs, points outside their own ROI: {bad}')
        return fn

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

    @staticmethod
    def _swap(s):
        return ('-' + s[1:]) if s.startswith('+') else (('+' + s[1:]) if s.startswith('-') else s)

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


# ----------------------------------------------------------------------------------------------------------------------
# reading the pipeline checkpoints (figs/<condition>/grain_XXXX.h5) without ebsdlib
# ----------------------------------------------------------------------------------------------------------------------
import io as _io
import pickle as _pickle


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, st): self.__dict__.update(st if isinstance(st, dict) else {'_st': st})


class _StubUnpickler(_pickle.Unpickler):
    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except Exception:
            return type(name, (_Stub,), {})


def _unpickle(group, key):
    return _StubUnpickler(_io.BytesIO(group[key][()].tobytes())).load()


def phi_class(phi, res, beta_star=6.8, res_tol=10.0):
    """Class of a rotation phi [deg] about d_s relative to the matrix, with the windows of the paper (Fig. 12):
    matrix |phi| < 5; twins within +-10 deg of phi_n of {112}..{115}, adjacent windows split at the mid-angle;
    kinks |phi| < 10 (~beta*) and 10-20 (~2 beta*); '' for an off-axis residual >= res_tol or no window."""
    if not np.isfinite(phi) or res >= res_tol:
        return ''
    a, sg = abs(phi), '+' if phi > 0 else '-'
    if a < 5:
        return 'M'
    if a < 10:
        return sg + 'K1'
    if a < 20:
        return sg + 'K2'
    cen = sorted((p, f'11{n}') for n, p in PHI_N.items())          # 31.6 (115), 38.9 (114), 50.5 (113), 70.5 (112)
    for i, (p, name) in enumerate(cen):
        lo = max(p - 10, (cen[i - 1][0] + p) / 2) if i else p - 10
        hi = min(p + 10, (p + cen[i + 1][0]) / 2) if i + 1 < len(cen) else p + 10
        if lo <= a < hi:
            return sg + name
    return ''


def load_pipeline_grain(path):
    """One grain checkpoint of the pipeline -> dict(labels (N,), parent g (3,3), d_s, root, clusters
    {id: dict(g, n, cls_pipeline, mode, phi, res, cls)}). phi: rotation about d_s relative to the parent with the
    convention of SyntheticEBSD (g_c = g_P R(d_s, phi)); cls: phi_class (pipeline windows)."""
    import h5py
    with h5py.File(path, 'r') as f:
        p = f['pickled']
        aux, rep = _unpickle(p, 'aux'), _unpickle(p, 'report')
        nm = _unpickle(p, 'notmerged_new_result').__dict__
    labels = np.asarray(nm['labels']).ravel()
    Gp = np.asarray(rep['parent_mat'], float)
    ds = np.asarray(aux['d_s'], float); ds /= np.linalg.norm(ds)
    twins = {r['cluster_b']: r for r in rep['twin_results'] if r['is_match']}
    kinks = {r['cluster_b']: r for r in rep.get('kink_results', []) if r['is_match']}
    matrix = set(rep.get('merged_fragments') or []) | {rep.get('merged_root')}
    out = {}
    for cid, G in nm['avg_orientations'].items():
        G = np.asarray(G, float)
        Q = np.einsum('ji,sjk,kl->sil', Gp, SYM, G)                   # g_P^T S g_c
        phi, r = phi_about_axis(Q, ds); k = int(np.argmin(r))
        ph = (phi[k] + 90) % 180 - 90
        cp = 'matrix' if cid in matrix else 'twin' if cid in twins else 'kink' if cid in kinks else 'noOR'
        out[int(cid)] = dict(g=G, n=int((labels == cid).sum()), cls_pipeline=cp,
                             mode=str(twins[cid].get('mode')) if cid in twins else '', phi=float(ph), res=float(r[k]),
                             cls=phi_class(ph, r[k]) or ('OA114' if (cid in twins and r[k] >= 10.0) else ''))
    return dict(labels=labels, parent=Gp, d_s=ds, root=rep.get('merged_root'), clusters=out,
                root_selection=aux.get('root_selection'))


def score_condition(cond_dir, truth, verbose=True):
    """Score all grain checkpoints of a pipeline condition folder (figs/synthetic/<condition>) against the ground
    truth (*_truth.npz of the same map). Per grain: d_s error, matrix error (parent vs true matrix of the grain with
    most points in it), area fraction of correctly classified points (sign-normalised per grain as in the pipeline,
    clustered B2 points), class confusion. Returns a list of dicts (one per grain)."""
    import glob as _glob
    V = SyntheticValidator(truth)
    names = np.array(V.class_names, dtype=object)[V.cls]
    res = []
    for path in sorted(_glob.glob(os.path.join(cond_dir, 'grain_*.h5'))):
        if os.path.basename(path) == 'grain_0000.h5':
            continue
        P = load_pipeline_grain(path)
        lab = P['labels']; on = (lab > 0) & V.valid
        tg = np.bincount(V.grain[on]).argmax()                         # true grain of this ROI
        Gt = V.grains[tg]
        ds_err = float(np.degrees(np.arccos(min(1.0, abs(P['d_s'] @ Gt['ds'])))))
        mat_err = float(disorientation(P['parent'][None], Gt['g_M'][None])[0])
        pred = np.array([''] * len(lab), dtype=object)
        for cid, c in P['clusters'].items():
            pred[lab == cid] = c['cls']
        # sign normalisation per grain: the sense covering most of the rotated area is +
        def norm(arr, m):
            rot = np.array([x.startswith('-') for x in arr[m]]).sum(), np.array([x.startswith('+') for x in arr[m]]).sum()
            if rot[0] > rot[1]:
                arr = arr.copy(); sw = SyntheticValidator._swap
                arr[m] = [sw(x) for x in arr[m]]
            return arr
        m = on & (V.grain == tg)
        predn = norm(pred, m); truthn = norm(names.copy(), m)
        acc = float(np.mean(predn[m] == truthn[m]))
        labs = sorted(set(truthn[m]) | set(predn[m]))
        conf = {a: {b: int(np.sum((truthn[m] == a) & (predn[m] == b))) for b in labs if np.sum((truthn[m] == a) & (predn[m] == b))}
                for a in labs}
        root = P['root']; rootcls = names[lab == root] if root is not None else []
        rootcls = max(set(rootcls), key=list(rootcls).count) if len(rootcls) else ''
        r = dict(grain_file=os.path.basename(path), true_grain=int(tg), n_points=int(m.sum()), ds_err_deg=ds_err,
                 matrix_err_deg=mat_err, root_true_class=rootcls, class_accuracy=acc, confusion=conf)
        res.append(r)
        if verbose:
            print(f"{r['grain_file']}: true grain {tg}, d_s error {ds_err:.2f} deg, matrix error {mat_err:.2f} deg "
                  f"(root = true '{rootcls}'), class accuracy {100 * acc:.1f} % of {r['n_points']} points")
            for a in labs:
                row = conf.get(a, {})
                if row:
                    print('   truth %-6s -> ' % (a or '-') + ', '.join(f"{b or '-'}: {n}" for b, n in sorted(row.items(), key=lambda x: -x[1])))
    return res
