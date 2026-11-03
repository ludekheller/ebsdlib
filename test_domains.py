"""Diagnostics of the domain functions of ebsdlib.py (rotation about d_s, classes, Sigma from dphi, laminate) against the reference implementations it was moved from.
  3.1  rotation_about_axis    vs micromod/figures/grain_substructure.rotation_about_axis (all clusters of the grains)
  3.3  twin_windows/phi_class vs fig10_grain_examples.windows / phi_class (dense phi grid)
  3.8  classify_domains       vs fig13_variant_map.grain_classes; psi_angle vs grain_examples_panels.psi_angle
  3.10 sigma_from_dphi        vs s3_rotation_pairs.py (Sigma3 share of the boundary length, psi < / >= 10 deg)
  3.9  laminate functions     vs the own functions of the pipeline notebook stage trstrain_maps before the move
                              (ebsd_figs_pipeline_v2_backup_before_domains.ipynb, cell source executed),
                              on the NiTi correspondence matrices (crystlibs, Waitz notation) and random orientations
Needs the ebsdlib environment (imports ebsdlib.py of this folder). Writes nothing.
Run:  MICROMOD_ROIS=auto python test_domains.py [state ...]       (default def30_500C def50_500C)"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
GITHUB = os.environ.get('GITHUB', '/home/lheller/Documents/GitHub')
MICROMOD = os.environ.get('MICROMOD', '/home/lheller/Documents/papers/mypapers/micromod')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.append(GITHUB)   # crystlibs, getphases, ...
import ebsdlib as dm                             # ebsdlib.py of this folder (functions of Sections 3.1-3.10)
sys.path.insert(0, os.path.join(MICROMOD, 'figures'))
import matplotlib; matplotlib.use('Agg')
import grain_substructure as gs, fig10_grain_examples as fge, fig13_variant_map as f13, grain_examples_panels as gep
states = sys.argv[1:] or ['def30_500C', 'def50_500C']
ok_all = True
def report(name, ok, extra=''):
    global ok_all; ok_all &= bool(ok); print(f'{name:48s} {"OK" if ok else "DIFFERENT"} {extra}')

# 3.3 windows and classes
report('twin_windows', dm.twin_windows() == fge.windows(), str([tuple(round(v, 2) if isinstance(v, float) else v for v in w) for w in dm.twin_windows()]))
grid = np.arange(-90, 90.001, 0.01)
report('phi_class (phi grid, 0.01 deg)', all(dm.phi_class(p) == fge.phi_class(p) for p in grid))
report('lcv_class (phi grid, 0.01 deg)', all(dm.lcv_class(p) == f13.lcv_class(p) for p in grid))

# 3.1, 3.8 on the grains
nclu, dphi_max, dres_max, cls_diff, psi_diff, sgn_diff = 0, 0.0, 0.0, 0, 0.0, 0
for st in states:
    for g in gs.load_state(st):
        import h5py
        path = os.path.join(gs.sdir(st), 'grain_%04d.h5' % g['grain'])
        with h5py.File(path, 'r') as f:
            ao = gs._load(f['pickled'], 'notmerged_new_result').__dict__['avg_orientations']
        cids = [r['cid'] for r in g['clusters']]
        Gc = np.array([np.asarray(ao[c], float) for c in cids])
        p, r = dm.rotation_about_axis(g['Gp'], Gc, g['d_s'])
        ref = np.array([[x['phi'], x['res']] for x in g['clusters']])
        nclu += len(cids)
        dphi_max = max(dphi_max, float(np.abs(p - ref[:, 0]).max())); dres_max = max(dres_max, float(np.abs(r - ref[:, 1]).max()))
        cls_diff += sum(a != b for a, b in zip([dm.classify_domains(g['clusters'])[c] for c in cids],
                                                [f13.grain_classes(g)[c] for c in cids]))
        sgn_diff += int(dm.signed_rotations(g['clusters']) != gep.signed_rows(g))
        psi_diff = max(psi_diff, abs(dm.psi_angle(g['Gp'], g['d_s'], g['axial']) - gep.psi_angle(g['Gp'], g['d_s'], g['axial'])))
report(f'rotation_about_axis ({nclu} clusters, {gs.ROISET})', dphi_max < 1e-9 and dres_max < 1e-9,
       f'max |dphi| {dphi_max:.1e}, |dres| {dres_max:.1e} deg')
report('signed_rotations', sgn_diff == 0, f'{sgn_diff} grains differ')
report('classify_domains', cls_diff == 0, f'{cls_diff} clusters differ')
report('psi_angle', psi_diff < 1e-12, f'max diff {psi_diff:.1e} deg')

# 3.10 Sigma3 from dphi (s3_rotation_pairs.py, its inputs and selection)
src = os.path.join(MICROMOD, 'figures', 'sf_twin_grains' + ('_auto' if gs.ROISET == 'auto' else ''))
c = pd.read_csv(os.path.join(src, 'kc_clusters.csv')); e = pd.read_csv(os.path.join(src, 'kc_edges.csv'))
c.loc[c.tag == 'M', ['phi', 'res']] = 0.0
D = {(s, g, i): (p, r) for s, g, i, p, r in zip(c.state, c.grain, c.cid, c.phi, c.res)}
for col, end, k in (('pa', 'a', 0), ('pb', 'b', 0), ('ra', 'a', 1), ('rb', 'b', 1)):
    e[col] = [D.get((s, g, i), (np.nan, np.nan))[k] for s, g, i in zip(e.state, e.grain, e[end])]
x = e[(e.ra < 5.0) & (e.rb < 5.0)].copy()
d = (x.pa - x.pb).abs().values; ref_s3 = np.abs(np.where(d > 90, 180 - d, d) - 70.53) <= 15 / np.sqrt(3)
new_s3 = dm.sigma_from_dphi(x.pa.values, x.pb.values, sigmas=('3',)) == '3'
report(f'sigma_from_dphi Sigma3 ({len(x)} boundaries)', (new_s3 == ref_s3).all(),
       f'Sigma3 share {100 * x.n[new_s3].sum() / x.n.sum():.1f} % (ideal 70.529 vs 70.53 of the script)')
lab = dm.sigma_from_dphi(x.pa.values, x.pb.values)
print('   all Sigma from dphi, share of the boundary length:',
      {s: round(float(100 * x.n[lab == s].sum() / x.n.sum()), 1) for s in ('3', '9', '11', '27a')})

# 3.9 laminate: notebook functions (cell of stage trstrain_maps) vs domains
try:
    from crystlibs.crystlib import B19p_B2_lattice_correspondence
except Exception:
    s_ = open(os.path.join(GITHUB, 'crystlibs', 'crystlib.py')).read()
    i = s_.index('def B19p_B2_lattice_correspondence('); j = s_.index('\ndef ', i + 10)
    ns_ = dict(np=np, inv=np.linalg.inv, array=np.array, dot=np.dot, sqrt=np.sqrt); exec(s_[i:j], ns_)
    B19p_B2_lattice_correspondence = ns_['B19p_B2_lattice_correspondence']
CId = B19p_B2_lattice_correspondence(notation='Waitz')[1]
nb = json.load(open(os.path.join(MICROMOD, 'figs', 'ebsd_figs_pipeline_v2_backup_before_domains.ipynb')))   # own functions, before ebsdlib.domains
cell = next(''.join(cc['source']) for cc in nb['cells'] if 'def _tr_lamellae' in ''.join(cc['source']))
defs = cell[cell.index('TR_FRAC_LARGE = 2 / 3'):cell.index('import importlib')]
ns = dict(np=np, mcolors=matplotlib.colors); exec(defs.replace('import textwrap\n', ''), ns)
class _D: OR = {'NiTi': {'CId': CId}}
report('variant_pairs_010', (dm.variant_pairs_010(CId) == ns['_tr_pairs'](_D())).all())
from scipy.spatial.transform import Rotation as Rot
rng = np.random.default_rng(1); M = Rot.random(2000, random_state=2).as_matrix(); ds = np.array([1., 1, 0]) / np.sqrt(2)
report('pair_index', (dm.pair_index(M, ds) == ns['_tr_pair_index'](M, ds)).all())
E = rng.normal(0.03, 0.04, (12, 2000))
lam, mx = dm.pair_laminate_strain(E, M, ds, CId)
pairs = ns['_tr_pairs'](_D()); k = ns['_tr_pair_index'](M, ds); n = np.arange(2000)
ea, eb = E[pairs[k, 0], n], E[pairs[k, 1], n]
ref = np.sqrt(2 / 3 * (1 + np.maximum(ea, eb)) ** 2 + 1 / 3 * (1 + np.minimum(ea, eb)) ** 2) - 1
report('pair_laminate_strain', np.allclose(lam, ref, rtol=0, atol=1e-15) and np.array_equal(mx, E.max(0)))
X, Y = np.meshgrid(np.arange(0, 20, 0.1), np.arange(0, 15, 0.0866)); X, Y = X.ravel(), Y.ravel()
roi = (X - 10) ** 2 + (Y - 7) ** 2 < 25
same = True
for t in range(20):
    Mp = M[t]; row = pairs[t % 6]; e12 = E[:, t]
    a = dm.lamella_pattern(Mp, row[0], row[1], e12, CId, roi, X, Y)
    b = ns['_tr_lamellae'](Mp, row[0], row[1], e12, CId, roi, X, Y)
    same &= np.array_equal(a[0], b[0], equal_nan=True) and np.allclose(a[1], b[1]) and \
        (a[2] == b[2] or (np.isnan(a[2]) and np.isnan(b[2]))) and a[3] == b[3]
report('lamella_pattern (20 orientations)', same)
print('\nALL IDENTICAL' if ok_all else '\nDIFFERENCES FOUND')
