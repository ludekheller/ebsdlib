"""Diagnostics of ebsdlib.detect_former_grains (automatic detection of the former austenite grains).
Runs the cells of micromod/figs/prior_grain_detection_test.ipynb verbatim (Steps 1b-7b, the automatic clean-up of
Step 11 without clicks, the export of Step 12) and detect_former_grains on the same whole-scan clustering, and checks
that the cluster partition and the exported polygons are identical; then compares the polygons with the corrected
export used by the pipeline (figs/<state>_auto/auto_rois.h5, after the hand corrections of Step 11).
Needs the ebsdlib environment (imports ebsdlib.py of this folder; micromod/figures/grain_substructure reads the pickled
clustering). Writes nothing.
Run:  python test_former_grains.py def30_500C      (or def50_500C; env GITHUB, MICROMOD for other locations)"""
import sys, json, os, collections, time, numpy as np, pandas as pd, h5py
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from matplotlib.path import Path
import importlib.util
GITHUB = os.environ.get('GITHUB', '/home/lheller/Documents/GitHub')
MICROMOD = os.environ.get('MICROMOD', '/home/lheller/Documents/papers/mypapers/micromod')
sys.path.insert(0, os.path.join(MICROMOD, 'figures')); import grain_substructure as gs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.append(GITHUB)   # crystlibs, getphases, ...
import ebsdlib as fg                             # ebsdlib.py of this folder (detect_former_grains)
NB = json.load(open(os.path.join(MICROMOD, 'figs', 'prior_grain_detection_test.ipynb')))
cell = lambda i: ''.join(NB['cells'][i]['source'])
c32 = cell(32).splitlines()
def block(lines, start, stop):
    i = next(k for k, l in enumerate(lines) if l.startswith(start)); j = next(k for k, l in enumerate(lines[i + 1:], i + 1) if l.startswith(stop))
    return '\n'.join(lines[i:j])
STATE = sys.argv[1] if len(sys.argv) > 1 else 'def30_500C'
ns = dict(np=np, pd=pd, h5py=h5py, os=os, json=json, time=time, collections=collections, cKDTree=cKDTree, Rot=Rot, Path=Path,
          gs=gs, STATE=STATE, USE_HAND=False, PROJ=MICROMOD, DATADIR=os.path.join(GITHUB, 'ebsdlib', 'notebooks', 'data'),
          SCANS={'def30_500C': 'def30_500C_x0.100_y0.087_c534_r483.ang', 'def50_500C': 'def50_500C_x0.060_y0.052_c445_r403.ang'},
          B2_PHASES_NEW=None, MAP2SAMPLE=np.eye(2), TRACE_MIN_UM=1e9, display=print)
for l in cell(2).splitlines():                        # Step 0 parameters
    if l.split('=')[0].strip() in ('RES_TOL', 'AXIS_TOL', 'LOW_MIS', 'MIN_AREA_DET_UM2', 'GAP_UM', 'MIN_CONTACT_UM', 'MIN_TWIN_UM', 'MIN_PART_UM2', 'MIN_SHOW_UM2'):
        exec(l, ns)
t0 = time.time()
exec(cell(4), ns); exec(cell(6), ns); exec(cell(8), ns); exec(cell(12), ns)
ns['cl_roi'] = collections.defaultdict(lambda: -1)
exec(cell(14), ns); exec(cell(18), ns); exec(cell(22), ns)
ns['state'] = dict(comp=dict(ns['comp']), dropped=set(), pinned={}, outlines={}, paints=[], history=[], log=[])
ns['_copy'] = __import__('copy'); ns['_csize'] = np.bincount(np.clip(ns['labels'], 0, None))
exec(block(c32, '_lut = ', '_rng_c'), ns); exec(block(c32, 'def _pix_grains', '_rng_c'), ns)
exec(block(c32, 'SURROUND_SHARE', 'AUTO_CLEAN'), ns); n_clean, _ = ns['clean_geometry']()
c34 = cell(34); exec(c34[:c34.index('\nAUTO = ')], ns)
ref_comp, ref_polys = dict(ns['state']['comp']), ns['polys']
print(f'notebook: {time.time() - t0:.0f} s, clean-up moves {n_clean}, {len(ref_polys)} ROIs')

t0 = time.time()
ang = os.path.join(ns['DATADIR'], ns['SCANS'][STATE])
with h5py.File(f"{ns['PROJ']}/figs/{STATE}/grain_0000.h5", 'r') as h:
    cr = gs._load(h['pickled'], 'clustering_result').__dict__
res = fg.detect_former_grains(ns['XY'], ns['Opix'], ns['isB2'], np.asarray(cr['labels']), cr['avg_orientations'], ns['STEP'],
                              cluster_phase=cr['_cluster_phases_id'])
print(f'module:   {time.time() - t0:.0f} s')
# ---- comparison: partition of the clusters (grain numbers may differ) and polygons
pairs = collections.Counter((ref_comp[c], res['grain_of_cluster'][c]) for c in ref_comp)
same_part = len(set(g for g, _ in pairs)) == len(pairs) == len(set(h for _, h in pairs))
diff = [c for c in ref_comp if res['grain_of_cluster'].get(c) is None]
print('clusters', len(ref_comp), '| grains notebook', len(set(ref_comp.values())), 'module', len(set(res['grain_of_cluster'].values())),
      '| identical cluster partition:', same_part, '| missing', len(diff))
P = res['polygons']
same_poly = len(P) == len(ref_polys) and all(a.shape == b.shape and np.allclose(a, b) for a, b in zip(P, ref_polys))
print('ROIs notebook', len(ref_polys), 'module', len(P), '| identical polygons:', same_poly)
# ---- agreement with the corrected export used by the pipeline (user's clicks)
with h5py.File(f"{ns['PROJ']}/figs/{STATE}_auto/auto_rois.h5", 'r') as h:
    U = [h['rois'][f'roi_{i}'][:] for i in range(h['rois'].attrs['n_rois'])]
XY = ns['XY']
lu = np.full(len(XY), -1); lm = np.full(len(XY), -1)
for i, q in enumerate(U): lu[Path(q).contains_points(XY)] = i
for i, q in enumerate(P): lm[Path(q).contains_points(XY)] = i
best = []
for i in range(len(U)):
    m = lu == i; v = lm[m]; v = v[v >= 0]
    if len(v) == 0: best.append(0); continue
    j = np.bincount(v).argmax(); inter = (m & (lm == j)).sum(); best.append(inter / (m | (lm == j)).sum())
best = np.array(best)
print(f'vs corrected export ({len(U)} ROIs): IoU median {np.median(best):.2f}, >= 0.9: {(best >= 0.9).sum()}, >= 0.7: {(best >= 0.7).sum()}')
