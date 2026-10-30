"""Diagnostics of ebsdlib/synthetic.py: Euler convention vs pyebsd, phi recovery from the written .ang, CSL of the
ideal junctions (compare with Table 3 of the functional paper), validator self-consistency, multi-grain map.
Runs without the ebsdlib dependencies (synthetic.py and pyebsd/ebsd/orientation.py are loaded by path).
Run:  python test_synthetic.py   (writes S1_ideal.ang, S1_pert.ang, S9_multi.ang + *_truth.* in the current folder)"""
import os, sys, time, importlib.util
import numpy as np
GITHUB = os.environ.get('GITHUB', '/home/lheller/Documents/GitHub')
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
_syn = _load('ebsdlib_synthetic', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic.py'))
globals().update({k: getattr(_syn, k) for k in dir(_syn) if not k.startswith('_')})
pyori = _load('pyori', os.path.join(GITHUB, 'pyebsd/ebsd/orientation.py'))
T = os.path.join(GITHUB, 'ebsdlib/notebooks/data/def20_austenite_plasticity_x0.100_y0.087_c206_r186.ang')

# 1) Euler convention identical to pyebsd
rng = np.random.default_rng(3); G = random_rotations(500, rng); R = G.transpose(0, 2, 1)
e = R_to_euler(R); Rp = pyori.euler_angles_to_rotation_matrix(*e, verbose=False)
print('1) euler vs pyebsd, max dev [deg]:', rot_angle(np.einsum('nij,nkj->nik', R, Rp)).max())

# 2) single grain, ideal
t0 = time.time()
syn = SyntheticEBSD(T, size_um=(20, 16), step_um=0.1, seed=1)
syn.add_grain(); syn.build(); res = syn.check(verbose=False)
print('2) build %.1f s' % (time.time() - t0), {k: res[k] for k in ('phi_err_max', 'residual_max', 'euler_roundtrip_max_deg', 'n_points', 'n_domains')})
print('   class areas', {k: round(v, 3) for k, v in res['class_area'].items()})

# 3) write and read back (pyebsd-like parser), orientations identical
syn.write('S1_ideal.ang')
d = np.loadtxt('S1_ideal.ang', comments='#')
Rr = pyori.euler_angles_to_rotation_matrix(d[:, 0], d[:, 1], d[:, 2], verbose=False); M = Rr.transpose(0, 2, 1)
print('3) read back: N', len(d), 'max dev [deg]', rot_angle(np.einsum('nij,nkj->nik', M, syn.g)).max(),
      ' x/y identical', np.allclose(d[:, 3], syn.X, atol=1e-4) and np.allclose(d[:, 4], syn.Y, atol=1e-4))
hdr = [l for l in open('S1_ideal.ang') if l.startswith('# NCOLS') or l.startswith('# NROWS') or l.startswith('# XSTEP') or l.startswith('# YSTEP')]
print('   header', ''.join(hdr).replace('\n', ' | '))

# 4) phi recovered from the file orientations (pipeline Eq. phic with symmetry search)
gM = syn.grains[0]['g_M']; ds = syn.grains[0]['ds']
Q = np.einsum('ji,sjk,nkl->nsil', gM, SYM, M)          # g_M^T S g
phi, r = phi_about_axis(Q, ds); k = r.argmin(1); phi = phi[np.arange(len(M)), k]
on = np.isfinite(syn.phi)
dphi = (phi[on] - syn.phi[on] + 90) % 180 - 90
print('4) phi from file vs truth: max |err| %.2e deg, residual max %.2e' % (np.abs(dphi).max(), r.min(1)[on].max()))

# 5) CSL of neighbour pairs between domains (ideal)
pairs, sig = syn.truth_csl_pairs()
cn = np.array(syn.class_names, dtype=object)[syn.cls]
import collections
cnt = collections.Counter((tuple(sorted((cn[i], cn[j]))), s) for (i, j), s in zip(pairs, sig))
print('5) junction -> Sigma (ideal):')
for (pr, s), n in sorted(cnt.items(), key=lambda x: -x[1])[:14]:
    print('   %-18s %-4s %d' % ('|'.join(pr), s or '-', n))

# 6) perturbed + validator self-consistency (truth as prediction -> perfect scores)
syn.perturb(noise_deg=0.5, gradient_deg_per_um=0.3, f_mis=0.01, f_ni=0.01, f_mart=0.3, w_mart_um=0.15)
syn.write('S1_pert.ang')
print('6) flags', {k: int((syn.flag == k).sum()) for k in range(4)})
V = SyntheticValidator('S1_pert_truth.npz')
print('   grains ARI (truth)', V.score_grains(V.grain)['ARI'])
print('   classes acc (truth)', V.score_classes(V.normalised_truth_classes())['accuracy'])
print('   phi (truth)', V.score_phi(V.phi * np.array([V.sense_flip()[g] for g in V.grain])))
dm = syn.domain.copy() + 1; dm[~V.valid] = 0
mc = V.match_clusters(dm); print('   clusters=domains: n', len(mc), 'min purity', min(v['purity'] for v in mc.values()))
nb = syn._neighbour_pairs(1)
noisy = disorientation(syn.g[nb[:, 0]], syn.g[nb[:, 1]])
same = (syn.domain[nb[:, 0]] == syn.domain[nb[:, 1]]) & (syn.flag[nb[:, 0]] == 0) & (syn.flag[nb[:, 1]] == 0)
print('   neighbour disorientation within domains: median %.2f, 99%% %.2f deg' % (np.median(noisy[same]), np.percentile(noisy[same], 99)))

# 7) multi-grain map with off-axis twin
syn2 = SyntheticEBSD(T, size_um=(40, 30), step_um=0.1, seed=7)
for i in range(6):
    syn2.add_grain(offaxis=dict(width_um=1.5, position_um=0.0) if i == 0 else None)
syn2.build(); r2 = syn2.check(verbose=False); syn2.perturb(noise_deg=0.5); syn2.write('S9_multi.ang')
print('7) multi-grain: N', r2['n_points'], 'domains', r2['n_domains'], 'phi_err', r2['phi_err_max'], 'areas', {k: round(v, 3) for k, v in r2['class_area'].items()})
