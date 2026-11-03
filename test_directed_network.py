#!/usr/bin/env python3
"""
Diagnostic of the directed twin network of ebsdlib.reconstruct_parent_orientation_directed on synthetic maps.

For every grain checkpoint of a pipeline condition (figs/synthetic/<condition>/grain_*.h5) and its ground truth
(*_truth.npz/json of ebsdlib.synthetic), every DIRECTED twin edge parent -> child of the not-merged network is checked:
  correct   the child is a twin domain (truth class +-11n) and the parent is its true host ('schmid' {114} edges) or,
            for an edge directed by triangle closure ({113}/{115} = kink + {114}), another host (matrix <-> kink);
  reversed  the parent is a twin domain whose true host (or other host) is the child;
  other     anything else (e.g. both twins, accidental links).
Reported per grain and per kind of direction ('schmid' {114} edges, 'closed' triangle edges), with the root class,
the number of sources and of open / conflicting closure edges.

Run:  python test_directed_network.py <cond_dir> <truth.npz> [<cond_dir> <truth.npz> ...]
e.g.  python test_directed_network.py ~/.../micromod/figs/synthetic/syn_S10_kink2_dir114cfs \\
          ~/.../ebsdlib/notebooks/data/synthetic/syn_S10_kink2_x0.100_y0.087_c401_r347_truth.npz
"""
import os, sys, glob, json, collections
import numpy as np
import h5py

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ebsdlib import synthetic as syn
except ImportError:                                      # without the dependencies of ebsdlib.py: the module alone
    import importlib.util
    _sp = importlib.util.spec_from_file_location('synthetic', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                         'synthetic.py'))
    syn = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(syn)


def check(cond_dir, truth):
    V = syn.SyntheticValidator(truth)
    meta = json.load(open(truth.replace('.npz', '.json')))
    host = {d['id']: d['host'] for d in meta['domains']}
    dcls = {d['id']: d['cls'] for d in meta['domains']}
    tot = collections.Counter()
    print(f'\n=== {os.path.basename(cond_dir)}')
    for path in sorted(glob.glob(os.path.join(cond_dir, 'grain_*.h5'))):
        if path.endswith('grain_0000.h5'):
            continue
        with h5py.File(path, 'r') as f:
            p = f['pickled']
            cr = syn._unpickle(p, 'clustering_result'); res = syn._unpickle(p, 'notmerged_result')
        lab = np.asarray(cr.labels).ravel()
        dom = {}
        for c in cr.avg_orientations:
            m = (lab == c) & V.valid
            if m.any():
                dom[int(c)] = int(collections.Counter(V.domain[m]).most_common(1)[0][0])
        tg = collections.Counter(V.grain[(lab > 0) & V.valid]).most_common(1)[0][0]
        hosts_of_grain = [d['id'] for d in meta['domains'] if d['grain'] == tg and d['host'] is None]

        def is_twin(c):
            return c in dom and dcls[dom[c]].lstrip('+-')[:2] == '11'

        def related(pa, ch, kind):    # schmid: pa = true host of ch; closure: pa = another host of the grain
            if not (is_twin(ch) and pa in dom):
                return False
            if kind == 'schmid':
                return dom[pa] == host[dom[ch]]
            return dom[pa] in hosts_of_grain and dom[pa] != host[dom[ch]]

        cnt = collections.Counter()
        for e in res['edges']:
            if e['type'] != 'twin':
                continue
            kind = e.get('closure', 'schmid')
            if e['direction'] == 'undirected':
                cnt[(kind if kind in ('open', 'conflict') else 'undirected', '')] += 1
                continue
            pa, ch = ((e['cluster_a'], e['cluster_b']) if e['direction'] == 'a_to_b'
                      else (e['cluster_b'], e['cluster_a']))
            verdict = 'correct' if related(pa, ch, kind) else 'reversed' if related(ch, pa, kind) else 'other'
            cnt[(kind, verdict)] += 1
        root = res['root_cluster_id']
        rc = dcls.get(dom.get(int(root)), '?') if root is not None else '?'
        nsrc = len(res.get('source_candidates', []))
        txt = ', '.join(f'{k} {v}: {n}' for (k, v), n in sorted(cnt.items()))
        print(f'{os.path.basename(path)} (true grain {tg}): root {root} = true {rc!r}, sources {nsrc}/{res["n_nodes_total"]}; {txt}')
        tot.update(cnt); tot[('root', rc)] += 1
    print('total:', ', '.join(f'{k} {v}: {n}' for (k, v), n in sorted(tot.items())))
    return tot


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or len(a) % 2:
        sys.exit(__doc__)
    for c, t in zip(a[::2], a[1::2]):
        check(os.path.expanduser(c), os.path.expanduser(t))
