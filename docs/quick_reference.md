# Quick Reference - EBSD Analysis Library

**Navigation:** [Home](../README.md) | [Complete Docs](./documentation_complete.md) | [Summary](./documentation_summary.md) | [Quick Ref Comprehensive](./quick_reference_comprehensive.md)

---

## Quick Import

```python
from ebsdlib import *
# Or
import ebsdlib
```

---

## Cluster Analysis

### remove_small_clusters()
```python
clean_labels = remove_small_clusters(labels, minidxs=5)
```
- **Purpose:** Remove clusters with fewer than `minidxs` pixels
- **Input:** `labels` (N,) int array
- **Output:** Filtered and relabeled array
- **Example:** `remove_small_clusters(np.array([1,1,2,2,2,3,0]), minidxs=3)` → `[1,1,2,2,2,0,0]`

---

## Orientation Processing

### reduce_to_fundzone() - Fast
```python
M_reduced = reduce_to_fundzone(M, symops)
```
- **Purpose:** Vectorized fundamental zone reduction
- **Input:** `M` (N,3,3), `symops` (Ns,3,3)
- **Speed:** 10-50× faster than slow version
- **Use when:** N > 100

### reduce_to_fundzone_slow() - Reference
```python
M_reduced = reduce_to_fundzone_slow(M, symops)
```
- **Purpose:** Loop-based fundamental zone reduction
- **Use when:** N < 100 or low memory

### find_best_symmetric_quat() - Numba
```python
q_best, M_best, min_ang = find_best_symmetric_quat(q, q_ref, symops, max_iter=10, tol=1e-6)
```
- **Purpose:** Find optimal symmetric quaternion
- **Input:** `q` (4,), `q_ref` (4,), `symops` (Ns,3,3)
- **Output:** Best quaternion, matrix, angle (degrees)
- **Speed:** 10-100× faster (JIT compiled)

---

## Habit Plane Analysis

### getNormals()
```python
normals = analyzer.getNormals(
    interface_trace,          # (3,) direction vector
    interfacenorm_trace,      # (3,) normal to trace
    LrI,                      # (3,3) inverse reciprocal lattice
    Lr,                       # (3,3) reciprocal lattice
    G2Sampl,                  # (3,3) crystal→sample transform
    angles=np.linspace(0, 180, 361),  # Angles to test
    maxdevfrom90deg=5.0,      # Max deviation from 90°
    maxmillerindex=3          # Max Miller index
)
```

**Returns dict with:**
- `'n_miller'`: Miller indices (M,3)
- `'n_vec_sampl'`: Normals in sample frame (M,3)
- `'HPvsTrace_angle'`: Angles vs trace (M,)

### getCorrespNormals()
```python
N_phase2, N_all_vars = analyzer.getCorrespNormals(
    N_guess,              # Dict from getNormals()
    interface_trace2,     # (3,) trace in phase 2
    LC,                   # (3,3) correspondence matrix
    Lr2,                  # (3,3) reciprocal lattice phase 2
    LCall                 # (3,3,Nvars) all variants
)
```

---

## Results Display

### printHPmatches()
```python
analyzer.printHPmatches(
    sel=[0,1,2],      # Selection indices
    ifaces=[0],       # Interface indices
    nodirs=False      # Skip directions?
)
```

**Prints:**
- Variant scores
- Transformation strains
- Miller indices with angles
- Misalignment values

### printCorresp()
```python
analyzer.printCorresp(
    sel=[0],              # Selections
    ifaces=[0],           # Interfaces
    printfor='keyma',     # 'keyau' or 'keyma'
    printvars='closest'   # 'closest', list, or None
)
```

**Prints:**
- EBSD vs calculated orientations
- Habit plane normals
- Directions in habit plane
- Correspondence verification

---

## Common Workflows

### Clean EBSD Data
```python
# 1. Remove small grains
labels = grain_detection(ebsd_data)
clean = remove_small_clusters(labels, minidxs=50)

# 2. Reduce orientations
M = orientations[clean > 0]
M_red = reduce_to_fundzone(M, cubic_symops)
```

### Analyze Interface
```python
# 1. Get candidate normals
N1 = analyzer.getNormals(trace1, normal1, LrI1, Lr1, G2S1)

# 2. Get corresponding normals
N2, N_all = analyzer.getCorrespNormals(N1, trace2, LC, Lr2, LCall)

# 3. Display matches
analyzer.printHPmatches(sel=[0])
analyzer.printCorresp(printvars='closest')
```

### Batch Processing
```python
# Process large dataset in chunks
def process_chunks(M, symops, chunk_size=10000):
    N = len(M)
    result = np.empty_like(M)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        result[i:end] = reduce_to_fundzone(M[i:end], symops)
    return result

M_reduced = process_chunks(M, symops)
```

---

## Typical Parameters

### NiTi Alloys
```python
maxdevfrom90deg = 5.0          # Habit plane tolerance
maxmillerindex = 3             # Low-index planes
min_grain_size = 10            # pixels
misorientation_tol = 2.0       # degrees
```

### Symmetry Operations
```python
cubic_ops = 24         # m-3m (austenite)
hexagonal_ops = 24     # 6/mmm
monoclinic_ops = 4     # 2/m (martensite)
```

---

## Data Shapes

| Object | Shape | Example |
|--------|-------|---------|
| Orientation | (3,3) | `np.eye(3)` |
| Orientations | (N,3,3) | `R.random(100).as_matrix()` |
| Quaternion | (4,) | `[0,0,0,1]` |
| Miller index | (3,) | `[1,1,0]` |
| Symops | (Ns,3,3) | `(24,3,3)` for cubic |
| Correspondence | (3,3,Nvars) | `(3,3,12)` for 12 variants |

---

## Performance Tips

### Speed
```python
# ✓ Fast (vectorized)
M_red = reduce_to_fundzone(M, symops)

# ✗ Slow (loop-based)
M_red = reduce_to_fundzone_slow(M, symops)

# ✓ Numba warmup
find_best_symmetric_quat(dummy_q, dummy_ref, symops)
```

### Memory
```python
# High memory: vectorized (fast)
reduce_to_fundzone(M, symops)  # O(Ns × N)

# Low memory: loop (slow)  
reduce_to_fundzone_slow(M, symops)  # O(1)

# Balanced: chunking
process_chunks(M, symops, chunk_size=5000)
```

---

## Constants

```python
from ebsdlib import _2PI, _COS60, _SIN60

_2PI = 6.283185...    # 2π
_COS60 = 0.5          # cos(60°)
_SIN60 = 0.866025...  # sin(60°)
```

---

## Common Errors

### Shape mismatch
```python
# Check dimensions
assert M.shape[-2:] == (3,3), "Must be (...,3,3)"
assert symops.shape == (Ns,3,3), "Must be (Ns,3,3)"
```

### Non-orthogonal matrix
```python
# Verify rotation matrix
det = np.linalg.det(M)
assert np.isclose(det, 1.0), "Det must be +1"
```

### Non-unit quaternion
```python
# Normalize quaternion
q = q / np.linalg.norm(q)
```

### Empty results
```python
# Relax search criteria
maxdevfrom90deg = 10.0  # Increase
maxmillerindex = 5      # Increase
```

---

## Quick Examples

### Remove small clusters
```python
labels = np.array([1,1,1,2,2,3,3,3,3,4,4,0,0])
clean = remove_small_clusters(labels, minidxs=3)
# [1,1,1,0,0,2,2,2,2,0,0,0,0]
```

### Reduce orientations
```python
from scipy.spatial.transform import Rotation as R
M = R.random(100).as_matrix()
M_red = reduce_to_fundzone(M, cubic_symops)
```

### Find best quaternion
```python
q = np.array([0.1, 0.2, 0.3, 0.9])
q = q / np.linalg.norm(q)
q_best, M_best, ang = find_best_symmetric_quat(
    q, np.array([0,0,0,1]), cubic_symops
)
```

### Generate habit plane candidates
```python
candidates = analyzer.getNormals(
    interface_trace=np.array([1,0,0]),
    interfacenorm_trace=np.array([0,1,0]),
    LrI=inv_recip_lattice,
    Lr=recip_lattice,
    G2Sampl=np.eye(3),
    maxdevfrom90deg=5.0,
    maxmillerindex=3
)
print(f"Found {len(candidates['n_miller'])} candidates")
```

---

## Function Summary

| Function | Purpose | Speed |
|----------|---------|-------|
| `remove_small_clusters()` | Filter clusters | Fast |
| `reduce_to_fundzone()` | Vectorized FZ reduction | Fast |
| `reduce_to_fundzone_slow()` | Loop FZ reduction | Slow |
| `find_best_symmetric_quat()` | Optimal quaternion | Very Fast (Numba) |
| `getNormals()` | Generate habit planes | Medium |
| `getCorrespNormals()` | Phase correspondence | Fast |
| `printHPmatches()` | Display results | - |
| `printCorresp()` | Show correspondence | - |

---

## One-Liners

```python
# Clean labels
clean = remove_small_clusters(labels, 5)

# Reduce orientations
M_red = reduce_to_fundzone(M, symops)

# Find best quat
q_best, M, ang = find_best_symmetric_quat(q, qref, symops)

# Get candidates
N = analyzer.getNormals(trace, norm, LrI, Lr, G2S)

# Correspond
N2 = analyzer.getCorrespNormals(N, trace2, LC, Lr2)

# Print
analyzer.printHPmatches()
analyzer.printCorresp(printvars='closest')
```

---

**Navigation:** [Home](../README.md) | [Complete Docs](./documentation_complete.md) | [Summary](./documentation_summary.md) | [Quick Ref Comprehensive](./quick_reference_comprehensive.md)

---

*Quick reference for ebsdlib v1.0 - June 2025*
