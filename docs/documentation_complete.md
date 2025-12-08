# Complete Documentation - EBSD Analysis Library

**Navigation:** [Home](../README.md) | [Summary](./documentation_summary.md) | [Quick Ref Comprehensive](./quick_reference_comprehensive.md) | [Quick Ref](./quick_reference.md)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Module Structure](#module-structure)
4. [Core Utilities](#core-utilities)
5. [Crystallographic Functions](#crystallographic-functions)
6. [Main Analysis Class](#main-analysis-class)
7. [Habit Plane Analysis](#habit-plane-analysis)
8. [Advanced Features](#advanced-features)
9. [Performance Optimization](#performance-optimization)
10. [Examples and Use Cases](#examples-and-use-cases)

---

## Introduction

### Purpose and Scope

The `ebsdlib` module is a specialized library for analyzing Electron Backscatter Diffraction (EBSD) data with a focus on phase transformations in crystalline materials. It was developed specifically for characterizing martensitic transformations in NiTi shape memory alloys but can be adapted for other material systems.

### Key Capabilities

- **Orientation Analysis**: Comprehensive tools for crystallographic orientation processing
- **Phase Transformation**: Specialized algorithms for martensitic transformation characterization
- **Habit Plane Detection**: Automated identification of habit planes between phases
- **Correspondence Relationships**: Determination of lattice correspondence matrices
- **Variant Analysis**: Identification and ranking of transformation variants
- **Microstructure Characterization**: Cluster analysis and interface detection

### Scientific Background

The library implements established crystallographic theory including:
- Fundamental zone reduction using symmetry operations
- Misorientation calculations via quaternion algebra
- Habit plane determination using trace analysis
- Correspondence variant theory for martensitic transformations

---

## Installation and Setup

### Required Dependencies

#### Core Scientific Libraries
```bash
pip install numpy>=1.20.0
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
pip install numba>=0.54.0
```

#### Crystallographic Libraries
```bash
pip install orix>=0.9.0  # Orientation and texture analysis
pip install crystals      # Crystal structure definitions
```

### Custom Module Dependencies

The library requires several custom modules that must be in your Python path:

1. **orilib**: Orientation, quaternion, and Euler angle utilities
2. **projlib**: Stereographic projection functions
3. **plotlib**: Crystallographic plotting utilities
4. **crystlib**: Crystallographic calculation functions
5. **effelconst**: Elastic property calculations
6. **getphases**: Phase identification from EBSD data

### Setup Instructions

```python
# Option 1: Add to system path
import sys
sys.path.append('/path/to/ebsdlib/directory')
import ebsdlib

# Option 2: Direct import from local directory
from ebsdlib import *

# Option 3: Selective imports
from ebsdlib import remove_small_clusters, reduce_to_fundzone
```

### Verification

```python
# Test installation
import ebsdlib
import numpy as np

# Test basic functionality
test_labels = np.array([1,1,2,2,2,3,0,0])
result = ebsdlib.remove_small_clusters(test_labels, minidxs=3)
print("Installation successful!" if result is not None else "Installation failed")
```

---

## Module Structure

### Constants

```python
_2PI = 2 * np.pi              # 2π constant
_COS60 = 0.5                  # cos(60°) for hexagonal calculations
_SIN60 = 0.5 * 3.0**0.5       # sin(60°) for hexagonal calculations
```

### Module Imports

The library integrates multiple specialized modules:

```python
from numba import njit                    # JIT compilation for performance
from orix import plot                     # Crystallographic plotting
from orix.quaternion import Orientation, Rotation, symmetry
from scipy.spatial.transform import Rotation as R
from orilib import *                      # Orientation operations
from projlib import *                     # Stereographic projections
from plotlib import *                     # Plotting utilities
from crystlib import *                    # Crystal calculations
from crystals import Crystal              # Crystal definitions
from effelconst import  * #effective elastic constants calculations
from getphases import getPhases           # Phase detection
```

### Architectural Overview

```
ebsdlib
├── Utility Functions
│   ├── Cluster Analysis
│   └── Data Filtering
├── Orientation Processing
│   ├── Fundamental Zone Reduction
│   ├── Symmetry Operations
│   └── Misorientation Calculations
├── Main Analysis Class
│   ├── Interface Detection
│   ├── Habit Plane Analysis
│   └── Correspondence Determination
└── Visualization
    ├── Crystallographic Plots
    └── Statistical Analysis
```

---

## Core Utilities

### remove_small_clusters()

**Purpose**: Filters labeled clusters by removing those below a minimum size threshold.

#### Function Signature
```python
def remove_small_clusters(labels, minidxs=5)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | array-like (N,) | Required | Integer array of cluster labels for each data point |
| `minidxs` | int | 5 | Minimum number of points required for cluster retention |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `new_labels` | ndarray (N,) | Filtered labels with small clusters set to 0 and remaining clusters relabeled contiguously |

#### Algorithm Details

1. **Cluster Counting**: Uses `np.unique()` with `return_counts=True` to identify cluster sizes
2. **Filtering**: Sets labels of clusters with count < `minidxs` to 0
3. **Relabeling**: Reassigns remaining clusters to contiguous labels (1, 2, 3, ...)

#### Complexity
- **Time**: O(N log N) due to unique value operations
- **Space**: O(N) for label arrays

#### Example Usage

```python
import numpy as np
from ebsdlib import remove_small_clusters

# Create sample labeled data
labels = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 0, 0])
print("Original labels:", labels)
# Output: [1 1 1 2 2 3 3 3 3 4 4 0 0]

# Remove clusters with fewer than 3 pixels
filtered = remove_small_clusters(labels, minidxs=3)
print("Filtered labels:", filtered)
# Output: [1 1 1 0 0 2 2 2 2 0 0 0 0]
# Cluster 1 (3 pixels) → kept as cluster 1
# Cluster 2 (2 pixels) → removed (set to 0)
# Cluster 3 (4 pixels) → kept and relabeled as cluster 2
# Cluster 4 (2 pixels) → removed (set to 0)
```

#### Edge Cases

```python
# Empty array
empty = np.array([])
result = remove_small_clusters(empty, minidxs=5)
# Returns: empty array

# All zeros
zeros = np.zeros(10, dtype=int)
result = remove_small_clusters(zeros, minidxs=5)
# Returns: array of zeros

# All clusters too small
small = np.array([1, 2, 3, 4, 5])
result = remove_small_clusters(small, minidxs=2)
# Returns: array of zeros
```

#### Applications

- **Noise Removal**: Eliminate spurious single-pixel classifications
- **Microstructure Cleaning**: Remove artifacts from grain detection
- **Statistical Filtering**: Focus analysis on significant features
- **Pre-processing**: Clean data before further analysis

---

### reduce_to_fundzone()

**Purpose**: Vectorized reduction of orientation matrices to the fundamental zone of a crystal symmetry group.

#### Function Signature
```python
def reduce_to_fundzone(M, symops)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `M` | ndarray (N, 3, 3) | Orientation matrices (sample→crystal reference frame) |
| `symops` | ndarray (Ns, 3, 3) | Symmetry operation matrices for the crystal system |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `M_reduced` | ndarray (N, 3, 3) | Reduced orientation matrices (symmetry-equivalent with minimum rotation to identity) |

#### Mathematical Basis

The fundamental zone is the region of orientation space where each crystallographic orientation is represented exactly once, accounting for crystal symmetry. The reduction finds the symmetry-equivalent orientation with the smallest rotation angle from identity.

**Mathematical Formulation**:

For each orientation matrix M and symmetry operation S:
1. Compute all equivalent orientations: M_eq = S @ M
2. Convert to quaternions: q = matrix_to_quaternion(M_eq)
3. Calculate rotation angle: θ = 2 * arccos(|q_w|)
4. Select equivalent with minimum θ

#### Algorithm Steps

```python
# Pseudo-code representation
1. Compute all equivalent orientations: M_eq[s,n] = symops[s] @ M[n] for all s, n
2. Convert to quaternions: q_eq = matrix_to_quat(M_eq)
3. Extract scalar component: w = q_eq[..., 3]
4. Compute rotation angles: ang = 2 * arccos(|w|)
5. Find minimum angle index per orientation: best_idx = argmin(ang, axis=0)
6. Select corresponding orientations: M_reduced[n] = M_eq[best_idx[n], n]
```

#### Performance Characteristics

- **Time Complexity**: O(Ns × N) where Ns = number of symmetry operations, N = number of orientations
- **Vectorization**: Uses `np.einsum` for efficient batch matrix multiplication
- **Memory**: O(Ns × N × 9) for storing equivalent orientations

#### Example Usage

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from ebsdlib import reduce_to_fundzone

# Example: Cubic symmetry (simplified - use actual cubic symmetry ops)
# Define cubic symmetry operations (24 operations)
symops = get_cubic_symmetry_operations()  # User-defined function

# Generate random orientations
N = 1000
orientations = R.random(N).as_matrix()

# Reduce to fundamental zone
reduced_orientations = reduce_to_fundzone(orientations, symops)

# Verify reduction
print(f"Original orientations: {orientations.shape}")
print(f"Reduced orientations: {reduced_orientations.shape}")
```

#### Comparison with Slow Version

```python
# Timing comparison
import time

M = R.random(500).as_matrix()
symops = get_cubic_symmetry_operations()

# Vectorized version
t0 = time.time()
M_fast = reduce_to_fundzone(M, symops)
t_fast = time.time() - t0

# Slow version
t0 = time.time()
M_slow = reduce_to_fundzone_slow(M, symops)
t_slow = time.time() - t0

print(f"Speedup: {t_slow/t_fast:.1f}x")
# Typical speedup: 10-50x depending on N and Ns
```

---

### reduce_to_fundzone_slow()

**Purpose**: Non-vectorized implementation of fundamental zone reduction (reference implementation).

#### Function Signature
```python
def reduce_to_fundzone_slow(M, symops)
```

#### Parameters

Same as `reduce_to_fundzone()`

#### Returns

Same as `reduce_to_fundzone()`

#### Implementation Details

Uses explicit loop over orientations instead of vectorization:

```python
# Algorithm outline
for each orientation i:
    for each symmetry operation s:
        compute equivalent: M_eq = s @ M[i]
        calculate angle to identity
        if angle < current_minimum:
            update best equivalent
    M_reduced[i] = best equivalent
```

#### When to Use

- **Small datasets**: When N < 100 orientations
- **Debugging**: Easier to step through and verify logic
- **Memory constraints**: Lower peak memory usage
- **Reference**: Verification of vectorized implementation

#### Performance Trade-offs

| Aspect | Slow Version | Fast Version |
|--------|--------------|--------------|
| Speed | O(Ns × N) sequential | O(Ns × N) vectorized |
| Memory | O(9) per iteration | O(Ns × N × 9) total |
| Readability | High | Medium |
| Debugging | Easy | Harder |

---

### find_best_symmetric_quat()

**Purpose**: Numba-accelerated iterative optimization to find the symmetry-equivalent quaternion closest to a reference.

#### Function Signature
```python
@njit
def find_best_symmetric_quat(q, q_ref, symops, max_iter=10, tol=1e-6)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | ndarray (4,) | Required | Input quaternion [x, y, z, w] |
| `q_ref` | ndarray (4,) | Required | Reference quaternion for comparison |
| `symops` | ndarray (Ns, 3, 3) | Required | Symmetry operation matrices |
| `max_iter` | int | 10 | Maximum optimization iterations |
| `tol` | float | 1e-6 | Convergence tolerance (degrees) |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `q_best` | ndarray (4,) | Optimal symmetry-equivalent quaternion |
| `M_best` | ndarray (3, 3) | Corresponding rotation matrix |
| `min_ang` | float | Minimum misorientation angle (degrees) |

#### Algorithm

Iterative greedy search:

```python
1. Initialize: q_best = q, min_ang = misorientation(q, q_ref)
2. For each iteration (up to max_iter):
   a. For each symmetry operation s:
      - Compute q_sym = s @ q
      - Calculate ang = misorientation(q_sym, q_ref)
      - If ang + tol < min_ang: update q_best, min_ang
   b. If no improvement: break (converged)
3. Return q_best, M_best, min_ang
```

#### Numba Optimization

The `@njit` decorator provides:
- **Just-in-Time compilation**: First call compiles to machine code
- **Type inference**: Automatic type deduction
- **No Python overhead**: Native execution speed
- **Typical speedup**: 10-100x over pure Python

#### Example Usage

```python
import numpy as np
from ebsdlib import find_best_symmetric_quat

# Define quaternions (normalized)
q = np.array([0.1, 0.2, 0.3, 0.9])
q = q / np.linalg.norm(q)

q_ref = np.array([0.0, 0.0, 0.0, 1.0])  # Identity

# Symmetry operations (example: cubic)
symops = get_cubic_symmetry_operations()

# Find best symmetric equivalent
q_best, M_best, min_angle = find_best_symmetric_quat(
    q, q_ref, symops, max_iter=20, tol=1e-7
)

print(f"Best quaternion: {q_best}")
print(f"Minimum angle: {min_angle:.4f}°")
print(f"Rotation matrix:\n{M_best}")
```

#### Convergence Characteristics

```python
# Monitoring convergence
angles = []
for iteration in range(max_iter):
    q_best, M_best, angle = find_best_symmetric_quat(
        q, q_ref, symops, max_iter=iteration+1
    )
    angles.append(angle)

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(angles)
plt.xlabel('Iteration')
plt.ylabel('Misorientation (deg)')
plt.title('Convergence of Symmetric Quaternion Search')
```

---

## Main Analysis Class

### Overview

The main analysis class (implementation details from lines 175-2749) provides comprehensive EBSD analysis capabilities. While the full class implementation is extensive, key methods include:

### Key Methods

#### Interface Analysis
- **findInterfaces()**: Detects phase boundaries in EBSD data
- **getHabitPlanes()**: Determines habit plane orientations
- **matchHabitPlanes()**: Matches habit planes between phases

#### Correspondence Analysis
- **printHPmatches()**: Displays habit plane matching results with scoring
- **printCorresp()**: Shows correspondence relationships between phases
- **getNormals()**: Generates candidate plane normals perpendicular to interface traces
- **getCorrespNormals()**: Calculates corresponding normals in the second phase

---

## Habit Plane Analysis

### getNormals()

**Purpose**: Generate candidate plane normals perpendicular to an interface trace direction.

#### Function Signature
```python
def getNormals(self, interface_trace, interfacenorm_trace, LrI, Lr, G2Sampl, 
               angles=np.linspace(0, 180, 361), maxdevfrom90deg=None, 
               maxmillerindex=None)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `interface_trace` | ndarray (3,) | Interface trace direction vector |
| `interfacenorm_trace` | ndarray (3,) | Normal to the interface trace |
| `LrI` | ndarray (3, 3) | Inverse reciprocal lattice matrix |
| `Lr` | ndarray (3, 3) | Reciprocal lattice matrix |
| `G2Sampl` | ndarray (3, 3) | Crystal to sample coordinate transformation |
| `angles` | ndarray | Rotation angles to test (default: 0-180° in 0.5° steps) |
| `maxdevfrom90deg` | float | Maximum deviation from 90° for habit plane (default: class attribute) |
| `maxmillerindex` | int | Maximum Miller index magnitude (default: class attribute) |

#### Returns

Dictionary with keys:
- `'n_vec'`: Normal vectors in crystal frame
- `'n_vec_sampl'`: Normal vectors in sample frame  
- `'n_miller'`: Miller indices (rounded)
- `'n_miller_normvec'`: Normalized Miller direction vectors
- `'n_miller_normvec_sampl'`: Normalized Miller vectors in sample frame
- `'HPvsTrace_angle'`: Angle between habit plane and trace

#### Algorithm

```python
1. For each angle in angles:
   a. Rotate interfacenorm_trace around interface_trace by angle
   b. Transform to sample coordinates
   c. Convert to Miller indices via inverse reciprocal lattice
   d. Round to nearest integer Miller indices
   e. Calculate normalized Miller direction vector
   f. Compute angle between habit plane and trace
   g. Filter by:
      - Maximum Miller index magnitude
      - Maximum deviation from 90°
      - Remove duplicates (parallel/antiparallel normals)
2. Return all qualifying candidates
```

#### Example Usage

```python
# Define interface geometry
interface_trace = np.array([1.0, 0.0, 0.0])  # Along x-axis
normal_to_trace = np.array([0.0, 1.0, 0.0])   # In y-direction

# Lattice parameters (example: cubic)
a = 3.0  # Angstroms
Lr = np.diag([2*np.pi/a, 2*np.pi/a, 2*np.pi/a])
LrI = np.linalg.inv(Lr)

# Crystal to sample transformation (identity for this example)
G2Sampl = np.eye(3)

# Generate candidate normals
normals_dict = analyzer.getNormals(
    interface_trace=interface_trace,
    interfacenorm_trace=normal_to_trace,
    LrI=LrI,
    Lr=Lr,
    G2Sampl=G2Sampl,
    angles=np.linspace(0, 180, 181),  # 1° steps
    maxdevfrom90deg=5.0,  # ±5° from perpendicular
    maxmillerindex=3      # Indices ≤ 3
)

print(f"Found {len(normals_dict['n_miller'])} candidate habit planes")
for i, miller in enumerate(normals_dict['n_miller']):
    angle = normals_dict['HPvsTrace_angle'][i]
    print(f"  {miller}: {angle:.2f}° from perpendicular")
```

---

### getCorrespNormals()

**Purpose**: Calculate corresponding normals in the second phase using orientation relationship matrices.

#### Function Signature
```python
def getCorrespNormals(self, N_guess, interface_trace2, LC, Lr2, LCall=None)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `N_guess` | dict | Candidate normals from `getNormals()` |
| `interface_trace2` | ndarray (3,) | Interface trace in second phase |
| `LC` | ndarray (3, 3) | Lattice correspondence matrix for closest variant |
| `Lr2` | ndarray (3, 3) | Reciprocal lattice matrix for second phase |
| `LCall` | ndarray (3, 3, Nvars) | Correspondence matrices for all variants (optional) |

#### Returns

If `LCall is None`:
- Dictionary of corresponding normals for closest variant

If `LCall is not None`:
- Tuple: (closest_variant_normals, all_variants_normals)

#### Example Usage

```python
# After getting normals in phase 1
N_austenite = analyzer.getNormals(...)

# Calculate corresponding normals in martensite
N_martensite, N_all_variants = analyzer.getCorrespNormals(
    N_guess=N_austenite,
    interface_trace2=trace_in_martensite,
    LC=correspondence_matrix_closest,
    Lr2=martensite_reciprocal_lattice,
    LCall=all_correspondence_matrices
)

# Examine results
print("Martensite normals (closest variant):")
for nm in N_martensite['n_miller']:
    print(f"  {nm}")
```

---

### printHPmatches()

**Purpose**: Display detailed habit plane matching results with quality scores.

#### Function Signature
```python
def printHPmatches(self, sel=None, ifaces=None, nodirs=False)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sel` | list of int | None | Selection indices to print (None = all) |
| `ifaces` | list of int | None | Interface indices to print (None = all) |
| `nodirs` | bool | False | If True, skip printing directions in habit plane |

#### Output Format

For each selection and interface, prints:
```
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Selection # X of Y
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Interface # X of Y
------------------------------------------------------------
Fitting Closest LCV Z: Score A, mean misalignment: B°
Transformation strain along [100]: C
Normals: misalignment: D°, (hkl)_A (angle1°) / (hkl)_M (angle2°)
Directions: misalignment: E°, [uvw]_A (angle3°) / [uvw]_M (angle4°)
====================================================================================
```

Where:
- **LCV**: Lattice Correspondence Variant number
- **Score**: Overall matching quality
- **Misalignment**: Angular deviation between measured and predicted
- **(hkl)_A, (hkl)_M**: Miller indices in austenite and martensite
- **[uvw]_A, [uvw]_M**: Direction indices in austenite and martensite

---

### printCorresp()

**Purpose**: Print correspondence relationships showing how crystallographic features transform between phases.

#### Function Signature
```python
def printCorresp(self, sel=None, ifaces=None, printfor=None, printvars=None)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sel` | list of int | None | Selection indices (None = all) |
| `ifaces` | list of int | None | Interface indices (None = all) |
| `printfor` | str | keyma | Phase to print correspondence for ('keyau' or 'keyma') |
| `printvars` | list/str | None | Variants to print (None = all, 'closest' = best fit only) |

#### Output Structure

```
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Selection # X
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Interface # Y
---------------------------------------------------------
Habit plane: misalignment: Z°, (hkl)_A (angle1°) / (hkl)_M (angle2°)
Best fitting variant is V
Habit plane normal: ebsd vs. calculated from austenite/direction in habit plane: ebsd vs. calculated from austenite
Variant W: (hkl)_M vs. (h'k'l')_M / [uvw]_M vs. [u'v'w']_M
-----------------------------------------
=====================================================
```

This allows verification that measured EBSD orientations match theoretical predictions from lattice correspondence theory.

---

## Advanced Features

### Transformation Strain Calculation

The library calculates transformation strains associated with martensitic variants:

```python
# Access transformation strain for closest variant
CV = analyzer.Sels[sel_idx]['Closest Variant']
strain_tensor = analyzer.Sels[sel_idx]['Transformation strain_allvars'][CV]

# Strain component along specific direction
strain_100 = strain_tensor[0]['along [1. 0. 0.] [-]']
print(f"Transformation strain along [100]: {strain_100:.4f}")
```

### Symmetry Operations

The library handles crystallographic symmetry through:

1. **Point Group Operations**: Rotation matrices representing crystal symmetries
2. **Martensite Symmetry**: Special symmetry operations for product phase
3. **Variant Generation**: Symmetrically equivalent orientation relationships

### Misorientation Analysis

```python
# Calculate misorientation between phases
misori_angle = quat_misori_deg(quat1, quat2)

# Find symmetry-equivalent orientation with minimum misorientation
q_best, M_best, min_angle = find_best_symmetric_quat(
    q, q_ref, symops, max_iter=10
)
```

---

## Performance Optimization

### Numba Acceleration

Functions decorated with `@njit` provide significant speedups:

```python
@njit
def find_best_symmetric_quat(q, q_ref, symops, max_iter=10, tol=1e-6):
    # Numba-compiled code runs at C speed
    ...
```

**Best Practices**:
- First call compiles the function (initial overhead)
- Subsequent calls execute at native speed
- Avoid Python objects inside `@njit` functions
- Use NumPy arrays exclusively

### Vectorization

Prefer vectorized operations over loops:

```python
# Slow: loop over orientations
for i in range(N):
    M_reduced[i] = process_orientation(M[i], symops)

# Fast: vectorized operation
M_eq = np.einsum("sab,nbc->snac", symops, M, optimize=True)
```

### Memory Management

For large EBSD datasets:

```python
# Process in chunks to manage memory
chunk_size = 10000
N_total = len(orientations)

for start in range(0, N_total, chunk_size):
    end = min(start + chunk_size, N_total)
    chunk = orientations[start:end]
    processed = reduce_to_fundzone(chunk, symops)
    results[start:end] = processed
```

---

## Examples and Use Cases

### Complete Workflow Example

```python
import numpy as np
from ebsdlib import *

# 1. Load EBSD data (pseudocode)
ebsd_data = load_ebsd('sample.ang')
orientations = ebsd_data['orientations']
phases = ebsd_data['phases']

# 2. Filter small grains
grain_labels = ebsd_data['grain_ids']
filtered_labels = remove_small_clusters(grain_labels, minidxs=50)

# 3. Separate phases
austenite_mask = phases == 1
martensite_mask = phases == 2

M_austenite = orientations[austenite_mask]
M_martensite = orientations[martensite_mask]

# 4. Reduce to fundamental zones
symops_cubic = get_cubic_symmetry_operations()
M_aus_reduced = reduce_to_fundzone(M_austenite, symops_cubic)

symops_mono = get_monoclinic_symmetry_operations()
M_mar_reduced = reduce_to_fundzone(M_martensite, symops_mono)

# 5. Analyze habit planes (pseudocode)
analyzer = EBSDAnalyzer(ebsd_data)
analyzer.findInterfaces()
analyzer.getHabitPlanes()
analyzer.matchVariants()

# 6. Print results
analyzer.printHPmatches(sel=[0,1,2])
analyzer.printCorresp(printvars='closest')
```

### Specific Use Case: Variant Identification

```python
# Identify transformation variant for each martensite region
variants = []
for region in martensite_regions:
    # Get orientations
    M_aus = region.parent_orientation
    M_mar = region.martensite_orientation
    
    # Test all variants
    best_variant = None
    min_misori = np.inf
    
    for var_idx in range(n_variants):
        M_predicted = correspondence_matrices[:,:,var_idx] @ M_aus
        misori = calculate_misorientation(M_mar, M_predicted)
        
        if misori < min_misori:
            min_misori = misori
            best_variant = var_idx
    
    variants.append({
        'variant': best_variant,
        'misorientation': min_misori,
        'region_id': region.id
    })

# Analyze variant statistics
variant_counts = np.bincount([v['variant'] for v in variants])
print(f"Variant distribution: {variant_counts}")
```

---

## Appendices

### A. Miller Index Conventions

Miller indices (hkl) represent planes:
- (100): YZ plane
- (010): XZ plane  
- (001): XY plane
- (111): Plane with equal intercepts

Direction indices [uvw] represent directions:
- [100]: X axis
- [010]: Y axis
- [001]: Z axis
- [111]: Body diagonal

### B. Quaternion Representation

Quaternions q = [x, y, z, w] represent rotations:
- (x, y, z): Rotation axis × sin(θ/2)
- w: cos(θ/2)
- Rotation angle: θ = 2 × arccos(w)

### C. Common Symmetry Groups

| Crystal System | Point Group | # Operations | Examples |
|----------------|-------------|--------------|----------|
| Cubic | m-3m (Oh) | 24 | Fe, Ni, austenite |
| Hexagonal | 6/mmm | 24 | Ti, Mg, Zn |
| Monoclinic | 2/m | 4 | B19' NiTi martensite |
| Orthorhombic | mmm | 8 | B19 NiTi |

### D. Typical Parameter Values

For NiTi shape memory alloys:

```python
# Habit plane search parameters
maxdevfrom90deg = 5.0  # degrees
maxmillerindex = 3     # reasonable for low-index planes
angle_step = 0.5       # degrees (good balance of accuracy/speed)

# Cluster filtering
min_grain_size = 10    # pixels
min_interface_length = 20  # pixels

# Variant matching
misorientation_tolerance = 2.0  # degrees
strain_tolerance = 0.01  # 1% strain
```

---

**Navigation:** [Home](../README.md) | [Summary](./documentation_summary.md) | [Quick Ref Comprehensive](./quick_reference_comprehensive.md) | [Quick Ref](./quick_reference.md)

---

*Documentation generated for ebsdlib v1.0 - June 2025*
