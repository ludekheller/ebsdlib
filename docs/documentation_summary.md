# Documentation Summary - EBSD Analysis Library

**Navigation:** [Home](../README.md) | [Complete Docs](./documentation_complete.md) | [Quick Ref Comprehensive](./quick_reference_comprehensive.md) | [Quick Ref](./quick_reference.md)

---

## Library Overview

**ebsdlib** is a Python library for EBSD (Electron Backscatter Diffraction) analysis specializing in phase transformation characterization, particularly martensitic transformations in NiTi shape memory alloys.

### Core Purpose

- Analyze crystallographic orientations from EBSD data
- Characterize habit planes at phase boundaries
- Determine orientation relationships between phases
- Identify and rank transformation variants

---

## Key Components

### 1. Cluster Analysis Utilities

#### `remove_small_clusters(labels, minidxs=5)`

Filters labeled data by removing clusters smaller than a threshold.

**Quick Example:**
```python
labels = np.array([1,1,1,2,2,3,3,3,3,0,0])
clean = remove_small_clusters(labels, minidxs=3)
# Result: clusters with <3 pixels removed
```

**Use Cases:**
- Remove noise from grain maps
- Clean up phase segmentation
- Filter artifacts before analysis

---

### 2. Orientation Processing

#### `reduce_to_fundzone(M, symops)`

Vectorized reduction of orientations to fundamental zone (fast).

**Key Features:**
- Batch processes N orientations simultaneously
- Uses symmetry operations to find equivalent orientations
- Selects orientation with minimum rotation from identity
- 10-50× faster than loop-based version

**Quick Example:**
```python
M = orientation_matrices  # (N, 3, 3)
symops = cubic_symmetry_ops  # (24, 3, 3)
M_reduced = reduce_to_fundzone(M, symops)
```

#### `reduce_to_fundzone_slow(M, symops)`

Non-vectorized reference implementation.

**When to Use:**
- Small datasets (N < 100)
- Debugging and verification
- Lower memory requirement scenarios

#### `find_best_symmetric_quat(q, q_ref, symops, max_iter=10, tol=1e-6)`

Numba-accelerated iterative search for optimal symmetric quaternion.

**Features:**
- JIT-compiled for speed (10-100× faster)
- Finds symmetry-equivalent quaternion closest to reference
- Returns quaternion, matrix, and misorientation angle

---

### 3. Habit Plane Analysis

The main analysis class provides sophisticated habit plane characterization:

#### `getNormals(interface_trace, interfacenorm_trace, LrI, Lr, G2Sampl, ...)`

Generates candidate plane normals perpendicular to interface traces.

**Parameters:**
- `interface_trace`: Direction along interface
- `LrI`, `Lr`: Reciprocal lattice matrices
- `G2Sampl`: Crystal-to-sample transformation
- `maxdevfrom90deg`: Tolerance for perpendicularity (default: 5°)
- `maxmillerindex`: Maximum Miller index (default: 3)

**Returns:** Dictionary containing:
- Miller indices of candidate planes
- Normal vectors in crystal and sample frames
- Angles between habit plane and trace

**Application:**
```python
candidates = analyzer.getNormals(
    interface_trace=trace_dir,
    interfacenorm_trace=normal_dir,
    LrI=inv_reciprocal_lattice,
    Lr=reciprocal_lattice,
    G2Sampl=crystal_to_sample,
    maxdevfrom90deg=5.0,
    maxmillerindex=3
)
```

#### `getCorrespNormals(N_guess, interface_trace2, LC, Lr2, LCall=None)`

Calculates corresponding normals in second phase using lattice correspondence.

**Purpose:**
- Transform habit plane predictions from austenite to martensite
- Check consistency across phase transformation
- Evaluate all transformation variants

**Returns:**
- Corresponding normals for closest variant
- Optionally: normals for all variants

---

### 4. Results Display

#### `printHPmatches(sel=None, ifaces=None, nodirs=False)`

Displays habit plane matching results with quality metrics.

**Output Includes:**
- Matching scores for each interface
- Closest lattice correspondence variant
- Transformation strain values
- Miller indices and angular deviations
- Habit plane normals and directions

**Example Output:**
```
Fitting Closest LCV 2: Score 8.5, mean misalignment: 1.3°
Transformation strain along [100]: 0.0425
Normals: misalignment: 0.8°, (0 1 1)_A (89.5°) / (0 2 1)_M (90.2°)
```

#### `printCorresp(sel=None, ifaces=None, printfor=None, printvars=None)`

Shows correspondence relationships between phases.

**Features:**
- Compare EBSD-measured vs. theoretically-predicted orientations
- Verify lattice correspondence theory
- Check multiple transformation variants
- Display both habit plane normals and directions

**Parameters:**
- `printfor`: Which phase to show ('keyau' or 'keyma')
- `printvars`: Which variants ('closest', list of indices, or None for all)

---

## Typical Workflow

```python
# 1. Load and prepare data
ebsd_data = load_ebsd_file('scan.ang')
orientations = ebsd_data['orientations']
phases = ebsd_data['phases']

# 2. Clean up microstructure
grain_labels = detect_grains(phases)
clean_labels = remove_small_clusters(grain_labels, minidxs=50)

# 3. Reduce orientations to fundamental zones
austenite_orientations = orientations[phases == 1]
martensite_orientations = orientations[phases == 2]

M_aus = reduce_to_fundzone(austenite_orientations, cubic_symops)
M_mar = reduce_to_fundzone(martensite_orientations, monoclinic_symops)

# 4. Initialize analyzer
analyzer = EBSDAnalyzer(ebsd_data)

# 5. Detect and analyze interfaces
analyzer.findInterfaces()
analyzer.getHabitPlanes()
analyzer.matchVariants()

# 6. Display results
analyzer.printHPmatches(sel=[0, 1, 2])  # First 3 selections
analyzer.printCorresp(printvars='closest')  # Best-fit variants only
```

---

## Performance Tips

### Speed Optimization

1. **Use vectorized functions:**
   ```python
   # Fast
   M_reduced = reduce_to_fundzone(M, symops)
   
   # Slow
   M_reduced = reduce_to_fundzone_slow(M, symops)
   ```

2. **Leverage Numba:**
   ```python
   # First call compiles (slow)
   result1 = find_best_symmetric_quat(q1, q_ref, symops)
   
   # Subsequent calls are fast
   result2 = find_best_symmetric_quat(q2, q_ref, symops)
   ```

3. **Process in chunks for large datasets:**
   ```python
   chunk_size = 10000
   for start in range(0, N_total, chunk_size):
       end = min(start + chunk_size, N_total)
       process_chunk(orientations[start:end])
   ```

### Memory Management

- Vectorized operations use more memory: O(Ns × N) vs O(1)
- Trade-off: Speed vs. memory usage
- For >100k orientations, consider chunking

---

## Common Parameters

### Habit Plane Analysis

```python
# Typical values for NiTi
maxdevfrom90deg = 5.0      # ±5° from perpendicular
maxmillerindex = 3         # Low-index planes only
angle_steps = 361          # 0.5° resolution (0-180°)
```

### Cluster Filtering

```python
# Grain/feature size thresholds
min_grain_pixels = 10      # Remove tiny grains
min_interface_pixels = 20  # Minimum interface length
```

### Variant Matching

```python
# Tolerance parameters
misorientation_tol = 2.0   # degrees
strain_tolerance = 0.01    # 1% strain difference
max_iterations = 10        # For iterative optimization
convergence_tol = 1e-6     # Convergence criterion
```

---

## Dependencies

### Required Packages

```bash
pip install numpy scipy matplotlib numba orix crystals
```

### Custom Modules

Must be in Python path:
- `orilib` - Orientation utilities
- `projlib` - Stereographic projections
- `plotlib` - Crystallographic plotting
- `crystlib` - Crystal calculations
- `effective_elastic_constants_functions`
- `getphases` - Phase identification

---

## Data Structures

### Orientation Matrices

- **Format:** (N, 3, 3) NumPy array
- **Convention:** Sample-to-crystal transformation
- **Requirement:** Proper orthogonal matrices (det = +1)

### Quaternions

- **Format:** [x, y, z, w] where (x,y,z) = axis × sin(θ/2), w = cos(θ/2)
- **Normalization:** Must be unit quaternions (|q| = 1)
- **Sign convention:** w ≥ 0 preferred

### Miller Indices

- **Planes:** (hkl) - parentheses
- **Directions:** [uvw] - square brackets
- **Families:** {hkl} and <uvw> - symmetrically equivalent

---

## Crystallographic Conventions

### Reference Frames

1. **Crystal Frame:** Aligned with crystal axes (a, b, c)
2. **Sample Frame:** Lab coordinate system (X, Y, Z)
3. **Transformation:** `v_sample = G2Sampl @ v_crystal`

### Symmetry Operations

- **Cubic (m-3m):** 24 operations
- **Hexagonal (6/mmm):** 24 operations
- **Monoclinic (2/m):** 4 operations
- **Representation:** 3×3 rotation matrices

### Misorientation

```python
# Angle between two orientations
theta = 2 * arccos(|q1 · q2|)

# Minimum misorientation (accounting for symmetry)
theta_min = min over all symmetric equivalents
```

---

## Scientific Context

### Martensitic Transformation

Phase transformation characterized by:
- **Diffusionless:** Atoms move cooperatively
- **Habit Plane:** Invariant plane between phases
- **Orientation Relationship:** Crystallographic correspondence
- **Variants:** Symmetrically equivalent transformation modes

### NiTi Shape Memory Alloys

- **Austenite:** B2 cubic structure (m-3m symmetry)
- **Martensite:** B19' monoclinic (2/m symmetry)
- **Variants:** Typically 12-24 depending on correspondence
- **Habit Planes:** Often near {011}_B2

### Lattice Correspondence

Relates lattice vectors between phases:
```
a_M = CP @ a_A  (plane normals)
u_M = CD @ u_A  (directions)
```

Where:
- CP: Correspondence matrix for planes
- CD: Correspondence matrix for directions
- Generally: CD = (CP⁻¹)ᵀ

---

## Troubleshooting

### Common Issues

**Problem:** "Small clusters not being removed"
```python
# Solution: Check label format (must be integers ≥ 0)
labels = labels.astype(int)
labels[labels < 0] = 0
```

**Problem:** "Fundamental zone reduction giving inconsistent results"
```python
# Solution: Ensure orientation matrices are proper rotations
for M in orientations:
    assert np.allclose(np.linalg.det(M), 1.0)
    assert np.allclose(M @ M.T, np.eye(3))
```

**Problem:** "No habit plane candidates found"
```python
# Solution: Relax search criteria
maxdevfrom90deg = 10.0  # Increase tolerance
maxmillerindex = 5      # Allow higher indices
```

### Performance Issues

**Slow processing:**
- Use `reduce_to_fundzone()` not `reduce_to_fundzone_slow()`
- Process in chunks if dataset > 100k orientations
- Ensure Numba is properly installed and compiling

**Memory errors:**
- Reduce chunk size
- Use slower but memory-efficient version
- Close unused arrays with `del`

---

## Further Reading

- **[Complete Documentation](./documentation_complete.md)** - Detailed API reference
- **[Quick Reference](./quick_reference_comprehensive.md)** - All function signatures
- **[Quick Cheat Sheet](./quick_reference.md)** - Common operations

### External Resources

- Orix documentation: https://orix.readthedocs.io/
- MTEX (MATLAB): https://mtex-toolbox.github.io/
- Crystallography textbooks on phase transformations
- EBSD analysis fundamentals

---

**Navigation:** [Home](../README.md) | [Complete Docs](./documentation_complete.md) | [Quick Ref Comprehensive](./quick_reference_comprehensive.md) | [Quick Ref](./quick_reference.md)

---

*Summary for ebsdlib v1.0 - June 2025*
