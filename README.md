# EBSD Analysis Library

**Author:** lheller  
**Version:** 1.0  
**Created:** June 13, 2025

## Overview

`ebsdlib` is a comprehensive Python library for Electron Backscatter Diffraction (EBSD) analysis, specifically designed for characterizing phase transformations in crystalline materials. The library focuses on martensitic transformations, particularly in NiTi shape memory alloys, providing advanced tools for crystallographic analysis, habit plane determination, and orientation relationship characterization.

## Features

- **Crystallographic Analysis**: Advanced orientation analysis and reduction to fundamental zones
- **Phase Transformation Characterization**: Tools for analyzing martensitic transformations
- **Habit Plane Analysis**: Automatic detection and characterization of habit planes
- **Orientation Relationships**: Determination of correspondence matrices between phases
- **Variant Analysis**: Identification and analysis of transformation variants
- **Cluster Analysis**: Pixel clustering utilities for microstructure characterization
- **Visualization**: Integrated plotting capabilities for crystallographic data

## Installation

### Prerequisites

```bash
pip install numpy scipy matplotlib numba
pip install orix  # For quaternion and orientation handling
pip install crystals  # For crystal structure definitions
```

### Additional Dependencies

The library requires the following custom modules (ensure they are in your Python path):
- `orilib`: Orientation, quaternion, and Euler angle utilities
- `projlib`: Stereographic projection functions
- `plotlib`: Crystallographic plotting utilities
- `crystlib`: Crystallographic calculation functions
- `effelconst`: Elastic constants calculations
- `getphases`: Phase identification utilities

### Installation

```bash
# Clone or download the repository
# Add ebsdlib.py to your Python path or project directory
import ebsdlib
```

## Quick Start

```python
import numpy as np
from ebsdlib import *

# Example 1: Remove small clusters from labeled data
labels = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 0])
filtered_labels = remove_small_clusters(labels, minidxs=3)
print(filtered_labels)  # Small clusters removed

# Example 2: Reduce orientations to fundamental zone
# Assuming you have orientation matrices M and symmetry operations symops
M_reduced = reduce_to_fundzone(M, symops)

# Example 3: Create EBSD analysis instance (assuming main class)
# analyzer = EBSDAnalyzer(data)
# analyzer.findInterfaces()
# analyzer.getHabitPlanes()
```

## Key Capabilities

### 1. Orientation Processing
- Fundamental zone reduction (vectorized and standard implementations)
- Symmetry-equivalent orientation finding
- Quaternion-based misorientation calculations

### 2. Cluster Analysis
- Small cluster removal with automatic relabeling
- Minimum cluster size filtering
- Contiguous label reassignment

### 3. Habit Plane Analysis
- Candidate plane normal generation
- Miller index determination
- Crystallographic correspondence calculation
- Misalignment quantification

### 4. Correspondence Relationships
- Phase-to-phase correspondence matrices
- Variant identification and ranking
- Transformation strain calculations
- Direction and normal correspondence

## Documentation

- **[Complete Documentation](./docs/documentation_complete.md)** - Comprehensive API reference with detailed descriptions
- **[Documentation Summary](./docs/documentation_summary.md)** - Concise overview of key features and functions
- **[Quick Reference (Comprehensive)](./docs/quick_reference_comprehensive.md)** - Detailed function signatures and parameters
- **[Quick Reference](./docs/quick_reference.md)** - Compact cheat sheet for common operations

## Main Components

### Utility Functions

| Function | Purpose |
|----------|---------|
| `remove_small_clusters()` | Filter clusters by minimum size |
| `reduce_to_fundzone()` | Vectorized fundamental zone reduction |
| `reduce_to_fundzone_slow()` | Standard fundamental zone reduction |
| `find_best_symmetric_quat()` | Find optimal symmetric quaternion |

### Analysis Methods

The library includes methods for:
- Interface detection and analysis
- Habit plane matching and scoring
- Correspondence printing and visualization
- Normal vector generation and matching

## Usage Examples

### Cluster Filtering

```python
# Remove clusters with fewer than 5 pixels
labels = your_label_array
filtered = remove_small_clusters(labels, minidxs=5)
```

### Orientation Reduction

```python
# Reduce orientations to fundamental zone
from scipy.spatial.transform import Rotation as R

# Your orientation matrices (N, 3, 3)
M = orientation_matrices

# Your symmetry operations (Ns, 3, 3)
symops = symmetry_operations

# Vectorized reduction (fast)
M_reduced = reduce_to_fundzone(M, symops)
```

### Habit Plane Analysis

```python
# Generate candidate plane normals
normals = analyzer.getNormals(
    interface_trace=trace_vector,
    interfacenorm_trace=normal_to_trace,
    LrI=inverse_reciprocal_lattice,
    Lr=reciprocal_lattice,
    G2Sampl=crystal_to_sample_matrix,
    maxdevfrom90deg=5.0,
    maxmillerindex=3
)
```

## Performance Considerations

- **Vectorized Operations**: Use `reduce_to_fundzone()` for batch processing (much faster than `reduce_to_fundzone_slow()`)
- **Numba Acceleration**: Functions decorated with `@njit` provide significant speedup
- **Memory Management**: Large EBSD datasets may require chunked processing

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional symmetry group support
- Extended visualization capabilities
- Performance optimizations
- Additional phase transformation models

## Citation

If you use this library in your research, please cite:

```
EBSD Analysis Library
Author: lheller
GitHub: [Repository URL]
Year: 2025
```

## License

[Specify your license here]

## Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Contact: [Your contact information]

## Version History

- **v1.0** (June 2025): Initial release
  - Core EBSD analysis functionality
  - Habit plane analysis
  - Orientation relationships
  - Cluster utilities

---

**See Also:**
- [Orix Documentation](https://orix.readthedocs.io/) - For orientation and symmetry operations
- [MTEX](https://mtex-toolbox.github.io/) - MATLAB toolbox for texture analysis
- Related crystallographic analysis tools
