#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:36:58 2025

@author: lheller

EBSD Analysis Library for phase transformation characterization
Main classes for crystallographic analysis of martensitic transformations,
particularly focused on NiTi shape memory alloys.
"""
from numba import njit
from orix import plot
from orix.quaternion import Orientation, Rotation, symmetry
from orix.vector import Vector3d
from orix import plot
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import matplotlib.path as mpltPath

#from orilib import * # Orientations, quaternions, Euler angles
#from projlib import * # Stereographic projections
#from plotlib import * # Crystallographic plotting
#from crystlib import * # Crystallographic calculations
#from effelconst import  * #effective elastic constants calculations


from crystals import Crystal
from numpy import sqrt
from crystlibs import * #importing orilib: Orientations, quaternions, Euler angles, projlib import: Stereographic projections, plotlib: Crystallographic plotting
from effelconst import * #effective elastic constants calculations
from getphases import getPhases
import pyebsd
from matplotlib.path import Path
import time
_2PI = 2 * np.pi
_COS60 = 0.5  # cos(60deg)
_SIN60 = 0.5 * 3.0**0.5  # sin(60deg)


import h5py
from scipy.spatial.transform import Rotation as ScipyRotation
def annotate_sample_transform(ax, sample_transform):
    """
    Append a note to ax's title reporting where the standard sample
    axes x=[1,0,0], y=[0,1,0], z=[0,0,1] land under sample_transform --
    i.e. the coordinate system the pole figure is actually plotted in,
    when a non-identity sample_transform was used.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes returned by plot_cluster_pole_figure (or any function
        following the same title-append convention).
    sample_transform : (3,3) array
        The same matrix passed as sample_transform to
        plot_cluster_pole_figure.
    """
    T = np.asarray(sample_transform)

    def _fmt(vec):
        return tuple(float(round(v, 2)) for v in vec)

    x_new = _fmt(T.dot([1.0, 0.0, 0.0]))
    y_new = _fmt(T.dot([0.0, 1.0, 0.0]))
    z_new = _fmt(T.dot([0.0, 0.0, 1.0]))

    note = f"plotted in rotated frame:\nx={x_new}, y={y_new}, z={z_new}"

    existing_title = ax.get_title()
    ax.set_title(f"{existing_title}\n{note}" if existing_title else note)
def _generate_distinct_colors(n):
    """
    Generate n visually distinct, evenly-spaced colors (as RGB tuples in
    [0,1]) using golden-angle hue stepping -- good spacing regardless of
    n, no need to know the count in advance.
    """
    import colorsys
    golden_ratio_conjugate = 0.618033988749895
    hue = 0.0
    colors = []
    for _ in range(n):
        hue = (hue + golden_ratio_conjugate) % 1.0
        colors.append(colorsys.hsv_to_rgb(hue, 0.85, 0.85))
    return colors
def save_to_hdf5(filename, data, key='data'):
    """
    Save data (dict or list of dicts) to HDF5.
    
    Parameters
    ----------
    filename : str
        Output HDF5 filename
    data : dict or list of dict
        Data to save
    key : str, optional
        Root key name. Default is 'data'.
    
    Examples
    --------
    >>> # Single dictionary
    >>> save_to_hdf5('data.h5', my_dict)
    
    >>> # List of dictionaries
    >>> save_to_hdf5('data.h5', [dict1, dict2, dict3])
    """
    with h5py.File(filename, 'w') as f:
        if isinstance(data, list):
            # List of dictionaries
            list_group = f.create_group(key)
            list_group.attrs['type'] = 'list_of_dicts'
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    item_group = list_group.create_group(f'item_{i}')
                    _write_dict(item_group, item)
                else:
                    raise ValueError(f"Item {i} in list is not a dictionary")
        elif isinstance(data, dict):
            # Single dictionary
            f.attrs['type'] = 'single_dict'
            _write_dict(f, data)
        else:
            raise ValueError("Data must be a dictionary or list of dictionaries")
    
    print(f"Saved to {filename}")


def load_from_hdf5(filename):
    """
    Load data (dict or list of dicts) from HDF5.
    
    Parameters
    ----------
    filename : str
        Input HDF5 filename
    
    Returns
    -------
    dict or list of dict
        Loaded data
    
    Examples
    --------
    >>> # Load (automatically detects if single dict or list)
    >>> data = load_from_hdf5('data.h5')
    """
    with h5py.File(filename, 'r') as f:
        # Check if it's a single dict or list of dicts
        if 'type' in f.attrs and f.attrs['type'] == 'single_dict':
            # Single dictionary
            return _read_dict(f)
        else:
            # Check for list of dicts
            for key in f.keys():
                item = f[key]
                if isinstance(item, h5py.Group) and 'type' in item.attrs and item.attrs['type'] == 'list_of_dicts':
                    # List of dictionaries
                    result = []
                    for i in range(len([k for k in item.keys() if k.startswith('item_')])):
                        if f'item_{i}' in item:
                            result.append(_read_dict(item[f'item_{i}']))
                    return result
            
            # If no type attribute, assume single dict for backward compatibility
            return _read_dict(f)


def _write_dict(group, dictionary):
    """Internal function to write a dictionary to an HDF5 group."""
    for key, value in dictionary.items():
        # Sanitize key
        safe_key = str(key).replace('/', '_').replace(' ', '_')
        
        if isinstance(value, dict):
            # Create subgroup for nested dict
            subgroup = group.create_group(safe_key)
            _write_dict(subgroup, value)
            
        elif isinstance(value, np.ndarray):
            # Save numpy arrays
            group.create_dataset(safe_key, data=value, compression='gzip')
            
        elif isinstance(value, list):
            # Handle lists
            if len(value) > 0:
                if isinstance(value[0], dict):
                    # List of dicts
                    list_group = group.create_group(safe_key)
                    list_group.attrs['list_type'] = 'dicts'
                    for i, item in enumerate(value):
                        item_group = list_group.create_group(f'item_{i}')
                        _write_dict(item_group, item)
                elif isinstance(value[0], np.ndarray):
                    # List of arrays - try to stack
                    try:
                        stacked = np.stack(value)
                        group.create_dataset(safe_key, data=stacked, compression='gzip')
                    except:
                        # Can't stack, save individually
                        list_group = group.create_group(safe_key)
                        list_group.attrs['list_type'] = 'arrays'
                        for i, arr in enumerate(value):
                            list_group.create_dataset(f'item_{i}', data=arr, compression='gzip')
                elif hasattr(value[0], '__class__') and 'Rotation' in value[0].__class__.__name__:
                    # List of Rotation objects
                    list_group = group.create_group(safe_key)
                    list_group.attrs['list_type'] = 'rotations'
                    for i, rot in enumerate(value):
                        _save_rotation_object(list_group, f'item_{i}', rot)
                else:
                    # Simple list - convert to array
                    try:
                        group.create_dataset(safe_key, data=np.array(value))
                    except:
                        group.attrs[safe_key] = str(value)
            else:
                # Empty list
                group.attrs[safe_key] = '[]'
                
        elif hasattr(value, '__class__') and 'Rotation' in value.__class__.__name__:
            # Handle any Rotation object
            _save_rotation_object(group, safe_key, value)
            
        elif isinstance(value, (int, float, tuple, bool, np.bool, np.integer, np.floating)):
            # Store as attribute for small scalars
            group.attrs[safe_key] = value
            
        elif isinstance(value, str):
            # Store strings as attributes
            group.attrs[safe_key] = value
            
        elif value is None:
            # Store None
            group.attrs[safe_key] = 'None'
            
        else:
            # Unknown type - store as string
            print(f"Warning: Unknown type {type(value)} for key {key}, storing as string")
            group.attrs[safe_key] = str(value)


def _save_rotation_object(group, key, rotation):
    """Internal function to save scipy or orix Rotation object."""
    rotation_type = rotation.__class__.__module__ + '.' + rotation.__class__.__name__
    
    if 'orix' in rotation_type:
        # orix Rotation - save as quaternions
        quaternions = rotation.data
        
        rot_group = group.create_group(key)
        rot_group.create_dataset('quaternions', data=quaternions, compression='gzip')
        rot_group.attrs['type'] = 'orix.Rotation'
        rot_group.attrs['shape'] = rotation.shape
        
    elif 'scipy' in rotation_type:
        # scipy Rotation - save as quaternions
        quaternions = rotation.as_quat()
        
        rot_group = group.create_group(key)
        rot_group.create_dataset('quaternions', data=quaternions, compression='gzip')
        rot_group.attrs['type'] = 'scipy.Rotation'
        
    else:
        # Unknown rotation type - try to save as matrix
        print(f"Warning: Unknown Rotation type {rotation_type}, trying to save as matrix")
        try:
            if hasattr(rotation, 'as_matrix'):
                matrix = rotation.as_matrix()
            elif hasattr(rotation, 'to_matrix'):
                matrix = rotation.to_matrix()
            else:
                matrix = np.array(rotation)
            
            group.create_dataset(key, data=matrix, compression='gzip')
            group[key].attrs['type'] = 'Rotation_matrix'
        except Exception as e:
            print(f"Error saving rotation: {e}")
            group.attrs[key] = str(rotation)


def _read_dict(group):
    """Internal function to read a dictionary from an HDF5 group."""
    # Try to import orix
    try:
        from orix.quaternion import Rotation as OrixRotation
        has_orix = True
    except ImportError:
        has_orix = False
    
    result = {}
    
    # Load datasets (arrays) and groups
    for key in group.keys():
        item = group[key]
        
        if isinstance(item, h5py.Group):
            # Check if it's a Rotation object
            if 'type' in item.attrs:
                rot_type = item.attrs['type']
                
                if rot_type == 'orix.Rotation':
                    # Reconstruct orix Rotation
                    quaternions = item['quaternions'][()]
                    
                    if has_orix:
                        if 'shape' in item.attrs:
                            shape = item.attrs['shape']
                            result[key] = OrixRotation(quaternions).reshape(shape)
                        else:
                            result[key] = OrixRotation(quaternions)
                    else:
                        result[key] = quaternions
                        
                elif rot_type == 'scipy.Rotation':
                    # Reconstruct scipy Rotation
                    quaternions = item['quaternions'][()]
                    result[key] = ScipyRotation.from_quat(quaternions)
                    
                elif rot_type == 'Rotation_matrix':
                    result[key] = item[()]
                    
            # Check if it's a list with type annotation
            elif 'list_type' in item.attrs:
                list_type = item.attrs['list_type']
                items = []
                
                for i in range(len([k for k in item.keys() if k.startswith('item_')])):
                    if f'item_{i}' in item:
                        sub_item = item[f'item_{i}']
                        
                        if list_type == 'dicts':
                            items.append(_read_dict(sub_item))
                        elif list_type == 'arrays':
                            items.append(sub_item[()])
                        elif list_type == 'rotations':
                            # Reconstruct Rotation
                            if 'type' in sub_item.attrs:
                                rot_type = sub_item.attrs['type']
                                if rot_type == 'orix.Rotation':
                                    quaternions = sub_item['quaternions'][()]
                                    if has_orix:
                                        items.append(OrixRotation(quaternions))
                                    else:
                                        items.append(quaternions)
                                elif rot_type == 'scipy.Rotation':
                                    quaternions = sub_item['quaternions'][()]
                                    items.append(ScipyRotation.from_quat(quaternions))
                
                result[key] = items
                
            # Check if it's a generic list of items (backward compatibility)
            elif all(k.startswith('item_') for k in item.keys()):
                items = []
                for i in range(len([k for k in item.keys() if k.startswith('item_')])):
                    if f'item_{i}' in item:
                        sub_item = item[f'item_{i}']
                        if isinstance(sub_item, h5py.Group):
                            if 'type' in sub_item.attrs:
                                # It's a Rotation
                                rot_type = sub_item.attrs['type']
                                if rot_type == 'orix.Rotation':
                                    quaternions = sub_item['quaternions'][()]
                                    if has_orix:
                                        items.append(OrixRotation(quaternions))
                                    else:
                                        items.append(quaternions)
                                elif rot_type == 'scipy.Rotation':
                                    quaternions = sub_item['quaternions'][()]
                                    items.append(ScipyRotation.from_quat(quaternions))
                            else:
                                # Regular nested dict
                                items.append(_read_dict(sub_item))
                        else:
                            # Dataset
                            items.append(sub_item[()])
                result[key] = items
            else:
                # Regular nested dict
                result[key] = _read_dict(item)
        else:
            # Dataset (array)
            result[key] = item[()]
    
    # Load attributes (scalars, strings)
    for key, value in group.attrs.items():
        if key in ['type', 'list_type', 'shape']:  # Skip internal attributes
            continue
        if value == 'None':
            result[key] = None
        elif value == '[]':
            result[key] = []
        else:
            result[key] = value
    
    return result


# Convenience aliases (backward compatible)
def save_dict_to_hdf5(filename, data):
    """Backward compatible: save single dictionary to HDF5."""
    save_to_hdf5(filename, data)


def load_dict_from_hdf5(filename):
    """Backward compatible: load from HDF5."""
    return load_from_hdf5(filename)


def print_dict_structure(obj, indent=0, name="root", max_depth=10, max_array_display=3,
                        collapse_levels=None, collapse_levels_labels=None, 
                        collapse_preset=None, current_path="", _skip_type_line=False,
                        indent_size=2):  # NEW PARAMETER
    """
    Print dictionary structure with detailed type information.
    
    Parameters
    ----------
    obj : any
        Object to analyze
    indent : int
        Indentation level (internal use)
    name : str
        Key name
    max_depth : int, optional
        Maximum nesting depth. Default is 10.
    max_array_display : int, optional
        Maximum array dimensions to show in detail. Default is 3.
    collapse_levels : list of str or int, optional
        Levels to collapse (show only first key).
    collapse_levels_labels : list of str, optional
        Labels for collapsed levels. Must have same length as collapse_levels.
    collapse_preset : str, optional
        Named preset for collapse_levels.
    current_path : str
        Current path in the structure (internal use)
    _skip_type_line : bool
        Internal parameter to avoid printing duplicate type lines
    indent_size : int, optional
        Number of spaces per indentation level. Default is 2.
    
    Examples
    --------
    >>> # Default 2 spaces per level
    >>> print_dict_structure(hierarchy, collapse_levels=[0,1],
    ...                      collapse_levels_labels=['parent_id', 'child_id'])
    
    >>> # Use 4 spaces per level (more readable)
    >>> print_dict_structure(hierarchy, collapse_levels=[0,1],
    ...                      collapse_levels_labels=['parent_id', 'child_id'],
    ...                      indent_size=4)
    
    >>> # Use 1 space per level (compact)
    >>> print_dict_structure(hierarchy, collapse_levels=[0,1],
    ...                      collapse_levels_labels=['parent_id', 'child_id'],
    ...                      indent_size=1)
    """
    # Preset definitions
    COLLAPSE_PRESETS = {
        'minimal': ([0], ['PARENT']),
        'compact': ([0, 'children', 'parent_side', 'child_side'], 
                   ['PARENT', 'CHILD', 'PARENT_SIDE', 'CHILD_SIDE']),
        'interfaces': (['parent_side', 'child_side'], 
                      ['PARENT_SIDE', 'CHILD_SIDE']),
        'hierarchy': ([0, 'children', 'parent_side', 'child_side', 'semi_roi_1', 'semi_roi_2'],
                     ['PARENT', 'CHILD', 'PARENT_SIDE', 'CHILD_SIDE', 'SEMI_ROI_1', 'SEMI_ROI_2']),
        'sides_only': (['parent_side', 'child_side'], 
                      ['PARENT_SIDE', 'CHILD_SIDE']),
        'all_lists': ([0, 'children'], 
                     ['PARENT', 'CHILD']),
    }
    
    if collapse_preset is not None:
        if collapse_preset in COLLAPSE_PRESETS:
            preset_levels, preset_labels = COLLAPSE_PRESETS[collapse_preset]
            if collapse_levels is None:
                collapse_levels = preset_levels
                collapse_levels_labels = preset_labels
            else:
                collapse_levels = list(collapse_levels) + list(preset_levels)
                if collapse_levels_labels is None:
                    collapse_levels_labels = [None] * len(collapse_levels)
                else:
                    collapse_levels_labels = list(collapse_levels_labels) + list(preset_labels)
        else:
            available = ', '.join(COLLAPSE_PRESETS.keys())
            print(f"Warning: Unknown preset '{collapse_preset}'. Available: {available}")
    
    if collapse_levels_labels is not None and collapse_levels is not None:
        if len(collapse_levels_labels) != len(collapse_levels):
            raise ValueError(f"collapse_levels_labels length ({len(collapse_levels_labels)}) "
                           f"must match collapse_levels length ({len(collapse_levels)})")
    
    # CHANGED: Use indent_size parameter
    prefix = " " * (indent * indent_size)
    
    if indent > max_depth:
        print(f"{prefix}... (max depth)")
        return
    
    if current_path:
        path = f"{current_path}/{name}"
    else:
        path = name
    
    # Check if this level should be collapsed
    should_collapse = False
    collapse_label = None
    root_label_replacement = None
    
    if collapse_levels is not None:
        for i, collapse_spec in enumerate(collapse_levels):
            if isinstance(collapse_spec, int):
                if indent == collapse_spec:
                    should_collapse = True
                    if collapse_levels_labels and i < len(collapse_levels_labels):
                        if indent == 0:
                            root_label_replacement = collapse_levels_labels[i]
                        else:
                            collapse_label = collapse_levels_labels[i]
                    break
            elif isinstance(collapse_spec, str):
                if collapse_spec == name or collapse_spec in path:
                    should_collapse = True
                    if collapse_levels_labels and i < len(collapse_levels_labels):
                        collapse_label = collapse_levels_labels[i]
                    break
    
    if root_label_replacement is not None:
        display_name = root_label_replacement
    else:
        display_name = name
    
    # Determine type and format
    if isinstance(obj, dict):
        if should_collapse and len(obj) > 0:
            # COLLAPSED DICT
            first_key = list(obj.keys())[0]
            first_value = obj[first_key]
            
            # Determine the type info for the CHILD
            if isinstance(first_value, dict):
                type_info = f"dict [{len(first_value)} keys]"
            elif isinstance(first_value, list):
                type_info = f"list [{len(first_value)} items]"
            elif isinstance(first_value, np.ndarray):
                shape_str = "×".join(map(str, first_value.shape))
                type_info = f"array[{shape_str}] <{first_value.dtype}>"
            elif isinstance(first_value, (int, np.integer)):
                type_info = "int"
            elif isinstance(first_value, (float, np.floating)):
                type_info = "float"
            elif isinstance(first_value, str):
                type_info = "str"
            elif isinstance(first_value, bool):
                type_info = "bool"
            else:
                type_info = type(first_value).__name__
            
            if root_label_replacement is not None:
                effective_name = root_label_replacement
            elif collapse_label:
                effective_name = collapse_label
            else:
                effective_name = first_key
            
            print(f"{prefix}{effective_name} → {type_info}")
            
            if isinstance(first_value, (dict, list)):
                print_dict_structure(first_value, indent + 1, name=effective_name, 
                                   max_depth=max_depth, max_array_display=max_array_display,
                                   collapse_levels=collapse_levels, 
                                   collapse_levels_labels=collapse_levels_labels,
                                   current_path=path, _skip_type_line=True,
                                   indent_size=indent_size)  # ADDED
        else:
            # NOT COLLAPSED
            if not _skip_type_line:
                print(f"{prefix}{display_name} → dict [{len(obj)} keys]")
            
            for key in obj.keys():
                print_dict_structure(obj[key], indent + 1, name=key, 
                                   max_depth=max_depth, max_array_display=max_array_display,
                                   collapse_levels=collapse_levels,
                                   collapse_levels_labels=collapse_levels_labels,
                                   current_path=path,
                                   indent_size=indent_size)  # ADDED
    
    elif isinstance(obj, list):
        if len(obj) == 0:
            if not _skip_type_line:
                print(f"{prefix}{display_name} → list [empty]")
        else:
            if should_collapse:
                # COLLAPSED LIST
                first_item = obj[0]
                
                # Determine the type info for the CHILD
                if isinstance(first_item, dict):
                    type_info = f"dict [{len(first_item)} keys]"
                elif isinstance(first_item, list):
                    type_info = f"list [{len(first_item)} items]"
                elif isinstance(first_item, np.ndarray):
                    shape_str = "×".join(map(str, first_item.shape))
                    type_info = f"array[{shape_str}] <{first_item.dtype}>"
                elif isinstance(first_item, (int, np.integer)):
                    type_info = "int"
                elif isinstance(first_item, (float, np.floating)):
                    type_info = "float"
                elif isinstance(first_item, str):
                    type_info = "str"
                elif isinstance(first_item, bool):
                    type_info = "bool"
                else:
                    type_info = type(first_item).__name__
                
                if root_label_replacement is not None:
                    effective_name = root_label_replacement
                elif collapse_label:
                    effective_name = collapse_label
                else:
                    effective_name = "[0]"
                
                print(f"{prefix}{effective_name} → {type_info}")
                
                if isinstance(first_item, (dict, list)):
                    print_dict_structure(first_item, indent + 1, name=effective_name, 
                                       max_depth=max_depth, max_array_display=max_array_display,
                                       collapse_levels=collapse_levels,
                                       collapse_levels_labels=collapse_levels_labels,
                                       current_path=path, _skip_type_line=True,
                                       indent_size=indent_size)  # ADDED
            else:
                # NOT COLLAPSED
                if not _skip_type_line:
                    print(f"{prefix}{display_name} → list [{len(obj)} items]")
                
                print_dict_structure(obj[0], indent + 1, name="[0]", 
                                   max_depth=max_depth, max_array_display=max_array_display,
                                   collapse_levels=collapse_levels,
                                   collapse_levels_labels=collapse_levels_labels,
                                   current_path=path,
                                   indent_size=indent_size)  # ADDED
    
    elif isinstance(obj, np.ndarray):
        shape_str = "×".join(map(str, obj.shape))
        if obj.ndim <= max_array_display:
            print(f"{prefix}{display_name} → array[{shape_str}] <{obj.dtype}>")
        else:
            print(f"{prefix}{display_name} → array[{shape_str}] <{obj.dtype}> ({obj.ndim}D)")
    
    elif isinstance(obj, (int, np.integer)):
        print(f"{prefix}{display_name} → int")
    
    elif isinstance(obj, (float, np.floating)):
        print(f"{prefix}{display_name} → float")
    
    elif isinstance(obj, str):
        max_len = 30
        if len(obj) <= max_len:
            print(f"{prefix}{display_name} → str: '{obj}'")
        else:
            print(f"{prefix}{display_name} → str[{len(obj)} chars]: '{obj[:max_len]}...'")
    
    elif isinstance(obj, bool):
        print(f"{prefix}{display_name} → bool: {obj}")
    
    elif isinstance(obj, tuple):
        print(f"{prefix}{display_name} → tuple[{len(obj)}]: {obj}")
    
    elif obj is None:
        print(f"{prefix}{display_name} → None")
    
    else:
        print(f"{prefix}{display_name} → {type(obj).__name__}")


def list_collapse_presets():
    """List all available collapse presets with descriptions."""
    presets = {
        'minimal': 'Collapse root level only → PARENT',
        'compact': 'Collapse root, children, sides → PARENT, CHILD, PARENT_SIDE, CHILD_SIDE',
        'interfaces': 'Collapse interface sides → PARENT_SIDE, CHILD_SIDE',
        'hierarchy': 'Collapse all repetitive structures with labels',
        'sides_only': 'Collapse parent_side and child_side → PARENT_SIDE, CHILD_SIDE',
        'all_lists': 'Collapse all list structures → PARENT, CHILD',
    }
    
    print("\nAvailable collapse presets:")
    print("=" * 70)
    for preset, description in presets.items():
        print(f"  '{preset}'")
        print(f"      {description}")
    print("=" * 70)

#Pixels clustering utilities

def remove_small_clusters(labels, minidxs=5):
    """
    Removes clusters smaller than minidxs pixels by setting their label to 0.

    Parameters
    ----------
    labels : (N,) int
        Cluster labels for each data point.
    minidxs : int
        Minimum number of points required for a cluster to remain.

    Returns
    -------
    new_labels : (N,) int
        Labels with small clusters removed (set to 0).
    """
    labels = np.asarray(labels)
    new_labels = labels.copy()
    if labels.size == 0:
        return new_labels

    unique, counts = np.unique(labels[labels > 0], return_counts=True)
    for lab, c in zip(unique, counts):
        if c < minidxs:
            new_labels[labels == lab] = 0

    # Optional: relabel remaining clusters contiguously (1..N)
    nonzero = np.unique(new_labels[new_labels > 0])
    relabel = {lab: i + 1 for i, lab in enumerate(nonzero)}
    for old, new in relabel.items():
        new_labels[new_labels == old] = new

    return new_labels


def reduce_to_fundzone(M, symops):
    """
    Vectorized reduction of orientations to the fundamental zone.
    Finds symmetry-equivalent rotation S@M with smallest rotation
    angle to identity.

    Parameters
    ----------
    M : (N,3,3)
        Orientation matrices (sample→crystal).
    symops : (Ns,3,3)
        List of symmetry operation matrices.

    Returns
    -------
    M_reduced : (N,3,3)
        Reduced orientations.
    """
    M = np.asarray(M)
    symops = np.asarray(symops)
    N, Ns = M.shape[0], symops.shape[0]

    # Compute all equivalent orientations: (Ns, N, 3, 3)
    M_eq = np.einsum("sab,nbc->snac", symops, M, optimize=True)

    # Flatten to quaternions for angular distance
    M_eq_flat = M_eq.reshape(Ns * N, 3, 3)
    q_eq = R.from_matrix(M_eq_flat).as_quat().reshape(Ns, N, 4)

    # measure angle to identity (w close to 1)
    w = np.abs(np.clip(q_eq[..., 3], -1.0, 1.0))
    ang = 2 * np.arccos(w)  # (Ns, N)

    # pick symmetry giving smallest angle
    best_idx = np.argmin(ang, axis=0)  # (N,)

    # use fancy indexing to select M_eq[best_idx, n]
    M_reduced = np.empty_like(M)
    for n in range(N):
        M_reduced[n] = M_eq[best_idx[n], n]

    return M_reduced


def reduce_to_fundzone_slow(M, symops):
    """
    Reduce each orientation to the closest equivalent in the fundamental zone.

    Parameters
    ----------
    M : (N,3,3)
        Orientation matrices (sample→crystal).
    symops : (Ns,3,3)
        List of symmetry operation matrices.

    Returns
    -------
    M_reduced : (N,3,3)
        Reduced orientations (symmetry-equivalent rotations).
    """
    N = len(M)
    Ns = len(symops)
    M_reduced = np.empty_like(M)
    R_crys = R.from_matrix(M)
    q = R_crys.as_quat()  # (x, y, z, w)

    for i in range(N):
        q_i = q[i]
        r_i = R.from_quat(q_i)
        # find equivalent with smallest rotation angle to identity
        min_angle = np.inf
        best = M[i]
        for s in symops:
            q_eq = R.from_matrix(s @ M[i]).as_quat()
            # measure angular distance to identity (or arbitrary reference)
            dot = abs(np.clip(np.dot(q_eq, [0, 0, 0, 1]), -1.0, 1.0))
            ang = 2 * np.arccos(dot)
            if ang < min_angle:
                min_angle = ang
                best = s @ M[i]
        M_reduced[i] = best
    return M_reduced







def cluster_colors(labels, cmap_name="tab20",type="255",transparency=True):
    """
    Generate unique RGB colors for each cluster label.

    Parameters
    ----------
    labels : array-like of shape (N,)
        Cluster labels (starting at 1; 0 means unassigned).
    cmap_name : str
        Name of matplotlib colormap, e.g. "tab20", "hsv", "nipy_spectral", "viridis".

    Returns
    -------
    colors : (N, 3)
        RGB color array in [0,1] range for each point.
    """
    n_clusters = labels.max()
    cmap = plt.get_cmap(cmap_name, n_clusters)
    colors = np.zeros((len(labels), 3))
    for i in range(1, n_clusters + 1):
        color = cmap(i - 1)[:3]  # ignore alpha
        colors[labels == i] = color

    if type=='255':
        colors=colors*255
        colors=colors.astype(int)
    if transparency:
        if type=='255':
            colors=np.hstack((colors,colors[:,0:1]*0+255))
        else:
            colors=np.hstack((colors,colors[:,0:1]*0+1))

    return colors

@njit
def find_cluster_neighbors_with_lengths_and_boundaries_numba_roi(labels_2d, inside_mask, phase_2d):
    """
    Identify neighboring clusters, shared boundary lengths, and boundary pixel coordinates
    using fixed-size arrays (no Python lists or typed Lists), now including ROI border detection
    and phase information.

    Parameters
    ----------
    labels_2d : (ny, nx) int32
        2D array of cluster labels. Label 0 is ignored.

    inside_mask : (ny, nx) bool
        True for pixels inside the region of interest (ROI), False outside.
        ROI can be any convex or concave shape.
    
    phase_2d : (ny, nx) int32
        2D array of phase labels corresponding to each pixel.

    Returns
    -------
    clusters : (n_clusters,) int32
        Unique cluster labels (>0).

    flat_neighbors : (total_neighbors,) int32
        Concatenated neighbor labels for all clusters.

    flat_lengths : (total_neighbors,) int32
        Boundary pixel counts per cluster-pair.

    slices : (n_clusters, 2) int32
        Each row gives (start_idx, end_idx) into flat arrays for that cluster.

    boundary_y : (N_boundary,) int32
        Y-coordinates of all boundary pixels.

    boundary_x : (N_boundary,) int32
        X-coordinates of all boundary pixels.

    boundary_nb : (N_boundary,) int32
        Neighbor label for each boundary pixel. ROI border pixels are marked as -1.
    
    boundary_nb_phase : (N_boundary,) int32
        Phase of the neighbor for each boundary pixel. ROI border pixels are marked as -1.

    boundary_slices : (n_clusters, 2) int32
        Each row gives (start_idx, end_idx) into boundary arrays for that cluster.

    Notes
    -----
    - Access boundaries for cluster i as:
        start, end = boundary_slices[i]
        y = boundary_y[start:end]
        x = boundary_x[start:end]
        nb = boundary_nb[start:end]
        nb_phase = boundary_nb_phase[start:end]
    - To distinguish boundary types:
        * ROI border: boundary_nb[k] == -1 and boundary_nb_phase[k] == -1
        * Same-phase: boundary_nb_phase[k] == phase_2d[y[k], x[k]]
        * Inter-phase: boundary_nb_phase[k] != phase_2d[y[k], x[k]] and boundary_nb_phase[k] != -1
    """
    ny, nx = labels_2d.shape

    # --- find unique clusters ---
    clusters_unique = np.unique(labels_2d)
    n_clusters = 0
    for k in range(len(clusters_unique)):
        if clusters_unique[k] > 0:
            clusters_unique[n_clusters] = clusters_unique[k]
            n_clusters += 1
    clusters_unique = clusters_unique[:n_clusters]

    max_label = clusters_unique[-1] + 1
    label_to_idx = -np.ones(max_label, dtype=np.int32)
    for i in range(n_clusters):
        label_to_idx[clusters_unique[i]] = i

    neighbors_counts = np.zeros((n_clusters, n_clusters), dtype=np.int32)

    dy = np.array([-1, 0, 1, 0, -1, -1, 1, 1], dtype=np.int32)
    dx = np.array([0, 1, 0, -1, -1, 1, 1, -1], dtype=np.int32)

    # --- first pass: count boundary pixels ---
    total_boundary = 0
    boundary_counts = np.zeros(n_clusters, dtype=np.int32)
    for y in range(ny):
        for x in range(nx):
            lab = labels_2d[y, x]
            if lab == 0:
                continue
            ci = label_to_idx[lab]
            for n in range(8):
                yy = y + dy[n]
                xx = x + dx[n]
                # check ROI and neighbors
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx or not inside_mask[yy, xx]:
                    total_boundary += 1
                    boundary_counts[ci] += 1
                    continue
                nl = labels_2d[yy, xx]
                if nl == 0 or nl == lab:
                    continue
                total_boundary += 1
                boundary_counts[ci] += 1

    # --- allocate arrays ---
    boundary_y = np.empty(total_boundary, dtype=np.int32)
    boundary_x = np.empty(total_boundary, dtype=np.int32)
    boundary_nb = np.empty(total_boundary, dtype=np.int32)
    boundary_nb_phase = np.empty(total_boundary, dtype=np.int32)  # NEW: phase of neighbor
    boundary_slices = np.empty((n_clusters, 2), dtype=np.int32)

    # set slice positions
    offset = 0
    for ci in range(n_clusters):
        boundary_slices[ci, 0] = offset
        offset += boundary_counts[ci]
        boundary_slices[ci, 1] = offset

    # temporary pointers
    cur_pos = np.zeros(n_clusters, dtype=np.int32)
    for ci in range(n_clusters):
        cur_pos[ci] = boundary_slices[ci, 0]

    # --- second pass: fill boundary arrays ---
    for y in range(ny):
        for x in range(nx):
            lab = labels_2d[y, x]
            if lab == 0:
                continue
            ci = label_to_idx[lab]
            for n in range(8):
                yy = y + dy[n]
                xx = x + dx[n]
                k = cur_pos[ci]
                # ROI border
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx or not inside_mask[yy, xx]:
                    boundary_y[k] = y
                    boundary_x[k] = x
                    boundary_nb[k] = -1
                    boundary_nb_phase[k] = -1  # NEW: ROI border has phase -1
                    cur_pos[ci] += 1
                    continue
                nl = labels_2d[yy, xx]
                if nl == 0 or nl == lab:
                    continue
                cj = label_to_idx[nl]
                neighbors_counts[ci, cj] += 1
                boundary_y[k] = y
                boundary_x[k] = x
                boundary_nb[k] = nl
                boundary_nb_phase[k] = phase_2d[yy, xx]  # NEW: store neighbor's phase
                cur_pos[ci] += 1

    # --- neighbor summary ---
    total_neighbors = 0
    for ci in range(n_clusters):
        for cj in range(n_clusters):
            if neighbors_counts[ci, cj] > 0:
                total_neighbors += 1

    flat_neighbors = np.empty(total_neighbors, dtype=np.int32)
    flat_lengths = np.empty(total_neighbors, dtype=np.int32)
    slices = np.empty((n_clusters, 2), dtype=np.int32)

    idx = 0
    for ci in range(n_clusters):
        start = idx
        for cj in range(n_clusters):
            if neighbors_counts[ci, cj] > 0:
                flat_neighbors[idx] = clusters_unique[cj]
                flat_lengths[idx] = neighbors_counts[ci, cj]
                idx += 1
        end = idx
        slices[ci, 0] = start
        slices[ci, 1] = end

    return (clusters_unique, flat_neighbors, flat_lengths, slices,
            boundary_y, boundary_x, boundary_nb, boundary_nb_phase, boundary_slices)

@njit
def find_cluster_neighbors_with_lengths_and_boundaries_numba_roi_ini(labels_2d, inside_mask):
    """
    Identify neighboring clusters, shared boundary lengths, and boundary pixel coordinates
    using fixed-size arrays (no Python lists or typed Lists), now including ROI border detection.

    Parameters
    ----------
    labels_2d : (ny, nx) int32
        2D array of cluster labels. Label 0 is ignored.

    inside_mask : (ny, nx) bool
        True for pixels inside the region of interest (ROI), False outside.
        ROI can be any convex or concave shape.

    Returns
    -------
    clusters : (n_clusters,) int32
        Unique cluster labels (>0).

    flat_neighbors : (total_neighbors,) int32
        Concatenated neighbor labels for all clusters.

    flat_lengths : (total_neighbors,) int32
        Boundary pixel counts per cluster-pair.

    slices : (n_clusters, 2) int32
        Each row gives (start_idx, end_idx) into flat arrays for that cluster.

    boundary_y : (N_boundary,) int32
        Y-coordinates of all boundary pixels.

    boundary_x : (N_boundary,) int32
        X-coordinates of all boundary pixels.

    boundary_nb : (N_boundary,) int32
        Neighbor label for each boundary pixel. ROI border pixels are marked as -1.

    boundary_slices : (n_clusters, 2) int32
        Each row gives (start_idx, end_idx) into boundary arrays for that cluster.

    Notes
    -----
    - Access boundaries for cluster i as:
        start, end = boundary_slices[i]
        y = boundary_y[start:end]
        x = boundary_x[start:end]
        nb = boundary_nb[start:end]
    """
    ny, nx = labels_2d.shape

    # --- find unique clusters ---
    clusters_unique = np.unique(labels_2d)
    n_clusters = 0
    for k in range(len(clusters_unique)):
        if clusters_unique[k] > 0:
            clusters_unique[n_clusters] = clusters_unique[k]
            n_clusters += 1
    clusters_unique = clusters_unique[:n_clusters]

    max_label = clusters_unique[-1] + 1
    label_to_idx = -np.ones(max_label, dtype=np.int32)
    for i in range(n_clusters):
        label_to_idx[clusters_unique[i]] = i

    neighbors_counts = np.zeros((n_clusters, n_clusters), dtype=np.int32)

    dy = np.array([-1, 0, 1, 0, -1, -1, 1, 1], dtype=np.int32)
    dx = np.array([0, 1, 0, -1, -1, 1, 1, -1], dtype=np.int32)

    # --- first pass: count boundary pixels ---
    total_boundary = 0
    boundary_counts = np.zeros(n_clusters, dtype=np.int32)
    for y in range(ny):
        for x in range(nx):
            lab = labels_2d[y, x]
            if lab == 0:
                continue
            ci = label_to_idx[lab]
            for n in range(8):
                yy = y + dy[n]
                xx = x + dx[n]
                # check ROI and neighbors
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx or not inside_mask[yy, xx]:
                    total_boundary += 1
                    boundary_counts[ci] += 1
                    continue
                nl = labels_2d[yy, xx]
                if nl == 0 or nl == lab:
                    continue
                total_boundary += 1
                boundary_counts[ci] += 1

    # --- allocate arrays ---
    boundary_y = np.empty(total_boundary, dtype=np.int32)
    boundary_x = np.empty(total_boundary, dtype=np.int32)
    boundary_nb = np.empty(total_boundary, dtype=np.int32)
    boundary_slices = np.empty((n_clusters, 2), dtype=np.int32)

    # set slice positions
    offset = 0
    for ci in range(n_clusters):
        boundary_slices[ci, 0] = offset
        offset += boundary_counts[ci]
        boundary_slices[ci, 1] = offset

    # temporary pointers
    cur_pos = np.zeros(n_clusters, dtype=np.int32)
    for ci in range(n_clusters):
        cur_pos[ci] = boundary_slices[ci, 0]

    # --- second pass: fill boundary arrays ---
    for y in range(ny):
        for x in range(nx):
            lab = labels_2d[y, x]
            if lab == 0:
                continue
            ci = label_to_idx[lab]
            for n in range(8):
                yy = y + dy[n]
                xx = x + dx[n]
                k = cur_pos[ci]
                # ROI border
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx or not inside_mask[yy, xx]:
                    boundary_y[k] = y
                    boundary_x[k] = x
                    boundary_nb[k] = -1
                    cur_pos[ci] += 1
                    continue
                nl = labels_2d[yy, xx]
                if nl == 0 or nl == lab:
                    continue
                cj = label_to_idx[nl]
                neighbors_counts[ci, cj] += 1
                boundary_y[k] = y
                boundary_x[k] = x
                boundary_nb[k] = nl
                cur_pos[ci] += 1

    # --- neighbor summary ---
    total_neighbors = 0
    for ci in range(n_clusters):
        for cj in range(n_clusters):
            if neighbors_counts[ci, cj] > 0:
                total_neighbors += 1

    flat_neighbors = np.empty(total_neighbors, dtype=np.int32)
    flat_lengths = np.empty(total_neighbors, dtype=np.int32)
    slices = np.empty((n_clusters, 2), dtype=np.int32)

    idx = 0
    for ci in range(n_clusters):
        start = idx
        for cj in range(n_clusters):
            if neighbors_counts[ci, cj] > 0:
                flat_neighbors[idx] = clusters_unique[cj]
                flat_lengths[idx] = neighbors_counts[ci, cj]
                idx += 1
        end = idx
        slices[ci, 0] = start
        slices[ci, 1] = end

    return (clusters_unique, flat_neighbors, flat_lengths, slices,
            boundary_y, boundary_x, boundary_nb, boundary_slices)




def cluster_map_sample_to_crystal_numba_with_neighbors(X, Y, Q, sym_quats,
                                                                  neighbors, Sel,
                                                                  ang_thr=5.0,
                                                                  dmax=1.5,
                                                                  minidxs=1):
    """
    EBSD clustering using explicit neighbor index matrix, limited to ROI.

    Parameters
    ----------
    X, Y : float32 arrays (N,)
        Pixel coordinates.
    Q : float64 array (N, 4)
        Orientation quaternions.
    sym_quats : float64 array (M, 4)
        Crystal symmetry quaternions.
    neighbors : int32 array (N, n_neigh)
        Neighbor indices per pixel (-1 for no neighbor).
    Sel : bool array (N,)
        True if pixel is inside ROI; False to exclude.
    ang_thr : float
        Misorientation threshold in degrees.
    dmax : float
        Maximum spatial neighbor distance.
    minidxs : int
        Minimum pixel count to keep cluster.

    Returns
    -------
    labels_new : int32 array (N,)
        Cluster labels (0 outside ROI or filtered clusters).
    com : float64 array (n_clusters, 2)
        Cluster centers of mass.
    flat_neighbors, flat_lengths, slices : flattened adjacency representation.
    """
    N = X.shape[0]
    labels = np.zeros(N, np.int32)
    cluster_id = 0
    dmax2 = dmax * dmax
    stacksize = 5*N
    stack = np.empty(stacksize, np.int32)

    com_x = np.zeros(N)
    com_y = np.zeros(N)
    counts = np.zeros(N)

    # --- 1. Flood-fill clustering within ROI ---
    for i in range(N):
        #if i % 10000 == 0:
        #    print("Clustering pixel ", i, " / ", N, end=',')
        if not Sel[i]:
            continue
        if labels[i] != 0:
            continue
        cluster_id += 1
        sp = 0
        stack[sp] = i
        sp += 1
        while sp > 0:
            sp -= 1
            j = stack[sp]
            if labels[j] != 0:
                continue
            if not Sel[j]:
                continue

            labels[j] = cluster_id
            cidx = cluster_id - 1
            com_x[cidx] += X[j]
            com_y[cidx] += Y[j]
            counts[cidx] += 1
            qj = Q[j]

            for n in range(neighbors.shape[1]):
                k = neighbors[j, n]
                if k < 0:
                    continue
                if not Sel[k]:
                    continue
                if labels[k] != 0:
                    continue
                dx = X[k] - X[j]
                dy = Y[k] - Y[j]
                if dx*dx + dy*dy > dmax2:
                    continue
                ang = misori_sym_deg_quats(qj, Q[k], sym_quats)
                if ang < ang_thr:
                    if sp < stacksize:  # Fixed: Check stack bounds before adding
                        #print(f'{i}/{sp}',end=',')
                        stack[sp] = k
                        sp += 1
    # --- 2. Filter small clusters & compute COM ---
    new_idx = -np.ones(cluster_id, dtype=np.int32)
    n_clusters_new = 0
    for c in range(cluster_id):
        if counts[c] >= minidxs:
            new_idx[c] = n_clusters_new
            n_clusters_new += 1

    labels_new = np.zeros(N, np.int32)
    com = np.zeros((n_clusters_new, 2))
    counts_new = np.zeros(n_clusters_new)

    for i in range(N):
        if not Sel[i]:
            continue
        c_old = labels[i] - 1
        if c_old < 0:
            continue
        c_new = new_idx[c_old]
        if c_new >= 0:
            labels_new[i] = c_new + 1
            com[c_new, 0] += X[i]
            com[c_new, 1] += Y[i]
            counts_new[c_new] += 1

    for c in range(n_clusters_new):
        if counts_new[c] > 0:
            com[c, 0] /= counts_new[c]
            com[c, 1] /= counts_new[c]

    # --- 3. Compute cluster adjacency ---
    label_to_idx = -np.ones(n_clusters_new + 1, dtype=np.int32)
    for idx in range(n_clusters_new):
        label_to_idx[idx + 1] = idx

    neighbors_counts = np.zeros((n_clusters_new, n_clusters_new), dtype=np.int32)

    for i in range(N):
        if not Sel[i]:
            continue
        li = labels_new[i]
        if li == 0:
            continue
        ci = label_to_idx[li]
        for n in range(neighbors.shape[1]):
            k = neighbors[i, n]
            if k < 0:
                continue
            if not Sel[k]:
                continue
            lj = labels_new[k]
            if lj == 0 or lj == li:
                continue
            cj = label_to_idx[lj]
            neighbors_counts[ci, cj] += 1

    # --- 4. Flatten adjacency arrays ---
    total_neighbors = 0
    for ci in range(n_clusters_new):
        for cj in range(n_clusters_new):
            if neighbors_counts[ci, cj] > 0:
                total_neighbors += 1

    flat_neighbors = np.empty(total_neighbors, dtype=np.int32)
    flat_lengths = np.empty(total_neighbors, dtype=np.int32)
    slices = np.empty((n_clusters_new, 2), dtype=np.int32)

    idx = 0
    for ci in range(n_clusters_new):
        start = idx
        for cj in range(n_clusters_new):
            if neighbors_counts[ci, cj] > 0:
                flat_neighbors[idx] = cj + 1
                flat_lengths[idx] = neighbors_counts[ci, cj]
                idx += 1
        end = idx
        slices[ci, 0] = start
        slices[ci, 1] = end

    return labels_new, com, flat_neighbors, flat_lengths, slices

def extract_boundaries_grouped_by_neighbor(xs, ys,
                                           clusters_unique,
                                           boundary_slices,
                                           boundary_x,
                                           boundary_y,
                                           boundary_nb,
                                           boundary_nb_phase=None,
                                           phase_2d=None,
                                           mark_outer=-1):
    """
    Group boundary pixels of each cluster by their neighboring cluster,
    including region borders as a special neighbor (default label = -1),
    and optionally track phase information.

    Parameters
    ----------
    xs, ys : array_like
        Unique sorted coordinates used in the label grid reconstruction.
    clusters_unique : (n_clusters,) int32
        Cluster labels.
    boundary_slices : (n_clusters, 2) int32
        Start and end indices for each cluster.
    boundary_x, boundary_y : (N_total_boundaries,) int32
        Boundary pixel grid indices.
    boundary_nb : (N_total_boundaries,) int32
        Neighbor cluster label for each boundary pixel.
    boundary_nb_phase : (N_total_boundaries,) int32, optional
        Phase of neighbor for each boundary pixel. If None, phase info is ignored.
    phase_2d : (ny, nx) int32, optional
        2D array of phase labels. Required if boundary_nb_phase is provided.
    mark_outer : int or str, optional
        Label used for ROI border (default = -1).

    Returns
    -------
    grouped_boundaries : list of dicts
        grouped_boundaries[i] is a dictionary for cluster i:
            {
                neighbor_label_1: np.array([[x1, y1], ...]),
                neighbor_label_2: np.array([...]),
                mark_outer: np.array([...])   # if cluster touches ROI border
            }
    
    grouped_boundary_phases : list of dicts (only if boundary_nb_phase provided)
        grouped_boundary_phases[i] is a dictionary for cluster i:
            {
                neighbor_label_1: phase_value,
                neighbor_label_2: phase_value,
                mark_outer: -1   # ROI border
            }
    
    cluster_phases : (n_clusters,) int32 (only if phase_2d provided)
        Phase of each cluster.
    """
    n_clusters = clusters_unique.shape[0]
    grouped_boundaries = []
    grouped_boundary_phases = [] if boundary_nb_phase is not None else None
    cluster_phases = np.zeros(n_clusters, dtype=np.int32) if phase_2d is not None else None
    
    # Get phase for each cluster if phase_2d is provided
    if phase_2d is not None:
        ny, nx = phase_2d.shape
        # Build label_to_idx mapping
        max_label = clusters_unique[-1] + 1 if len(clusters_unique) > 0 else 1
        label_to_idx = -np.ones(max_label, dtype=np.int32)
        for i in range(n_clusters):
            label_to_idx[clusters_unique[i]] = i
        
        # Find phase for each cluster by scanning phase_2d
        labels_2d = np.zeros((ny, nx), dtype=np.int32)
        for y in range(ny):
            for x in range(nx):
                lab = phase_2d[y, x]  # This should actually be labels_2d if available
                # We need to find the phase from the first occurrence of each cluster
        
        # Alternative: find phase from boundary pixels
        for ci in range(n_clusters):
            start, end = boundary_slices[ci]
            if end > start:
                # Get phase from first boundary pixel's cluster location
                bx_first = boundary_x[start]
                by_first = boundary_y[start]
                cluster_phases[ci] = phase_2d[by_first, bx_first]

    for ci in range(n_clusters):
        start, end = boundary_slices[ci]
        bx = boundary_x[start:end]
        by = boundary_y[start:end]
        nb = boundary_nb[start:end]
        
        Xb = xs[bx]
        Yb = ys[by]

        cluster_boundary = {}
        cluster_boundary_phase = {} if boundary_nb_phase is not None else None
        
        if boundary_nb_phase is not None:
            nb_phase = boundary_nb_phase[start:end]
            
            for j in range(len(nb)):
                nbl = nb[j]
                nbl_phase = nb_phase[j]
                
                if nbl == 0 or nbl == -1:
                    nbl = mark_outer  # mark ROI border
                    nbl_phase = -1
                
                if nbl not in cluster_boundary:
                    cluster_boundary[nbl] = []
                    cluster_boundary_phase[nbl] = nbl_phase
                
                cluster_boundary[nbl].append((Xb[j], Yb[j]))
        else:
            # Original behavior without phase
            for j in range(len(nb)):
                nbl = nb[j]
                if nbl == 0:
                    nbl = mark_outer  # mark ROI border
                if nbl not in cluster_boundary:
                    cluster_boundary[nbl] = []
                cluster_boundary[nbl].append((Xb[j], Yb[j]))

        for nbl in cluster_boundary:
            cluster_boundary[nbl] = np.array(cluster_boundary[nbl], dtype=np.float64)

        grouped_boundaries.append(cluster_boundary)
        if cluster_boundary_phase is not None:
            grouped_boundary_phases.append(cluster_boundary_phase)

    # Return appropriate outputs based on what was provided
    if boundary_nb_phase is not None and phase_2d is not None:
        return grouped_boundaries, grouped_boundary_phases, cluster_phases
    elif boundary_nb_phase is not None:
        return grouped_boundaries, grouped_boundary_phases
    else:
        return grouped_boundaries
    
def extract_boundaries_grouped_by_neighbor_ini(xs, ys,
                                           clusters_unique,
                                           boundary_slices,
                                           boundary_x,
                                           boundary_y,
                                           boundary_nb,
                                           mark_outer=-1):
    """
    Group boundary pixels of each cluster by their neighboring cluster,
    including region borders as a special neighbor (default label = -1).

    Parameters
    ----------
    xs, ys : array_like
        Unique sorted coordinates used in the label grid reconstruction.
    clusters_unique : (n_clusters,) int32
        Cluster labels.
    boundary_slices : (n_clusters, 2) int32
        Start and end indices for each cluster.
    boundary_x, boundary_y : (N_total_boundaries,) int32
        Boundary pixel grid indices.
    boundary_nb : (N_total_boundaries,) int32
        Neighbor cluster label for each boundary pixel.
    mark_outer : int or str, optional
        Label used for ROI border (default = -1).

    Returns
    -------
    grouped_boundaries : list of dicts
        grouped_boundaries[i] is a dictionary for cluster i:
            {
                neighbor_label_1: np.array([[x1, y1], ...]),
                neighbor_label_2: np.array([...]),
                mark_outer: np.array([...])   # if cluster touches ROI border
            }
    """
    n_clusters = clusters_unique.shape[0]
    grouped_boundaries = []

    for ci in range(n_clusters):
        start, end = boundary_slices[ci]
        bx = boundary_x[start:end]
        by = boundary_y[start:end]
        nb = boundary_nb[start:end]

        Xb = xs[bx]
        Yb = ys[by]

        cluster_boundary = {}
        for j in range(len(nb)):
            nbl = nb[j]
            if nbl == 0:
                nbl = mark_outer  # mark ROI border
            if nbl not in cluster_boundary:
                cluster_boundary[nbl] = []
            cluster_boundary[nbl].append((Xb[j], Yb[j]))

        for nbl in cluster_boundary:
            cluster_boundary[nbl] = np.array(cluster_boundary[nbl], dtype=np.float64)

        grouped_boundaries.append(cluster_boundary)

    return grouped_boundaries


def prepare_boundaries_for_numba(grouped_boundaries, grouped_boundary_phases=None):
    """
    Converts grouped_boundaries dict into flat arrays for Numba processing,
    optionally including phase information.

    Parameters
    ----------
    grouped_boundaries : list of dicts
        grouped_boundaries[i] corresponds to cluster i and has keys:
            neighbor_label: np.array([[x, y], ...])
            mark_outer: np.array([[x, y], ...]) optional for ROI border
    
    grouped_boundary_phases : list of dicts, optional
        grouped_boundary_phases[i] corresponds to cluster i and has keys:
            neighbor_label: phase_value (int)
            If None, phase information is not included in output.

    Returns
    -------
    boundary_coords : (N_boundary, 2) float64
        X, Y coordinates of all boundaries for all clusters.

    boundary_cluster : (N_boundary,) int32
        Cluster index each boundary pixel belongs to.

    cluster_offsets : (n_clusters, 2) int32
        Start/end indices in boundary_coords for each cluster.
    
    boundary_neighbor : (N_boundary,) int32 (only if grouped_boundary_phases provided)
        Neighbor label for each boundary pixel.
    
    boundary_neighbor_phase : (N_boundary,) int32 (only if grouped_boundary_phases provided)
        Phase of neighbor for each boundary pixel.
    """
    n_clusters = len(grouped_boundaries)
    counts = np.zeros(n_clusters, dtype=np.int32)

    # count boundary points for each cluster
    for i, gb in enumerate(grouped_boundaries):
        n_points = 0
        for key, arr in gb.items():
            if arr.shape[0] > 0:
                n_points += arr.shape[0]
        counts[i] = n_points

    total_points = np.sum(counts)
    boundary_coords = np.empty((total_points, 2), dtype=np.float64)
    boundary_cluster = np.empty(total_points, dtype=np.int32)
    cluster_offsets = np.empty((n_clusters, 2), dtype=np.int32)
    
    # Allocate phase arrays if phase info is provided
    if grouped_boundary_phases is not None:
        boundary_neighbor = np.empty(total_points, dtype=np.int32)
        boundary_neighbor_phase = np.empty(total_points, dtype=np.int32)

    idx = 0
    for i, gb in enumerate(grouped_boundaries):
        cluster_offsets[i, 0] = idx
        
        if grouped_boundary_phases is not None:
            gb_phase = grouped_boundary_phases[i]
            
            for neighbor_label, arr in gb.items():
                if arr.shape[0] > 0:
                    n_points = arr.shape[0]
                    boundary_coords[idx:idx+n_points, :] = arr
                    boundary_cluster[idx:idx+n_points] = i
                    boundary_neighbor[idx:idx+n_points] = neighbor_label
                    boundary_neighbor_phase[idx:idx+n_points] = gb_phase[neighbor_label]
                    idx += n_points
        else:
            # Original behavior without phase
            for key, arr in gb.items():
                if arr.shape[0] > 0:
                    boundary_coords[idx:idx+arr.shape[0], :] = arr
                    boundary_cluster[idx:idx+arr.shape[0]] = i
                    idx += arr.shape[0]
        
        cluster_offsets[i, 1] = idx

    if grouped_boundary_phases is not None:
        return boundary_coords, boundary_cluster, cluster_offsets, boundary_neighbor, boundary_neighbor_phase
    else:
        return boundary_coords, boundary_cluster, cluster_offsets

def prepare_boundaries_for_numba_ini(grouped_boundaries):
    """
    Converts grouped_boundaries dict into flat arrays for Numba processing.

    Parameters
    ----------
    grouped_boundaries : list of dicts
        grouped_boundaries[i] corresponds to cluster i and has keys:
            neighbor_label: np.array([[x, y], ...])
            mark_outer: np.array([[x, y], ...]) optional for ROI border

    Returns
    -------
    boundary_coords : (N_boundary, 2) float64
        X, Y coordinates of all boundaries for all clusters.

    boundary_cluster : (N_boundary,) int32
        Cluster index each boundary pixel belongs to.

    cluster_offsets : (n_clusters, 2) int32
        Start/end indices in boundary_coords for each cluster.
    """
    n_clusters = len(grouped_boundaries)
    counts = np.zeros(n_clusters, dtype=np.int32)

    # count boundary points for each cluster
    for i, gb in enumerate(grouped_boundaries):
        n_points = 0
        for key, arr in gb.items():
            if arr.shape[0] > 0:
                n_points += arr.shape[0]
        counts[i] = n_points

    total_points = np.sum(counts)
    boundary_coords = np.empty((total_points, 2), dtype=np.float64)
    boundary_cluster = np.empty(total_points, dtype=np.int32)
    cluster_offsets = np.empty((n_clusters, 2), dtype=np.int32)

    idx = 0
    for i, gb in enumerate(grouped_boundaries):
        cluster_offsets[i, 0] = idx
        for key, arr in gb.items():
            if arr.shape[0] > 0:
                boundary_coords[idx:idx+arr.shape[0], :] = arr
                boundary_cluster[idx:idx+arr.shape[0]] = i
                idx += arr.shape[0]
        cluster_offsets[i, 1] = idx

    return boundary_coords, boundary_cluster, cluster_offsets

@njit
def point_in_polygon(x, y, poly_x, poly_y):
    n = poly_x.size
    inside = False
    j = n - 1
    for i in range(n):
        if ((poly_y[i] > y) != (poly_y[j] > y)) and \
           (x < (poly_x[j] - poly_x[i]) * (y - poly_y[i]) / (poly_y[j] - poly_y[i] + 1e-12) + poly_x[i]):
            inside = not inside
        j = i
    return inside

#@njit
# With phase information
def representative_points_from_grouped_boundaries(grouped_boundaries, 
                                                  grouped_boundary_phases=None,
                                                  cluster_phases=None,
                                                  n_samples=1000):
    """
    Compute a representative point inside each cluster polygon from grouped boundaries,
    optionally including phase information.

    Parameters
    ----------
    grouped_boundaries : list of dicts
        grouped_boundaries[i] is a dict for cluster i:
            {
                neighbor_label_1: np.array([[x1, y1], ...]),
                neighbor_label_2: np.array([[x2, y2], ...]),
                'mark_outer': np.array([[x, y], ...])  # optional
            }
    grouped_boundary_phases : list of dicts, optional
        grouped_boundary_phases[i] is a dict for cluster i:
            {
                neighbor_label_1: phase_value,
                neighbor_label_2: phase_value,
                ...
            }
    cluster_phases : array-like, optional
        Phase of each cluster (n_clusters,).
    n_samples : int
        Number of random points to try if centroid lies outside the polygon.

    Returns
    -------
    rep_points : list of tuples
        rep_points[i] = (x, y) representative point inside cluster i.
    
    rep_point_phases : (n_clusters,) int32 (only if cluster_phases provided)
        Phase of each cluster.
    
    rep_point_info : list of dicts (only if grouped_boundary_phases provided)
        rep_point_info[i] contains:
            {
                'position': (x, y),
                'cluster_phase': phase_value,
                'neighbor_phases': list of unique neighboring phases,
                'has_roi_border': bool,
                'has_same_phase_neighbors': bool,
                'has_inter_phase_neighbors': bool
            }
    """
    rep_points = []
    rep_point_info = [] if grouped_boundary_phases is not None else None

    rng = np.random.default_rng()

    for i, cluster_dict in enumerate(grouped_boundaries):
        # Merge all boundary coordinates
        coords_list = []
        for key, arr in cluster_dict.items():
            if arr.size > 0:
                coords_list.append(arr)

        if len(coords_list) == 0:
            rep_points.append((np.nan, np.nan))
            if rep_point_info is not None:
                rep_point_info.append({
                    'position': (np.nan, np.nan),
                    'cluster_phase': cluster_phases[i] if cluster_phases is not None else None,
                    'neighbor_phases': [],
                    'has_roi_border': False,
                    'has_same_phase_neighbors': False,
                    'has_inter_phase_neighbors': False
                })
            continue

        all_coords = np.vstack(coords_list)
        bx, by = all_coords[:, 0], all_coords[:, 1]

        # Create polygon path
        poly = Path(all_coords)
        # Try centroid first
        cx, cy = np.mean(bx), np.mean(by)
        if poly.contains_point((cx, cy)):
            rep_point = (cx, cy)
        else:
            # Bounding box
            xmin, xmax = np.min(bx), np.max(bx)
            ymin, ymax = np.min(by), np.max(by)

            # Sample random points inside bounding box
            found = False
            for _ in range(n_samples):
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                if poly.contains_point((x, y)):
                    rep_point = (x, y)
                    found = True
                    break

            if not found:
                # fallback: first boundary pixel
                rep_point = (bx[0], by[0])
        
        rep_points.append(rep_point)
        
        # Collect phase information if available
        if rep_point_info is not None:
            cluster_phase = cluster_phases[i] if cluster_phases is not None else None
            
            # Get neighbor phases
            neighbor_phases_dict = grouped_boundary_phases[i]
            neighbor_phases = list(neighbor_phases_dict.values())
            unique_neighbor_phases = list(set(neighbor_phases))
            
            # Check boundary types
            has_roi_border = -1 in neighbor_phases
            has_same_phase = cluster_phase in neighbor_phases if cluster_phase is not None else False
            has_inter_phase = any(p != cluster_phase and p != -1 for p in neighbor_phases) if cluster_phase is not None else False
            
            rep_point_info.append({
                'position': rep_point,
                'cluster_phase': cluster_phase,
                'neighbor_phases': [p for p in unique_neighbor_phases if p != -1],  # Exclude ROI border marker
                'has_roi_border': has_roi_border,
                'has_same_phase_neighbors': has_same_phase,
                'has_inter_phase_neighbors': has_inter_phase
            })

    if rep_point_info is not None:
        return rep_points, rep_point_info
    else:
        return rep_points



@njit
def representative_points_from_grouped_boundaries_numba_ini(
    boundary_coords, boundary_nb, cluster_offsets, roi_border_label=-1
):
    """
    Compute a representative point inside each cluster polygon for potentially non-convex clusters,
    ignoring boundary points marked as ROI-border.

    Parameters
    ----------
    boundary_coords : (N_boundary, 2) float64
        X,Y coordinates of all boundary pixels for all clusters.
    boundary_nb : (N_boundary,) int32
        Neighbor label of each boundary pixel. ROI-border points have label roi_border_label.
    cluster_offsets : (n_clusters, 2) int32
        Start/end indices of boundary pixels for each cluster in boundary arrays.
    roi_border_label : int
        Label indicating ROI-border pixels (default -1).

    Returns
    -------
    rep_points : (n_clusters, 2) float64
        Representative point (x,y) inside each cluster polygon.
    """
    n_clusters = cluster_offsets.shape[0]
    rep_points = np.empty((n_clusters, 2), dtype=np.float64)

    for i in range(n_clusters):
        start, end = cluster_offsets[i]
        n_pts = end - start
        if n_pts == 0:
            rep_points[i, 0] = np.nan
            rep_points[i, 1] = np.nan
            continue

        # Compute centroid ignoring ROI-border pixels
        cx = 0.0
        cy = 0.0
        count = 0
        for j in range(start, end):
            if boundary_nb[j] == roi_border_label:
                continue
            cx += boundary_coords[j, 0]
            cy += boundary_coords[j, 1]
            count += 1

        if count > 0:
            cx /= count
            cy /= count
        else:
            # fallback: use all boundary pixels if all touch ROI border
            for j in range(start, end):
                cx += boundary_coords[j, 0]
                cy += boundary_coords[j, 1]
            cx /= n_pts
            cy /= n_pts

        rep_points[i, 0] = cx
        rep_points[i, 1] = cy

    return rep_points



def representative_points_from_grouped_boundaries_ini(grouped_boundaries, n_samples=1000):
    """
    Compute a representative point inside each cluster polygon from grouped boundaries.

    Parameters
    ----------
    grouped_boundaries : list of dicts
        grouped_boundaries[i] is a dict for cluster i:
            {
                neighbor_label_1: np.array([[x1, y1], ...]),
                neighbor_label_2: np.array([[x2, y2], ...]),
                'mark_outer': np.array([[x, y], ...])  # optional
            }
    n_samples : int
        Number of random points to try if centroid lies outside the polygon.

    Returns
    -------
    rep_points : list of tuples
        rep_points[i] = (x, y) representative point inside cluster i.
    """
    rep_points = []

    rng = np.random.default_rng()

    for cluster_dict in grouped_boundaries:
        # Merge all boundary coordinates
        coords_list = []
        for key, arr in cluster_dict.items():
            if arr.size > 0:
                coords_list.append(arr)
        #if 'mark_outer' in cluster_dict and cluster_dict['mark_outer'].size > 0:
        #    coords_list.append(cluster_dict['mark_outer'])

        if len(coords_list) == 0:
            rep_points.append((np.nan, np.nan))
            continue

        all_coords = np.vstack(coords_list)
        bx, by = all_coords[:, 0], all_coords[:, 1]

        # Create polygon path
        poly = Path(all_coords)
        # Try centroid first
        cx, cy = np.mean(bx), np.mean(by)
        if poly.contains_point((cx, cy)):
            rep_points.append((cx, cy))
            continue

        # Bounding box
        xmin, xmax = np.min(bx), np.max(bx)
        ymin, ymax = np.min(by), np.max(by)

        # Sample random points inside bounding box
        found = False
        for _ in range(n_samples):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if poly.contains_point((x, y)):
                rep_points.append((x, y))
                found = True
                break

        if not found:
            # fallback: first boundary pixel
            rep_points.append((bx[0], by[0]))

    return rep_points



def representative_point(boundary_x, boundary_y, n_samples=1000):
    """
    Find a point guaranteed to lie inside a (possibly non-convex) polygon.

    Parameters
    ----------
    boundary_x, boundary_y : 1D arrays
        Coordinates of the polygon boundary.
    n_samples : int
        Number of random points to try inside bounding box.

    Returns
    -------
    x_rep, y_rep : float
        A point inside the polygon.
    """
    # Create polygon path
    poly = Path(np.column_stack((boundary_x, boundary_y)))

    # Bounding box
    xmin, xmax = np.min(boundary_x), np.max(boundary_x)
    ymin, ymax = np.min(boundary_y), np.max(boundary_y)

    # Try centroid first
    cx, cy = np.mean(boundary_x), np.mean(boundary_y)
    if poly.contains_point((cx, cy)):
        return cx, cy

    # Otherwise, sample random points inside bounding box
    rng = np.random.default_rng()
    for _ in range(n_samples):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if poly.contains_point((x, y)):
            return x, y

    # Fallback: return first boundary point if all else fails
    return boundary_x[0], boundary_y[0]

@njit
def find_cluster_neighbors_with_lengths_numba(labels_2d):
    """
    Fast Numba-compatible version using preallocated arrays and slices.

    Parameters
    ----------
    labels_2d : (ny, nx) int32
        Cluster labels on 2D grid. 0 is ignored.

    Returns
    -------
    clusters : (n_clusters,) int32
        Unique cluster labels
    neighbors_list : list of arrays
        neighbors_list[i] = array of cluster labels neighboring cluster[i]
    lengths_list : list of arrays
        lengths_list[i] = number of shared boundary pixels with each neighbor
    slices : list of tuples
        slices[i] = (start_idx, end_idx) of neighbors in the flat arrays
    flat_neighbors : (total_neighbors,) int32
        All neighbor labels concatenated
    flat_lengths : (total_neighbors,) int32
        All boundary lengths concatenated
    """
    ny, nx = labels_2d.shape

    # --- find unique clusters ---
    clusters_unique = np.unique(labels_2d)
    n_clusters = 0
    for k in range(len(clusters_unique)):
        if clusters_unique[k] > 0:
            clusters_unique[n_clusters] = clusters_unique[k]
            n_clusters += 1
    clusters_unique = clusters_unique[:n_clusters]

    # mapping label -> index
    max_label = clusters_unique[-1] + 1
    label_to_idx = -np.ones(max_label, dtype=np.int32)
    for idx in range(n_clusters):
        label_to_idx[clusters_unique[idx]] = idx

    # --- adjacency array to count boundary pixels ---
    neighbors_counts = np.zeros((n_clusters, n_clusters), dtype=np.int32)

    # 8-connectivity
    dy = np.array([-1, 0, 1, 0, -1, -1, 1, 1], dtype=np.int32)
    dx = np.array([0, 1, 0, -1, -1, 1, -1, 1], dtype=np.int32)

    for y in range(ny):
        for x in range(nx):
            label = labels_2d[y, x]
            if label == 0:
                continue
            ci = label_to_idx[label]
            for n in range(8):
                yy = y + dy[n]
                xx = x + dx[n]
                if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                    continue
                neighbor_label = labels_2d[yy, xx]
                if neighbor_label == 0 or neighbor_label == label:
                    continue
                cj = label_to_idx[neighbor_label]
                neighbors_counts[ci, cj] += 1

    # --- count total neighbors ---
    total_neighbors = 0
    neighbors_per_cluster = np.zeros(n_clusters, dtype=np.int32)
    for ci in range(n_clusters):
        count = 0
        for cj in range(n_clusters):
            if neighbors_counts[ci, cj] > 0:
                count += 1
        neighbors_per_cluster[ci] = count
        total_neighbors += count

    # --- preallocate flat arrays ---
    flat_neighbors = np.empty(total_neighbors, dtype=np.int32)
    flat_lengths = np.empty(total_neighbors, dtype=np.int32)
    slices = []

    idx = 0
    for ci in range(n_clusters): 
        start_idx = idx
        for cj in range(n_clusters):
            if neighbors_counts[ci, cj] > 0:
                flat_neighbors[idx] = clusters_unique[cj]
                flat_lengths[idx] = neighbors_counts[ci, cj]
                idx += 1
        end_idx = idx
        slices.append((start_idx, end_idx))

    return clusters_unique, flat_neighbors, flat_lengths, slices

def build_neighbors_dict(clusters, flat_neighbors, slices):
    """
    Build a dictionary mapping cluster -> set of neighboring clusters.

    Parameters
    ----------
    clusters : array of int
        Unique cluster labels
    flat_neighbors : array of int
        Concatenated neighbor labels from all clusters
    slices : list of tuples
        slices[i] = (start_idx, end_idx) in flat_neighbors for cluster[i]

    Returns
    -------
    neighbors : dict[int, set[int]]
        neighbors[cluster_label] = set of neighboring cluster labels
    """
    neighbors = dict()
    for i, cluster in enumerate(clusters):
        start, end = slices[i]
        #neighbors[cluster] = set(flat_neighbors[start:end])
        neighbors[int(cluster)] = set(int(x) for x in flat_neighbors[start:end])
    return neighbors
    
def compute_cluster_boundary_misorientations(avg_M, neighbors, symops):
    """
    Compute the minimum misorientation angle (in degrees) between neighboring clusters,
    considering crystal symmetry operations.

    Parameters
    ----------
    avg_M : dict[int, np.ndarray(3,3)]
        Average orientation matrix (sample→crystal) for each cluster label.
    neighbors : dict[int, set[int]]
        Neighbor map as produced by `find_cluster_neighbors_with_lengths`.
    symops : np.ndarray(Ns,3,3)
        List of symmetry operation matrices.

    Returns
    -------
    miso_angles : dict[(int,int), float]
        Mean misorientation angles between neighboring clusters (deg).
    """
    symops = np.asarray(symops)
    miso_angles = {}

    for a, nbs in neighbors.items():
        Ma = avg_M[a]
        for b in nbs:
            if (a, b) in miso_angles or (b, a) in miso_angles:
                continue  # avoid duplicates

            Mb = avg_M[b]

            # Relative rotation sample→crystal to crystal→sample
            R_ab = Mb @ Ma.T  # rotation from a to b in crystal frame

            # Apply all symmetry equivalents: S * R_ab
            # shape (Ns,3,3)
            R_eq = np.einsum('sij,jk->sik', symops, R_ab, optimize=True)

            # Compute rotation angles for all symmetry equivalents
            q = R.from_matrix(R_eq).as_quat()
            w = np.clip(np.abs(q[:, 3]), -1.0, 1.0)
            ang = 2 * np.arccos(w)
            ang_deg = np.degrees(ang)

            # Take minimum disorientation angle
            miso_angles[(a, b)] = np.min(ang_deg)

    return miso_angles

def compute_cluster_boundary_misorientations_fast(avg_M, neighbors, symops):
    """
    Vectorized computation of misorientation angles (deg) between neighboring clusters
    considering symmetry operations, without explicit per-cluster loops.

    Parameters
    ----------
    avg_M : dict[int, (3,3) ndarray]
        Average orientation matrices (sample→crystal) per cluster.
    neighbors : dict[int, set[int]]
        Neighbor relationships.
    symops : (Ns,3,3) ndarray
        Symmetry operation matrices.

    Returns
    -------
    miso_angles : dict[(int,int), float]
        Minimum misorientation angle (degrees) between neighboring clusters.
    """
    # --- 1. Extract all unique neighbor pairs
    pairs = set()
    for a, nbs in neighbors.items():
        for b in nbs:
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    pairs = sorted(pairs)
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {}

    # --- 2. Stack orientation matrices for vectorized computation
    A = np.stack([avg_M[a] for a, _ in pairs])  # (P,3,3)
    B = np.stack([avg_M[b] for _, b in pairs])  # (P,3,3)

    # --- 3. Compute relative rotations R_ab = B @ A.T (vectorized)
    R_ab = np.einsum("pij,pkj->pik", B, A, optimize=True)  # (P,3,3)

    # --- 4. Apply all symmetry operators at once
    # Shape: (Ns, P, 3, 3) → (Ns*P, 3, 3)
    R_eq = np.einsum("sij,pjk->spik", symops, R_ab, optimize=True).reshape(-1, 3, 3)

    # --- 5. Convert to quaternions & compute rotation angles
    q = R.from_matrix(R_eq).as_quat().reshape(len(symops), n_pairs, 4)
    w = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
    ang = 2 * np.arccos(w)  # radians
    ang_deg = np.degrees(ang)  # (Ns, P)

    # --- 6. Select minimum angle for each pair
    min_ang = np.min(ang_deg, axis=0)

    # --- 7. Return results in dictionary
    return {pairs[i]: min_ang[i] for i in range(n_pairs)} 


class EBSDData(getPhases):
    """
    Main container for EBSD scan data.
    
    Design principles:
    - Immutable core data (X, Y, orientations)
    - Lazy evaluation for derived properties
    - Caching of expensive computations
    """
        
    
    def __init__(self, phase_info=None, symmetries=None):
        """
        Parameters
        ----------
        X, Y : array (N,)
            Spatial coordinates
        orientations : array (N, 3, 3) or (N, 4)
            Rotation matrices or quaternions
        phases_id : array (N,)
            Phase identifier labels for each point
        phase_info : dict, optional
            Metadata about phases {phase_id: {'name': ..., 'crystal_system': ...}}
        symmetries : dict, optional
            Symmetry operations {phase_id: symmetry_operations}
        quality : array (N,), optional
            Quality metrics for each point
        """
        getPhases.__init__(self)
        self.setAttributes(PhaseNames={self.austenite:'Austenite',self.martensite:'Martensite'})
        self.setAttributes(PhaseCols={self.austenite:['r','m'],self.martensite:['b','c']})
        #setattr(self,'rois',{'masks':[]})
        self.rois = type("rois", (),{})()
        self.rois.masks = []
        self.rois.masks_by_phase = []
        self.phase_info = phase_info or {}
        self.symmetries = symmetries or {}
        
        # Cached properties
        self._grid_2d = None
        self._phase_2d = None
        self._neighbors = None
        self._inside_mask = None
        self._coord_to_idx = None  # NEW: Coordinate to index mapping
        
    def setAttributes(self,**kwargs):    
        """
        Set attributes for the EBSD analyzer.
        
        Args:
            **kwargs: Key-value pairs of attributes to set
        """
        self.__dict__.update(kwargs)    

    def getEBSDdata(self,X, Y, orientations, phases, quality=None):
        self._X = np.asarray(X, dtype=np.float32)
        self._Y = np.asarray(Y, dtype=np.float32)
        self._orientations = np.asarray(orientations)
        self._phases_id = np.asarray(phases, dtype=np.int32)
        self._quality = quality if quality is not None else np.ones(len(X))
        # Validate data
        self._validate()
    def readEBSDdata(self,filename):
        self._ebsdData = pyebsd.load_scandata(filename)
        #self._X = np.asarray(scan.X, dtype=np.float32)
        #self._Y = np.asarray(scan.Y, dtype=np.float32)
        #self._orientations = np.asarray(scan.M)
        #self._phases_id = np.asarray(scan.phase)
        #self._phi1 = np.asarray(scan.Euler1, dtype=np.float32)
        #self._Phi = np.asarray(scan.Euler2, dtype=np.float32)
        #self._phi2 = np.asarray(scan.Euler3, dtype=np.float32)
        #self._quality = np.asarray(scan.IQ, dtype=np.float32)#quality if quality is not None else np.ones(len(X))
        N = len(self._ebsdData.X)
        self._quaternions = np.zeros((N,4))

        for i in range(N):
            self._quaternions[i] = mat_to_quat(self._ebsdData.M[i])
        #del scan
        
        # Validate data
        SelVerts, SelPaths = self.set_allroi()
        self.selector = selectROI(None)
        self.selector.selVerts.append(SelVerts)
        self.selector.selPaths.append(SelPaths)
        self._validate()

    def _validate(self):
        """Validate data consistency."""
        N = len(self._ebsdData.X)
        assert len(self._ebsdData.Y) == N, "X and Y must have same length"
        assert len(self._ebsdData.M) == N, "Orientations length mismatch"
        assert len(self._ebsdData.phase) == N, "Phases length mismatch"
        assert len(self._ebsdData.IQ) == N, "Quality length mismatch"
    
    @property
    def N(self):
        """Number of points."""
        return len(self._ebsdData.X)
    
    @property
    def X(self):
        return self._ebsdData.X
    
    @property
    def Y(self):
        return self._ebsdData.Y
    
    @property
    def orientations(self):
        return self._ebsdData.M
    
    @property
    def phases_id(self):
        return self._ebsdData.phase
    
    @property
    def quality(self):
        return self._ebsdData.IQ
    
    @property
    def unique_phases_id(self):
        return np.unique(self._ebsdData.phase)
    @property
    def unique_phases_names(self):
        return list(self.phase_ids.keys())
    
    @property
    def phi1(self):
        return self._ebsdData.Euler1
    @property
    def Phi(self):
        return self._ebsdData.Euler2
    
    @property
    def phi2(self):
        return self._ebsdData.Euler3
    
    @property
    def quaternions(self):
        return self._quaternions
    
    def setPhaseID(self,phase_names_ids):
        """
        create dictionariy phase_ids  providing phase id (from ebsd scan) for phase abbreviation (phase abbreviation is assigned in getphases class)
        create dictionariy phase_names  providing phase name for phase id
        phase ids =  np.unique(phases_id)
        phaseids = {"phase abbreviation1":phase_id1, "phase abbreviation2":phase_id2}
        """
        
        self.phase_ids = {}
        self.phase_names = {}
        for name in phase_names_ids.keys():
            self.phase_ids[name] = phase_names_ids[name]
            self.phase_names[phase_names_ids[name]] = name
        self.sym_quats_dict={}
        for key in self.phase_names.keys():
            self.sym_quats_dict[key] = self.phases[self.phase_names[key]]['sym_quats']

    def getPhaseID(self,name):
        return self.phase_ids[name]
    
    def getPhaseName(self,name):
        return self.phase_names[name]
    
    def getMask(self,roi, phase):
        if type(phase)==int:
            phase=self.phase_names[phase]
        if roi is None:
            if phase is None:
                sel = self.rois.masks[0]
            else:
                sel = self.rois.masks_by_phase[0][phase]
        else:
            if phase is None:
                sel=self.rois.masks[roi]
            else:
                sel=self.rois.masks_by_phase[roi][phase]
        return sel,phase
    def plot_colmap(self,d=[1,0,0], tiling=None, scalebar=True,globalScale=False, roi=None, phase=None, mask=None, fig=None, ax=None, **kwargs):
        if tiling is None:
            if self._ebsdData.grid.lower() == 'hexgrid':
                tiling == "hex"
            else:
                tiling = "rect"

        sel,phase = self.getMask(roi, phase)
        if mask is None:
            mask=sel
        if fig is None and ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.68])
        if globalScale:
            ax.set_xlim((self.X.min(),self.X.max()))
            ax.set_ylim((self.Y.max(),self.Y.min()))
        colmap = self._ebsdData.plot_IPF_lh(d=d,tiling=tiling,scalebar=scalebar,d_IPF=None,ax=ax,sel=mask,**kwargs)#(gray=scan.IQ)

        return fig,ax

    def getIPFcolors(self, d, orientations, roi=None, phase=None):
        sel,phase = self.getMask(roi, phase)
        
        d_IPF = orilistMult(orientations,d).T
        d_IPF[d_IPF[:,2]<0,:]=-1*d_IPF[d_IPF[:,2]<0,:]
        d_IPF[:,1] = np.abs(d_IPF[:,1])
        w = Vector3d(d_IPF)
        if phase == None:
            phases = self.unique_phases_names
        else:
            phases=[phase]
        pg_laues=[]
        for phase in phases:
            pg_laues.append(self.phases[phase]['symmetry'])
        
        dirkey=plot.DirectionColorKeyTSL(pg_laues[0])
        Colors=dirkey.direction2color(w)*0
        
        for phase,pg_laue in zip(phases,pg_laues):
            dirkey=plot.DirectionColorKeyTSL(pg_laue)
            Colors[self.phases_id==self.getPhaseID(phase),:]=dirkey.direction2color(w[self.phases_id==self.getPhaseID(phase)])

        Colors=Colors*255
        Colors=Colors.astype(int)
        Colors=np.hstack((Colors,Colors[:,0:1]*0+255))

        return Colors, phases, pg_laues


    def plot_IPF(self,d, tiling=None, scalebar=True,globalScale=False, roi=None, phase=None, orientations=None, mask=None,fig=None, ax=None, **kwargs):
        if tiling is None:
            if self._ebsdData.grid.lower() == 'hexgrid':
                tiling == "hex"
            else:
                tiling = "rect"

        sel,phase = self.getMask(roi, phase)
        #print(sel.shape)
        #print(mask.shape)
        if mask is None:
            mask=sel
        if orientations is None:
            orientations = self.orientations
        Colors, phases, pg_laues = self.getIPFcolors(d, orientations, roi=roi,phase=phase)
        if fig is None and ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.68])
        ax.set_title(f'Inverse pole figure for direction {d} in sample coordinate system')
        
        if globalScale:
            ax.set_xlim((self.X.min(),self.X.max()))
            ax.set_ylim((self.Y.max(),self.Y.min()))
        ipfmap = self._ebsdData.plot_IPF_lh(d=d,tiling=tiling,scalebar=scalebar,d_IPF=None,color=Colors,ax=ax,sel=mask,**kwargs)#(gray=scan.IQ)
        #print(ax.get_xlim())
        #print(ax.get_ylim())
        rc = {"font.size": 8}
        with plt.rc_context(rc):  # Temporarily reduce font size
            for pgi, pg_laue in enumerate(pg_laues):
                ax_ckey = fig.add_axes(
                    [0.2+0.2*pgi, 0.85, 0.1, 0.1], projection="ipf", symmetry=pg_laue, zorder=2
                )
                ax_ckey.plot_ipf_color_key(show_title=True)
                ax_ckey.patch.set_facecolor("None")

        return fig, ax
            
    def select_rois(self, d, phases='all',tiling=None, scalebar=True, roi=None, phase=None,):
        """
        select ROIS from an IPF by polygons
        """
        fig, ax = self.plot_IPF(d, tiling=tiling, scalebar=scalebar, roi=roi, phase=phase)
        self.selector = selectROI(ax)
        #selector2 = PolygonSelector(ax,onselect)#, lambda *args: None)
        plt.show()    


        #selector.disconnect()
        #selector2 = LassoSelector(ax, lambda *args: None)
        print("Click on the figure to create a polygon.")
        print("Press the 'esc' key to start a new polygon.")
        print("Try holding the 'shift' key to move all of the vertices.")
        print("Try holding the 'ctrl' key to move a single vertex.")
        #print(selector.selVerts)
        return 

    def set_rois(self):
        """
        Set selected ROIS as masks
        """
        if len(self.rois.masks)==0:
            SelVerts, SelPaths = self.set_allroi()
        if len(self.rois.masks_by_phase)==0:
            mask_by_phase={}
            for phase_id in self.phase_names:
                mask_by_phase[self.phase_names[phase_id]] = (self.phases_id==phase_id)*self.rois.masks[0]
            self.rois.masks_by_phase.append(mask_by_phase)

        for selPath, selVert in zip(self.selector.selPaths,self.selector.selVerts):
            self.rois.masks.append(selPath.contains_points(np.vstack((self.X,self.Y)).T))
            mask_by_phase={}
            for phase_id in self.phase_names:
                mask_by_phase[self.phase_names[phase_id]] = (self.phases_id==phase_id)*self.rois.masks[-1]
            self.rois.masks_by_phase.append(mask_by_phase)
        self.selector.selPaths.insert(0,selPath)
        self.selector.selVerts.insert(0,selVert)
        mask_by_phase={}
        

    def set_allroi(self):
        SelVerts = np.array([[ 0.     ,   0.],
        [ 0.     , self._ebsdData.Y.max()],
        [self._ebsdData.X.max()     , self._ebsdData.Y.max()],
        [self._ebsdData.X.max()     ,   0.],
        [ 0.     ,   0.]])
        SelPaths = Path(SelVerts)
        #self.rois.masks.append(SelPaths.contains_points(np.vstack((self.X,self.Y)).T))
        #self.rois.allroidx = len(self.rois.masks)-1
        self.rois.masks.append(SelPaths.contains_points(np.vstack((self.X,self.Y)).T))
        return SelVerts, SelPaths
        

    def rm_rois(self):
        """
        Remove all previously set ROIS
        """
        self.rois.masks = []
        self.rois.masks_by_phase = []
        self.selector.selVerts = []
        self.selector.selPaths= []

    def get_grid_2d(self, force_recompute=False):
        """
        Get 2D grid representation (lazy evaluation).
        
        Returns
        -------
        xs, ys : arrays
            Unique sorted coordinates
        labels_2d : array (ny, nx)
            2D grid (initially zeros, filled by clustering)
        phase_2d : array (ny, nx)
            Phase labels in 2D
        x_map, y_map : dict
            Coordinate to index mapping
        """
        if self._grid_2d is None or force_recompute:
            xs, ys = np.unique(self._ebsdData.X), np.unique(self._ebsdData.Y)
            x_map = {x: i for i, x in enumerate(xs)}
            y_map = {y: i for i, y in enumerate(ys)}
            ny, nx = len(ys), len(xs)
            
            labels_2d = np.zeros((ny, nx), dtype=np.int32)
            phase_2d = np.zeros((ny, nx), dtype=np.int32)
            # NEW: Create coordinate to original index mapping
            coord_to_idx = {}
            for i in range(self.N):
                coord_to_idx[(self.X[i], self.Y[i])] = i

            self._grid_2d = {
                'xs': xs, 'ys': ys,
                'labels_2d': labels_2d,
                'phase_2d': phase_2d,
                'x_map': x_map, 'y_map': y_map,
                'shape': (ny, nx),
                'coord_to_idx': coord_to_idx  # NEW
                }
            
        return self._grid_2d
    def set_rois2d(self, roi):
        """
        Set region of interest.
        
        Parameters
        ----------
        roi_polygon : matplotlib.path.Path or array (M, 2)
            ROI boundary coordinates
        """
        #if not isinstance(roi_polygon, Path):
        #    roi_polygon = Path(roi_polygon)
        grid = self.get_grid_2d(roi)
        xs, ys = grid['xs'], grid['ys']
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        inside = self.selector.selPaths[roi].contains_points(points)
        #inside = self.rois.masks[roi]
        self._inside_mask = inside.reshape(grid['shape'])
    
    def get_inside_mask2d(self):
        """Get ROI mask (all True if no ROI set)."""
        if self._inside_mask is None:
            grid = self.get_grid_2d()
            self._inside_mask = np.ones(grid['shape'], dtype=bool)
        return self._inside_mask
    
    def get_phase_subset(self, phase_id):
        """Get data for specific phase."""
        mask = self.__ebsdData.phase == phase_id
        return EBSDSubset(self, mask)
    
    def get_neighbors_oim(self, distance):
        """
        Returns list of relative indices of neighboring pixels for hexagonal grid.
        
        Args:
            distance (int): Neighbor distance in pixels
            
        Returns:
            tuple: (j_list, i_list) of neighbor indices
            
        From pyebsd - implements OIM convention for hexagonal grids.
        """
        if self._ebsdData.grid.lower() == "hexgrid":
            R60 = np.array(
                [[_COS60, -_SIN60], [_SIN60, _COS60]]
            )  # 60 degrees rotation matrix

            j_list = np.arange(-distance, distance, 2)
            i_list = np.full(j_list.shape, -distance)

            xy = np.vstack([j_list * _COS60, i_list * _SIN60])

            j_list, i_list = list(j_list), list(i_list)

            for r in range(1, 6):
                xy = np.dot(R60, xy)  # 60 degrees rotation
                j_list += list((xy[0] / _COS60).round(0).astype(int))
                i_list += list((xy[1] / _SIN60).round(0).astype(int))
        else:  # sqrgrid
            R90 = np.array([[0, -1], [1, 0]], dtype=int)  # 90 degrees rotation matrix
            xy = np.vstack(
                [
                    np.arange(-distance, distance, dtype=int),
                    np.full(2 * distance, -distance, dtype=int),
                ]
            )

            j_list, i_list = list(xy[0]), list(xy[1])

            for r in range(1, 4):
                xy = np.dot(R90, xy)
                j_list += list(xy[0])
                i_list += list(xy[1])

        return j_list, i_list

    def compute_neighbors(
        self, distance=1, perimeteronly=True, distance_convention="OIM", roi=None, sel=None
    ):
        """
        Get indices of neighboring pixels for every pixel at given distance.
        
        Args:
            distance (int): Neighbor distance
            perimeteronly (bool): Only include perimeter pixels
            distance_convention (str): 'OIM' or 'fixed' distance convention
            roi: Selection mask for pixels
            
        Returns:
            array: Neighbor indices array
            
        From pyebsd - calculates neighbor relationships for EBSD data.
        """
        if distance_convention.lower() == "oim":
            _get_neighbors = self.get_neighbors_oim
        else:
            raise Exception(
                'get_neighbors: unknown distance convention "{}"'.format(
                    distance_convention
                )
            )

        if perimeteronly:
            # only pixels in the perimeter
            j_shift, i_shift = _get_neighbors(distance)
        else:
            # including inner pixels
            j_shift, i_shift = [], []
            for d in range(1, distance + 1):
                j_sh, i_sh = _get_neighbors(d)
                j_shift += j_sh
                i_shift += i_sh

        n_neighbors = len(j_shift)
        if roi is None:
            if sel is None:
                sel = self.rois.masks[0]
        else:
            sel = self.rois.masks[roi]
        # x
        j_neighbors = np.full((self._ebsdData.N, n_neighbors), -1, dtype=int)
        j_neighbors[sel] = np.add.outer(self._ebsdData.j[sel], j_shift)
        # y
        i_neighbors = np.full((self._ebsdData.N, n_neighbors), -1, dtype=int)
        i_neighbors[sel] = np.add.outer(self._ebsdData.i[sel], i_shift)

        # i, j out of allowed range
        outliers = (
            (j_neighbors < 0)
            | (j_neighbors >= self._ebsdData.ncols)
            | (i_neighbors < 0)
            | (i_neighbors >= self._ebsdData.nrows)
        )

        self.neighbors_ind = np.full((self._ebsdData.N, n_neighbors), -1, dtype=int)
        self.neighbors_ind[sel] = self._ebsdData.ij_to_index(i_neighbors[sel], j_neighbors[sel])
        self.neighbors_ind[outliers] = -1
        self.neighbors_ind = self.neighbors_ind.astype(int)
        return self.neighbors_ind
    def get_neighbors_fixed(self, distance):
        """
        Returns list of relative indices of the neighboring pixels for
        a given distance in pixels
        """


        neighbors_hexgrid_fixed = [
        # 1st neighbors
        [[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]],
        # 2nd neighbors
        [[3, 1], [0, 2], [-3, 1], [-3, -1], [0, -2], [3, -1]],
        # 3rd neighbors and so on...
        [[4, 0], [2, 2], [-2, 2], [-4, 0], [-2, -2], [2, -2]],
        [
            [5, 1],
            [4, 2],
            [1, 3],
            [-1, 3],
            [-4, 2],
            [-5, 1],
            [-5, -1],
            [-4, -2],
            [-1, -3],
            [1, -3],
            [4, -2],
            [5, -1],
        ],
        [[6, 0], [3, 3], [-3, 3], [-6, 0], [-3, -3], [3, -3]],
        [[6, 2], [0, 4], [-6, 2], [-6, -2], [0, -4], [6, -2]],
        [
            [7, 1],
            [5, 3],
            [2, 4],
            [-2, 4],
            [-5, 3],
            [-7, 1],
            [-7, -1],
            [-5, -3],
            [-2, -4],
            [2, -4],
            [5, -3],
            [7, -1],
        ],
        [[8, 0], [4, 4], [-4, 4], [-8, 0], [-4, -4], [4, -4]],
        [
            [8, 2],
            [7, 3],
            [1, 5],
            [-1, 5],
            [-7, 3],
            [-8, 2],
            [-8, -2],
            [-7, -3],
            [-1, -5],
            [1, -5],
            [7, -3],
            [8, -2],
        ],
        [
            [9, 1],
            [6, 4],
            [3, 5],
            [-3, 5],
            [-6, 4],
            [-9, 1],
            [-9, -1],
            [-6, -4],
            [-3, -5],
            [3, -5],
            [6, -4],
            [9, -1],
        ],
        [[10, 0], [5, 5], [-5, 5], [-10, 0], [-5, -5], [5, -5]],
        [[9, 3], [0, 6], [-9, 3], [-9, -3], [0, -6], [9, -3]],
        [
            [10, 2],
            [8, 4],
            [2, 6],
            [-2, 6],
            [-8, 4],
            [-10, 2],
            [-10, -2],
            [-8, -4],
            [-2, -6],
            [2, -6],
            [8, -4],
            [10, -2],
        ],
        [
            [11, 1],
            [7, 5],
            [4, 6],
            [-4, 6],
            [-7, 5],
            [-11, 1],
            [-11, -1],
            [-7, -5],
            [-4, -6],
            [4, -6],
            [7, -5],
            [11, -1],
        ],
        # 15th neighbors
        [[12, 0], [6, 6], [-6, 6], [-12, 0], [-6, -6], [6, -6]],
        ]
        self._ebsdData.neighbors_hexgrid_fixed = neighbors_hexgrid_fixed
        self._ebsdData._n_neighbors_hexgrid_fixed = len(neighbors_hexgrid_fixed)


        if self.scan.grid.lower() == "hexgrid":
            if distance > self._ebsdData._n_neighbors_hexgrid_fixed:
                raise Exception(
                    "get_neighbors_fixed not supported for distance > {}".format(
                        self._ebsdData._n_neighbors_hexgrid_fixed
                    )
                )
            j_list, i_list = list(zip(*self._ebsdData.neighbors_hexgrid_fixed[distance - 1]))
        else:
            raise Exception(
                "get_neighbors_fixed not yet supported for grid type {}".format(
                    self._ebsdData.grid
                )
            )
        return list(j_list), list(i_list)

    def get_distance_neighbors(self, distance, distance_convention="OIM"):
        """
        Returns distance, in um, to the n-th (distance-th) neighbor

        Arguments
        ---------
        distance : int
            Distance with respect to the central pixel defined in terms of
            the nearest neighbor, i.e., distance = 3 represents the 3rd
            closest neighbor pixels
        distance_convention : str (optional)
            Distance convention used for selecting the neighboring pixels.
            Two possible values are allowed: 'OIM' or 'fixed'.
            The OIM convention is used by the TSL OIM software and is
            explained in its manual. 'fixed' stands for fixed distance,
            meaning that the neighbors are defined based on a fixed
            distance from the central pixel.
            Default : OIM

        Returns
        -------
        d : float
            Distance, in um, to the n-th (distance-th) neighbor
        """
        if distance_convention.lower() == "oim":
            j, i = self.get_neighbors_oim(distance)
        elif distance_convention.lower() == "fixed":
            j, i = self.get_neighbors_fixed(distance)
        else:
            raise Exception(
                ("get_distance_neighbors: unknown distance convention " '"{}"').format(
                    distance_convention
                )
            )
        #print(j)
        if self._ebsdData.grid.lower() == "hexgrid":
            #d = 0.5 * ((self.scan.dx*np.array(j)) ** 2 + 3.0 * (self.scan.dy*np.array(i)) ** 2) ** 0.5
            d = self._ebsdData.dx*0.5 * ((np.array(j)) ** 2 + 3.0 * (np.array(i)) ** 2) ** 0.5
        else:  # sqrgrid
            d = ((self._ebsdData.dx*np.array(j)) ** 2 + (self._ebsdData.dy*np.array(i)) ** 2) ** 0.5

        return d.mean()    
    def get_distance_neighbors_xy(self, distance, distance_convention="OIM"):
        """
        Returns distance, in um, to the n-th (distance-th) neighbor

        Arguments
        ---------
        distance : int
            Distance with respect to the central pixel defined in terms of
            the nearest neighbor, i.e., distance = 3 represents the 3rd
            closest neighbor pixels
        distance_convention : str (optional)
            Distance convention used for selecting the neighboring pixels.
            Two possible values are allowed: 'OIM' or 'fixed'.
            The OIM convention is used by the TSL OIM software and is
            explained in its manual. 'fixed' stands for fixed distance,
            meaning that the neighbors are defined based on a fixed
            distance from the central pixel.
            Default : OIM

        Returns
        -------
        d : float
            Distance, in um, to the n-th (distance-th) neighbor
        """
        if distance_convention.lower() == "oim":
            j, i = self.get_neighbors_oim(distance)
        elif distance_convention.lower() == "fixed":
            j, i = self.get_neighbors_fixed(distance)
        else:
            raise Exception(
                ("get_distance_neighbors: unknown distance convention " '"{}"').format(
                    distance_convention
                )
            )
        #print(j)
        if self._ebsdData.grid.lower() == "hexgrid":
            #d = 0.5 * ((self.scan.dx*np.array(j)) ** 2 + 3.0 * (self.scan.dy*np.array(i)) ** 2) ** 0.5
            dx = self._ebsdData.dx*np.array(j)
            dy = self._ebsdData.dy*np.array(i)
        else:  # sqrgrid
            dx = self._ebsdData.dx*np.array(j)
            dy = self._ebsdData.dy*np.array(i)

        return dx,dy   
    def get_distance_neighbors_ij(self, distance, distance_convention="OIM"):
        """
        Returns distance, in um, to the n-th (distance-th) neighbor

        Arguments
        ---------
        distance : int
            Distance with respect to the central pixel defined in terms of
            the nearest neighbor, i.e., distance = 3 represents the 3rd
            closest neighbor pixels
        distance_convention : str (optional)
            Distance convention used for selecting the neighboring pixels.
            Two possible values are allowed: 'OIM' or 'fixed'.
            The OIM convention is used by the TSL OIM software and is
            explained in its manual. 'fixed' stands for fixed distance,
            meaning that the neighbors are defined based on a fixed
            distance from the central pixel.
            Default : OIM

        Returns
        -------
        d : float
            Distance, in um, to the n-th (distance-th) neighbor
        """
        if distance_convention.lower() == "oim":
            j, i = self.get_neighbors_oim(distance)
        elif distance_convention.lower() == "fixed":
            j, i = self.get_neighbors_fixed(distance)
        else:
            raise Exception(
                ("get_distance_neighbors: unknown distance convention " '"{}"').format(
                    distance_convention
                )
            )
        
        return np.array(i),np.array(j)    
    def ij_to_index(self, i, j):
        """
        i, j grid positions to pixel index (self.index)

        Parameters
        ----------
        i : int or numpy ndarray
            Column number (y coordinate) according to grid description below
        j : int or numpy ndarray
            Row number (x coordinate) according to grid description below

        Returns
        -------
        index : int or numpy ndarray
            Pixel index

        Grid description for HexGrid:
        -----------------------------
        o : ncols_odd
        c : ncols_odd + ncols_even
        r : nrows
        n : total number of pixels

        ===================================
                    index
        0     1     2       o-2   o-1
        *     *     *  ...   *     *
            o    o+1            c-1
            *     *     ...      *
        c    c+1   c+2     c+o-2 c+o-1
        *     *     *  ...   *     *
                        .
                        .
                        .      n-1
            *     *     ...      *

        ===================================
                    j, i
        0  1  2  3  4   j         m-1
        *     *     *  ...   *     *   0

            *     *     ...      *      1

        *     *     *  ...   *     *   2
                        .
                        .              i
                        .
            *     *     ...      *     r-1

        Grid description for SqrGrid
        ----------------------------
        c : ncols_odd = ncols_even
        r : nrows
        n : total number of pixels

        ===================================
                    index
        0     1     2       c-2   c-1
        *     *     *  ...   *     *
        c    c+1   c+2     2c-2  2c-1
        *     *     *  ...   *     *
                        .
                        .
                        .   n-2   n-1
        *     *     *  ...   *     *

        ===================================
                    j, i
        0     1     2   j   n-2   n-1
        *     *     *  ...   *     *   0

        *     *     *        *     *   1
                        .
                        .              i
                        .
        *     *     *  ...   *     *  r-1

        """
        if self._ebsdData.grid.lower() == "hexgrid":
            index = (i // 2) * self._ebsdData.ncols + (j // 2)
            # ncols_odd > ncols_even is the normal situation
            if self._ebsdData.ncols_odd > self._ebsdData.ncols_even:
                index += (j % 2) * self._ebsdData.ncols_odd
                forbidden = i % 2 != j % 2  # forbidden i, j pairs
            else:
                index += (1 - j % 2) * self._ebsdData.ncols_odd
                forbidden = i % 2 == j % 2
            # This turns negative every i, j pair where j > ncols
            index *= 1 - self._ebsdData.N * (j // self._ebsdData.ncols)
            # Turns forbidden values negative
            index = np.array(index)
            index[forbidden] = -1
            if index.ndim == 0:
                index = int(index)
        else:
            index = i * self._ebsdData.ncols + j
        return index
    def get_coord_to_idx_map(self):
        """
        Get coordinate to index mapping.
        
        Returns
        -------
        coord_to_idx : dict
            Mapping {(x, y): i} from coordinates to original array indices
        """
        grid = self.get_grid_2d()
        return grid['coord_to_idx']
    
    def coords_to_indices(self, coords):
        """
        Convert array of coordinates to original array indices.
        
        Parameters
        ----------
        coords : array (M, 2)
            Array of (x, y) coordinates
        
        Returns
        -------
        indices : array (M,)
            Original array indices. -1 if coordinate not found.
        """
        coord_to_idx = self.get_coord_to_idx_map()
        indices = np.empty(len(coords), dtype=np.int32)
        
        for i, (x, y) in enumerate(coords):
            # Convert to same dtype as stored coordinates
            key = (float(x), float(y))
            indices[i] = coord_to_idx.get(key, -1)
        
        return indices

class selectROI(object):
    """Select indices from a matplotlib collection using `PolygonSelector`.

    Selected ROI
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    """

    def __init__(self, ax):
        if ax is not None:
            self.canvas = ax.figure.canvas
        
            # Ensure that we have separate colors for each object
        
            self.poly = PolygonSelector(ax, self.onselect)
        self.selVerts = []
        self.selPaths = []

    def onselect(self, verts):
        path = Path(verts)

        if len(self.selVerts)>0:
            ifin=True
            for vert2,vert1 in zip(verts,self.selVerts[-1]):
                if np.sum(np.array(vert2)-vert1)!=0:
                    ifin=False    
            if not ifin:
                verts = np.array(verts)
                verts = np.vstack((verts,verts[0,:]))
                self.selVerts.append(verts)
                self.selPaths.append(mpltPath.Path(verts))
        else:
            verts = np.array(verts)
            verts = np.vstack((verts,verts[0,:]))
            self.selVerts.append(verts)
            self.selPaths.append(mpltPath.Path(verts))

        #print(self.verts)
        #print(self.selVerts)
        #print(self.selPaths)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()

        
class EBSDSubset:
    """
    Lightweight view of EBSDData subset.
    Useful for phase-specific analysis without copying data.
    """
    
    def __init__(self, parent: EBSDData, mask: np.ndarray):
        self.parent = parent
        self.mask = mask
        self.indices = np.where(mask)[0]
    
    @property
    def X(self):
        return self.parent.X[self.mask]
    
    @property
    def Y(self):
        return self.parent.Y[self.mask]
    
    @property
    def orientations(self):
        return self.parent.orientations[self.mask]
    
    @property
    def phases_id(self):
        return self.parent.phases_id[self.mask]
    
    @property
    def N(self):
        return np.sum(self.mask)
    
    @property
    def phi1(self):
        return self.parent.phi1[self.mask]
    @property
    def Phi(self):
        return self.parent.Phi[self.mask]
    
    @property
    def phi2(self):
        return self.parent.phi2[self.mask]
    
    @property
    def quaternions(self):
        return self.parent.quaternions[self.mask]
    

# ============================================================================
# CLUSTERING ALGORITHMS
# ============================================================================
from abc import ABC, abstractmethod

class ClusteringAlgorithm(ABC):
    """
    Abstract base class for clustering algorithms.
    
    Design pattern: Strategy pattern
    - Allows easy swapping of algorithms
    - Common interface for all clustering methods
    """
    
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.parameters = {}
    
    @abstractmethod
    def fit(self, data: EBSDData) -> 'ClusteringResult':
        """
        Perform clustering.
        
        Parameters
        ----------
        data : EBSDData
            Input EBSD data
        
        Returns
        -------
        result : ClusteringResult
            Clustering results
        """
        pass
    
    def set_parameters(self, **kwargs):
        """Set algorithm parameters."""
        self.parameters.update(kwargs)
        return self


class MisorientationClustering(ClusteringAlgorithm):
    """
    Misorientation-based clustering with spatial constraints.
    
    This is the main clustering algorithm from your original code.
    """
    
    def __init__(self, ang_thr=5.0, dmax=1.5, minidxs=1,roi=None,distance=1, perimeteronly=True, distance_convention="OIM"):
        super().__init__("Misorientation Clustering")
        self.ang_thr = ang_thr
        self.dmax = dmax
        self.minidxs = minidxs
        self.roi = roi
        self.distance = distance
        self.perimeteronly=perimeteronly
        self.distance_convention = distance_convention
        
    
    def fit(self, data: EBSDData) -> 'ClusteringResult':
        """Perform misorientation-based clustering."""
        # Get data
        X, Y = data.X, data.Y
        #print(X)
        if self.roi is None:
            sel = data.rois.masks[0]
        else:
            sel=data.rois.masks[self.roi]

        nph = data.unique_phases_id.shape[0]
        if not isinstance(self.ang_thr,np.ndarray) and not isinstance(self.ang_thr,list):
            self.ang_thr = [self.ang_thr]*nph

        if not isinstance(self.dmax,np.ndarray) and not isinstance(self.dmax,list):
            self.dmax = [self.dmax]*nph

        if not isinstance(self.minidxs,np.ndarray) and not isinstance(self.minidxs,list):
            self.minidxs = [self.minidxs]*nph

        if not isinstance(self.distance,np.ndarray) and not isinstance(self.distance,list):
            self.distance = [self.distance]*nph

        if not isinstance(self.perimeteronly,np.ndarray) and not isinstance(self.perimeteronly,list):
            self.perimeteronly = [self.perimeteronly]*nph
        neighbors=[]
        for distance, perimeteronly in zip(self.distance,self.perimeteronly):
            neighbors.append(data.compute_neighbors(distance=distance, perimeteronly=perimeteronly, distance_convention=self.distance_convention, roi=self.roi))
        # Perform clustering (using your existing algorithm)
        labels, com, cluster_phase_map = self._cluster_multiphase(
            data.X, data.Y, data.quaternions, data.sym_quats_dict,
            neighbors, sel, data.phases_id, data.phase_ids,
            ang_thr=self.ang_thr,
            dmax=self.dmax,
            minidxs=self.minidxs)
        # Create result object
        result = ClusteringResult(labels, self, data, 
                                  parameters={
                                    'ang_thr': self.ang_thr,
                                    'dmax': self.dmax,
                                    'minidxs': self.minidxs,
                                    'roi':self.roi,
                                    'distance':self.distance,
                                    'perimeteronly':self.perimeteronly,
                                    'distance_convention':self.distance_convention,
                                    }, cluster_phases_id=cluster_phase_map, com=com)
        
        return result
    
    def _cluster_multiphase(self,X, Y, Q, sym_quats_dict, neighbors, Sel, phases_id, phase_ids,
                       ang_thr=5.0, dmax=1.5, minidxs=1):
        """
        Multi-phase clustering wrapper that runs clustering sequentially for each phase.
        
        Parameters
        ----------
        X, Y : float32 arrays (N,)
            Pixel coordinates.
        Q : float64 array (N, 4)
            Orientation quaternions.
        sym_quats_dict : dict
            Dictionary mapping phase_id to symmetry quaternions.
            Example: {1: cubic_sym_quats, 2: monoclinic_sym_quats}
        neighbors : int32 array (N, n_neigh)
            Neighbor indices per pixel (-1 for no neighbor).
        Sel : bool array (N,)
            True if pixel is inside ROI; False to exclude.
        phases_id : int32 array (N,)
            Phase identifier for each pixel.
        ang_thr : float
            Misorientation threshold in degrees.
        dmax : float
            Maximum spatial neighbor distance.
        minidxs : int
            Minimum pixel count to keep cluster.
        
        Returns
        -------
        labels_combined : int32 array (N,)
            Combined cluster labels across all phases (0 outside ROI).
        com_combined : float64 array (n_clusters_total, 2)
            Cluster centers of mass for all phases.
        cluster_phase_map : dict
            Mapping from cluster label to phase_id.
        """
        N = X.shape[0]
        labels_combined = np.zeros(N, dtype=np.int32)
        com_list = []
        cluster_phase_map = {}
        
        unique_phases = np.unique(phases_id)
        unique_phases = np.array(list((phase_ids.values())))
        current_cluster_offset = 0
        
        print(f"Clustering {len(unique_phases)} phases sequentially...")
        
        for pi, phase_id in enumerate(unique_phases):
            print(f"  Phase {phase_id}...", end=' ')
            
            # Create phase mask
            phase_mask = (phases_id == phase_id) & Sel
            
            if not np.any(phase_mask):
                print("No pixels in ROI, skipping.")
                continue
            
            # Get symmetry operations for this phase
            if phase_id not in sym_quats_dict:
                print(f"Warning: No symmetry operations for phase {phase_id}, skipping.")
                continue
            
            sym_quats = sym_quats_dict[phase_id]
            
            # Run clustering for this phase only
            labels_phase, com_phase, flat_neighbors, flat_lengths, slices = \
                self._cluster_map_sample_to_crystal_numba_with_neighbors_single_phase(
                    X, Y, Q, sym_quats,
                    neighbors[pi], phase_mask,
                    ang_thr=ang_thr[pi],
                    dmax=dmax[pi],
                    minidxs=minidxs[pi]
                )
            
            # Count clusters found
            n_clusters_phase = np.max(labels_phase)
            print(f"{n_clusters_phase} clusters")
            
            if n_clusters_phase == 0:
                continue
            
            # Offset cluster labels to make them unique across phases
            labels_phase_nonzero = labels_phase > 0
            labels_combined[labels_phase_nonzero] = labels_phase[labels_phase_nonzero] + current_cluster_offset
            
            # Store COM
            com_list.append(com_phase)
            
            # Map cluster labels to phase
            for cluster_label in range(1, n_clusters_phase + 1):
                cluster_phase_map[cluster_label + current_cluster_offset] = phase_id
            
            # Update offset for next phase
            current_cluster_offset += n_clusters_phase
        
        # Combine all COMs
        if com_list:
            com_combined = np.vstack(com_list)
        else:
            com_combined = np.zeros((0, 2))
        
        total_clusters = len(cluster_phase_map)
        print(f"Total: {total_clusters} clusters across all phases")
        
        return labels_combined, com_combined, cluster_phase_map

    def _cluster_map_sample_to_crystal_numba_with_neighbors_single_phase(self,X, Y, Q, sym_quats,
                                                                        neighbors, Sel,
                                                                        ang_thr=5.0,
                                                                        dmax=1.5,
                                                                        minidxs=1):
        """
        EBSD clustering for a SINGLE phase using explicit neighbor index matrix.
        This is the original function, works on one phase at a time.

        Parameters
        ----------
        X, Y : float32 arrays (N,)
            Pixel coordinates.
        Q : float64 array (N, 4)
            Orientation quaternions.
        sym_quats : float64 array (M, 4)
            Crystal symmetry quaternions for this phase.
        neighbors : int32 array (N, n_neigh)
            Neighbor indices per pixel (-1 for no neighbor).
        Sel : bool array (N,)
            True if pixel is inside ROI; False to exclude.
        ang_thr : float
            Misorientation threshold in degrees.
        dmax : float
            Maximum spatial neighbor distance.
        minidxs : int
            Minimum pixel count to keep cluster.

        Returns
        -------
        labels_new : int32 array (N,)
            Cluster labels (0 outside ROI or filtered clusters).
        com : float64 array (n_clusters, 2)
            Cluster centers of mass.
        flat_neighbors, flat_lengths, slices : flattened adjacency representation.
        """
        N = X.shape[0]
        labels = np.zeros(N, np.int32)
        cluster_id = 0
        dmax2 = dmax * dmax
        stacksize = 5*N
        stack = np.empty(stacksize, np.int32)

        com_x = np.zeros(N)
        com_y = np.zeros(N)
        counts = np.zeros(N)

        # --- 1. Flood-fill clustering within ROI ---
        for i in range(N):
            if not Sel[i]:
                continue
            if labels[i] != 0:
                continue
            cluster_id += 1
            sp = 0
            stack[sp] = i
            sp += 1
            while sp > 0:
                sp -= 1
                j = stack[sp]
                if labels[j] != 0:
                    continue
                if not Sel[j]:
                    continue

                labels[j] = cluster_id
                cidx = cluster_id - 1
                com_x[cidx] += X[j]
                com_y[cidx] += Y[j]
                counts[cidx] += 1
                qj = Q[j]

                for n in range(neighbors.shape[1]):
                    k = neighbors[j, n]
                    if k < 0:
                        continue
                    if not Sel[k]:
                        continue
                    if labels[k] != 0:
                        continue
                    dx = X[k] - X[j]
                    dy = Y[k] - Y[j]
                    if dx*dx + dy*dy > dmax2:
                        continue
                    ang = misori_sym_deg_quats(qj, Q[k], sym_quats)
                    if ang < ang_thr:
                        if sp < stacksize:
                            stack[sp] = k
                            sp += 1
                            
        # --- 2. Filter small clusters & compute COM ---
        new_idx = -np.ones(cluster_id, dtype=np.int32)
        n_clusters_new = 0
        for c in range(cluster_id):
            if counts[c] >= minidxs:
                new_idx[c] = n_clusters_new
                n_clusters_new += 1

        labels_new = np.zeros(N, np.int32)
        com = np.zeros((n_clusters_new, 2))
        counts_new = np.zeros(n_clusters_new)

        for i in range(N):
            if not Sel[i]:
                continue
            c_old = labels[i] - 1
            if c_old < 0:
                continue
            c_new = new_idx[c_old]
            if c_new >= 0:
                labels_new[i] = c_new + 1
                com[c_new, 0] += X[i]
                com[c_new, 1] += Y[i]
                counts_new[c_new] += 1

        for c in range(n_clusters_new):
            if counts_new[c] > 0:
                com[c, 0] /= counts_new[c]
                com[c, 1] /= counts_new[c]

        # --- 3. Compute cluster adjacency ---
        label_to_idx = -np.ones(n_clusters_new + 1, dtype=np.int32)
        for idx in range(n_clusters_new):
            label_to_idx[idx + 1] = idx

        neighbors_counts = np.zeros((n_clusters_new, n_clusters_new), dtype=np.int32)

        for i in range(N):
            if not Sel[i]:
                continue
            li = labels_new[i]
            if li == 0:
                continue
            ci = label_to_idx[li]
            for n in range(neighbors.shape[1]):
                k = neighbors[i, n]
                if k < 0:
                    continue
                if not Sel[k]:
                    continue
                lj = labels_new[k]
                if lj == 0 or lj == li:
                    continue
                cj = label_to_idx[lj]
                neighbors_counts[ci, cj] += 1

        # --- 4. Flatten adjacency arrays ---
        total_neighbors = 0
        for ci in range(n_clusters_new):
            for cj in range(n_clusters_new):
                if neighbors_counts[ci, cj] > 0:
                    total_neighbors += 1

        flat_neighbors = np.empty(total_neighbors, dtype=np.int32)
        flat_lengths = np.empty(total_neighbors, dtype=np.int32)
        slices = np.empty((n_clusters_new, 2), dtype=np.int32)

        idx = 0
        for ci in range(n_clusters_new):
            start = idx
            for cj in range(n_clusters_new):
                if neighbors_counts[ci, cj] > 0:
                    flat_neighbors[idx] = cj + 1
                    flat_lengths[idx] = neighbors_counts[ci, cj]
                    idx += 1
            end = idx
            slices[ci, 0] = start
            slices[ci, 1] = end

        return labels_new, com, flat_neighbors, flat_lengths, slices
    

# ============================================================================
# RESULTS CONTAINERS
# ============================================================================

class ClusteringResult:
    """Container for clustering results with analysis methods."""
    
    def __init__(self, labels, algorithm, data, parameters=None, 
                 cluster_phases_id=None, com=None):
        """
        Parameters
        ----------
        labels : array (N,)
            Cluster labels
        algorithm : ClusteringAlgorithm
            Algorithm that produced these results
        data : EBSDData
            Original EBSD data
        parameters : dict, optional
            Algorithm parameters used
        cluster_phases_id : dict, optional
            Pre-computed mapping {cluster_label: phase_id}
            If provided, skips recomputation
        com : array (n_clusters, 2), optional
            Pre-computed cluster centers of mass
            If provided, skips recomputation
        """
        self.labels = labels

        self.algorithm = algorithm
        self.data = data
        self.parameters = parameters or {}
        
        # Cached analyses - can be provided or computed later
        self._clusters_unique = None
        self._cluster_sizes = None
        self._cluster_sizes_by_phase = None
        self._cluster_phases_id = cluster_phases_id  # ← Pre-computed if provided
        self._com = com  # ← Pre-computed if provided
        self._average_orientations = None
        self.get_phase_labels()
        #self.getColors()
        self.getAvgOri()
        # ========== NEW: Cached morphological properties ==========
        self._cluster_areas = None
        self._cluster_perimeters = None
        self._cluster_equivalent_diameters = None
        self._cluster_sphericities = None
        # ==========================================================
        # Detect grid type from EBSD data
        if hasattr(data, '_ebsdData') and hasattr(data._ebsdData, 'grid'):
            grid_type = data._ebsdData.grid.lower()
            self.is_hexagonal = (grid_type == 'hexgrid')
            self.grid_type = grid_type
        else:
            self.is_hexagonal = False
            self.grid_type = 'sqrgrid'
        # ==========================================================

    @property
    def n_clusters(self):
        return len(self.get_unique_clusters())
    
    def _getMask(self, roi=None,cluster_id=None, phase=None):
        if roi is None:
            roimask = self.data.rois.masks[0]
        else:
            roimask = self.data.rois.masks[roi]
        if cluster_id is None:
            clustermask = self.data.rois.masks[0]
        else:
            if isinstance(cluster_id, np.ndarray) or isinstance(cluster_id, list):
                isonephase=True
                phasecid = self.data.phase_names[self.cluster_phases_id[cluster_id[0]]]
                clustermask = (self.data.rois.masks[0]*0).astype(bool)
                notinphase=[]
                for cid in cluster_id:
                    clustermask+=self.get_cluster_mask(cid)
                    if phasecid!=self.data.phase_names[self.cluster_phases_id[cid]]:
                        isonephase=False
                        #notinphase.append(cid)
                    else:
                        if phase is not None:
                            if phasecid!=phase:
                                isonephase=False
                                notinphase.append(cid)
                if isonephase and phase is None:
                    phase = phasecid
                elif not isonephase and phase is None:
                    print(f'Warning: Clusters do not belong to a single phase')
                else:
                    if not isonephase and phase is not None:
                        print(f'Warning: Clusters {notinphase} do not belong to the phase {phase}')
            else:
                if phase is None:
                    phase = self.data.phase_names[self.cluster_phases_id[cluster_id]]
                else:
                    if self.data.phase_names[self.cluster_phases_id[cluster_id]] != phase:
                        print(f'Warning: Cluster does not belong to the phase {phase}')

                clustermask = self.get_cluster_mask(cluster_id)
                
        if phase is None:
            phasemask = self.data.rois.masks[0]
        else:
            phasemask =self.data.rois.masks_by_phase[0][phase]
        if phase is None:
            if self.unique_phases.shape[0]==0:
                phase = self.data.phase_names[self.unique_phases[0]]

        return roimask*clustermask*phasemask, phase
        
    def get_unique_clusters(self):
        """Get unique cluster labels (excluding 0)."""
        if self._clusters_unique is None:
            self._clusters_unique = np.unique(self.labels[self.labels > 0])
        return self._clusters_unique
    
    def get_cluster_sizes(self):
        """Get size of each cluster."""
        if self._cluster_sizes is None:
            clusters = self.get_unique_clusters()
            self._cluster_sizes = {
                c: np.sum(self.labels == c) for c in clusters
            }
        return self._cluster_sizes
    def get_cluster_sizes_by_phase(self,phase):
        """Get size of each cluster for a phase."""
        if self._cluster_sizes_by_phase is None:
            clusters = self.labels_by_phase[phase]
            self._cluster_sizes_by_phase = {
                c: np.sum(self.labels == c) for c in clusters
            }
        return self._cluster_sizes_by_phase


    @property
    def unique_phases(self):
        return np.array(list((self.data.phase_ids.values())))#np.unique(self.data.phases_id)
           
    @property
    def cluster_phases_id(self):
        """Get phase_id of each cluster (lazy evaluation if not pre-computed)."""
        if self._cluster_phases_id is None:
            # Compute only if not provided during initialization
            clusters = self.get_unique_clusters()
            self._cluster_phases_id = {}
            for c in clusters:
                mask = self.labels == c
                phases = self.data.phases_id[mask]
                self._cluster_phases_id[c] = np.bincount(phases).argmax()
        return self._cluster_phases_id
    
    @cluster_phases_id.setter
    def cluster_phases_id(self, value):
        """Allow manual setting of cluster phase mapping."""
        self._cluster_phases_id = value
    
    @property
    def com(self):
        """Get cluster centers of mass (lazy evaluation if not pre-computed)."""
        if self._com is None:
            # Compute only if not provided during initialization
            clusters = self.get_unique_clusters()
            self._com = np.zeros((len(clusters), 2))
            for i, c in enumerate(clusters):
                mask = self.labels == c
                self._com[i, 0] = np.mean(self.data.X[mask])
                self._com[i, 1] = np.mean(self.data.Y[mask])
        return self._com
    
    @com.setter
    def com(self, value):
        """Allow manual setting of centers of mass."""
        self._com = value
    
    @property
    def cluster_sizes(self):
        """Get size of each cluster."""
        if self._cluster_sizes is None:
            clusters = self.get_unique_clusters()
            self._cluster_sizes = {
                c: np.sum(self.labels == c) for c in clusters
            }
        return self._cluster_sizes
    
    def getAvgOri(self, max_iter=10, tol=1e-6):
        #get average cluster orientations
        self.avg_orientations={}
        self.avg_quats={}
        for phase in self.labels_by_phase.keys():
            avg_M_dict, avg_q_dict, M_best_dict=self.average_orientations(phase, max_iter=max_iter, tol=tol)
            self.avg_orientations.update(avg_M_dict)
            self.avg_quats.update(avg_q_dict)
    def get_cluster_mask(self, cluster_id):
        """Get boolean mask for a specific cluster."""
        return self.labels == cluster_id
    
    def get_phase_labels(self):
        """Group labels by phase_id"""
        self.labels_by_phase={}
        for phase_id in np.array(list((self.data.phase_ids.values()))):#np.unique(self.data.phases_id):
            self.labels_by_phase[self.data.phase_names[phase_id]] = np.array([label for label in self.get_unique_clusters() if int(self._cluster_phases_id[label])==phase_id])
            #self.labels_by_phase[phase_id] = 
    
    def filter_by_size(self, min_size):
        """Create new result with small clusters removed."""
        from copy import deepcopy
        new_result = deepcopy(self)
        new_result.labels = remove_small_clusters(self.labels, min_size)
        # Invalidate cached properties
        new_result._clusters_unique = None
        new_result._cluster_sizes = None
        new_result._cluster_phases_id = None  # Need to recompute after filtering
        new_result._com = None  # Need to recompute after filtering

        # ========== NEW: Invalidate morphological properties ==========
        new_result._cluster_areas = None
        new_result._cluster_perimeters = None
        new_result._cluster_equivalent_diameters = None
        new_result._cluster_sphericities = None
        # ==============================================================

        return new_result
    
    def update_grid_2d(self):
        """Update 2D grid in data object with cluster labels."""
        grid = self.data.get_grid_2d()
        x_map, y_map = grid['x_map'], grid['y_map']
        inside_mask = self.data.get_inside_mask2d()
        
        for i in range(self.data.N):
            j, k = y_map[self.data.Y[i]], x_map[self.data.X[i]]
            if inside_mask[j, k]:
                grid['labels_2d'][j, k] = self.labels[i]
                grid['phase_2d'][j, k] = self.data.phases_id[i]
    # --- Numba-compatible average orientation ---
    def average_orientations(self, phase, ref_idx=0, max_iter=10, tol=1e-6,q_ref=None):
        N = self.data.quaternions.shape[0]
        unique_labels = self.labels_by_phase[phase]
        symops = np.array(self.data.phases[phase]['symops'])
        #print(symops)
        
        avg_q_dict = {}
        avg_M_dict = {}
        M_best_dict = {}  # new: stores all best-symmetric matrices per cluster

        for i in range(unique_labels.shape[0]):
            lab = unique_labels[i]
            if lab == 0: 
                continue

            # collect indices of current cluster
            idxs = np.where(self.labels == lab)[0]
            q_mean,M_mean,M_best_cluster,q_best_cluster= get_avg_orientations(self.data.quaternions[idxs], symops, ref_idx=ref_idx, max_iter=max_iter, tol=tol,q_ref=q_ref)
            #n_pix = idxs.shape[0]

            ## reference quaternion = first in cluster
            #q_ref = self.data.quaternions[idxs[0]]#mat_to_quat(M[idxs[0]])
            ##q_ref = mat_to_quat(np.eye(3))
            
            
            ## store best-symmetric matrices for this cluster
            #M_best_cluster = np.zeros((n_pix,3,3))
            #q_sum = np.zeros(4)

            #for j in range(n_pix):
            #    q_best, M_best, _ = find_best_symmetric_quat(self.data.quaternions[idxs[j]], q_ref, symops, max_iter, tol)
            #    if np.dot(q_best, q_ref) < 0:
            #        q_best *= -1.0
            #    q_sum += q_best
            #    M_best_cluster[j] = M_best

            ## average quaternion
            #q_mean = q_sum / np.linalg.norm(q_sum)
            M_mean = quat_to_mat(q_mean)

            avg_q_dict[lab] = q_mean
            avg_M_dict[lab] = M_mean
            M_best_dict[lab] = M_best_cluster

        return avg_M_dict, avg_q_dict, M_best_dict
    # ============================================================================
    # MORPHOLOGICAL ANALYSIS METHODS
    # ============================================================================

    
    def get_cluster_areas(self, pixel_size=None, use_hexagonal=None, units='pixels'):
        """
        Get area of each cluster.
        
        Parameters
        ----------
        pixel_size : float, optional
            Size of one pixel in physical units (e.g., micrometers).
            For square grids only. Ignored if use_hexagonal=True.
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData for hexagonal grid.
            If None, automatically detected from data._ebsdData.grid.
        units : str, optional
            Unit string for the measurement (e.g., 'μm', 'nm').
            Default is 'pixels'.
        
        Returns
        -------
        cluster_areas : dict
            Dictionary {cluster_label: area}
        """
        # Determine if using hexagonal grid
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        # Calculate area per pixel
        if use_hexagonal:
            if not hasattr(self.data, '_ebsdData'):
                raise ValueError("Hexagonal grid requested but data._ebsdData not found")
            if not hasattr(self.data._ebsdData, 'dx') or not hasattr(self.data._ebsdData, 'dy'):
                raise ValueError("Hexagonal grid requested but data._ebsdData.dx and dy not found")
            
            dx = self.data._ebsdData.dx
            dy = self.data._ebsdData.dy
            # Area per hexagonal pixel (standard formula for hexagonal grid)
            area_per_pixel = dx * dy * np.sqrt(3) / 2
            area_label = f"{units}"
        elif pixel_size is not None:
            # Square grid with physical units
            area_per_pixel = pixel_size ** 2
            area_label = f"{units}²"
        else:
            # Square grid in pixels
            area_per_pixel = 1.0
            area_label = "pixels"
        
        if self._cluster_areas is None or pixel_size is not None or use_hexagonal:
            clusters = self.get_unique_clusters()
            cluster_areas = {}
            
            for c in clusters:
                n_pixels = np.sum(self.labels == c)
                area = n_pixels * area_per_pixel
                cluster_areas[c] = area
            
            # Cache only if using default settings
            if pixel_size is None and not use_hexagonal:
                self._cluster_areas = cluster_areas
            
            return cluster_areas
        
        return self._cluster_areas
    
    def get_cluster_perimeters_from_boundaries(self, boundary_result, pixel_size=None, 
                                          use_hexagonal=None, boundary_type='all',
                                          warn_zero_perimeter=True):
        """
        Get perimeter of each cluster using actual boundary coordinates.
        
        This is more accurate than edge counting as it uses the actual
        detected boundaries from boundary analysis.
        
        Parameters
        ----------
        boundary_result : BoundaryResult
            Boundary analysis results
        pixel_size : float, optional
            Size of one pixel in physical units (for square grids)
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
        boundary_type : str, optional
            Type of boundaries to count:
            - 'all': All boundaries including ROI (default)
            - 'same_phase': Only boundaries with same-phase neighbors (excludes ROI and inter-phase)
            - 'exclude_interphase': ROI + same-phase boundaries (excludes inter-phase)
            - 'interphase_only': Only inter-phase boundaries (excludes ROI and same-phase)
        warn_zero_perimeter : bool, optional
            If True, prints warning when clusters have zero perimeter. Default is True.
        
        Returns
        -------
        cluster_perimeters : dict
            Dictionary {cluster_label: perimeter}
        """
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        # Validate boundary_type
        valid_types = ['all', 'same_phase', 'exclude_interphase', 'interphase_only']
        if boundary_type not in valid_types:
            raise ValueError(f"boundary_type must be one of {valid_types}")
        
        # Calculate edge length
        if use_hexagonal:
            dx = self.data._ebsdData.dx
            dy = self.data._ebsdData.dy
            edge_length = (dx + dy) / 2
        elif pixel_size is not None:
            edge_length = pixel_size
        else:
            edge_length = 1.0
        
        cluster_perimeters = {}
        zero_perimeter_clusters = []
        
        # Iterate through all clusters in boundary result
        for i, cluster_id in enumerate(boundary_result.clusters):
            cluster_phase_id = boundary_result.cluster_phases_id[i]
            
            # Get all boundaries for this cluster
            boundaries = boundary_result.grouped_boundaries[i]
            boundary_phases = boundary_result.grouped_boundary_phases_id[i]
            
            # Sum up boundary lengths based on boundary_type
            total_boundary_pixels = 0
            
            for neighbor_id, coords in boundaries.items():
                count_this_boundary = False
                
                if neighbor_id == -1:
                    # ROI boundary
                    if boundary_type in ['all', 'exclude_interphase']:
                        count_this_boundary = True
                else:
                    # Cluster-to-cluster boundary
                    neighbor_phase_id = boundary_phases[neighbor_id]
                    is_same_phase = (neighbor_phase_id == cluster_phase_id)
                    
                    if boundary_type == 'all':
                        count_this_boundary = True
                    elif boundary_type == 'same_phase':
                        count_this_boundary = is_same_phase
                    elif boundary_type == 'exclude_interphase':
                        count_this_boundary = is_same_phase
                    elif boundary_type == 'interphase_only':
                        count_this_boundary = not is_same_phase
                
                if count_this_boundary:
                    total_boundary_pixels += len(coords)
            
            # Convert to actual length
            perimeter = total_boundary_pixels * edge_length
            cluster_perimeters[cluster_id] = perimeter
            
            # Track zero perimeter clusters
            if perimeter == 0 and boundary_type != 'all':
                zero_perimeter_clusters.append(cluster_id)
        
        # Warn about zero perimeter clusters
        if warn_zero_perimeter and len(zero_perimeter_clusters) > 0:
            print(f"\nWARNING: {len(zero_perimeter_clusters)} clusters have zero perimeter with boundary_type='{boundary_type}'")
            if boundary_type == 'same_phase':
                print(f"These clusters have no same-phase neighbors (surrounded by different phases).")
            elif boundary_type == 'exclude_interphase':
                print(f"These clusters are completely surrounded by different phases with no ROI contact.")
            elif boundary_type == 'interphase_only':
                print(f"These clusters have no inter-phase boundaries (surrounded by same phase).")
            
            if len(zero_perimeter_clusters) <= 10:
                print(f"  Cluster IDs: {zero_perimeter_clusters}")
            else:
                print(f"  Cluster IDs: {zero_perimeter_clusters[:10]}... (and {len(zero_perimeter_clusters)-10} more)")
            print(f"Consider using boundary_type='all' or filtering these clusters.\n")
        
        return cluster_perimeters
    
    
    def get_cluster_perimeters(self, pixel_size=None, use_hexagonal=None, 
                            boundary_type='all', warn_zero_perimeter=True):
        """
        Get perimeter of each cluster.
        
        Perimeter is calculated by counting exposed edges (not boundary pixels).
        For each pixel in a cluster, we count how many of its edges border
        non-cluster pixels or the ROI boundary.
        
        Parameters
        ----------
        pixel_size : float, optional
            Size of one pixel in physical units.
            For square grids only. Ignored if use_hexagonal=True.
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
            If None, automatically detected from data._ebsdData.grid.
        boundary_type : str, optional
            Type of boundaries to count:
            - 'all': All boundaries (default)
            - 'same_phase': Only boundaries with same-phase neighbors
            - 'exclude_interphase': All boundaries except inter-phase boundaries
            - 'interphase_only': Only inter-phase boundaries
        warn_zero_perimeter : bool, optional
            If True, prints warning when clusters have zero perimeter. Default is True.
        
        Returns
        -------
        cluster_perimeters : dict
            Dictionary {cluster_label: perimeter}
        """
        # Determine if using hexagonal grid
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        # Validate boundary_type
        valid_types = ['all', 'same_phase', 'exclude_interphase', 'interphase_only']
        if boundary_type not in valid_types:
            raise ValueError(f"boundary_type must be one of {valid_types}")
        
        # Calculate edge length
        if use_hexagonal:
            if not hasattr(self.data, '_ebsdData'):
                raise ValueError("Hexagonal grid requested but data._ebsdData not found")
            if not hasattr(self.data._ebsdData, 'dx') or not hasattr(self.data._ebsdData, 'dy'):
                raise ValueError("Hexagonal grid requested but data._ebsdData.dx and dy not found")
            
            dx = self.data._ebsdData.dx
            dy = self.data._ebsdData.dy
            edge_length = (dx + dy) / 2
        elif pixel_size is not None:
            edge_length = pixel_size
        else:
            edge_length = 1.0
        
        # Need 2D grid to calculate perimeter
        grid = self.data.get_grid_2d()
        labels_2d = grid['labels_2d']
        inside_mask = self.data.get_inside_mask2d()
        
        # Update grid with current labels
        self.update_grid_2d()
        
        clusters = self.get_unique_clusters()
        cluster_perimeters = {}
        
        # Create phase map for quick lookup
        phase_2d = np.zeros_like(labels_2d)
        for i in range(self.data.N):
            j, k = grid['y_map'][self.data.Y[i]], grid['x_map'][self.data.X[i]]
            if inside_mask[j, k]:
                phase_2d[j, k] = self.data.phases_id[i]
        
        # Define 4-connectivity neighborhood (orthogonal only)
        dy_array = np.array([-1, 0, 1, 0], dtype=np.int32)
        dx_array = np.array([0, 1, 0, -1], dtype=np.int32)
        
        ny, nx = labels_2d.shape
        
        zero_perimeter_clusters = []
        
        for c in clusters:
            edge_count = 0
            cluster_phase = self.cluster_phases_id[c]
            
            # Find all pixels in this cluster
            cluster_mask = labels_2d == c
            ys, xs = np.where(cluster_mask)
            
            for y, x in zip(ys, xs):
                # Count how many edges of this pixel should be counted
                for n in range(4):  # Check all 4 orthogonal neighbors
                    yy = y + dy_array[n]
                    xx = x + dx_array[n]
                    
                    count_this_edge = False
                    
                    if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                        # Out of bounds
                        if boundary_type in ['all', 'exclude_interphase']:
                            count_this_edge = True
                    elif not inside_mask[yy, xx]:
                        # Outside ROI
                        if boundary_type in ['all', 'exclude_interphase']:
                            count_this_edge = True
                    elif labels_2d[yy, xx] != c:
                        # Different cluster - check phase
                        neighbor_phase = phase_2d[yy, xx]
                        is_same_phase = (neighbor_phase == cluster_phase)
                        
                        if boundary_type == 'all':
                            count_this_edge = True
                        elif boundary_type == 'same_phase':
                            count_this_edge = is_same_phase
                        elif boundary_type == 'exclude_interphase':
                            count_this_edge = is_same_phase
                        elif boundary_type == 'interphase_only':
                            count_this_edge = not is_same_phase
                    
                    if count_this_edge:
                        edge_count += 1
            
            perimeter = edge_count * edge_length
            cluster_perimeters[c] = perimeter
            
            # Track zero perimeter clusters
            if perimeter == 0 and boundary_type != 'all':
                zero_perimeter_clusters.append(c)
        
        # Warn about zero perimeter clusters
        if warn_zero_perimeter and len(zero_perimeter_clusters) > 0:
            print(f"\nWARNING: {len(zero_perimeter_clusters)} clusters have zero perimeter with boundary_type='{boundary_type}'")
            if boundary_type == 'same_phase':
                print(f"These clusters have no same-phase neighbors (surrounded by different phases).")
            elif boundary_type == 'exclude_interphase':
                print(f"These clusters are completely surrounded by different phases with no ROI contact.")
            elif boundary_type == 'interphase_only':
                print(f"These clusters have no inter-phase boundaries (surrounded by same phase).")
            
            if len(zero_perimeter_clusters) <= 10:
                print(f"  Cluster IDs: {zero_perimeter_clusters}")
            else:
                print(f"  Cluster IDs: {zero_perimeter_clusters[:10]}... (and {len(zero_perimeter_clusters)-10} more)")
            print(f"Consider using boundary_type='all' or filtering these clusters.\n")
        
        return cluster_perimeters
    
    def get_cluster_equivalent_diameters(self, pixel_size=None, use_hexagonal=None):
        """
        Get equivalent diameter of each cluster.
        
        Equivalent diameter is the diameter of a circle with the same area.
        Formula: d_eq = 2 * sqrt(Area / π)
        
        Parameters
        ----------
        pixel_size : float, optional
            Size of one pixel in physical units.
            For square grids only. Ignored if use_hexagonal=True.
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
            If None, automatically detected from data._ebsdData.grid.
        
        Returns
        -------
        cluster_equivalent_diameters : dict
            Dictionary {cluster_label: equivalent_diameter}
        """
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        if self._cluster_equivalent_diameters is None or pixel_size is not None or use_hexagonal:
            areas = self.get_cluster_areas(pixel_size=pixel_size, use_hexagonal=use_hexagonal)
            
            cluster_equivalent_diameters = {}
            for c, area in areas.items():
                d_eq = 2 * np.sqrt(area / np.pi)
                cluster_equivalent_diameters[c] = d_eq
            
            # Cache only if using default settings
            if pixel_size is None and not use_hexagonal:
                self._cluster_equivalent_diameters = cluster_equivalent_diameters
            
            return cluster_equivalent_diameters
        
        return self._cluster_equivalent_diameters
    
    def get_cluster_sphericities(self, pixel_size=None, use_hexagonal=None, 
                                boundary_result=None, boundary_type='all'):
        """
        Get sphericity (circularity) of each cluster.
        
        Sphericity is a measure of how circular a cluster is.
        Formula: sphericity = 4π × Area / Perimeter²
        
        Values range from 0 to 1:
        - 1.0 = perfect circle
        - < 1.0 = irregular shape
        
        Parameters
        ----------
        pixel_size : float, optional
            Size of one pixel in physical units.
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
        boundary_result : BoundaryResult, optional
            If provided, uses actual boundary coordinates for more accurate perimeter.
        boundary_type : str, optional
            Type of boundaries to count: 'all', 'same_phase', 
            'exclude_interphase', 'interphase_only'. Default is 'all'.
        
        Returns
        -------
        cluster_sphericities : dict
            Dictionary {cluster_label: sphericity}
            Note: Clusters with zero perimeter will have sphericity = 0.0
        """
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        areas = self.get_cluster_areas(pixel_size=pixel_size, use_hexagonal=use_hexagonal)
        
        # Choose perimeter calculation method
        if boundary_result is not None:
            perimeters = self.get_cluster_perimeters_from_boundaries(
                boundary_result, pixel_size=pixel_size, 
                use_hexagonal=use_hexagonal, boundary_type=boundary_type,
                warn_zero_perimeter=False  # Don't warn here, warn below
            )
        else:
            perimeters = self.get_cluster_perimeters(
                pixel_size=pixel_size, use_hexagonal=use_hexagonal,
                boundary_type=boundary_type, warn_zero_perimeter=False
            )
        
        cluster_sphericities = {}
        zero_perimeter_count = 0
        
        for c in self.get_unique_clusters():
            area = areas[c]
            perimeter = perimeters[c]
            
            if perimeter > 0:
                sphericity = 4 * np.pi * area / (perimeter ** 2)
                sphericity = min(sphericity, 1.0)
            else:
                sphericity = 0.0
                zero_perimeter_count += 1
            
            cluster_sphericities[c] = sphericity
        
        # Warning for zero perimeter clusters
        if zero_perimeter_count > 0:
            print(f"\nWARNING: {zero_perimeter_count} clusters have sphericity = 0 due to zero perimeter")
            print(f"  (boundary_type='{boundary_type}')")
            print(f"  These clusters will be excluded from sphericity statistics.\n")
        
        return cluster_sphericities


    def get_cluster_morphology(self, cluster_id, pixel_size=None, use_hexagonal=None, 
                            units='pixels', boundary_result=None, boundary_type='all'):
        """
        Get complete morphological information for a specific cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster label
        pixel_size : float, optional
            Size of one pixel in physical units (for square grids)
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
        units : str, optional
            Unit string for measurements
        boundary_result : BoundaryResult, optional
            If provided, uses actual boundary coordinates for perimeter.
        boundary_type : str, optional
            Type of boundaries to count: 'all', 'same_phase', 
            'exclude_interphase', 'interphase_only'. Default is 'all'.
        
        Returns
        -------
        morphology : dict
            Dictionary containing morphological properties
        """
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        if cluster_id not in self.get_unique_clusters():
            raise ValueError(f"Cluster {cluster_id} not found")
        
        areas = self.get_cluster_areas(pixel_size=pixel_size, use_hexagonal=use_hexagonal, units=units)
        
        if boundary_result is not None:
            perimeters = self.get_cluster_perimeters_from_boundaries(
                boundary_result, pixel_size=pixel_size, 
                use_hexagonal=use_hexagonal, boundary_type=boundary_type,
                warn_zero_perimeter=False
            )
        else:
            perimeters = self.get_cluster_perimeters(
                pixel_size=pixel_size, use_hexagonal=use_hexagonal,
                boundary_type=boundary_type, warn_zero_perimeter=False
            )
        
        equiv_diameters = self.get_cluster_equivalent_diameters(pixel_size=pixel_size, use_hexagonal=use_hexagonal)
        sphericities = self.get_cluster_sphericities(
            pixel_size=pixel_size, use_hexagonal=use_hexagonal,
            boundary_result=boundary_result, boundary_type=boundary_type
        )
        
        cluster_idx = np.where(self.get_unique_clusters() == cluster_id)[0][0]
        phase_id = self.cluster_phases_id[cluster_id]
        
        return {
            'cluster_id': cluster_id,
            'n_pixels': self.cluster_sizes[cluster_id],
            'area': areas[cluster_id],
            'perimeter': perimeters[cluster_id],
            'equivalent_diameter': equiv_diameters[cluster_id],
            'sphericity': sphericities[cluster_id],
            'center_of_mass': tuple(self.com[cluster_idx]),
            'phase_id': phase_id,
            'phase_name': self.data.phase_names[phase_id],
            'grid_type': self.grid_type,
            'units': units if (pixel_size is not None or use_hexagonal) else 'pixels',
            'boundary_type': boundary_type
        }


    def get_all_cluster_morphologies(self, pixel_size=None, use_hexagonal=None, units='pixels',
                                    boundary_result=None, boundary_type='all'):
        """
        Get morphological information for all clusters.
        
        Parameters
        ----------
        pixel_size : float, optional
            Size of one pixel in physical units (for square grids)
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
        units : str, optional
            Unit string for measurements
        boundary_result : BoundaryResult, optional
            If provided, uses actual boundary coordinates for perimeter.
        boundary_type : str, optional
            Type of boundaries to count: 'all', 'same_phase', 
            'exclude_interphase', 'interphase_only'. Default is 'all'.
        
        Returns
        -------
        morphologies : list of dict
            List of morphology dictionaries, one per cluster
        """
        return [
            self.get_cluster_morphology(c, pixel_size=pixel_size, 
                                    use_hexagonal=use_hexagonal, units=units,
                                    boundary_result=boundary_result,
                                    boundary_type=boundary_type)
            for c in self.get_unique_clusters()
        ]


    def print_cluster_morphology(self, cluster_id, pixel_size=None, use_hexagonal=None, 
                                units='μm', boundary_result=None, boundary_type='all'):
        """
        Print detailed morphological information for a specific cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster label to print
        pixel_size : float, optional
            Size of one pixel in physical units (for square grids)
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
        units : str, optional
            Unit string for measurements. Default is 'μm'.
        boundary_result : BoundaryResult, optional
            If provided, uses actual boundary coordinates for perimeter.
        boundary_type : str, optional
            Type of boundaries to count: 'all', 'same_phase', 
            'exclude_interphase', 'interphase_only'. Default is 'all'.
        """
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        # Get morphology data
        morph = self.get_cluster_morphology(
            cluster_id, 
            pixel_size=pixel_size, 
            use_hexagonal=use_hexagonal, 
            units=units,
            boundary_result=boundary_result,
            boundary_type=boundary_type
        )
        
        # Determine area units
        if use_hexagonal:
            dx = self.data._ebsdData.dx
            dy = self.data._ebsdData.dy
            area_units = f"{units}"
            grid_info = f"Hexagonal grid (dx={dx}, dy={dy})"
        elif pixel_size is not None:
            area_units = f"{units}²"
            grid_info = f"Square grid (pixel size={pixel_size} {units})"
        else:
            area_units = "pixels"
            units = "pixels"
            grid_info = "Square grid (pixel units)"
        
        # Describe boundary type
        boundary_desc = {
            'all': 'All boundaries (ROI + same-phase + inter-phase)',
            'same_phase': 'Same-phase only (intra-phase grain boundaries)',
            'exclude_interphase': 'ROI + same-phase (excludes inter-phase)',
            'interphase_only': 'Inter-phase only'
        }
        
        # Print formatted output
        print(f"\n{'='*70}")
        print(f"Cluster {morph['cluster_id']} - Morphological Analysis")
        print(f"{'='*70}")
        
        print(f"\nBasic Information:")
        print(f"  Phase:             {morph['phase_name']} (ID: {morph['phase_id']})")
        print(f"  Grid type:         {morph['grid_type']}")
        print(f"  Measurement:       {grid_info}")
        print(f"  Boundary type:     {boundary_desc[boundary_type]}")
        if boundary_result is not None:
            print(f"  Perimeter method:  Boundary-based (accurate)")
        else:
            print(f"  Perimeter method:  Edge-counting")
        
        print(f"\nSize Metrics:")
        print(f"  Number of pixels:  {morph['n_pixels']}")
        print(f"  Area:              {morph['area']:.2f} {area_units}")
        
        if morph['perimeter'] > 0:
            print(f"  Perimeter:         {morph['perimeter']:.2f} {units}")
        else:
            print(f"  Perimeter:         {morph['perimeter']:.2f} {units} [ZERO - see warnings]")
        
        print(f"\nShape Metrics:")
        print(f"  Equiv. diameter:   {morph['equivalent_diameter']:.2f} {units}")
        
        if morph['sphericity'] > 0:
            print(f"  Sphericity:        {morph['sphericity']:.4f} (0=irregular, 1=circular)")
            
            # Interpret sphericity
            if morph['sphericity'] > 0.9:
                shape_desc = "nearly circular"
            elif morph['sphericity'] > 0.7:
                shape_desc = "fairly circular"
            elif morph['sphericity'] > 0.5:
                shape_desc = "moderately irregular"
            else:
                shape_desc = "highly irregular"
            print(f"  Shape:             {shape_desc}")
        else:
            print(f"  Sphericity:        {morph['sphericity']:.4f} [ZERO - no valid perimeter]")
            print(f"  Shape:             Cannot determine (zero perimeter)")
        
        print(f"\nLocation:")
        print(f"  Center of mass:    ({morph['center_of_mass'][0]:.2f}, {morph['center_of_mass'][1]:.2f})")
        
        print(f"{'='*70}\n")
        
        return morph


    def print_cluster_morphology_summary(self, pixel_size=None, use_hexagonal=None, units='μm',
                                        boundary_result=None, boundary_type='all',
                                        sort_by='area', ascending=False, 
                                        max_clusters=None, exclude_zero_perimeter=False):
        """
        Print formatted summary of cluster morphologies.
        
        Parameters
        ----------
        pixel_size : float, optional
            Size of one pixel in physical units (for square grids)
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
        units : str, optional
            Unit string for measurements
        boundary_result : BoundaryResult, optional
            If provided, uses actual boundary coordinates for perimeter.
        boundary_type : str, optional
            Type of boundaries to count: 'all', 'same_phase', 
            'exclude_interphase', 'interphase_only'. Default is 'all'.
        sort_by : str, optional
            Sort by: 'area', 'perimeter', 'equivalent_diameter', 'sphericity', 'cluster_id'
            Default is 'area'
        ascending : bool, optional
            Sort in ascending order. Default is False (descending).
        max_clusters : int, optional
            Maximum number of clusters to print. If None, prints all.
        exclude_zero_perimeter : bool, optional
            If True, excludes clusters with zero perimeter from the summary.
            Default is False.
        """
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        morphologies = self.get_all_cluster_morphologies(
            pixel_size=pixel_size, use_hexagonal=use_hexagonal, units=units,
            boundary_result=boundary_result, boundary_type=boundary_type
        )
        
        # Filter out zero perimeter if requested
        if exclude_zero_perimeter:
            original_count = len(morphologies)
            morphologies = [m for m in morphologies if m['perimeter'] > 0]
            excluded_count = original_count - len(morphologies)
            if excluded_count > 0:
                print(f"\nNote: Excluded {excluded_count} clusters with zero perimeter from summary.")
        
        if len(morphologies) == 0:
            print("\nNo clusters to display (all have zero perimeter).")
            return
        
        # Sort
        if sort_by in ['area', 'perimeter', 'equivalent_diameter', 'sphericity', 'n_pixels']:
            morphologies.sort(key=lambda x: x[sort_by], reverse=not ascending)
        elif sort_by == 'cluster_id':
            morphologies.sort(key=lambda x: x['cluster_id'], reverse=not ascending)
        
        # Print header
        print(f"\n{'='*110}")
        print(f"Cluster Morphology Summary (sorted by {sort_by}, {'ascending' if ascending else 'descending'})")
        print(f"{'='*110}")
        
        grid_type = morphologies[0]['grid_type'] if morphologies else self.grid_type
        print(f"Grid type: {grid_type}")
        
        if use_hexagonal:
            dx = self.data._ebsdData.dx
            dy = self.data._ebsdData.dy
            print(f"Hexagonal grid: dx={dx}, dy={dy}")
            area_units = f"{units}"
        elif pixel_size is not None:
            print(f"Pixel size: {pixel_size} {units}")
            area_units = f"{units}²"
        else:
            area_units = "pixels"
            units = "pixels"
        
        boundary_desc = {
            'all': 'All boundaries',
            'same_phase': 'Same-phase only',
            'exclude_interphase': 'Excludes inter-phase',
            'interphase_only': 'Inter-phase only'
        }
        print(f"Boundary type: {boundary_desc[boundary_type]}")
        
        if boundary_result is not None:
            print(f"Perimeter method: Boundary-based (accurate)")
        else:
            print(f"Perimeter method: Edge-counting")
        
        # Column headers
        print(f"\n{'ID':>6} {'Phase':>8} {'Pixels':>8} {'Area':>12} {'Perimeter':>12} "
            f"{'Equiv.Diam':>12} {'Sphericity':>12}")
        print(f"{'':>6} {'':>8} {'':>8} {f'({area_units})':>12} {f'({units})':>12} "
            f"{f'({units})':>12} {'(0-1)':>12}")
        print(f"{'-'*110}")
        
        # Print clusters
        n_to_print = len(morphologies) if max_clusters is None else min(max_clusters, len(morphologies))
        
        for i, morph in enumerate(morphologies[:n_to_print]):
            phase_name = self.data.phase_names[morph['phase_id']]
            
            # Mark zero perimeter clusters
            perim_str = f"{morph['perimeter']:>12.2f}"
            if morph['perimeter'] == 0:
                perim_str = f"{morph['perimeter']:>11.2f}*"
            
            sph_str = f"{morph['sphericity']:>12.4f}"
            if morph['sphericity'] == 0:
                sph_str = f"{morph['sphericity']:>11.4f}*"
            
            print(f"{morph['cluster_id']:>6} {phase_name:>8} {morph['n_pixels']:>8} "
                f"{morph['area']:>12.2f} {perim_str} "
                f"{morph['equivalent_diameter']:>12.2f} {sph_str}")
        
        if max_clusters is not None and len(morphologies) > max_clusters:
            print(f"... and {len(morphologies) - max_clusters} more clusters")
        
        # Print statistics
        print(f"\n{'-'*110}")
        print(f"Summary Statistics:")
        print(f"  Total clusters: {len(morphologies)}")
        
        areas = [m['area'] for m in morphologies]
        perimeters = [m['perimeter'] for m in morphologies if m['perimeter'] > 0]
        diameters = [m['equivalent_diameter'] for m in morphologies]
        sphericities = [m['sphericity'] for m in morphologies if m['sphericity'] > 0]
        
        zero_perim_count = len([m for m in morphologies if m['perimeter'] == 0])
        if zero_perim_count > 0:
            print(f"  Clusters with zero perimeter: {zero_perim_count} (*marked in table)")
        
        print(f"  Area - Mean: {np.mean(areas):.2f}, Std: {np.std(areas):.2f}, "
            f"Min: {np.min(areas):.2f}, Max: {np.max(areas):.2f} {area_units}")
        
        if len(perimeters) > 0:
            print(f"  Perimeter - Mean: {np.mean(perimeters):.2f}, Std: {np.std(perimeters):.2f}, "
                f"Min: {np.min(perimeters):.2f}, Max: {np.max(perimeters):.2f} {units}")
        else:
            print(f"  Perimeter - All clusters have zero perimeter!")
        
        print(f"  Equiv. Diameter - Mean: {np.mean(diameters):.2f}, Std: {np.std(diameters):.2f}, "
            f"Min: {np.min(diameters):.2f}, Max: {np.max(diameters):.2f} {units}")
        
        if len(sphericities) > 0:
            print(f"  Sphericity - Mean: {np.mean(sphericities):.4f}, Std: {np.std(sphericities):.4f}, "
                f"Min: {np.min(sphericities):.4f}, Max: {np.max(sphericities):.4f} (n={len(sphericities)})")
        else:
            print(f"  Sphericity - All clusters have zero sphericity!")
        
        print(f"{'='*110}\n")


    def filter_by_sphericity(self, min_sphericity, pixel_size=None, use_hexagonal=None,
                            boundary_result=None, boundary_type='all'):
        """
        Create new result with non-circular clusters removed.
        
        Parameters
        ----------
        min_sphericity : float
            Minimum sphericity (0-1). Clusters below this are removed.
        pixel_size : float, optional
            Size of one pixel in physical units (for square grids)
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData.
        boundary_result : BoundaryResult, optional
            If provided, uses actual boundary coordinates for perimeter.
        boundary_type : str, optional
            Type of boundaries to count: 'all', 'same_phase', 
            'exclude_interphase', 'interphase_only'. Default is 'all'.
        
        Returns
        -------
        new_result : ClusteringResult
            New result with filtered clusters
        """
        from copy import deepcopy
        
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        sphericities = self.get_cluster_sphericities(
            pixel_size=pixel_size, use_hexagonal=use_hexagonal,
            boundary_result=boundary_result, boundary_type=boundary_type
        )
        
        # Create new labels array
        new_labels = self.labels.copy()
        
        for cluster_id, sphericity in sphericities.items():
            if sphericity < min_sphericity:
                new_labels[new_labels == cluster_id] = 0
        
        new_result = deepcopy(self)
        new_result.labels = new_labels
        
        # Invalidate cached properties
        new_result._clusters_unique = None
        new_result._cluster_sizes = None
        new_result._cluster_phases_id = None
        new_result._com = None
        new_result._cluster_areas = None
        new_result._cluster_perimeters = None
        new_result._cluster_equivalent_diameters = None
        new_result._cluster_sphericities = None
        
        return new_result  
    
    
    def plot_cluster_ipf4cubic(self, cluster_id, direction=(0, 0, 1), phase=None, roi=None,
                                equalarea=False, scale='sqrt', nlevels=10,
                                fig=None, ax=None, cmap=None, return_val=False,
                                plot_avg=False, avg_kwargs=None,
                                zoom_to_triangle=True, zoom_margin=0.05, **kwargs):
        """
        ... (same docstring as before) ...

        Notes
        -----
        If ax is provided (reused from a previous call), the new title line
        is appended below the existing title (separated by a newline)
        instead of replacing it — useful for overlaying multiple
        clusters/directions on the same axes. If ax is created fresh, this
        has no visible effect since the starting title is empty.
        """
        mask, resolved_phase = self._getMask(roi=roi, cluster_id=cluster_id, phase=phase)
        idxs = np.where(mask)[0]

        if idxs.shape[0] == 0:
            raise ValueError(f"No pixels found for cluster_id={cluster_id} (phase={resolved_phase})")

        pixel_quats = self.data.quaternions[idxs]
        M = np.array([quat_to_mat(q) for q in pixel_quats])  # (n_pixels, 3, 3), sample -> crystal

        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

        data = np.einsum('nij,j->ni', M, direction).T  # (3, n_pixels), crystal-frame

        if cmap is None:
            cmap = get_cmap([(1, 1, 1), (1, 0, 0)])

        result = stereaotriangle_hist(data, equalarea=equalarea, scale=scale, nlevels=nlevels,
                                    fig=fig, ax=ax, cmap=cmap, return_val=True, **kwargs)
        fig, ax, spsel, hist = result

        cluster_id_list = [cluster_id] if isinstance(cluster_id, (int, np.integer)) else list(cluster_id)
        n_clusters = len(cluster_id_list)

        if plot_avg:
            style = {
                'marker': 'o', 'markersize': 20, 'markeredgewidth': 2,
                'markeredgecolor': 'k', 'markerfacecolor': 'b', 'alpha': 0.7,
            }
            if avg_kwargs:
                style.update(avg_kwargs)

            avg_mats = np.array([self.avg_orientations[c] for c in cluster_id_list
                                if c in self.avg_orientations])
            missing = [c for c in cluster_id_list if c not in self.avg_orientations]
            if missing:
                print(f"Warning: no average orientation available for clusters {missing}, skipping overlay for them.")

            if avg_mats.shape[0] > 0:
                data_avg = np.einsum('nij,j->ni', avg_mats, direction).T
                #data_avg = data_avg[:, data_avg[2, :] > 0]

                if data_avg.shape[1] > 0:
                    if equalarea:
                        spsel_avg = equalarea_intotriangle_fast(data_avg)
                    else:
                        spsel_avg = stereoprojection_intotriangle_fast(data_avg)
                    ax.plot(spsel_avg[0, :], spsel_avg[1, :], linestyle='None', **style)

        if zoom_to_triangle:
            xlim, ylim = stereotriangle_bbox(equalarea=equalarea, margin=zoom_margin)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect('equal')

        new_title = (f"IPF {tuple(round(float(v), 2) for v in direction)} "
                    f"— {resolved_phase}, {n_clusters} cluster(s) of id(s) {cluster_id_list}, {idxs.shape[0]} px")
        existing_title = ax.get_title()
        ax.set_title(f"{existing_title}\n{new_title}" if existing_title else new_title)

        if return_val:
            return fig, ax, spsel, hist
        return fig, ax
    def plot_cluster_pole_figure_ini(self, cluster_id, uvw=(1, 0, 0), phase=None, roi=None,
                              equalarea=False, scale='sqrt', nlevels=10, hemi='upper',
                              fig=None, ax=None, cmap=None, return_val=False,
                              plot_avg=False, avg_kwargs=None, **kwargs):
        """
        ... (same docstring as before) ...

        Notes
        -----
        If ax is provided (reused from a previous call), the new title line
        is appended below the existing title (separated by a newline)
        instead of replacing it — useful for overlaying multiple
        clusters/directions on the same axes and keeping a running label of
        what's been plotted. If ax is created fresh, this has no visible
        effect since the starting title is empty.
        """
        if ax is None:
            fig, ax = plt.subplots()

        mask, resolved_phase = self._getMask(roi=roi, cluster_id=cluster_id, phase=phase)
        idxs = np.where(mask)[0]

        if idxs.shape[0] == 0:
            raise ValueError(f"No pixels found for cluster_id={cluster_id} (phase={resolved_phase})")

        pixel_quats = self.data.quaternions[idxs]
        M = np.array([quat_to_mat(q) for q in pixel_quats])  # (n_pixels, 3, 3), sample -> crystal
        Minv = M.transpose(0, 2, 1)                           # crystal -> sample

        symops = np.array(self.data.phases[resolved_phase]['symops'])

        uvw = np.asarray(uvw, dtype=float)
        uvwsb = set(tuple(v) for v in np.round(np.dot(symops, uvw), 8))
        uvwsb = np.asarray(list(uvwsb))

        XYZ = np.tensordot(Minv, uvwsb.T, axes=[[-1], [-2]]).transpose([0, 2, 1])
        XYZ = XYZ.reshape(-1, 3).T  # (3, n_pixels * n_variants)

        if cmap is None:
            cmap = get_cmap([(1, 1, 1), (1, 0, 0)])

        result = polefigure_hist(XYZ, equalarea=equalarea, scale=scale, nlevels=nlevels,
                                hemi=hemi, fig=fig, ax=ax, cmap=cmap, return_val=True, **kwargs)
        fig, ax, sp, hist = result

        cluster_id_list = [cluster_id] if isinstance(cluster_id, (int, np.integer)) else list(cluster_id)
        n_clusters = len(cluster_id_list)

        if plot_avg:
            style = {
                'marker': 'o', 'markersize': 20, 'markeredgewidth': 2,
                'markeredgecolor': 'k', 'markerfacecolor': 'b', 'alpha': 0.7,
            }
            if avg_kwargs:
                style.update(avg_kwargs)

            avg_mats = np.array([self.avg_orientations[c] for c in cluster_id_list
                                if c in self.avg_orientations])
            missing = [c for c in cluster_id_list if c not in self.avg_orientations]
            if missing:
                print(f"Warning: no average orientation available for clusters {missing}, skipping overlay for them.")

            if avg_mats.shape[0] > 0:
                avg_Minv = avg_mats.transpose(0, 2, 1)
                XYZavg = np.tensordot(avg_Minv, uvwsb.T, axes=[[-1], [-2]]).transpose([0, 2, 1])
                XYZavg = XYZavg.reshape(-1, 3).T

                if hemi == 'upper':
                    XYZavg = XYZavg[:, XYZavg[2, :] >= 0]
                else:
                    XYZavg = XYZavg[:, XYZavg[2, :] <= 0]

                if XYZavg.shape[1] > 0:
                    if equalarea:
                        sp_avg = equalarea_directions(XYZavg)
                    else:
                        sp_avg = stereoprojection_directions(XYZavg)
                    ax.plot(sp_avg[0, :], sp_avg[1, :], linestyle='None', **style)

        new_title = (f"Pole figure {tuple(int(v) for v in np.round(uvw))} "
                    f"— {resolved_phase}, {n_clusters} cluster(s) of id(s) {cluster_id_list}, {idxs.shape[0]} px")
        existing_title = ax.get_title()
        ax.set_title(f"{existing_title}\n{new_title}" if existing_title else new_title)

        if return_val:
            return fig, ax, sp, hist
        return fig, ax

    def plot_cluster_pole_figure(self, cluster_id, uvw=(1, 0, 0), phase=None, roi=None,
                                equalarea=False, hemi='upper', fig=None, ax=None,
                                pixel_plot='symbols', pixel_pooled=False,
                                pixel_symbol_kwargs=None, pixel_hist_kwargs=None,
                                avg_plot=None, avg_symbol_kwargs=None, avg_hist_kwargs=None,
                                sample_transform=None, min_size=None,
                                show_legend=True, return_val=False):
        """
        Plot pole positions for one or more clusters, on a stereographic
        (Wulff) or equal-area (Schmidt) net.

        Pixel orientations and average orientations are controlled
        INDEPENDENTLY: each can be shown as symbols (scatter), a density
        histogram, both, or omitted entirely -- with independent style.

        Parameters
        ----------
        cluster_id : int, list of int, or None/'all'
            Cluster label(s). None or 'all' selects every cluster in
            `phase` (min_size-filtered), sorted by pixel size descending.
        uvw : array-like (3,)
            Crystal direction/pole to plot.
        phase : str, optional
            Required if cluster_id is None/'all'. Otherwise inferred via
            self._getMask.
        roi : optional
            ROI index, passed to self._getMask.
        equalarea : bool
            Equal-area (Schmidt) vs. stereographic (Wulff) net/projection.
        hemi : {'upper', 'lower'}
            Which hemisphere to plot.
        fig, ax : matplotlib Figure, Axes, optional
            Existing fig/ax. If ax is None, a new one is created (net drawn
            once via wulffnet/schmidtnet).

        pixel_plot : {None, 'symbols', 'histogram', 'both'}, optional
            How to show individual PIXEL orientations. Default 'symbols'.
        pixel_pooled : bool, optional
            Only matters when multiple clusters are selected. If False
            (default), each cluster is plotted separately with its own
            auto-generated color/colormap (one histogram per cluster if
            pixel_plot involves 'histogram'). If True, ALL selected
            clusters' pixels are POOLED into a single combined scatter/
            histogram with one shared color/colormap -- i.e. "orientations
            of all pixels of all clusters" as one dataset.
        pixel_symbol_kwargs : dict, optional
            Style for pixel symbols (used if pixel_plot in
            {'symbols','both'}). Defaults: {'color': None, 's': 2,
            'alpha': 0.5}. 'color' is ignored (auto per-cluster instead)
            when pixel_pooled=False and multiple clusters are selected.
            Extra keys forwarded to ax.scatter.
        pixel_hist_kwargs : dict, optional
            Histogram settings for pixels (used if pixel_plot in
            {'histogram','both'}). Defaults: {'scale': 'sqrt', 'nlevels': 10,
            'bins': 256, 'cmap': None, 'vmax': None}. 'cmap' is ignored
            (auto white->hue per cluster instead) when pixel_pooled=False
            and multiple clusters are selected; default pooled/single-
            cluster cmap is white->red. 'vmax', if given, caps the contour
            level range (see polefigure_plot); default None uses the
            histogram's own max. Extra keys forwarded to ax.contourf.

        avg_plot : {None, 'symbols', 'histogram', 'both'}, optional
            How to show cluster AVERAGE orientations. Default None (off).
        avg_symbol_kwargs : dict, optional
            Style for average-orientation symbols (used if avg_plot in
            {'symbols','both'}). Defaults: {'marker': 'o', 'markersize': 20,
            'markeredgewidth': 2, 'markeredgecolor': 'k',
            'markerfacecolor': 'b', 'alpha': 0.7}. 'markerfacecolor' is
            ignored (auto per-cluster instead) when pixel_pooled=False and
            multiple clusters are selected.
        avg_hist_kwargs : dict, optional
            Histogram settings for average orientations (used if avg_plot
            in {'histogram','both'}). Defaults: {'scale': 'sqrt',
            'nlevels': 10, 'bins': 256, 'cmap': None, 'vmax': None} (default
            cmap white->blue). ALWAYS POOLED across all selected clusters
            into ONE histogram regardless of pixel_pooled (a per-cluster
            histogram of a single average point isn't meaningful), and
            ALWAYS WEIGHTED by each cluster's pixel size (self.cluster_sizes)
            -- so the histogram reflects the population of pixels each
            average represents, not just a count of cluster averages.

        sample_transform : (3,3) array, optional
            Rotation applied to every projected SAMPLE-frame vector (pixel
            and average), after the crystal->sample transform (M.T).
        min_size : int, optional
            Minimum pixel count for a cluster to be included, when
            cluster_id is None/'all' or an explicit list/'all'.
        show_legend : bool, optional
            Show a legend mapping colors to cluster IDs, only shown when
            pixel_pooled=False and multiple clusters are plotted. Default True.
        return_val : bool, optional
            If True, also return a dict with whatever was computed:
            {'pixel': {...}, 'avg': {...}}, keyed by cluster_id when
            per-cluster, or under 'pooled' when pooled. Each pixel/avg
            histogram entry includes both 'hist' (the array) and
            'contourf' (the QuadContourSet artist, for building a colorbar
            via fig.colorbar(entry['contourf'], ax=ax)).

        Returns
        -------
        fig, ax : matplotlib Figure, Axes
        results : dict, only if return_val=True

        Notes
        -----
        self.data.quaternions (via quat_to_mat) give M, sample -> crystal
        (v_crystal = M @ v_sample). Pole figures need crystal -> sample:
        v_sample = M.T @ uvw. sample_transform is applied after that.
        """
        if ax is None:
            if equalarea:
                fig, ax = schmidtnet(ax=None, basedirs=False, facecolor='None')
            else:
                fig, ax = wulffnet(ax=None, basedirs=False, facecolor='None')
        fig = ax.figure

        uvw = np.asarray(uvw, dtype=float)
        sizes = self.cluster_sizes

        if cluster_id is None or (isinstance(cluster_id, str) and cluster_id == 'all'):
            if phase is None:
                raise ValueError("phase is required when cluster_id is None or 'all'")
            cluster_list = sorted(self.labels_by_phase[phase], key=lambda c: -sizes.get(c, 0))
        elif isinstance(cluster_id, (int, np.integer)):
            cluster_list = [cluster_id]
        else:
            cluster_list = sorted(cluster_id, key=lambda c: -sizes.get(c, 0))

        if min_size is not None:
            cluster_list = [c for c in cluster_list if sizes.get(c, 0) >= min_size]
        if not cluster_list:
            raise ValueError("No clusters left after filtering (check phase/min_size).")

        # ---- resolve style defaults ----
        p_sym = {'color': None, 's': 2, 'alpha': 0.5}
        if pixel_symbol_kwargs:
            p_sym.update(pixel_symbol_kwargs)

        p_hist = {'scale': 'sqrt', 'nlevels': 10, 'bins': 256, 'cmap': None, 'vmax': None}
        if pixel_hist_kwargs:
            p_hist.update(pixel_hist_kwargs)

        a_sym = {'marker': 'o', 'markersize': 20, 'markeredgewidth': 2,
                'markeredgecolor': 'k', 'markerfacecolor': 'b', 'alpha': 0.7}
        if avg_symbol_kwargs:
            a_sym.update(avg_symbol_kwargs)

        a_hist = {'scale': 'sqrt', 'nlevels': 10, 'bins': 256, 'cmap': None, 'vmax': None}
        if avg_hist_kwargs:
            a_hist.update(avg_hist_kwargs)

        use_distinct_colors = (not pixel_pooled) and len(cluster_list) > 1
        colors = _generate_distinct_colors(len(cluster_list)) if use_distinct_colors else [None] * len(cluster_list)

        def _project(XYZ):
            if hemi == 'upper':
                XYZ = XYZ[:, XYZ[2, :] >= 0]
            else:
                XYZ = XYZ[:, XYZ[2, :] <= 0]
            if XYZ.shape[1] == 0:
                return np.zeros((2, 0))
            return equalarea_directions(XYZ) if equalarea else stereoprojection_directions(XYZ)

        def _cluster_pixel_XYZ(c):
            mask, resolved_phase = self._getMask(roi=roi, cluster_id=c, phase=phase)
            idxs = np.where(mask)[0]
            if idxs.shape[0] == 0:
                return None, None, None
            pixel_quats = self.data.quaternions[idxs]
            M = np.array([quat_to_mat(q) for q in pixel_quats])
            Minv = M.transpose(0, 2, 1)
            symops = np.array(self.data.phases[resolved_phase]['symops'])
            uvwsb = set(tuple(v) for v in np.round(np.dot(symops, uvw), 8))
            uvwsb = np.asarray(list(uvwsb))
            XYZ = np.tensordot(Minv, uvwsb.T, axes=[[-1], [-2]]).transpose([0, 2, 1])
            XYZ = XYZ.reshape(-1, 3).T
            if sample_transform is not None:
                XYZ = np.asarray(sample_transform) @ XYZ
            return XYZ, uvwsb, idxs.shape[0]

        def _cluster_avg_XYZ(c, uvwsb):
            if c not in self.avg_orientations:
                return None
            avg_M = self.avg_orientations[c]
            XYZavg = (avg_M.T).dot(uvwsb.T).T.T  # (3, n_variants)
            if sample_transform is not None:
                XYZavg = np.asarray(sample_transform) @ XYZavg
            return XYZavg

        results = {'pixel': {}, 'avg': {}}
        legend_handles = []

        # ==================== PIXEL PLOTTING ====================
        if pixel_plot is not None:
            if use_distinct_colors:
                for c, base_color in zip(cluster_list, colors):
                    XYZ, uvwsb, n_px = _cluster_pixel_XYZ(c)
                    if XYZ is None:
                        continue
                    cmap = get_cmap([(1, 1, 1), base_color])
                    if pixel_plot in ('symbols', 'both'):
                        sp = _project(XYZ)
                        style = {k: v for k, v in p_sym.items() if k != 'color'}
                        ax.scatter(sp[0, :], sp[1, :], color=base_color, **style)
                        results['pixel'].setdefault(c, {})['sp'] = sp
                    if pixel_plot in ('histogram', 'both'):
                        hist, sp, extent = polefigure_histogram(
                            XYZ, equalarea=equalarea, scale=p_hist['scale'],
                            bins=p_hist['bins'], hemi=hemi)
                        extra = {k: v for k, v in p_hist.items()
                                if k not in ('scale', 'nlevels', 'bins', 'cmap', 'vmax')}
                        fig, ax, _cs = polefigure_plot(hist, equalarea=equalarea, nlevels=p_hist['nlevels'],
                                                        vmax=p_hist['vmax'], fig=fig, ax=ax, draw_net=False,
                                                        cmap=cmap, **extra)
                        results['pixel'].setdefault(c, {})['hist'] = hist
                        results['pixel'].setdefault(c, {})['contourf'] = _cs
                    legend_handles.append(
                        plt.Line2D([0], [0], marker='o', linestyle='None',
                                markerfacecolor=base_color, markeredgecolor='k',
                                label=f"{c} ({n_px} px)")
                    )
            else:
                XYZ_all = []
                uvwsb_ref = None
                for c in cluster_list:
                    XYZ, uvwsb, n_px = _cluster_pixel_XYZ(c)
                    if XYZ is None:
                        continue
                    XYZ_all.append(XYZ)
                    uvwsb_ref = uvwsb
                if XYZ_all:
                    XYZ_all = np.concatenate(XYZ_all, axis=1)
                    if pixel_plot in ('symbols', 'both'):
                        sp = _project(XYZ_all)
                        ax.scatter(sp[0, :], sp[1, :], **p_sym)
                        results['pixel'].setdefault('pooled', {})['sp'] = sp
                    if pixel_plot in ('histogram', 'both'):
                        cmap = p_hist['cmap'] or get_cmap([(1, 1, 1), (1, 0, 0)])
                        hist, sp, extent = polefigure_histogram(
                            XYZ_all, equalarea=equalarea, scale=p_hist['scale'],
                            bins=p_hist['bins'], hemi=hemi)
                        extra = {k: v for k, v in p_hist.items()
                                if k not in ('scale', 'nlevels', 'bins', 'cmap', 'vmax')}
                        fig, ax, _cs = polefigure_plot(hist, equalarea=equalarea, nlevels=p_hist['nlevels'],
                                                        vmax=p_hist['vmax'], fig=fig, ax=ax, draw_net=False,
                                                        cmap=cmap, **extra)
                        results['pixel'].setdefault('pooled', {})['hist'] = hist
                        results['pixel'].setdefault('pooled', {})['contourf'] = _cs

        # ==================== AVERAGE ORIENTATION PLOTTING ====================
        if avg_plot is not None:
            avg_style_no_face = {k: v for k, v in a_sym.items() if k != 'markerfacecolor'}

            if avg_plot in ('symbols', 'both'):
                for c, base_color in zip(cluster_list, colors):
                    _, uvwsb, _ = _cluster_pixel_XYZ(c)
                    if uvwsb is None:
                        continue
                    XYZavg = _cluster_avg_XYZ(c, uvwsb)
                    if XYZavg is None:
                        continue
                    sp_avg = _project(XYZavg)
                    if sp_avg.shape[1] == 0:
                        continue
                    if use_distinct_colors:
                        ax.plot(sp_avg[0, :], sp_avg[1, :], linestyle='None',
                                markerfacecolor=base_color, **avg_style_no_face)
                    else:
                        ax.plot(sp_avg[0, :], sp_avg[1, :], linestyle='None', **a_sym)
                    results['avg'].setdefault(c, {})['sp'] = sp_avg

            if avg_plot in ('histogram', 'both'):
                # ALWAYS pooled across cluster_list, weighted by pixel size
                XYZavg_all = []
                weights_all = []
                for c in cluster_list:
                    _, uvwsb, _ = _cluster_pixel_XYZ(c)
                    if uvwsb is None:
                        continue
                    XYZavg = _cluster_avg_XYZ(c, uvwsb)
                    if XYZavg is None:
                        continue
                    n_variants = XYZavg.shape[1]
                    XYZavg_all.append(XYZavg)
                    weights_all.append(np.full(n_variants, sizes.get(c, 0), dtype=float))
                if XYZavg_all:
                    XYZavg_all = np.concatenate(XYZavg_all, axis=1)
                    weights_all = np.concatenate(weights_all)
                    cmap = a_hist['cmap'] or get_cmap([(1, 1, 1), (0, 0, 1)])
                    hist, sp, extent = polefigure_histogram(
                        XYZavg_all, weights=weights_all, equalarea=equalarea,
                        scale=a_hist['scale'], bins=a_hist['bins'], hemi=hemi)
                    extra = {k: v for k, v in a_hist.items()
                            if k not in ('scale', 'nlevels', 'bins', 'cmap', 'vmax')}
                    fig, ax, _cs = polefigure_plot(hist, equalarea=equalarea, nlevels=a_hist['nlevels'],
                                                    vmax=a_hist['vmax'], fig=fig, ax=ax, draw_net=False,
                                                    cmap=cmap, **extra)
                    results['avg']['pooled_hist'] = hist
                    results['avg']['pooled_contourf'] = _cs

        if show_legend and use_distinct_colors and legend_handles:
            ax.legend(handles=legend_handles, loc='upper left',
                    bbox_to_anchor=(1.02, 1.0), fontsize=8, title='Cluster (size)')

        n_clusters = len(cluster_list)
        total_px = sum(sizes.get(c, 0) for c in cluster_list)
        new_title = (f"Pole figure {tuple(int(v) for v in np.round(uvw))} "
                    f"— {n_clusters} cluster(s), {total_px} px")
        existing_title = ax.get_title()
        ax.set_title(f"{existing_title}\n{new_title}" if existing_title else new_title)

        return (fig, ax, results) if return_val else (fig, ax)

    def symmetrize_to_reference_ini(self, reference, cluster_ids, phase=None,
                                ref_lattice_direction=None,
                                return_clustering_result=False):
        """
        Re-express pixel orientations (and average orientations) of the
        given clusters using a SINGLE, per-cluster symmetry operation that
        brings each cluster's own (already self-consistent) average
        orientation closest to a fixed reference orientation.

        Unlike resolving each pixel independently against the reference
        (which can split a single physically coherent cluster's pixels
        across multiple different symmetric branches when the reference
        sits near-equidistant between two candidate branches -- confirmed
        empirically: one real dataset showed a cluster's 1408 pixels split
        across 6 different symops despite an internal pixel-to-pixel spread
        of only ~6 deg to its own orientation), this finds ONE symop per
        cluster (via the cluster's converged average, self.avg_orientations)
        and applies it uniformly to every pixel in that cluster. This
        guarantees the relabeling can never mix branches within a single
        cluster.

        Since this only ever applies a single rigid relabeling per cluster,
        ALL misorientation-derived quantities between any two of the
        processed clusters (pole coincidences, disorientation angles, twin
        checks) are mathematically guaranteed unchanged -- only the
        reference frame used for display shifts.

        Does NOT modify self.data.quaternions or self.avg_orientations in
        place -- returns a new dict (or, if return_clustering_result=True,
        a new ClusteringResult with a shallow-copied self.data whose
        quaternions array has been overwritten only for the affected
        pixels).

        Parameters
        ----------
        reference : int, (3,3) array, or (4,) array
            Fixed reference orientation. If int, treated as a cluster ID
            and self.avg_orientations[reference] is used. If (3,3), used
            directly as an orientation matrix. If (4,), used directly as a
            quaternion.
        cluster_ids : int, list of int, or 'all'
            Cluster(s) to resolve relative to `reference`. If 'all', uses
            every cluster in `phase` (phase then required).
        phase : str, optional
            Phase for symops lookup. If None, inferred from the first
            cluster via self.cluster_phases_id. All clusters in cluster_ids
            should share this phase.
        ref_lattice_direction : array-like (3,), optional
            If given, the REFERENCE orientation is first replaced by the
            symmetrically-equivalent variant equiRefOri such that
            equiRefOri.T @ ref_lattice_direction (i.e. this crystal-frame
            direction expressed in the SAMPLE frame) is closest to sample
            [0,0,1] (ND). All symmetry-equivalent variants of `reference`
            are searched; the one maximizing the dot product with [0,0,1]
            is selected as the new reference before proceeding. Default
            None (use `reference` as-is).

            Note: ref_lattice_direction is treated as already Cartesian in
            the crystal frame -- for a non-cubic phase where you're
            starting from Miller indices, convert via the phase's structure
            matrix first (ref_lattice_direction =
            self.data.phases[phase]['L'].dot(uvw)), same as
            get_zone_axis_transform does.
        return_clustering_result : bool, optional
            If False (default), return the plain result dict. If True,
            instead return a NEW ClusteringResult whose pixel quaternions
            (for the affected pixels only) and avg_orientations/avg_quats
            (for the affected clusters only) reflect the symmetrized
            values. self and self.data are left untouched -- both self and
            self.data are shallow-copied, with self.data's underlying
            quaternion array (self.data._quaternions, since `quaternions`
            is a read-only property) specifically deep-copied before the
            affected pixels are overwritten.

        Returns
        -------
        result : dict {cluster_id: dict}, if return_clustering_result=False
            'idxs' : pixel indices into self.data arrays
            'pixel_quats', 'pixel_mats' : symmetrized pixel orientations
            'avg_quat', 'avg_mat' : symmetrized cluster average
            'symop_idx' : index into this phase's symops of the single
                operation applied to this cluster
        new_result : ClusteringResult, if return_clustering_result=True
        """
        import copy
        from scipy.spatial.transform import Rotation as _R

        if isinstance(cluster_ids, str) and cluster_ids == 'all':
            if phase is None:
                raise ValueError("phase is required when cluster_ids='all'")
            cluster_ids = list(self.labels_by_phase[phase])
        elif isinstance(cluster_ids, (int, np.integer)):
            cluster_ids = [cluster_ids]

        if phase is None:
            phase = self.data.phase_names[self.cluster_phases_id[cluster_ids[0]]]
        symops = np.array(self.data.phases[phase]['symops'])

        if isinstance(reference, (int, np.integer)):
            if reference not in self.avg_orientations:
                raise ValueError(f"No average orientation available for reference cluster {reference}")
            ref_mat = self.avg_orientations[reference]
        else:
            reference = np.asarray(reference)
            if reference.shape == (3, 3):
                ref_mat = reference
            elif reference.shape == (4,):
                ref_mat = quat_to_mat(reference)
            else:
                raise ValueError(f"reference must be a cluster id, (3,3) matrix, or (4,) quaternion, "
                                f"got shape {reference.shape}")

        if ref_lattice_direction is not None:
            v = np.asarray(ref_lattice_direction, dtype=float)
            v = v / np.linalg.norm(v)
            equi_mats = np.einsum('sij,jk->sik', symops, ref_mat)  # (Ns,3,3), S @ ref_mat
            sample_dirs = np.einsum('sji,j->si', equi_mats, v)     # (M.T @ v) per variant
            target = np.array([0.0, 0.0, 1.0])
            dots = np.clip(sample_dirs @ target, -1.0, 1.0)
            ref_mat = equi_mats[np.argmax(dots)]

        result = {}
        for c in cluster_ids:
            idxs = np.where(self.labels == c)[0]
            if idxs.shape[0] == 0 or c not in self.avg_orientations:
                continue

            M_c = self.avg_orientations[c]

            # find the SINGLE symop S minimizing disorientation between
            # S @ M_c and ref_mat -- applied uniformly to the whole cluster
            candidates = np.einsum('sij,jk->sik', symops, M_c)   # (Ns,3,3), S @ M_c
            rel = np.einsum('sij,kj->sik', candidates, ref_mat)   # candidates @ ref_mat.T
            q = _R.from_matrix(rel).as_quat()
            w = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
            ang = 2.0 * np.arccos(w)
            best_s = int(np.argmin(ang))
            S = symops[best_s]

            new_avg_mat = S.dot(M_c)

            pixel_mats_old = np.array([quat_to_mat(qi) for qi in self.data.quaternions[idxs]])
            pixel_mats_new = np.einsum('ij,njk->nik', S, pixel_mats_old)
            pixel_quats_new = np.array([mat_to_quat(m) for m in pixel_mats_new])

            result[c] = {
                'idxs': idxs,
                'pixel_quats': pixel_quats_new,
                'pixel_mats': pixel_mats_new,
                'avg_quat': mat_to_quat(new_avg_mat),
                'avg_mat': new_avg_mat,
                'symop_idx': best_s,
            }

        if not return_clustering_result:
            return result

        new_data = copy.copy(self.data)
        new_quaternions = self.data.quaternions.copy()
        for c, r in result.items():
            new_quaternions[r['idxs']] = r['pixel_quats']
        new_data._quaternions = new_quaternions

        new_result = copy.copy(self)
        new_result.data = new_data
        new_result.avg_orientations = dict(self.avg_orientations)
        new_result.avg_quats = dict(self.avg_quats)
        for c, r in result.items():
            new_result.avg_orientations[c] = r['avg_mat']
            new_result.avg_quats[c] = r['avg_quat']

        return new_result

    def symmetrize_clusters(self, cluster_ids='all', phase=None,
                            max_iter=10, tol=1e-6,
                            ref_lattice_direction=None, reference_cluster_id=None,
                            return_clustering_result=False):
        """
        ... (same docstring as before, plus a note:)

        NOTE: the branch-search loops below (when ref_lattice_direction is
        given) are restricted to PROPER rotations only (det > 0), using
        symops_proper -- a filtered copy of the phase's symops, NOT a
        modification of the original. This is required because the search
        result gets applied to M_mean/M_best_cluster and then converted via
        mat_to_quat, which assumes a proper rotation matrix; feeding it an
        improper (det=-1, reflection) matrix silently produces a garbage
        quaternion with no error raised -- confirmed empirically: every
        pixel in one affected cluster showed ~20 deg of round-trip
        corruption despite the pre-conversion matrix math being correct
        (~1.8 deg). The get_avg_orientations call below is UNCHANGED and
        still receives the full (proper+improper) symops set, since it
        already handles that correctly and other code (e.g. disorimat)
        depends on self.data.phases[phase]['symops'] containing the full set.
        """
        import copy

        if isinstance(cluster_ids, str) and cluster_ids == 'all':
            if phase is None:
                raise ValueError("phase is required when cluster_ids='all'")
            cluster_ids = list(self.labels_by_phase[phase])
        elif isinstance(cluster_ids, (int, np.integer)):
            cluster_ids = [cluster_ids]

        if phase is None:
            phase = self.data.phase_names[self.cluster_phases_id[cluster_ids[0]]]
        symops = np.array(self.data.phases[phase]['symops'])  # full set, unchanged, for get_avg_orientations

        dets = np.array([np.linalg.det(s) for s in symops])
        symops_proper = symops[dets > 0]  # local filtered copy, for the search loops only

        v = None
        if ref_lattice_direction is not None:
            L = np.asarray(self.data.phases[phase]['L'])
            uvw = np.asarray(ref_lattice_direction, dtype=float)
            v = L.dot(uvw)
            v = v / np.linalg.norm(v)

        v_ref = None
        ref_entry = None
        if reference_cluster_id is not None:
            if v is None:
                raise ValueError("reference_cluster_id requires ref_lattice_direction to also be given")
            ref_idxs = np.where(self.labels == reference_cluster_id)[0]
            if ref_idxs.shape[0] == 0:
                raise ValueError(f"reference_cluster_id {reference_cluster_id} has no pixels")

            q_mean_ref, M_mean_ref, M_best_cluster_ref, q_best_cluster_ref = get_avg_orientations(
                self.data.quaternions[ref_idxs], symops, ref_idx=0,
                max_iter=max_iter, tol=tol
            )
            v_ref = M_mean_ref.T.dot(v)
            ref_entry = {
                'idxs': ref_idxs,
                'avg_mat': M_mean_ref,
                'avg_quat': q_mean_ref,
                'pixel_mats': M_best_cluster_ref,
                'pixel_quats': q_best_cluster_ref,
            }

        sizes = self.cluster_sizes
        cluster_ids = [c for c in cluster_ids if c != reference_cluster_id]
        cluster_ids = sorted(cluster_ids, key=lambda c: -sizes.get(c, 0))

        result = {}
        best_sym_ref = None

        for c in cluster_ids:
            idxs = np.where(self.labels == c)[0]
            if idxs.shape[0] == 0:
                continue

            q_mean, M_mean, M_best_cluster, q_best_cluster = get_avg_orientations(
                self.data.quaternions[idxs], symops, ref_idx=0,
                max_iter=max_iter, tol=tol
            )

            if v is not None and v_ref is not None:
                best_dot = -np.inf
                best_sym = None

                if c == cluster_ids[0]:
                    for sym_ref in symops_proper:
                        target = M_mean_ref.T.dot(sym_ref.dot(v))
                        for sym in symops_proper:
                            v_c = M_mean.T.dot(sym.dot(v))
                            dot = abs(v_c.dot(target))
                            if dot > best_dot:
                                best_dot = dot
                                best_sym = sym
                                best_sym_ref = sym_ref
                else:
                    target = M_mean_ref.T.dot(best_sym_ref.dot(v))
                    for sym in symops_proper:
                        v_c = M_mean.T.dot(sym.dot(v))
                        dot = abs(v_c.dot(target))
                        if dot > best_dot:
                            best_dot = dot
                            best_sym = sym

                M_mean = best_sym.T.dot(M_mean)
                q_mean = mat_to_quat(M_mean)
                M_best_cluster = np.einsum('ij,njk->nik', best_sym.T, M_best_cluster)
                q_best_cluster = np.array([mat_to_quat(m) for m in M_best_cluster])

            result[c] = {
                'idxs': idxs,
                'pixel_quats': q_best_cluster,
                'pixel_mats': M_best_cluster,
                'avg_quat': q_mean,
                'avg_mat': M_mean,
            }

        if ref_entry is not None and best_sym_ref is not None:
            M_mean_ref_new = best_sym_ref.T.dot(ref_entry['avg_mat'])
            q_mean_ref_new = mat_to_quat(M_mean_ref_new)
            M_best_ref_new = np.einsum('ij,njk->nik', best_sym_ref.T, ref_entry['pixel_mats'])
            q_best_ref_new = np.array([mat_to_quat(m) for m in M_best_ref_new])

            result[reference_cluster_id] = {
                'idxs': ref_entry['idxs'],
                'pixel_quats': q_best_ref_new,
                'pixel_mats': M_best_ref_new,
                'avg_quat': q_mean_ref_new,
                'avg_mat': M_mean_ref_new,
            }
        elif ref_entry is not None:
            result[reference_cluster_id] = {
                'idxs': ref_entry['idxs'],
                'pixel_quats': ref_entry['pixel_quats'],
                'pixel_mats': ref_entry['pixel_mats'],
                'avg_quat': ref_entry['avg_quat'],
                'avg_mat': ref_entry['avg_mat'],
            }

        if not return_clustering_result:
            return result

        new_data = copy.copy(self.data)
        new_quaternions = self.data.quaternions.copy()
        for c, r in result.items():
            new_quaternions[r['idxs']] = r['pixel_quats']
        new_data._quaternions = new_quaternions

        new_result = copy.copy(self)
        new_result.data = new_data
        new_result.avg_orientations = dict(self.avg_orientations)
        new_result.avg_quats = dict(self.avg_quats)
        for c, r in result.items():
            new_result.avg_orientations[c] = r['avg_mat']
            new_result.avg_quats[c] = r['avg_quat']

        return new_result    


    def get_symmetrize_to_reference_oris(self, reference, cluster_ids, phase=None, max_iter=1, tol=1e-6):
        """
        ... (same docstring, plus in the cluster_ids entry:)

        cluster_ids : int, list of int, or 'all'
            Cluster(s) to resolve relative to `reference`. If 'all', uses
            every cluster in `phase` (phase is then required).
        """
        if isinstance(cluster_ids, str) and cluster_ids == 'all':
            if phase is None:
                raise ValueError("phase is required when cluster_ids='all'")
            cluster_ids = list(self.labels_by_phase[phase])
        elif isinstance(cluster_ids, (int, np.integer)):
            cluster_ids = [cluster_ids]

        if phase is None:
            phase = self.data.phase_names[self.cluster_phases_id[cluster_ids[0]]]
        symops = np.array(self.data.phases[phase]['symops'])

        if isinstance(reference, (int, np.integer)):
            if reference not in self.avg_orientations:
                raise ValueError(f"No average orientation available for reference cluster {reference}")
            ref_quat = self.avg_quats[reference]
        else:
            reference = np.asarray(reference)
            if reference.shape == (3, 3):
                ref_quat = mat_to_quat(reference)
            elif reference.shape == (4,):
                ref_quat = reference
            else:
                raise ValueError(f"reference must be a cluster id, (3,3) matrix, or (4,) quaternion, "
                                f"got shape {reference.shape}")

        result = {}
        for c in cluster_ids:
            idxs = np.where(self.labels == c)[0]
            if idxs.shape[0] == 0:
                continue

            q_mean, M_mean, M_best_cluster, q_best_cluster = get_avg_orientations(
                self.data.quaternions[idxs], symops, ref_idx=0,
                max_iter=max_iter, tol=tol, q_ref=ref_quat
            )

            result[c] = {
                'idxs': idxs,
                'pixel_quats': q_best_cluster,
                'pixel_mats': M_best_cluster,
                'avg_quat': q_mean,
                'avg_mat': M_mean,
            }

        return result

    def get_cluster_pixel_orientations(self, cluster_id):
        """
        Get individual pixel orientations for a specific cluster, alongside
        the cluster's average orientation.

        Parameters
        ----------
        cluster_id : int
            Cluster label

        Returns
        -------
        result : dict
            'cluster_id' : int
            'idxs' : (n_pixels,) ndarray
                Indices into self.data arrays for pixels in this cluster.
            'X', 'Y' : (n_pixels,) ndarray
                Pixel coordinates.
            'pixel_quats' : (n_pixels, 4) ndarray
                Individual per-pixel orientation quaternions.
            'pixel_mats' : (n_pixels, 3, 3) ndarray
                Individual per-pixel orientation matrices.
            'avg_quat' : (4,) ndarray or None
                Cluster average orientation, quaternion form.
            'avg_mat' : (3, 3) ndarray or None
                Cluster average orientation, matrix form.
            'phase_id' : int
            'phase_name' : str
        """
        if cluster_id not in self.get_unique_clusters():
            raise ValueError(f"Cluster {cluster_id} not found")

        idxs = np.where(self.labels == cluster_id)[0]
        pixel_quats = self.data.quaternions[idxs]
        pixel_mats = np.array([quat_to_mat(q) for q in pixel_quats])

        phase_id = self.cluster_phases_id[cluster_id]

        return {
            'cluster_id': cluster_id,
            'idxs': idxs,
            'X': self.data.X[idxs],
            'Y': self.data.Y[idxs],
            'pixel_quats': pixel_quats,
            'pixel_mats': pixel_mats,
            'avg_quat': self.avg_quats.get(cluster_id),
            'avg_mat': self.avg_orientations.get(cluster_id),
            'phase_id': phase_id,
            'phase_name': self.data.phase_names[phase_id],
        }

    def get_zone_axis_transform(self, cluster_id, uvw=(0, 0, 1), second_uvw=None, phase=None):
        """
        Build a sample_transform matrix for plot_cluster_pole_figure that
        aligns the pole-figure viewing axis (always +z) EXACTLY with a
        given zone axis [uvw] of a reference cluster's crystal, and
        OPTIONALLY also fixes the remaining azimuthal "twist" around that
        axis by aligning a second zone axis [second_uvw] as closely as
        possible to sample +x.

        Composing this as sample_transform = R_twist @ R_align @ M1
        (M1 = reference cluster's avg_orientations, sample->crystal) means:
        - The reference cluster's own [uvw] direction lands EXACTLY on the
        pole figure's viewing axis (z).
        - If second_uvw is given, the reference cluster's own [second_uvw]
        direction lands with its in-plane (xy) component pointing along
        +x -- i.e. this fully and exactly constrains the reference's
        orientation in the plotted frame (up to the residual ambiguity of
        [second_uvw] having a component along z itself, which is
        unavoidable and only matters if the two directions are close to
        parallel).
        - Without second_uvw, aligning [uvw] to z alone leaves that
        azimuthal twist UNCONSTRAINED -- the minimal-rotation formula used
        picks an arbitrary one, which is why a second target you also
        care about (e.g. from ref_lattice_directions in
        symmetrize_to_reference) can end up nowhere near where you'd
        expect even though the primary [uvw] axis is correctly centered.

        Unlike symmetrize_to_reference's ref_lattice_directions (which
        picks the closest DISCRETE crystal-symmetry-equivalent branch, so
        alignment is only approximate, limited by how close a symmetric
        variant happens to be to the target), this applies a CONTINUOUS
        rotation, so both directions land exactly at their targets (as
        exactly as floating point allows) regardless of how close any
        symmetric branch happens to be.

        Parameters
        ----------
        cluster_id : int
            Reference cluster whose crystal frame defines the zone axis(es).
        uvw : array-like (3,), optional
            Zone axis in Miller-index (real-space lattice) notation, in the
            reference cluster's crystal frame, aligned EXACTLY to sample z.
            Default (0,0,1).
        second_uvw : array-like (3,), optional
            A second zone axis, also in Miller-index notation. If given,
            an additional rotation about the now-fixed z axis is applied so
            that this direction's in-plane (xy) component points along
            sample +x. Default None (no twist-fixing; z-alignment only,
            original behavior).
        phase : str, optional
            Phase of the reference cluster, to look up L. If None, inferred
            from cluster_id via self.cluster_phases_id.

        Returns
        -------
        sample_transform : (3,3) ndarray
            Pass directly as sample_transform to plot_cluster_pole_figure.
        """
        from scipy.spatial.transform import Rotation as _R

        if cluster_id not in self.avg_orientations:
            raise ValueError(f"No average orientation available for cluster {cluster_id}")

        if phase is None:
            phase = self.data.phase_names[self.cluster_phases_id[cluster_id]]

        L = np.asarray(self.data.phases[phase]['L'])
        uvw = np.asarray(uvw, dtype=float)

        zone_axis = L.dot(uvw)
        norm = np.linalg.norm(zone_axis)
        if norm < 1e-12:
            raise ValueError(f"Zone axis {uvw} maps to a near-zero vector under L for phase {phase}")
        zone_axis = zone_axis / norm

        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(zone_axis, z)
        axis_norm = np.linalg.norm(axis)
        dot = np.clip(zone_axis @ z, -1.0, 1.0)

        if axis_norm < 1e-8:
            if dot > 0:
                R_align = np.eye(3)
            else:
                R_align = _R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
        else:
            angle = np.arccos(dot)
            R_align = _R.from_rotvec(axis / axis_norm * angle).as_matrix()

        M1 = self.avg_orientations[cluster_id]
        transform = R_align @ M1

        if second_uvw is not None:
            second_uvw = np.asarray(second_uvw, dtype=float)
            second_axis = L.dot(second_uvw)
            second_norm = np.linalg.norm(second_axis)
            if second_norm < 1e-12:
                raise ValueError(f"second_uvw {second_uvw} maps to a near-zero vector under L for phase {phase}")
            second_axis = second_axis / second_norm

            # where second_axis's pole currently lands, after z-alignment
            # but before twist-fixing: crystal -> sample (M1.T) -> current
            # display frame (transform)
            w = transform.dot(M1.T.dot(second_axis))

            w_proj = w - z * w[2]
            w_proj_norm = np.linalg.norm(w_proj)
            if w_proj_norm < 1e-8:
                print(f"Warning: second_uvw {second_uvw} is (nearly) parallel to uvw {uvw} "
                        f"after z-alignment -- cannot fix azimuthal twist from a direction with "
                        f"no in-plane component. Returning z-alignment only.")
            else:
                current_angle = np.arctan2(w_proj[1], w_proj[0])
                R_twist = _R.from_rotvec(-current_angle * z).as_matrix()
                transform = R_twist @ transform
        return transform

    def check_twin_relationship(self, cluster_a, cluster_b, system, mode,
                                tol_deg=5.0, symmetric=True):
        """
        Check whether the orientation relationship between two clusters'
        average orientations corresponds to a known twinning mode, via the
        rotation axis/angle of the relative misorientation compared against
        the mode's tabulated crystallographic vectors.

        Convention: M = G2 @ G1.T (G1, G2 = self.avg_orientations for
        cluster_a, cluster_b) transforms a vector expressed in cluster_a's
        crystal frame into cluster_b's crystal frame, for the same physical
        (sample-frame) vector. Its rotation axis/angle (axis taken in
        [0,180] via a sign-consistent quaternion, so axis is unique up to
        the natural line ambiguity of a rotation axis) is compared against
        THREE independent conditions, ANY ONE of which is sufficient to
        report a match:

        1. axis ~ K2_a and angle ~ shear_angle (both within tol_deg)
        -- rotation about the twinning shear-plane normal K2, by the
        characteristic shear angle.
        2. axis ~ K1_a and angle ~ 180 deg (within tol_deg)
        -- type I twin: 180 deg rotation about the twin plane normal.
        3. axis ~ eta1_a and angle ~ 180 deg (within tol_deg)
        -- type II twin: 180 deg rotation about the shear direction.

        K1_a, K2_a, eta1_a are each lists of 12 crystallographically
        equivalent vectors (matching the 12 equivalent C_a variants of the
        original correspondence-matrix formulation this replaces) -- for
        each condition, the extracted rotation axis is compared against ALL
        12 variants and the closest one is used (axis_dev = minimum over
        variants), with the matching variant index recorded.

        Axis comparison is SIGN-INDEPENDENT (|axis . target| used), treating
        both the extracted rotation axis and the stored crystallographic
        vectors as undirected lines -- appropriate for the two 180 deg
        conditions (where R(n,180)=R(-n,180) exactly) and assumed also
        appropriate for K2_a/shear_angle unless K2_a's sign is meaningful in
        your convention, in which case this would need adjusting.

        NOTE: no additional phase-symmetry search is applied to G1/G2 (or to
        M) before extracting axis/angle -- this compares the SPECIFIC
        symmetric representative that G1, G2 happen to be stored as. If
        average orientations weren't resolved to a common symmetric branch
        beforehand (e.g. via symmetrize_to_reference), a true match could be
        missed because a symmetrically-equivalent rotation would satisfy a
        condition but the untransformed one doesn't.

        Parameters
        ----------
        cluster_a, cluster_b : int
            Cluster IDs. cluster_a is treated as the "matrix" (parent),
            cluster_b as the "twin". Both should be the same phase --
            twin relationships are only meaningful within one phase.
        system : str
            Alloy/system key, e.g. 'NiTi'.
        mode : str
            Twin mode key, e.g. '114'. Looks up
            self.data.twinSys[system][mode]['K1_a'], ['K2_a'], ['eta1_a']
            (each a list of 12 equivalent (3,) vectors), and
            ['shear_angle'] (scalar, degrees). Call getTwinSys() first if
            self.data.twinSys hasn't been populated yet.
        tol_deg : float, optional
            Deviation angle below which BOTH axis and angle must fall for a
            condition to count as a match. Default 5.0 degrees.
        symmetric : bool, optional
            If True (default), also test the reverse direction (G1 as twin,
            G2 as matrix). Direction/condition selection logic:
            - If a match exists in only one direction, that direction is
            used, regardless of raw deviation scores.
            - If both directions match, the direction with the smaller
            combined (axis_dev + angle_dev) for its best condition is
            used.
            - If neither direction matches, the direction with the smaller
            combined deviation is used purely for diagnostic reporting
            (angle_dev/axis_dev of the closest attempt); is_match stays
            False either way.
            This ensures a genuine match is never discarded in favor of a
            numerically closer non-match in the other direction.

        Returns
        -------
        result : dict
            'cluster_a', 'cluster_b' : IDs, reordered so cluster_a is
                whichever role gave the selected direction's "matrix" side
            'system', 'mode' : as given
            'is_match' : bool
                True if ANY of the three conditions matched (within
                tol_deg) in the selected direction.
            'best_condition' : str
                Name of the best-matching condition for the selected
                direction ('K2_shear', 'K1_180', or 'eta1_180'), by
                smallest combined (axis_dev + angle_dev).
            'axis_dev_deg', 'angle_dev_deg' : float
                Deviations for best_condition.
            'checks' : dict {condition_name: dict}
                All three conditions' results for the selected direction,
                each with 'axis_dev_deg', 'angle_dev_deg', 'is_match',
                'variant_idx' (which of the 12 equivalent target vectors
                gave the smallest axis deviation).
            'axis' : (3,) ndarray
                Extracted rotation axis (unit vector) for the selected
                direction.
            'angle_deg' : float
                Extracted rotation angle for the selected direction.
            'swapped' : bool
                True if the reverse direction (G1 as twin, G2 as matrix)
                was selected.
        """
        from scipy.spatial.transform import Rotation as _R

        if cluster_a not in self.avg_orientations or cluster_b not in self.avg_orientations:
            raise ValueError(f"Average orientation not available for cluster {cluster_a} or {cluster_b}")

        phase_a = self.data.phase_names[self.cluster_phases_id[cluster_a]]
        phase_b = self.data.phase_names[self.cluster_phases_id[cluster_b]]
        if phase_a != phase_b:
            print(f"Warning: cluster {cluster_a} (phase {phase_a}) and cluster {cluster_b} "
                f"(phase {phase_b}) are different phases; twin relationships are only "
                f"meaningful within a single phase.")

        G1 = self.avg_orientations[cluster_a]
        G2 = self.avg_orientations[cluster_b]

        twin_data = self.data.twinSys[system][mode]
        K1_a = np.asarray(twin_data['K1_a'], dtype=float)      # (12, 3)
        K2_a = np.asarray(twin_data['K2_a'], dtype=float)      # (12, 3)
        eta1_a = np.asarray(twin_data['eta1_a'], dtype=float)  # (12, 3)
        shear_angle = float(twin_data['shear_angle'])

        def _axis_angle(M):
            q = _R.from_matrix(M).as_quat()  # [x, y, z, w]
            if q[3] < 0:
                q = -q
            w = np.clip(q[3], -1.0, 1.0)
            angle = 2.0 * np.degrees(np.arccos(w))
            s = np.sqrt(max(1.0 - w * w, 0.0))
            if s < 1e-8:
                axis = np.array([0.0, 0.0, 1.0])  # angle ~0, axis undefined; arbitrary
            else:
                axis = q[:3] / s
            return axis, angle

        def _condition_dev(axis, angle, target_axes, target_angle):
            T = np.asarray(target_axes, dtype=float)
            T = T / np.linalg.norm(T, axis=1, keepdims=True)
            cosang = np.clip(np.abs(T @ axis), -1.0, 1.0)  # (12,)
            axis_devs = np.degrees(np.arccos(cosang))
            variant_idx = int(np.argmin(axis_devs))
            axis_dev = float(axis_devs[variant_idx])
            angle_dev = abs(angle - target_angle)
            return axis_dev, angle_dev, variant_idx

        def _evaluate(M):
            axis, angle = _axis_angle(M)
            checks = {}
            for name, (t_axes, t_angle) in {
                'K2_shear': (K2_a, shear_angle),
                'K1_180': (K1_a, 180.0),
                'eta1_180': (eta1_a, 180.0),
            }.items():
                axis_dev, angle_dev, variant_idx = _condition_dev(axis, angle, t_axes, t_angle)
                checks[name] = {
                    'axis_dev_deg': axis_dev,
                    'angle_dev_deg': angle_dev,
                    'is_match': (axis_dev < tol_deg) and (angle_dev < tol_deg),
                    'variant_idx': variant_idx,
                }
            best_name = min(checks, key=lambda k: checks[k]['axis_dev_deg'] + checks[k]['angle_dev_deg'])
            return axis, angle, checks, best_name

        M_fwd = G2 @ G1.T
        axis_fwd, angle_fwd, checks_fwd, best_fwd = _evaluate(M_fwd)
        score_fwd = checks_fwd[best_fwd]['axis_dev_deg'] + checks_fwd[best_fwd]['angle_dev_deg']
        match_fwd = any(v['is_match'] for v in checks_fwd.values())

        axis, angle, checks, best_condition, score = axis_fwd, angle_fwd, checks_fwd, best_fwd, score_fwd
        swapped = False
        is_match = match_fwd

        if symmetric:
            M_rev = G1 @ G2.T
            axis_rev, angle_rev, checks_rev, best_rev = _evaluate(M_rev)
            score_rev = checks_rev[best_rev]['axis_dev_deg'] + checks_rev[best_rev]['angle_dev_deg']
            match_rev = any(v['is_match'] for v in checks_rev.values())

            if match_fwd and match_rev:
                # both match: prefer the tighter fit
                if score_rev < score_fwd:
                    axis, angle, checks, best_condition, score = axis_rev, angle_rev, checks_rev, best_rev, score_rev
                    swapped = True
                is_match = True
            elif match_rev and not match_fwd:
                # only rev matches: take it regardless of raw score
                axis, angle, checks, best_condition, score = axis_rev, angle_rev, checks_rev, best_rev, score_rev
                swapped = True
                is_match = True
            elif not match_rev and not match_fwd:
                # neither matches: fall back to whichever is numerically closer,
                # purely for diagnostic reporting; is_match stays False
                if score_rev < score_fwd:
                    axis, angle, checks, best_condition, score = axis_rev, angle_rev, checks_rev, best_rev, score_rev
                    swapped = True
            # else: match_fwd and not match_rev -> keep fwd as-is (already set above)

        return {
            'cluster_a': cluster_b if swapped else cluster_a,
            'cluster_b': cluster_a if swapped else cluster_b,
            'system': system,
            'mode': mode,
            'is_match': is_match,
            'best_condition': best_condition,
            'axis_dev_deg': checks[best_condition]['axis_dev_deg'],
            'angle_dev_deg': checks[best_condition]['angle_dev_deg'],
            'checks': checks,
            'axis': axis,
            'angle_deg': angle,
            'swapped': swapped,
        }   
    
    def check_twin_relationship_ini(self, cluster_a, cluster_b, system, mode,
                                tol_deg=5.0, symmetric=True):
        """
        Check whether the orientation relationship between two clusters'
        average orientations corresponds to a known twinning mode.

        Convention: M = G2 @ G1.T (G1, G2 = self.avg_orientations for
        cluster_a, cluster_b) transforms a vector expressed in cluster_a's
        crystal frame into cluster_b's crystal frame, for the same physical
        (sample-frame) vector -- the same convention as the stored twin
        matrices C_a = -I + 2*outer(a1_a, a1_a), which transform real
        vectors from the "matrix" (parent) crystal frame to the "twin"
        crystal frame.

        M is compared against all 12 crystallographically equivalent C_a
        variants stored in self.data.twinSys[system][mode]['C_a'] (call
        getTwinSys() first if this hasn't been populated yet). Assumes the
        12 stored variants already span the relevant symmetry equivalents
        -- no additional phase-symmetry search is applied to G1/G2
        themselves here.

        Parameters
        ----------
        cluster_a, cluster_b : int
            Cluster IDs. cluster_a is treated as the "matrix" (parent),
            cluster_b as the "twin". Both should be the same phase --
            twin relationships are only meaningful within one phase.
        system : str
            Alloy/system key, e.g. 'NiTi'.
        mode : str
            Twin mode key, e.g. '114'.
        tol_deg : float, optional
            Deviation angle below which the relationship is reported as a
            match. Default 5.0 degrees.
        symmetric : bool, optional
            If True (default), also test the reverse direction (G1 as twin,
            G2 as matrix) and report whichever direction gives the smaller
            deviation -- useful since which cluster is "matrix" vs "twin"
            is often not known a priori.

        Returns
        -------
        result : dict
            'cluster_a', 'cluster_b' : IDs, reordered so cluster_a is
                whichever role gave the best match's "matrix" side
            'system', 'mode' : as given
            'angle_deg' : float
                Minimum deviation angle (degrees) to the best-matching
                C_a variant.
            'best_variant_idx' : int
                Index (0-11) of the best-matching C_a variant.
            'is_match' : bool
                True if angle_deg < tol_deg.
            'swapped' : bool
                True if the reverse direction gave the better match.
        """
        from scipy.spatial.transform import Rotation as _R

        if cluster_a not in self.avg_orientations or cluster_b not in self.avg_orientations:
            raise ValueError(f"Average orientation not available for cluster {cluster_a} or {cluster_b}")

        phase_a = self.data.phase_names[self.cluster_phases_id[cluster_a]]
        phase_b = self.data.phase_names[self.cluster_phases_id[cluster_b]]
        if phase_a != phase_b:
            print(f"Warning: cluster {cluster_a} (phase {phase_a}) and cluster {cluster_b} "
                f"(phase {phase_b}) are different phases; twin relationships are only "
                f"meaningful within a single phase.")

        G1 = self.avg_orientations[cluster_a]
        G2 = self.avg_orientations[cluster_b]

        C_a_list = np.asarray(self.data.twinSys[system][mode]['C_a'])  # (12, 3, 3)

        def _min_angle(M, C_list):
            rel = np.einsum('ij,njk->nik', M, C_list.transpose(0, 2, 1))  # M @ C.T per variant
            q = _R.from_matrix(rel).as_quat()
            w = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
            ang = np.degrees(2 * np.arccos(w))
            idx = int(np.argmin(ang))
            return float(ang[idx]), idx

        M_fwd = G2 @ G1.T
        angle_fwd, idx_fwd = _min_angle(M_fwd, C_a_list)

        swapped = False
        angle, idx = angle_fwd, idx_fwd

        if symmetric:
            M_rev = G1 @ G2.T
            angle_rev, idx_rev = _min_angle(M_rev, C_a_list)
            if angle_rev < angle_fwd:
                angle, idx = angle_rev, idx_rev
                swapped = True

        return {
            'cluster_a': cluster_b if swapped else cluster_a,
            'cluster_b': cluster_a if swapped else cluster_b,
            'system': system,
            'mode': mode,
            'angle_deg': angle,
            'best_variant_idx': idx,
            'is_match': angle < tol_deg,
            'swapped': swapped,
        }

    def map_twin_relationships(self, phase, system, modes, tol_deg=5.0, min_size=None,
                                cluster_ids=None):
        """
        Check the orientation relationship between EVERY pair of clusters
        (within one phase) against one or more twinning modes, in one
        vectorized pass per mode rather than looping check_twin_relationship
        pair-by-pair.

        Same convention as check_twin_relationship: for clusters a, b with
        average orientations G1, G2, M = G2 @ G1.T is compared against the
        12 crystallographically equivalent C_a variants stored in
        self.data.twinSys[system][mode]['C_a']. Both directions (a-as-matrix
        vs b-as-matrix) are tested; the reverse direction's M is simply
        M.T, so it costs nothing extra to include.

        Parameters
        ----------
        phase : str
            Phase to restrict clusters to. Twin relationships are only
            meaningful within a single phase.
        system : str
            Alloy/system key into self.data.twinSys, e.g. 'NiTi'.
        modes : str or list of str
            Twin mode key(s) to check, e.g. '114' or ['114', 'TypeII'].
            Call getTwinSys() beforehand if self.data.twinSys isn't
            populated yet.
        tol_deg : float, optional
            Deviation angle below which a pair is reported as a match.
            Default 5.0 degrees.
        min_size : int, optional
            Clusters smaller than this (pixels) are excluded entirely.
        cluster_ids : list of int, optional
            Explicit cluster subset to check, overriding phase-based
            selection. Must still all belong to `phase`.

        Returns
        -------
        pairs_results : list of dict
            One entry per cluster pair (i < j in cluster_ids order), each:
            'cluster_a', 'cluster_b' : IDs, reordered so cluster_a is
                whichever role gave the best match's "matrix" side
            'best_mode' : str
                Mode with the smallest deviation angle across all `modes`
                tested for this pair.
            'angle_deg' : float
                That minimum deviation angle.
            'best_variant_idx' : int
                Index (0-11) of the best-matching C_a variant, for best_mode.
            'is_match' : bool
                True if angle_deg < tol_deg.
            'swapped' : bool
                True if the reverse direction gave the better match.
            'angles_by_mode' : dict {mode: angle_deg}
                Per-mode minimum angle, for inspecting near-misses against
                other modes too.
        matches : list of dict
            Subset of pairs_results where is_match is True, sorted by
            angle_deg ascending (convenience view).
        """
        from scipy.spatial.transform import Rotation as _R

        if isinstance(modes, str):
            modes = [modes]

        if cluster_ids is None:
            cluster_ids = list(self.labels_by_phase[phase])
        cluster_ids = [c for c in cluster_ids if c in self.avg_orientations]
        if min_size is not None:
            sizes = self.cluster_sizes
            cluster_ids = [c for c in cluster_ids if sizes.get(c, 0) >= min_size]

        n = len(cluster_ids)
        if n < 2:
            return [], []

        G = np.array([self.avg_orientations[c] for c in cluster_ids])  # (n,3,3)
        pair_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
        P = len(pair_idx)
        ii = np.array([i for i, _ in pair_idx])
        jj = np.array([j for _, j in pair_idx])

        M_fwd = np.einsum('pij,pkj->pik', G[jj], G[ii])  # (P,3,3) = G[j] @ G[i].T
        M_rev = M_fwd.transpose(0, 2, 1)                  # = G[i] @ G[j].T

        def _min_angle_batch(M_batch, C_list):
            # M_batch: (P,3,3), C_list: (12,3,3) -> angle (P,), idx (P,)
            C_T = C_list.transpose(0, 2, 1)
            rel = np.einsum('pij,njk->pnik', M_batch, C_T)      # (P,12,3,3)
            q = _R.from_matrix(rel.reshape(-1, 3, 3)).as_quat().reshape(P, len(C_list), 4)
            w = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
            ang = np.degrees(2 * np.arccos(w))                   # (P,12)
            idx = np.argmin(ang, axis=1)
            return ang[np.arange(P), idx], idx

        # per-mode best angle/variant/direction for every pair
        angles_by_mode = {}    # mode -> (P,) angle
        per_mode_best = {}     # mode -> (angle(P,), idx(P,), swapped(P,) bool)

        for mode in modes:
            C_a_list = np.asarray(self.data.twinSys[system][mode]['C_a'])  # (12,3,3)
            angle_fwd, idx_fwd = _min_angle_batch(M_fwd, C_a_list)
            angle_rev, idx_rev = _min_angle_batch(M_rev, C_a_list)

            swapped = angle_rev < angle_fwd
            angle = np.where(swapped, angle_rev, angle_fwd)
            idx = np.where(swapped, idx_rev, idx_fwd)

            angles_by_mode[mode] = angle
            per_mode_best[mode] = (angle, idx, swapped)

        # across modes, pick the best (smallest angle) per pair
        stacked_angles = np.stack([angles_by_mode[m] for m in modes], axis=1)  # (P, n_modes)
        best_mode_idx = np.argmin(stacked_angles, axis=1)

        pairs_results = []
        for p in range(P):
            m = modes[best_mode_idx[p]]
            angle, idx, swapped = per_mode_best[m]
            cluster_a = cluster_ids[jj[p]] if swapped[p] else cluster_ids[ii[p]]
            cluster_b = cluster_ids[ii[p]] if swapped[p] else cluster_ids[jj[p]]

            pairs_results.append({
                'cluster_a': int(cluster_a),
                'cluster_b': int(cluster_b),
                'best_mode': m,
                'angle_deg': float(angle[p]),
                'best_variant_idx': int(idx[p]),
                'is_match': bool(angle[p] < tol_deg),
                'swapped': bool(swapped[p]),
                'angles_by_mode': {mode: float(angles_by_mode[mode][p]) for mode in modes},
            })

        matches = sorted([r for r in pairs_results if r['is_match']], key=lambda r: r['angle_deg'])

        return pairs_results, matches
    def merge_clusters_by_orientation(self, threshold_deg=2.0, phase=None, min_size=None,
                                        return_disor_matrices=False):
        """
        Merge clusters within the same phase whose average orientations are
        close, with NO spatial/adjacency constraint. Uses your orilib.disorimat
        for the disorientation calculation (full symmetry group, including
        improper operations, handled internally by disorimat — pass the
        complete symops set as stored in self.data.phases[phase]['symops'],
        NOT pre-filtered to proper rotations only).

        Uses greedy COMPLETE-linkage grouping: a cluster only joins a group if
        its disorientation to EVERY existing member is below threshold_deg,
        so all pairs within a merged group are guaranteed below threshold
        (unlike single-linkage/union-find chaining).

        Actually relabels self.labels, so downstream analysis transparently
        sees merged super-clusters.

        Parameters
        ----------
        threshold_deg : float
            Disorientation threshold (degrees). Default 2.0.
        phase : str or list of str, optional
            Phase(s) to merge within. Default: all phases in
            self.labels_by_phase. Never merges across phases.
        min_size : int, optional
            Clusters smaller than this (pixels) are excluded from merge
            consideration and kept as-is.
        return_disor_matrices : bool, optional
            If True, also return the NxN disorientation matrices used for
            grouping (one per phase processed).

        Returns
        -------
        new_result : ClusteringResult
        merge_groups : dict {phase: {root_id: [fragment cluster ids]}}
        disor_matrices : dict {phase: (cluster_ids, matrix)}, only if
            return_disor_matrices=True.
        """
        import copy

        if phase is None:
            phases_to_process = list(self.labels_by_phase.keys())
        elif isinstance(phase, str):
            phases_to_process = [phase]
        else:
            phases_to_process = list(phase)

        if not hasattr(self, 'avg_orientations') or self.avg_orientations is None:
            self.getAvgOri()

        sizes = self.cluster_sizes
        new_labels = self.labels.copy()
        merge_groups_all = {}
        disor_matrices = {}

        for ph in phases_to_process:
            cluster_ids = [c for c in self.labels_by_phase[ph] if c in self.avg_orientations]
            if min_size is not None:
                cluster_ids = [c for c in cluster_ids if sizes.get(c, 0) >= min_size]

            if len(cluster_ids) < 2:
                merge_groups_all[ph] = {int(c): [int(c)] for c in cluster_ids}
                if return_disor_matrices:
                    disor_matrices[ph] = (cluster_ids, np.zeros((len(cluster_ids),) * 2))
                continue

            # disorimat extends symops with improper operations internally,
            # so pass the full stored symmetry set as-is (not filtered to
            # proper rotations), and as a plain list (it does .append()
            # internally).
            symops = list(np.array(self.data.phases[ph]['symops']))

            M = np.array([self.avg_orientations[c] for c in cluster_ids])
            n = M.shape[0]

            # --- build NxN disorientation matrix using disorimat directly ---
            disor_matrix = np.asarray(disorimat(M, symops), dtype=float)
            np.fill_diagonal(disor_matrix, 0.0)  # numerical safety; diagonal should already be ~0

            if return_disor_matrices:
                disor_matrices[ph] = (cluster_ids, disor_matrix)

            # --- greedy complete-linkage grouping using the matrix ---
            order = sorted(range(n), key=lambda i: -sizes.get(cluster_ids[i], 0))
            groups = []

            for i in order:
                placed = False
                for g in groups:
                    if np.all(disor_matrix[i, g] < threshold_deg):
                        g.append(i)
                        placed = True
                        break
                if not placed:
                    groups.append([i])

            phase_groups = {}
            for g in groups:
                frag_ids = [cluster_ids[i] for i in g]
                root = min(frag_ids)
                phase_groups[root] = frag_ids
            merge_groups_all[ph] = {int(r): [int(f) for f in fr] for r, fr in phase_groups.items()}

            for root, fragments in phase_groups.items():
                for frag in fragments:
                    if frag != root:
                        new_labels[self.labels == frag] = root

        new_result = copy.copy(self)
        new_result.labels = new_labels

        new_result._clusters_unique = None
        new_result._cluster_sizes = None
        new_result._cluster_sizes_by_phase = None
        new_result._cluster_phases_id = None
        new_result._com = None
        new_result._cluster_areas = None
        new_result._cluster_perimeters = None
        new_result._cluster_equivalent_diameters = None
        new_result._cluster_sphericities = None

        _ = new_result.cluster_phases_id
        new_result.get_phase_labels()
        new_result.getAvgOri()
        _ = new_result.com

        if return_disor_matrices:
            return new_result, merge_groups_all, disor_matrices
        return new_result, merge_groups_all


    def merge_clusters_by_orientation_single_linkage(self, threshold_deg=2.0, phase=None, min_size=None):
        """
        Merge clusters within the same phase whose average orientations are
        close, with NO spatial/adjacency constraint — unlike flood-fill
        clustering, merged clusters need not touch or even be neighbours.

        SINGLE-LINKAGE (union-find): a cluster merges into a group if it is
        within threshold_deg of AT LEAST ONE existing member. This means
        disorientation is only guaranteed to be below threshold_deg along a
        *chain* of links — two clusters at opposite ends of a merged group
        can end up with a disorientation well above threshold_deg if they
        were connected only through intermediate clusters. See
        merge_clusters_by_orientation() for the complete-linkage variant that
        avoids this chaining behavior.

        Same union-find/disorientation logic as reconstruct_physical_grains_ini(),
        but actually relabels self.labels, so downstream analysis
        (get_cluster_areas, get_cluster_sphericities, avg_quats, printCorresp,
        etc.) transparently sees merged super-clusters rather than the
        original spatial fragments.

        Parameters
        ----------
        threshold_deg : float
            Disorientation threshold (degrees, proper-rotation symmetry of
            each phase) below which two clusters are merged. Default 2.0.
        phase : str or list of str, optional
            Phase(s) to merge within. Default: all phases in
            self.labels_by_phase. Clusters are only ever merged within the
            same phase — never across phases.
        min_size : int, optional
            If given, clusters smaller than this (pixels) are excluded from
            merge consideration and kept as-is. Prevents tiny noise clusters
            dragging larger ones together near the threshold.

        Returns
        -------
        new_result : ClusteringResult
            New result with merged labels. self is left untouched.
        merge_groups : dict {phase: {root_id: [fragment cluster ids]}}
            Merge groups actually applied, per phase.
        """
        import copy
        from scipy.spatial.transform import Rotation as _R
        from collections import defaultdict

        if phase is None:
            phases_to_process = list(self.labels_by_phase.keys())
        elif isinstance(phase, str):
            phases_to_process = [phase]
        else:
            phases_to_process = list(phase)

        if not hasattr(self, 'avg_orientations') or self.avg_orientations is None:
            self.getAvgOri()

        sizes = self.cluster_sizes
        new_labels = self.labels.copy()
        merge_groups_all = {}

        for ph in phases_to_process:
            cluster_ids = [c for c in self.labels_by_phase[ph] if c in self.avg_orientations]
            if min_size is not None:
                cluster_ids = [c for c in cluster_ids if sizes.get(c, 0) >= min_size]

            if len(cluster_ids) < 2:
                merge_groups_all[ph] = {int(c): [int(c)] for c in cluster_ids}
                continue

            symops_all = np.array(self.data.phases[ph]['symops'])
            symops = symops_all[np.array([round(np.linalg.det(s)) == 1
                                        for s in symops_all])]

            M = np.stack([self.avg_orientations[c] for c in cluster_ids])
            pairs = [(i, j) for i in range(len(cluster_ids))
                            for j in range(i + 1, len(cluster_ids))]

            parent_map = {c: c for c in cluster_ids}

            def _find(x):
                while parent_map[x] != x:
                    parent_map[x] = parent_map[parent_map[x]]
                    x = parent_map[x]
                return x

            if pairs:
                A = np.stack([M[i] for i, _ in pairs])
                B = np.stack([M[j] for _, j in pairs])
                R_ab = np.einsum("pij,pkj->pik", B, A)
                R_eq = np.einsum("sij,pjk->spik", symops, R_ab).reshape(-1, 3, 3)
                q = _R.from_matrix(R_eq).as_quat().reshape(len(symops), len(pairs), 4)
                w = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
                ang = np.degrees(2 * np.arccos(w))
                min_ang = ang.min(axis=0)

                for (i, j), d in zip(pairs, min_ang):
                    if d < threshold_deg:
                        a, b = cluster_ids[i], cluster_ids[j]
                        ra, rb = _find(a), _find(b)
                        if ra != rb:
                            parent_map[rb] = ra

            groups = defaultdict(list)
            for c in cluster_ids:
                groups[_find(c)].append(c)

            groups = {max(fr, key=lambda cid: sizes.get(cid, 0)): fr for fr in groups.values()}
            merge_groups_all[ph] = {int(r): [int(f) for f in fr] for r, fr in groups.items()}

            for root, fragments in groups.items():
                for frag in fragments:
                    if frag != root:
                        new_labels[self.labels == frag] = root

        new_result = copy.copy(self)
        new_result.labels = new_labels

        # Invalidate caches so everything recomputes on the merged labels
        new_result._clusters_unique = None
        new_result._cluster_sizes = None
        new_result._cluster_sizes_by_phase = None
        new_result._cluster_phases_id = None
        new_result._com = None
        new_result._cluster_areas = None
        new_result._cluster_perimeters = None
        new_result._cluster_equivalent_diameters = None
        new_result._cluster_sphericities = None

        _ = new_result.cluster_phases_id   # force lazy recompute before get_phase_labels
        new_result.get_phase_labels()
        new_result.getAvgOri()             # recompute avg orientation/quat per merged cluster
        _ = new_result.com

        return new_result, merge_groups_all
    def print_clusters_by_size(self, phase=None, ascending=False):
        """
        Print clusters sorted by pixel count, with their phase.

        Parameters
        ----------
        phase : str or list of str, optional
            Restrict to cluster(s) in this phase (or phases). Default: all
            phases.
        ascending : bool, optional
            Sort smallest-first. Default False (largest first).
        """
        if phase is None:
            cluster_ids = self.get_unique_clusters()
        elif isinstance(phase, str):
            cluster_ids = self.labels_by_phase[phase]
        else:
            cluster_ids = np.concatenate([self.labels_by_phase[p] for p in phase])

        sizes = self.cluster_sizes
        phases = self.cluster_phases_id
        phase_names = self.data.phase_names

        ordered = sorted(cluster_ids, key=lambda c: sizes[c], reverse=not ascending)

        print(f"{'Cluster':>8} {'Size (px)':>10} {'Phase':>8}")
        print("-" * 30)
        for c in ordered:
            print(f"{c:>8} {sizes[c]:>10} {phase_names[phases[c]]:>8}")
    # ============================================================================
    # LAMELLAR CLUSTER IDENTIFICATION METHODS
    # ============================================================================

    def analyze_cluster_elongation(self, cluster_id, use_sklearn=False, min_width_threshold=1.0,min_pixesl_threshold=3):
        """
        Analyze elongation and principal direction of a specific cluster.
        
        Uses PCA on all cluster pixels (not just boundary) to find the principal
        axis and calculate aspect ratio.
        
        Parameters
        ----------
        cluster_id : int
            Cluster label
        use_sklearn : bool, optional
            If True, uses scikit-learn PCA (more robust).
            If False, uses NumPy eigendecomposition (no extra dependencies).
            Default is False.
        min_width_threshold : float, optional
            Minimum width (in pixels) for valid cluster.
            Clusters narrower than this are considered degenerate (1D artifacts).
            Default is 1.0 pixel.
        min_pixesl_threshold : int, optional
            Minimum number of pixels for valid cluster.
            Clusters smaller than this are ignored.
            Default is 3 pixels.
        
        Returns
        -------
        elongation_info : dict or None
            Dictionary containing elongation properties, or None if:
            - Cluster too small (< 3 pixels)
            - Cluster too narrow (width < min_width_threshold)
        """
        # Get all pixels in this cluster
        mask = self.labels == cluster_id
        n_pixels = np.sum(mask)
        
        if n_pixels < min_pixesl_threshold:
            return None
        
        # Get coordinates of all pixels in cluster
        coords = np.column_stack([self.data.X[mask], self.data.Y[mask]])
        
        # Calculate centroid
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid
        
        # PCA to find principal directions
        if use_sklearn:
            try:
                from sklearn.decomposition import PCA
                
                # Fit PCA
                pca = PCA(n_components=2)
                pca.fit(coords_centered)
                
                # Get principal components
                principal_direction = pca.components_[0]
                perpendicular_direction = pca.components_[1]
                eigenvalues = pca.explained_variance_
                
            except ImportError:
                print("Warning: scikit-learn not available, falling back to NumPy")
                use_sklearn = False
        
        if not use_sklearn:
            # NumPy implementation
            cov = np.cov(coords_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Sort by eigenvalue (largest first)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Principal direction (long axis)
            principal_direction = eigenvectors[:, 0]
            perpendicular_direction = eigenvectors[:, 1]
        
        # Ensure real values (in case of numerical issues)
        principal_direction = np.real(principal_direction)
        perpendicular_direction = np.real(perpendicular_direction)
        eigenvalues = np.real(eigenvalues)
        
        # Calculate angle
        principal_angle = np.arctan2(principal_direction[1], principal_direction[0]) * 180 / np.pi
        if principal_angle < 0:
            principal_angle += 180
        
        # Project points onto principal axes
        projections_principal = np.dot(coords_centered, principal_direction)
        projections_perpendicular = np.dot(coords_centered, perpendicular_direction)
        
        # Calculate extents
        length = np.max(projections_principal) - np.min(projections_principal)
        width = np.max(projections_perpendicular) - np.min(projections_perpendicular)
        
        # Validate width - filter out degenerate (1D) clusters
        if min_width_threshold is not None:
            if width <= min_width_threshold:
                # Cluster is too narrow (likely 1D artifact, single-pixel line, or scanning issue)
                # Return None to exclude from analysis
                return None
        
        # Calculate aspect ratio (width is guaranteed to be >= min_width_threshold)
        #print(width)
        aspect_ratio = length / width# if width > 1e-9 else 999.9
        
        # Calculate area directly (fast, no perimeter calculation)
        if self.is_hexagonal:
            if hasattr(self.data, '_ebsdData') and hasattr(self.data._ebsdData, 'dx'):
                dx = self.data._ebsdData.dx
                dy = self.data._ebsdData.dy
                area = n_pixels * dx * dy * np.sqrt(3) / 2
            else:
                area = float(n_pixels)  # Fallback to pixel count
        else:
            area = float(n_pixels)  # In pixels
        
        # End points (extremes along principal axis)
        max_proj_idx = np.argmax(projections_principal)
        min_proj_idx = np.argmin(projections_principal)
        end_point_1 = coords[max_proj_idx]
        end_point_2 = coords[min_proj_idx]
        
        return {
            'cluster_id': cluster_id,
            'principal_direction': principal_direction,
            'perpendicular_direction': perpendicular_direction,
            'principal_angle': principal_angle,
            'aspect_ratio': aspect_ratio,
            'length': length,
            'width': width,
            'centroid': centroid,
            'eigenvalues': eigenvalues,
            'end_point_1': end_point_1,
            'end_point_2': end_point_2,
            'area': area,
            'n_pixels': n_pixels,
            'pca_method': 'sklearn' if use_sklearn else 'numpy'
        }

    def get_pixel_to_average_misorientation(self, cluster_id):
        """
        Misorientation (deg) of every individual pixel in a cluster to that
        cluster's average orientation (self.avg_orientations[cluster_id]),
        using the full symmetry of the cluster's phase.

        Useful for quantifying how much true pixel-level spread exists
        within a cluster -- e.g. after merge_clusters_by_orientation, where
        the merge criterion only bounds distances between cluster AVERAGES,
        not the spread of the underlying pixel populations, so large
        apparent spread post-merge is expected and this quantifies it
        directly.

        Parameters
        ----------
        cluster_id : int

        Returns
        -------
        angles : (n_pixels,) ndarray
            Misorientation (deg) of each pixel to the cluster average.
        """
        from scipy.spatial.transform import Rotation as _R

        if cluster_id not in self.avg_orientations:
            raise ValueError(f"No average orientation available for cluster {cluster_id}")

        idxs = np.where(self.labels == cluster_id)[0]
        if idxs.shape[0] == 0:
            raise ValueError(f"Cluster {cluster_id} has no pixels")

        phase_id = self.cluster_phases_id[cluster_id]
        phase_name = self.data.phase_names[phase_id]
        symops_all = np.array(self.data.phases[phase_name]['symops'])
        symops = symops_all[np.array([round(np.linalg.det(s)) == 1 for s in symops_all])]

        avg_M = self.avg_orientations[cluster_id]

        pixel_quats = self.data.quaternions[idxs]
        M_pix = np.array([quat_to_mat(q) for q in pixel_quats])  # (P,3,3)

        R_ab = np.einsum('pij,kj->pik', M_pix, avg_M)  # M_pix @ avg_M.T per pixel
        R_eq = np.einsum('sij,pjk->spik', symops, R_ab).reshape(-1, 3, 3)
        q = _R.from_matrix(R_eq).as_quat().reshape(len(symops), len(idxs), 4)
        w = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
        ang = np.degrees(2 * np.arccos(w))
        min_ang = np.min(ang, axis=0)

        return min_ang


    def summarize_pixel_to_average_misorientation(self, cluster_ids=None, phase=None):
        """
        Compute mean/std/max misorientation of pixels to their cluster's
        average orientation, for one or more clusters.

        Parameters
        ----------
        cluster_ids : int or list of int, optional
            Clusters to evaluate. Default: all clusters (in `phase` if
            given, else all phases).
        phase : str, optional
            Restrict to clusters of this phase. Ignored if cluster_ids is
            given explicitly.

        Returns
        -------
        summary : dict {cluster_id: dict}
            Each value: {'n_pixels', 'mean_deg', 'std_deg', 'max_deg'}
        """
        if cluster_ids is None:
            cluster_ids = list(self.labels_by_phase[phase]) if phase is not None \
                        else list(self.get_unique_clusters())
        elif isinstance(cluster_ids, (int, np.integer)):
            cluster_ids = [cluster_ids]

        summary = {}
        for c in cluster_ids:
            angles = self.get_pixel_to_average_misorientation(c)
            summary[int(c)] = {
                'n_pixels': int(angles.shape[0]),
                'mean_deg': float(np.mean(angles)),
                'std_deg': float(np.std(angles)),
                'max_deg': float(np.max(angles)),
            }
        return summary


    def print_pixel_to_average_misorientation_summary(self, cluster_ids=None, phase=None,
                                                        sort_by='mean_deg', ascending=False):
        """
        Print per-cluster pixel-to-average misorientation statistics,
        sorted by the worst (or best) spread first.

        Parameters
        ----------
        cluster_ids, phase : see summarize_pixel_to_average_misorientation.
        sort_by : {'mean_deg', 'std_deg', 'max_deg', 'n_pixels'}, optional
            Default 'mean_deg'.
        ascending : bool, optional
            Default False (largest spread first).
        """
        summary = self.summarize_pixel_to_average_misorientation(cluster_ids, phase)
        ordered = sorted(summary.items(), key=lambda kv: kv[1][sort_by], reverse=not ascending)

        print(f"{'Cluster':>8} {'Pixels':>8} {'Mean (deg)':>11} {'Std (deg)':>10} {'Max (deg)':>10}")
        print("-" * 52)
        for c, s in ordered:
            print(f"{c:>8} {s['n_pixels']:>8} {s['mean_deg']:>11.3f} {s['std_deg']:>10.3f} {s['max_deg']:>10.3f}")

        return summary

    def identify_lamellar_clusters(self, phase_id=None, phase_name=None, 
                                min_aspect_ratio=None,
                                pixel_size=None, use_hexagonal=None,
                                use_sklearn=False,
                                min_width_threshold=1.0,
                                show_progress=True,sort_by='cluster_id',min_pixesl_threshold=3):
        """
        Analyze elongation of all clusters for a specific phase.
        
        Returns elongation information for ALL clusters, not just lamellar ones.
        Users can filter by aspect ratio afterwards.
        
        Parameters
        ----------
        phase_id : int, optional
            Phase ID to analyze
        phase_name : str, optional
            Phase name to analyze (alternative to phase_id)
        min_aspect_ratio : float, optional
            If provided, marks clusters as 'is_lamellar' based on this threshold.
            If None, all clusters are analyzed without filtering.
            Default is None.
        pixel_size : float, optional
            Size of one pixel in physical units (for square grids).
            If provided, length, width, and area are reported in physical units.
            For square grids only. Ignored if use_hexagonal=True.
        use_hexagonal : bool, optional
            If True, uses dx and dy from data._ebsdData for hexagonal grids.
            If None, automatically detected from data._ebsdData.grid.
        use_sklearn : bool, optional
            If True, uses scikit-learn PCA (more robust).
            If False, uses NumPy eigendecomposition.
            Default is False.
        min_width_threshold : float, optional
            Minimum width (in pixels or physical units depending on pixel_size) 
            for valid clusters. Clusters narrower than this are excluded.
            Default is 1.0.
        min_pixesl_threshold : int, optional
            Minimum number of pixels for valid cluster.
            Clusters smaller than this are ignored.
            Default is 3 pixels.

        show_progress : bool, optional
            If True, shows progress bar. Default is True.
        sort_by : str, optional
            Sort results by:
            - 'aspect_ratio': Most elongated first (default)
            - 'cluster_id': By cluster ID (ascending)
            - 'length': Longest first
            - 'area': Largest first
            Default is 'cluster_id'.
        
        Returns
        -------
        cluster_elongations : list of dict
            List of elongation information for ALL valid clusters in the phase.
            Each dict contains:
            - All geometric properties (length, width, area) in physical units if pixel_size provided
            - 'units' field indicating the measurement units
            - 'is_lamellar' flag if min_aspect_ratio was provided
        
        Examples
        --------
        >>> # Get all cluster elongations in pixel units
        >>> all_clusters = clustering_result.identify_lamellar_clusters(
        ...     phase_name='Ferrite'
        ... )
        >>> 
        >>> # Get elongations in physical units (micrometers)
        >>> all_clusters = clustering_result.identify_lamellar_clusters(
        ...     phase_name='Ferrite',
        ...     pixel_size=0.5,  # 0.5 μm per pixel
        ...     min_aspect_ratio=2.5
        ... )
        >>> print(f"Length: {all_clusters[0]['length']:.2f} μm")
        """
        # Determine if using hexagonal grid
        if use_hexagonal is None:
            use_hexagonal = self.is_hexagonal
        
        # Calculate scale factors for physical units
        if use_hexagonal:
            if not hasattr(self.data, '_ebsdData'):
                raise ValueError("Hexagonal grid requested but data._ebsdData not found")
            if not hasattr(self.data._ebsdData, 'dx') or not hasattr(self.data._ebsdData, 'dy'):
                raise ValueError("Hexagonal grid requested but data._ebsdData.dx and dy not found")
            
            dx = self.data._ebsdData.dx
            dy = self.data._ebsdData.dy
            # For hexagonal: use average of dx and dy for length scaling
            length_scale = (dx + dy) / 2
            # Area per pixel for hexagonal
            area_per_pixel = dx * dy * np.sqrt(3) / 2
            units = "physical"  # Assume dx, dy are in physical units
        elif pixel_size is not None:
            # Square grid with physical units
            length_scale = pixel_size
            area_per_pixel = pixel_size ** 2
            units = "physical"
        else:
            # Pixel units (no scaling)
            length_scale = 1.0
            area_per_pixel = 1.0
            units = "pixels"
        
        # Determine phase
        if phase_name is not None:
            # Find phase_id from name
            phase_id = None
            for pid, pname in self.data.phase_names.items():
                if pname == phase_name:
                    phase_id = pid
                    break
            if phase_id is None:
                raise ValueError(f"Phase '{phase_name}' not found")
        elif phase_id is None:
            raise ValueError("Must specify either phase_id or phase_name")
        
        # Get phase name
        phase_name = self.data.phase_names[phase_id]
        
        # Get all clusters in this phase
        if phase_name not in self.labels_by_phase:
            return []
        
        phase_clusters = self.labels_by_phase[phase_name]
        
        n_clusters = len(phase_clusters)
        if show_progress:
            print(f"\nAnalyzing elongation of {n_clusters} clusters in phase '{phase_name}'...")
            if min_aspect_ratio is not None:
                print(f"Lamellar threshold: aspect ratio >= {min_aspect_ratio}")
            print(f"Width threshold: >= {min_width_threshold} {units}")
            print(f"Units: {units}")
            if use_hexagonal:
                print(f"Hexagonal grid: dx={dx}, dy={dy}")
            elif pixel_size is not None:
                print(f"Pixel size: {pixel_size}")
        
        cluster_elongations = []
        n_lamellar = 0
        n_degenerate = 0
        
        for i, cluster_id in enumerate(phase_clusters):
            if show_progress and (i % max(1, n_clusters // 20) == 0 or i == n_clusters - 1):
                progress = (i + 1) / n_clusters * 100
                if min_aspect_ratio is not None:
                    print(f"  Progress: {i+1}/{n_clusters} ({progress:.1f}%) - "
                        f"Found {n_lamellar} lamellar, {n_degenerate} degenerate", end='\r')
                else:
                    print(f"  Progress: {i+1}/{n_clusters} ({progress:.1f}%) - "
                        f"{n_degenerate} degenerate excluded", end='\r')
            
            # Analyze elongation (returns in pixel units)
            elongation = self.analyze_cluster_elongation(
                cluster_id, 
                use_sklearn=use_sklearn,
                min_width_threshold=min_width_threshold / length_scale if min_width_threshold is not None else None,# Convert threshold to pixels
                min_pixesl_threshold=min_pixesl_threshold  
            )
            
            if elongation is None:
                n_degenerate += 1
                continue
            
            # Scale geometric properties to physical units
            length_physical = elongation['length'] * length_scale
            width_physical = elongation['width'] * length_scale
            area_physical = elongation['n_pixels'] * area_per_pixel
            
            # Aspect ratio is dimensionless (unchanged)
            aspect_ratio = elongation['aspect_ratio']
            
            # Determine if lamellar
            
            if min_aspect_ratio is not None:
                is_lamellar = aspect_ratio >= min_aspect_ratio
                if is_lamellar:
                    n_lamellar += 1
            else:
                is_lamellar = True
                n_lamellar += 1
            
            # Add to list with physical units
            cluster_elongations.append({
                'cluster_id': cluster_id,
                'phase_id': phase_id,
                'phase_name': phase_name,
                'aspect_ratio': aspect_ratio,  # Dimensionless
                'length': length_physical,  # In physical units
                'width': width_physical,  # In physical units
                'orientation_angle': elongation['principal_angle'],  # Degrees
                'principal_direction': elongation['principal_direction'],  # Unit vector
                'perpendicular_direction': elongation['perpendicular_direction'],  # Unit vector
                'centroid': elongation['centroid'],  # In pixel coordinates
                'end_point_1': elongation['end_point_1'],  # In pixel coordinates
                'end_point_2': elongation['end_point_2'],  # In pixel coordinates
                'area': area_physical,  # In physical units
                'n_pixels': elongation['n_pixels'],  # Count (dimensionless)
                'eigenvalues': elongation['eigenvalues'],  # In pixel^2 units
                'pca_method': elongation['pca_method'],
                'is_lamellar': is_lamellar,
                'units': units,  # 'pixels' or 'physical'
                'elongation_info': elongation,  # Original data in pixels
                'length_scale': length_scale
            })
        
        if show_progress:
            print()  # New line after progress
            if min_aspect_ratio is not None:
                print(f"✓ Complete: {n_lamellar} lamellar clusters "
                    f"({n_lamellar/n_clusters*100:.1f}% of phase clusters)")
            print(f"  Valid clusters analyzed: {len(cluster_elongations)}")
            print(f"  Degenerate clusters excluded: {n_degenerate}\n")
        
        # ========== UPDATED SORTING ==========
        # Sort based on requested criterion
        if sort_by == 'cluster_id':
            cluster_elongations.sort(key=lambda x: x['cluster_id'])
        elif sort_by == 'aspect_ratio':
            cluster_elongations.sort(key=lambda x: x['aspect_ratio'], reverse=True)
        elif sort_by == 'length':
            cluster_elongations.sort(key=lambda x: x['length'], reverse=True)
        elif sort_by == 'area':
            cluster_elongations.sort(key=lambda x: x['area'], reverse=True)
        elif sort_by == 'width':
            cluster_elongations.sort(key=lambda x: x['width'], reverse=True)
        else:
            print(f"Warning: Unknown sort_by='{sort_by}', using 'aspect_ratio'")
            cluster_elongations.sort(key=lambda x: x['aspect_ratio'], reverse=True)
        # ====================================
        self.lamellar_child_clusters = cluster_elongations
        return cluster_elongations
    def merge_lamellar_fragments(self, lamellar_child_clusters,
                                    parent_child_hierarchy,
                                    angle_tol_deg=10.0,
                                    collinearity_tol=2.0,
                                    gap_tol_factor=5.0,
                                    min_pixels_threshold=0):
            """
            Merge fragmented lamellar clusters that belong to the same physical lamella.

            Segmentation sometimes splits one physical lamella into several clusters
            along its length (indexing gaps, orientation spread, low-CI regions).
            This method identifies such fragments and returns a merged
            lamellar_child_clusters list where each physical lamella is represented
            by a single entry with combined geometry.

            Two lamellar clusters A and B are merged if ALL of the following hold:

            1. **Same parent grain** — both are children of the same austenite cluster
            in parent_child_hierarchy.
            2. **Co-directional** — the angle between their principal_direction vectors
            is within angle_tol_deg (or within 180 - angle_tol_deg, since direction
            is unsigned).
            3. **Collinear** — the component of the centroid-to-centroid vector
            perpendicular to the shared principal direction is less than
            collinearity_tol × mean_width of the two clusters.
            4. **Gap is small** — the gap between the nearest endpoints along the
            principal direction is less than gap_tol_factor × mean_width.

            Merging is transitive: if A merges with B and B merges with C, all three
            form one group (union-find).

            Parameters
            ----------
            lamellar_child_clusters : list of dict
                Output of identify_lamellar_clusters(). Each dict must contain:
                'cluster_id', 'principal_direction', 'perpendicular_direction',
                'centroid', 'end_point_1', 'end_point_2', 'length', 'width',
                'n_pixels', 'orientation_angle', 'aspect_ratio', and all other
                fields produced by identify_lamellar_clusters().
            parent_child_hierarchy : list of dict
                Output of boundary_result.get_parent_child_hierarchy_with_boundaries().
                Used to determine which parent grain each lamella belongs to.
            angle_tol_deg : float
                Maximum angle (degrees) between principal directions to consider
                two clusters co-directional. Default 10.0.
            collinearity_tol : float
                Maximum perpendicular offset between centroids, expressed as a
                multiple of the mean cluster width. Default 2.0.
            gap_tol_factor : float
                Maximum gap between nearest endpoints along the principal direction,
                expressed as a multiple of the mean cluster width. Default 5.0.
                Increase for lamellae with large unindexed gaps.
            min_pixels_threshold : int
                Minimum pixel count for a merged cluster to be retained.
                Default 0 (keep all).

            Returns
            -------
            merged_clusters : list of dict
                New lamellar_child_clusters list. Each entry has the same fields as
                the input, plus:
                - 'fragment_ids' : list of int
                    Original cluster_id values that were merged into this entry.
                    Single-fragment entries have a one-element list.
                - 'is_merged' : bool
                    True if this entry represents two or more merged fragments.
                Entries are sorted by cluster_id (root of each merged group).

            Notes
            -----
            The merged geometry is computed as:
            - principal_direction : weighted mean of fragment directions
            (weight = n_pixels), re-normalised. Fragment directions are
            sign-aligned before averaging so anti-parallel vectors don't cancel.
            - centroid : weighted mean of fragment centroids (weight = n_pixels).
            - end_point_1 / end_point_2 : the two extreme projections along the
            merged principal direction — i.e. the actual tips of the merged lamella.
            - length : distance between the two extreme endpoints.
            - width : weighted mean of fragment widths (weight = n_pixels).
            - n_pixels : sum of fragment pixel counts.
            - aspect_ratio : merged length / merged width.
            - orientation_angle : angle of merged principal_direction in degrees.
            - perpendicular_direction : 90° rotation of merged principal_direction.
            - All other scalar fields (phase_id, phase_name, units, length_scale,
            pca_method) are taken from the fragment with the most pixels.

            Example
            -------
            >>> merged = clustering_result.merge_lamellar_fragments(
            ...     lamellar_child_clusters,
            ...     parent_child_hierarchy_with_boundaries,
            ...     angle_tol_deg=10.0,
            ...     collinearity_tol=2.0,
            ...     gap_tol_factor=5.0,
            ... )
            >>> print(f"{len(lamellar_child_clusters)} clusters -> "
            ...       f"{len(merged)} after merging")
            >>> n_merged = sum(1 for c in merged if c['is_merged'])
            >>> print(f"{n_merged} multi-fragment lamellae")
            """
            from collections import defaultdict

            if not lamellar_child_clusters:
                return []

            # ── Build lookup: child_id → parent_id ───────────────────────────
            # A child can touch multiple parents (touching_parents case), but
            # for merging we use the primary (enclosing) parent.
            child_to_parent = {}
            for hier in parent_child_hierarchy:
                pid = hier['parent_cluster_id']
                for child in hier.get('children', []):
                    cid = child['child_cluster_id']
                    rel = child.get('relationship', 'enclosed')
                    # Prefer 'enclosed' over 'touching'
                    if cid not in child_to_parent or rel == 'enclosed':
                        child_to_parent[cid] = pid

            # ── Index lamellar clusters by id ─────────────────────────────────
            by_id = {c['cluster_id']: c for c in lamellar_child_clusters}
            ids   = [c['cluster_id'] for c in lamellar_child_clusters]

            # ── Group by parent so we only compare within the same grain ──────
            by_parent = defaultdict(list)
            for cid in ids:
                pid = child_to_parent.get(cid, -1)
                by_parent[pid].append(cid)

            # ── Union-Find ────────────────────────────────────────────────────
            uf = {cid: cid for cid in ids}

            def _find(x):
                while uf[x] != x:
                    uf[x] = uf[uf[x]]
                    x = uf[x]
                return x

            def _union(x, y):
                rx, ry = _find(x), _find(y)
                if rx != ry:
                    uf[ry] = rx

            # ── Pairwise merge test within each parent ────────────────────────
            cos_tol = np.cos(np.radians(angle_tol_deg))

            for pid, group_ids in by_parent.items():
                if len(group_ids) < 2:
                    continue

                for i, cid_a in enumerate(group_ids):
                    a = by_id[cid_a]
                    d_a   = np.array(a['principal_direction'], dtype=float)
                    c_a   = np.array(a['centroid'],            dtype=float)
                    ep1_a = np.array(a['end_point_1'],         dtype=float)
                    ep2_a = np.array(a['end_point_2'],         dtype=float)
                    w_a   = float(a['width'])

                    for cid_b in group_ids[i + 1:]:
                        b = by_id[cid_b]
                        d_b   = np.array(b['principal_direction'], dtype=float)
                        c_b   = np.array(b['centroid'],            dtype=float)
                        ep1_b = np.array(b['end_point_1'],         dtype=float)
                        ep2_b = np.array(b['end_point_2'],         dtype=float)
                        w_b   = float(b['width'])
                        mean_w = (w_a + w_b) / 2.0

                        # ── Test 1: co-directional ─────────────────────────
                        dot = abs(np.dot(d_a, d_b))   # abs → unsigned angle
                        if dot < cos_tol:
                            continue

                        # Use d_a as the reference direction (sign-align d_b)
                        d_ref = d_a
                        if np.dot(d_a, d_b) < 0:
                            d_ref_for_b = -d_b
                        else:
                            d_ref_for_b = d_b

                        # ── Test 2: collinear ──────────────────────────────
                        # Perpendicular component of centroid separation
                        delta_c   = c_b - c_a
                        perp_comp = delta_c - np.dot(delta_c, d_ref) * d_ref
                        perp_dist = np.linalg.norm(perp_comp)
                        if perp_dist > collinearity_tol * mean_w:
                            continue

                        # ── Test 3: gap along principal direction ──────────
                        # Project all four endpoints onto d_ref
                        proj = lambda ep: float(np.dot(ep - c_a, d_ref))
                        p1a, p2a = proj(ep1_a), proj(ep2_a)
                        p1b, p2b = proj(ep1_b), proj(ep2_b)

                        # Interval of A: [min_a, max_a], of B: [min_b, max_b]
                        min_a, max_a = min(p1a, p2a), max(p1a, p2a)
                        min_b, max_b = min(p1b, p2b), max(p1b, p2b)

                        # Gap = positive if intervals don't overlap
                        gap = max(0.0, max(min_a, min_b) - min(max_a, max_b))
                        if gap > gap_tol_factor * mean_w:
                            continue

                        # All tests passed — merge
                        _union(cid_a, cid_b)

            # ── Collect groups ────────────────────────────────────────────────
            groups = defaultdict(list)
            for cid in ids:
                groups[_find(cid)].append(cid)

            # ── Build merged cluster entries ──────────────────────────────────
            merged_clusters = []

            for root, fragment_ids in sorted(groups.items()):
                fragments = [by_id[fid] for fid in fragment_ids]

                if len(fragments) == 1:
                    # No merge — copy with extra fields
                    entry = dict(fragments[0])
                    entry['fragment_ids'] = [fragments[0]['cluster_id']]
                    entry['is_merged']    = False
                    merged_clusters.append(entry)
                    continue

                # ── Weighted merge geometry ───────────────────────────────
                weights  = np.array([float(f['n_pixels']) for f in fragments])
                weights /= weights.sum()

                # Dominant fragment (most pixels) — source of scalar metadata
                dominant = fragments[int(np.argmax([f['n_pixels'] for f in fragments]))]

                # Sign-align all principal directions to the dominant fragment
                d_dom = np.array(dominant['principal_direction'], dtype=float)
                dirs  = []
                for f in fragments:
                    d = np.array(f['principal_direction'], dtype=float)
                    dirs.append(d if np.dot(d, d_dom) >= 0 else -d)

                # Weighted mean direction, re-normalised
                d_merged = np.average(np.stack(dirs), axis=0, weights=weights)
                d_norm   = np.linalg.norm(d_merged)
                if d_norm < 1e-10:
                    d_merged = d_dom
                else:
                    d_merged /= d_norm

                # Perpendicular direction (90° CCW rotation in 2D)
                d_perp = np.array([-d_merged[1], d_merged[0]])

                # Weighted centroid
                centroids = np.stack([np.array(f['centroid'], dtype=float)
                                    for f in fragments])
                centroid_merged = np.average(centroids, axis=0, weights=weights)

                # Extreme endpoints along merged principal direction
                all_endpoints = []
                for f in fragments:
                    all_endpoints.append(np.array(f['end_point_1'], dtype=float))
                    all_endpoints.append(np.array(f['end_point_2'], dtype=float))

                projections = [float(np.dot(ep - centroid_merged, d_merged))
                            for ep in all_endpoints]
                ep1_merged  = all_endpoints[int(np.argmin(projections))]
                ep2_merged  = all_endpoints[int(np.argmax(projections))]

                length_merged = float(np.linalg.norm(ep2_merged - ep1_merged))
                width_merged  = float(np.average(
                    [f['width'] for f in fragments], weights=weights))
                n_pixels_merged = int(sum(f['n_pixels'] for f in fragments))
                ar_merged = length_merged / width_merged if width_merged > 0 else 0.0
                angle_merged = float(
                    np.degrees(np.arctan2(d_merged[1], d_merged[0])))

                if n_pixels_merged < min_pixels_threshold:
                    continue

                entry = {
                    # Identity
                    'cluster_id':           root,
                    'fragment_ids':         sorted(fragment_ids),
                    'is_merged':            True,
                    # Phase info from dominant fragment
                    'phase_id':             dominant['phase_id'],
                    'phase_name':           dominant['phase_name'],
                    'units':                dominant['units'],
                    'length_scale':         dominant['length_scale'],
                    'pca_method':           dominant['pca_method'],
                    'is_lamellar':          True,
                    # Merged geometry
                    'principal_direction':  d_merged,
                    'perpendicular_direction': d_perp,
                    'centroid':             centroid_merged,
                    'end_point_1':          ep1_merged,
                    'end_point_2':          ep2_merged,
                    'length':               length_merged,
                    'width':                width_merged,
                    'n_pixels':             n_pixels_merged,
                    'aspect_ratio':         ar_merged,
                    'orientation_angle':    angle_merged,
                    # eigenvalues not meaningful after merge
                    'eigenvalues':          None,
                    'elongation_info':      None,
                    'area':                 float(sum(f['area'] for f in fragments)),
                }
                merged_clusters.append(entry)

            return sorted(merged_clusters, key=lambda x: x['cluster_id'])    
    def get_lamellar_cluster_info(self, cluster_id, use_sklearn=False, elongation_cache=None):
        """
        Get detailed lamellar information for a specific cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster label
        use_sklearn : bool, optional
            If True, uses scikit-learn PCA. Default is False.
            Ignored if elongation_cache is provided.
        elongation_cache : dict or list, optional
            Pre-computed elongation results from identify_lamellar_clusters.
            If provided as dict: {cluster_id: elongation_info}
            If provided as list: results from identify_lamellar_clusters
            If None, computes elongation on the fly.
        
        Returns
        -------
        info : dict
            Complete lamellar cluster information
        
        Examples
        --------
        >>> # Without cache (computes PCA)
        >>> info = clustering_result.get_lamellar_cluster_info(cluster_id=42)
        >>> 
        >>> # With cache (reuses PCA from identify_lamellar_clusters)
        >>> all_clusters = clustering_result.identify_lamellar_clusters(
        ...     phase_name='Ferrite'
        ... )
        >>> info = clustering_result.get_lamellar_cluster_info(
        ...     cluster_id=42,
        ...     elongation_cache=all_clusters
        ... )
        """
        # Try to get from cache
        elongation_data = None
        
        if elongation_cache is not None:
            if isinstance(elongation_cache, dict):
                # Cache is a dictionary {cluster_id: data}
                elongation_data = elongation_cache.get(cluster_id)
            elif isinstance(elongation_cache, list):
                # Cache is a list of results from identify_lamellar_clusters
                for cluster_data in elongation_cache:
                    if cluster_data['cluster_id'] == cluster_id:
                        elongation_data = cluster_data
                        break
        
        # If not in cache, compute it
        if elongation_data is None:
            elongation = self.analyze_cluster_elongation(cluster_id, use_sklearn=use_sklearn)
            
            if elongation is None:
                return None
            
            # Build elongation_data from elongation
            phase_id = self.cluster_phases_id[cluster_id]
            
            elongation_data = {
                'cluster_id': cluster_id,
                'phase_id': phase_id,
                'phase_name': self.data.phase_names[phase_id],
                'aspect_ratio': elongation['aspect_ratio'],
                'length': elongation['length'],
                'width': elongation['width'],
                'orientation_angle': elongation['principal_angle'],
                'principal_direction': elongation['principal_direction'],
                'perpendicular_direction': elongation['perpendicular_direction'],
                'centroid': elongation['centroid'],
                'end_point_1': elongation['end_point_1'],
                'end_point_2': elongation['end_point_2'],
                'area': elongation['area'],
                'n_pixels': elongation['n_pixels'],
                'eigenvalues': elongation['eigenvalues'],
                'pca_method': elongation['pca_method'],
                'elongation_info': elongation
            }
        
        # Get morphology for perimeter and sphericity
        morphology = self.get_cluster_morphology(elongation_data['cluster_id'])
        
        # Calculate side lines (equations for the two long edges)
        centroid = elongation_data['centroid']
        perpendicular_dir = elongation_data['perpendicular_direction']
        width = elongation_data['width']
        
        # Two parallel lines at ±width/2 from centroid
        side_1_point = centroid + perpendicular_dir * (width / 2)
        side_2_point = centroid - perpendicular_dir * (width / 2)
        
        return {
            'cluster_id': elongation_data['cluster_id'],
            'phase_id': elongation_data['phase_id'],
            'phase_name': elongation_data['phase_name'],
            
            # Size metrics
            'n_pixels': elongation_data['n_pixels'],
            'area': elongation_data['area'],
            'perimeter': morphology['perimeter'],
            
            # Elongation metrics
            'aspect_ratio': elongation_data['aspect_ratio'],
            'length': elongation_data['length'],
            'width': elongation_data['width'],
            'is_lamellar': elongation_data['aspect_ratio'] >= 2.5,
            
            # Orientation
            'orientation_angle': elongation_data['orientation_angle'],
            'principal_direction': elongation_data['principal_direction'],
            'perpendicular_direction': elongation_data['perpendicular_direction'],
            
            # Position
            'centroid': elongation_data['centroid'],
            'center_of_mass': morphology['center_of_mass'],
            'end_point_1': elongation_data['end_point_1'],
            'end_point_2': elongation_data['end_point_2'],
            
            # Shape quality
            'eigenvalues': elongation_data['eigenvalues'],
            'sphericity': morphology['sphericity'],
            
            # Side lines
            'side_1_point': side_1_point,
            'side_2_point': side_2_point,
            
            # Method info
            'pca_method': elongation_data.get('pca_method', 'unknown'),
            'units': elongation_data.get('units', 'pixels'),
            
            # Complete data
            'elongation_info': elongation_data.get('elongation_info', elongation_data),
            'morphology': morphology
        }

    def print_lamellar_clusters_summary(self, phase_id=None, phase_name=None,
                                    min_aspect_ratio=2.5,
                                    sort_by='aspect_ratio',
                                    max_clusters=None,
                                    use_sklearn=False):
        """
        Print summary of lamellar clusters for a phase.
        
        Parameters
        ----------
        phase_id : int, optional
            Phase ID to analyze
        phase_name : str, optional
            Phase name to analyze
        min_aspect_ratio : float, optional
            Minimum aspect ratio. Default is 2.5.
        sort_by : str, optional
            Sort by: 'aspect_ratio', 'length', 'area', 'orientation_angle'
            Default is 'aspect_ratio'.
        max_clusters : int, optional
            Maximum clusters to print
        use_sklearn : bool, optional
            If True, uses scikit-learn PCA. Default is False.
        """
        lamellar = self.identify_lamellar_clusters(
            phase_id=phase_id,
            phase_name=phase_name,
            min_aspect_ratio=min_aspect_ratio,
            use_sklearn=use_sklearn
        )
        
        if len(lamellar) == 0:
            phase_str = phase_name or self.data.phase_names[phase_id]
            print(f"\nNo lamellar clusters found in phase '{phase_str}' "
                f"(aspect ratio >= {min_aspect_ratio})")
            return
        
        # Sort
        if sort_by in lamellar[0]:
            lamellar.sort(key=lambda x: x[sort_by], reverse=True)
        
        phase_str = lamellar[0]['phase_name']
        pca_method = lamellar[0]['pca_method']
        
        print(f"\n{'='*90}")
        print(f"Lamellar Clusters in Phase '{phase_str}' (min aspect ratio: {min_aspect_ratio})")
        print(f"PCA method: {pca_method}")
        print(f"{'='*90}")
        print(f"Total lamellar clusters: {len(lamellar)}")
        
        print(f"\n{'ID':>6} {'Pixels':>8} {'Area':>10} {'Length':>10} {'Width':>8} "
            f"{'Aspect':>8} {'Angle':>8}")
        print(f"{'':>6} {'':>8} {'':>10} {'':>10} {'':>8} {'Ratio':>8} {'(°)':>8}")
        print(f"{'-'*90}")
        
        n_to_print = len(lamellar) if max_clusters is None else min(max_clusters, len(lamellar))
        
        for info in lamellar[:n_to_print]:
            print(f"{info['cluster_id']:>6} {info['n_pixels']:>8} {info['area']:>10.2f} "
                f"{info['length']:>10.2f} {info['width']:>8.2f} "
                f"{info['aspect_ratio']:>8.2f} {info['orientation_angle']:>8.1f}")
        
        if max_clusters and len(lamellar) > max_clusters:
            print(f"... and {len(lamellar) - max_clusters} more lamellar clusters")
        
        # Statistics
        print(f"\n{'-'*90}")
        print(f"Statistics:")
        
        aspect_ratios = [l['aspect_ratio'] for l in lamellar]
        lengths = [l['length'] for l in lamellar]
        widths = [l['width'] for l in lamellar]
        angles = [l['orientation_angle'] for l in lamellar]
        
        print(f"  Aspect ratio - Mean: {np.mean(aspect_ratios):.2f}, "
            f"Std: {np.std(aspect_ratios):.2f}, "
            f"Min: {np.min(aspect_ratios):.2f}, Max: {np.max(aspect_ratios):.2f}")
        print(f"  Length - Mean: {np.mean(lengths):.2f}, "
            f"Std: {np.std(lengths):.2f}, "
            f"Min: {np.min(lengths):.2f}, Max: {np.max(lengths):.2f}")
        print(f"  Width - Mean: {np.mean(widths):.2f}, "
            f"Std: {np.std(widths):.2f}, "
            f"Min: {np.min(widths):.2f}, Max: {np.max(widths):.2f}")
        print(f"  Orientation - Mean: {np.mean(angles):.1f}°, "
            f"Std: {np.std(angles):.1f}°")
        
        print(f"{'='*90}\n")
        
        return lamellar
    
    def plot_aspect_ratio_distribution(self,all_clusters, max_aspect_ratio=20.0, 
                                    lamellar_threshold=2.5, figsize=(14, 5)):
        """
        Plot aspect ratio distribution with optional cropping.
        
        Parameters
        ----------
        all_clusters : list of dict
            Results from identify_lamellar_clusters
        max_aspect_ratio : float, optional
            Maximum aspect ratio to display (crops outliers).
            Default is 20.0. Set to None for no cropping.
        lamellar_threshold : float, optional
            Threshold to mark on plots. Default is 2.5.
        figsize : tuple, optional
            Figure size
        
        Returns
        -------
        fig, axes : matplotlib figure and axes
        stats : dict
            Statistics dictionary
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract aspect ratios
        aspect_ratios = np.array([c['aspect_ratio'] for c in all_clusters])
        
        # Apply cropping if requested
        if max_aspect_ratio is not None:
            aspect_ratios_cropped = np.clip(aspect_ratios, None, max_aspect_ratio)
            n_cropped = np.sum(aspect_ratios > max_aspect_ratio)
            cropped = True
        else:
            aspect_ratios_cropped = aspect_ratios
            n_cropped = 0
            cropped = False
        
        if n_cropped > 0:
            print(f"\nCropped {n_cropped} clusters with aspect ratio > {max_aspect_ratio}")
            print(f"  Original max: {np.max(aspect_ratios):.2f}")
            print(f"  Cropped max: {np.max(aspect_ratios_cropped):.2f}")
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(aspect_ratios_cropped, bins=50, edgecolor='black', alpha=0.7, 
                color='steelblue')
        ax1.axvline(lamellar_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Lamellar threshold ({lamellar_threshold})')
        ax1.set_xlabel('Aspect Ratio', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        
        title = 'Aspect Ratio Distribution'
        if cropped:
            title += f' (cropped at {max_aspect_ratio})'
        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add annotation for cropped values
        if n_cropped > 0:
            ax1.text(0.98, 0.98, f'{n_cropped} values\n> {max_aspect_ratio}',
                    transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=9)
        
        # Cumulative distribution
        sorted_ar = np.sort(aspect_ratios_cropped)
        cumulative = np.arange(1, len(sorted_ar) + 1) / len(sorted_ar) * 100
        
        ax2.plot(sorted_ar, cumulative, linewidth=2.5, color='steelblue')
        ax2.axvline(lamellar_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Lamellar threshold ({lamellar_threshold})')
        ax2.axhline(50, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Aspect Ratio', fontsize=11)
        ax2.set_ylabel('Cumulative %', fontsize=11)
        
        title = 'Cumulative Distribution'
        if cropped:
            title += f' (cropped at {max_aspect_ratio})'
        ax2.set_title(title, fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add percentile markers
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            value = np.percentile(aspect_ratios_cropped, p)
            ax2.plot(value, p, 'ro', markersize=8, zorder=5)
            ax2.text(value, p + 3, f'P{p}={value:.1f}', 
                    fontsize=8, ha='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='red', alpha=0.7))
        
        plt.tight_layout()
        
        # Calculate statistics
        stats = {
            'n_total': len(aspect_ratios),
            'n_cropped': n_cropped,
            'original': {
                'mean': np.mean(aspect_ratios),
                'median': np.median(aspect_ratios),
                'std': np.std(aspect_ratios),
                'min': np.min(aspect_ratios),
                'max': np.max(aspect_ratios),
            },
            'cropped': {
                'mean': np.mean(aspect_ratios_cropped),
                'median': np.median(aspect_ratios_cropped),
                'std': np.std(aspect_ratios_cropped),
                'min': np.min(aspect_ratios_cropped),
                'max': np.max(aspect_ratios_cropped),
            },
            'percentiles': {p: np.percentile(aspect_ratios_cropped, p) 
                        for p in [10, 25, 50, 75, 90, 95, 99]},
            'n_lamellar': np.sum(aspect_ratios >= lamellar_threshold),
            'pct_lamellar': np.sum(aspect_ratios >= lamellar_threshold) / len(aspect_ratios) * 100
        }
        
        # Print statistics
        print(f"\n{'='*70}")
        print(f"Aspect Ratio Statistics")
        print(f"{'='*70}")
        print(f"\nTotal clusters: {stats['n_total']}")
        if cropped:
            print(f"Cropped at: {max_aspect_ratio}")
            print(f"Clusters cropped: {n_cropped} ({n_cropped/stats['n_total']*100:.1f}%)")
        
        print(f"\nOriginal data:")
        for key, val in stats['original'].items():
            print(f"  {key.capitalize():8s}: {val:.2f}")
        
        if cropped:
            print(f"\nCropped data:")
            for key, val in stats['cropped'].items():
                print(f"  {key.capitalize():8s}: {val:.2f}")
        
        print(f"\nPercentiles:")
        for p, val in stats['percentiles'].items():
            print(f"  P{p:2d}: {val:.2f}")
        
        print(f"\nClassification (threshold = {lamellar_threshold}):")
        print(f"  Lamellar (AR >= {lamellar_threshold}): {stats['n_lamellar']} ({stats['pct_lamellar']:.1f}%)")
        print(f"  Non-lamellar (AR < {lamellar_threshold}): {stats['n_total']-stats['n_lamellar']} ({100-stats['pct_lamellar']:.1f}%)")
        
        print(f"{'='*70}\n")
        
        return fig, (ax1, ax2), stats
    def reconstruct_physical_grains_ini(self, phase, threshold_deg=2.0):
        """
        Reconstruct physical grains by merging segmentation fragments.

        EBSD segmentation splits a single physical grain into multiple
        cluster IDs when a martensite lamella interrupts indexing continuity.
        Pairs with austenite disorientation < threshold_deg are merged.

        Uses the same M = G2 @ G1.T convention as
        compute_cluster_boundary_misorientations_fast().

        Parameters
        ----------
        phase : str
            Phase label of the parent phase (e.g. 'A' for austenite).
        symops : (Ns,3,3) ndarray
            Proper rotation symmetry operations for the phase.
            For cubic m-3m: orix.quaternion.symmetry.Oh.to_matrix()
        threshold_deg : float
            Disorientation below which two cluster IDs are treated as
            fragments of the same physical grain. Default 2.0 degrees.

        Returns
        -------
        grain_groups : dict {root_id: [list of fragment cluster IDs]}
            Each key is the representative ID of a physical grain.
            Single-fragment grains have a one-element list.
        disor_matrix : dict {(id_a, id_b): angle_deg}
            All pairwise disorientations between parent cluster IDs (a < b).

        Example
        -------
        >>> from orix.quaternion import symmetry
        >>> symops = symmetry.Oh.to_matrix()
        >>> grain_groups, disor = clustering_result.reconstruct_physical_grains(
        ...     phase='A', threshold_deg=2.0)
        >>> print(f"{len(clustering_result.avg_orientations)} IDs "
        ...       f"→ {len(grain_groups)} physical grains")
        """
        from scipy.spatial.transform import Rotation as _R
        #symops = np.array(self.data.phases[phase]['symops'])
        symops_all = np.array(self.data.phases[phase]['symops'])
        symops = symops_all[np.array([round(np.linalg.det(s)) == 1 
                                    for s in symops_all])]
        phase_id = self.data.phase_ids[phase]
        # Get all cluster IDs of this phase
        parent_ids = [
            cid for cid, ph in self.cluster_phases_id.items()
            if ph == phase_id
            and cid in self.avg_orientations
        ]

        # All pairwise disorientations (not just neighbours)
        pairs = [(a, b) for i, a in enumerate(parent_ids)
                        for b in parent_ids[i+1:]]
        disor_matrix = {}
        if pairs:
            A = np.stack([self.avg_orientations[a] for a, _ in pairs])
            B = np.stack([self.avg_orientations[b] for _, b in pairs])
            R_ab = np.einsum("pij,pkj->pik", B, A)          # (P,3,3)
            R_eq  = np.einsum("sij,pjk->spik", symops, R_ab
                            ).reshape(-1, 3, 3)            # (Ns*P,3,3)
            q   = _R.from_matrix(R_eq).as_quat(
                ).reshape(len(symops), len(pairs), 4)
            w   = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
            ang = np.degrees(2 * np.arccos(w))               # (Ns, P)
            min_ang = ang.min(axis=0)
            for (a, b), d in zip(pairs, min_ang):
                disor_matrix[(a, b)] = float(d)

        # Union-Find merge
        parent_map = {p: p for p in parent_ids}

        def _find(x):
            while parent_map[x] != x:
                parent_map[x] = parent_map[parent_map[x]]
                x = parent_map[x]
            return x

        for (a, b), d in disor_matrix.items():
            if d < threshold_deg:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent_map[rb] = ra

        from collections import defaultdict
        grain_groups = defaultdict(list)
        for p in parent_ids:
            grain_groups[_find(p)].append(p)

        return dict(grain_groups), disor_matrix
    def reconstruct_physical_grains(self, phase, boundary_result,
                                  child_phase='M', threshold_deg=2.0):
        """
        Reconstruct physical grains by merging segmentation fragments.

        EBSD segmentation splits a single physical grain into multiple cluster
        IDs when a martensite lamella interrupts indexing continuity across it.
        Only parent cluster pairs that co-border the same child cluster in
        boundary_result are considered as merge candidates — this is a strict
        spatial constraint that avoids merging orientationally similar but
        spatially separate grains.

        For each candidate pair the disorientation is computed using
        M = G2 @ G1.T minimised over proper rotation symmetry operations
        (det = +1 only). Pairs below threshold_deg are merged via union-find.

        Parameters
        ----------
        phase : str
            Phase label of the parent phase (e.g. 'A' for austenite).
        boundary_result : BoundaryResult
            Output of BoundaryAnalyzer().analyze(clustering_result).
            Used to identify which parent clusters co-border the same child.
        child_phase : str
            Phase label of the child phase (e.g. 'M' for martensite).
            Default 'M'.
        threshold_deg : float
            Disorientation below which two cluster IDs are treated as
            fragments of the same physical grain. Default 2.0 degrees.

        Returns
        -------
        grain_groups : dict {root_id: [list of fragment cluster IDs]}
            Each key is the representative ID of a physical grain.
            Single-fragment grains have a one-element list.
        disor_matrix : dict {(id_a, id_b): angle_deg}
            Disorientations for all evaluated candidate pairs (a < b).
            Only pairs that co-border the same child cluster are included.

        Notes
        -----
        Workflow::

            boundary_result = BoundaryAnalyzer().analyze(clustering_result)
            grain_groups, disor = clustering_result.reconstruct_physical_grains(
                phase='A', boundary_result=boundary_result,
                child_phase='M', threshold_deg=2.0)
            id_map = clustering_result.merge_physical_grains(grain_groups)
            # Re-run BoundaryAnalyzer after merge:
            boundary_result = BoundaryAnalyzer().analyze(clustering_result)
        """
        from scipy.spatial.transform import Rotation as _R
        from collections import defaultdict

        # ── Phase and symmetry setup ──────────────────────────────────────────
        symops_all = np.array(self.data.phases[phase]['symops'])
        symops     = symops_all[np.array([round(np.linalg.det(s)) == 1
                                        for s in symops_all])]
        phase_id       = self.data.phase_ids[phase]
        child_phase_id = self.data.phase_ids[child_phase]

        # All parent cluster IDs of the target phase that have orientations
        parent_ids    = [
            cid for cid, ph in self.cluster_phases_id.items()
            if ph == phase_id and cid in self.avg_orientations
        ]
        parent_id_set = set(parent_ids)

        # ── Find candidate pairs via boundary_result ──────────────────────────
        # For each child cluster: collect all parent-phase neighbours.
        # Every pair of parents that co-border the same child is a candidate
        # for merging — the lamella ran between them, causing the split.
        candidate_pairs = set()

        br_clusters = boundary_result.clusters  # array of cluster IDs

        for i, cid in enumerate(br_clusters):

            # Only process child-phase clusters
            if boundary_result.cluster_phases_id[i] != child_phase_id:
                continue

            # Neighbours of this child cluster and their phase IDs
            nb_phases = boundary_result.grouped_boundary_phases_id[i]

            # Parent-phase neighbours that are in our analysis set
            parent_neighbours = [
                nb_id for nb_id, nb_phase in nb_phases.items()
                if nb_id != -1                    # skip scan edge
                and nb_phase == phase_id
                and nb_id in parent_id_set
            ]

            # Every pair of these parents is a merge candidate
            for j, pid_a in enumerate(parent_neighbours):
                for pid_b in parent_neighbours[j + 1:]:
                    candidate_pairs.add(
                        (min(pid_a, pid_b), max(pid_a, pid_b)))

        pairs = sorted(candidate_pairs)

        if not pairs:
            # No candidate pairs found — each parent is its own physical grain
            return {p: [p] for p in parent_ids}, {}

        # ── Vectorised disorientation for all candidate pairs ─────────────────
        # M = G2 @ G1.T,  disorientation = min over symops of arccos(|w|)
        A    = np.stack([self.avg_orientations[a] for a, _ in pairs])  # (P,3,3)
        B    = np.stack([self.avg_orientations[b] for _, b in pairs])  # (P,3,3)

        R_ab = np.einsum("pij,pkj->pik", B, A)                        # (P,3,3)
        R_eq = np.einsum("sij,pjk->spik", symops, R_ab
                        ).reshape(-1, 3, 3)                           # (Ns*P,3,3)

        q      = _R.from_matrix(R_eq).as_quat(
                ).reshape(len(symops), len(pairs), 4)
        w      = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
        ang    = np.degrees(2 * np.arccos(w))                          # (Ns, P)
        min_ang = ang.min(axis=0)                                      # (P,)

        disor_matrix = {
            (a, b): float(d) for (a, b), d in zip(pairs, min_ang)
        }

        # ── Union-Find merge below threshold ──────────────────────────────────
        parent_map = {p: p for p in parent_ids}

        def _find(x):
            while parent_map[x] != x:
                parent_map[x] = parent_map[parent_map[x]]  # path compression
                x = parent_map[x]
            return x

        for (a, b), d in disor_matrix.items():
            if d < threshold_deg:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent_map[rb] = ra

        grain_groups = defaultdict(list)
        for p in parent_ids:
            grain_groups[_find(p)].append(p)
        self.parent_grain_groups = dict(grain_groups)  # store for potential later use
        self.parent_disor_matrix = disor_matrix      # store for potential later use
        return dict(grain_groups), disor_matrix
    
    def merge_physical_grains(self, grain_groups, label='grain'):
        """
        Register physical grain/lamella groups without modifying cluster IDs.

        Instead of relabelling self.labels, this method only stores the
        group membership mapping on self. Original cluster IDs are preserved
        throughout — all downstream code (BoundaryAnalyzer, singleHP,
        get_parent_child_hierarchy_with_boundaries) continues to work with
        the original IDs without modification.

        For merged groups, the average orientation is recomputed from the
        combined pixel data of all fragments using get_avg_orientations().

        Parameters
        ----------
        grain_groups : dict {root_id: [list of fragment cluster IDs]}
            Output of reconstruct_physical_grains() or
            reconstruct_physical_lamellas(). root_id is the representative
            ID of the physical grain (fragment with most pixels or lowest ID).
        label : str
        Storage label. Use 'grain' for austenite, 'lamella' for martensite.
        Results stored as self.{label}_to_clusters, self.cluster_to_{label} etc.

        Returns
        -------
        cluster_to_grain : dict {cluster_id: root_id}
            Maps every cluster ID to the root ID of its physical grain.
            Single-fragment grains map to themselves.
            Use this for aggregation in analysis methods.

        Stores on self
        --------------
        self.cluster_to_grain : dict {cluster_id: root_id}
            Same as return value — available for downstream methods.
        self.grain_to_clusters : dict {root_id: [cluster_ids]}
            Reverse mapping — all fragment IDs for each physical grain.
        self.grain_avg_orientations : dict {root_id: (3,3) ndarray}
            Average orientation matrix per physical grain.
            For single-fragment grains: same as self.avg_orientations[root_id].
            For merged grains: recomputed from combined pixel data.
        self.grain_avg_quats : dict {root_id: quaternion}
            Average quaternion per physical grain.

        Notes
        -----
        self.labels is NOT modified. All original cluster IDs remain valid.
        No cache invalidation is needed.
        No downstream structures need updating.

        Example
        -------
        >>> grain_groups, disor = clustering_result.reconstruct_physical_grains(
        ...     phase='A', boundary_result=boundary_result, child_phase='M')
        >>> cluster_to_grain = clustering_result.merge_physical_grains(
        ...     grain_groups, parent_phase='A', child_phase='M')
        >>> # Get physical grain ID for any cluster:
        >>> phys_id = cluster_to_grain[some_cluster_id]
        >>> # Get all fragments of a physical grain:
        >>> fragments = clustering_result.grain_to_clusters[phys_id]
        """
        # ── Build cluster → grain mapping ────────────────────────────────
        cluster_to_grain = {}
        for root_id, fragments in grain_groups.items():
            for frag_id in fragments:
                cluster_to_grain[int(frag_id)] = int(root_id)

        # ── Build grain → clusters reverse mapping ────────────────────────
        grain_to_clusters = {
            int(root_id): [int(f) for f in fragments]
            for root_id, fragments in grain_groups.items()
        }

        # ── Recompute average orientations for merged groups ──────────────
        grain_avg_orientations = {}
        grain_avg_quats        = {}

        for root_id, fragments in grain_groups.items():
            root_id = int(root_id)
            if len(fragments) == 1:
                # Single fragment — use existing orientation directly
                frag_id = int(fragments[0])
                grain_avg_orientations[root_id] = self.avg_orientations.get(frag_id)
                grain_avg_quats[root_id]        = self.avg_quats.get(frag_id)
            else:
                # Multi-fragment — recompute from combined pixel data
                # Collect pixel indices for all fragments
                all_idxs = np.concatenate([
                    np.where(self.labels == int(fid))[0]
                    for fid in fragments
                ])
                if len(all_idxs) > 0:
                    try:
                        # Get phase label from root fragment
                        phase_id    = self.cluster_phases_id.get(int(root_id))
                        if phase_id is None:
                            # Try first fragment
                            phase_id = self.cluster_phases_id.get(int(fragments[0]))
                        phase_label = self.data.phase_names[phase_id]
                        symops_all  = np.array(
                            self.data.phases[phase_label]['symops'])
                        symops = symops_all[np.array([
                            round(np.linalg.det(s)) == 1
                            for s in symops_all])]
                        q_mean, M_mean, _, _ = get_avg_orientations(
                            self.data.quaternions[all_idxs], symops)
                        grain_avg_orientations[root_id] = M_mean
                        grain_avg_quats[root_id]        = q_mean
                    except Exception:
                        # Fallback to root fragment orientation
                        grain_avg_orientations[root_id] = \
                            self.avg_orientations.get(int(root_id))
                        grain_avg_quats[root_id] = \
                            self.avg_quats.get(int(root_id))
                else:
                    grain_avg_orientations[root_id] = \
                        self.avg_orientations.get(int(root_id))
                    grain_avg_quats[root_id] = \
                        self.avg_quats.get(int(root_id))

        # Store under label-specific attributes
        setattr(self, f'cluster_to_{label}',       cluster_to_grain)
        setattr(self, f'{label}_to_clusters',       grain_to_clusters)
        setattr(self, f'{label}_avg_orientations',  grain_avg_orientations)
        setattr(self, f'{label}_avg_quats',         grain_avg_quats)

        return cluster_to_grain

    def reconstruct_physical_lamellas(self, phase, boundary_result,
                                    parent_phase,
                                    parent_groups=None,
                                    threshold_deg=2.0,
                                    angle_tol_deg=10.0,
                                    collinearity_tol=2.0,
                                    gap_tol_factor=5.0):
        """
        Reconstruct physical martensite lamellae by merging segmentation fragments.

        A lamella is split into multiple clusters when low-CI or unindexed pixels
        interrupt the indexing along its length. Two child clusters A and B are
        merged if ALL of the following hold:

        1. **Direct neighbours** — A and B share a boundary in boundary_result,
        i.e. they are spatially adjacent (the gap between them is indexing noise).
        OR they share at least one common parent-phase neighbour and are
        sufficiently close geometrically (catches cases where the gap cluster
        was re-labelled as parent phase).
        2. **Same parent grain** — both A and B border at least one common
        parent-phase cluster (they are inside the same austenite grain).
        3. **Co-directional** — principal directions within angle_tol_deg.
        4. **Collinear** — perpendicular centroid offset < collinearity_tol × mean width.
        5. **Close** — gap along principal direction < gap_tol_factor × mean width.
        6. **Co-oriented** — disorientation between average orientations < threshold_deg,
        using proper rotation symmetry operations of the child phase.

        Merging is transitive via union-find.

        Parameters
        ----------
        phase : str
            Phase label of the child (lamella) phase, e.g. 'M'.
        boundary_result : BoundaryResult
            Output of BoundaryAnalyzer().analyze(clustering_result).
        parent_phase : str
            Phase label of the parent phase, e.g. 'A'.
        threshold_deg : float
            Maximum disorientation (degrees) between fragment average orientations.
            Default 2.0.
        angle_tol_deg : float
            Maximum angle (degrees) between principal directions. Default 10.0.
        collinearity_tol : float
            Maximum perpendicular offset in units of mean width. Default 2.0.
        gap_tol_factor : float
            Maximum gap along principal direction in units of mean width. Default 5.0.

        Returns
        -------
        lamella_groups : dict {root_id: [list of fragment cluster IDs]}
            Each key is the representative ID of a physical lamella.
            Single-fragment entries have a one-element list.
        disor_matrix : dict {(id_a, id_b): angle_deg}
            Disorientations for all candidate pairs that passed geometric tests.

        Notes
        -----
        This method does NOT modify self.labels. Call merge_physical_grains()
        afterwards with the returned lamella_groups to apply the relabelling,
        then re-run BoundaryAnalyzer and identify_lamellar_clusters.

        Workflow::

            boundary_result = BoundaryAnalyzer().analyze(clustering_result)
            lamella_groups, disor = clustering_result.reconstruct_physical_lamellas(
                phase='M', boundary_result=boundary_result, parent_phase='A')
            id_map = clustering_result.merge_physical_grains(
                lamella_groups, child_phase='A')
            clustering_result.update_grid_2d()
            boundary_result = BoundaryAnalyzer().analyze(clustering_result)
            lamellar_child_clusters = clustering_result.identify_lamellar_clusters(...)
        """
        from scipy.spatial.transform import Rotation as _R
        from collections import defaultdict

        # ── Phase and symmetry setup ──────────────────────────────────────────
        symops_all = np.array(self.data.phases[phase]['symops'])
        symops     = symops_all[np.array([round(np.linalg.det(s)) == 1
                                        for s in symops_all])]
        phase_id        = self.data.phase_ids[phase]
        parent_phase_id = self.data.phase_ids[parent_phase]

        # ── Build cluster → physical grain lookup ────────────────────────────
        # If parent_groups provided, use physical grain membership for same-grain
        # check. Otherwise fall back to individual cluster co-bordering.
        if parent_groups is not None:
            cluster_to_phys_grain = {}
            for root_id, fragments in parent_groups.items():
                for frag_id in fragments:
                    cluster_to_phys_grain[int(frag_id)] = int(root_id)
        else:
            cluster_to_phys_grain = None
            
        # All child cluster IDs with known orientations
        child_ids     = [
            cid for cid, ph in self.cluster_phases_id.items()
            if ph == phase_id and cid in self.avg_orientations
        ]
        child_id_set  = set(child_ids)

        # ── Pre-compute elongation for all child clusters ─────────────────────
        # (principal_direction, centroid, end_points, width needed for geometry)
        elongation = {}
        for cid in child_ids:
            e = self.analyze_cluster_elongation(cid)
            if e is not None:
                elongation[cid] = e
        # Only keep clusters with valid elongation
        child_ids    = [cid for cid in child_ids if cid in elongation]
        child_id_set = set(child_ids)

        # ── Build boundary lookups from boundary_result ───────────────────────
        br_clusters  = np.array(boundary_result.clusters)
        br_index     = {int(cid): i for i, cid in enumerate(br_clusters)}

        def _boundaries(cid):
            """Return grouped_boundaries dict for cluster cid, or {}."""
            idx = br_index.get(int(cid))
            return boundary_result.grouped_boundaries[idx] if idx is not None else {}

        def _boundary_phases(cid):
            """Return grouped_boundary_phases_id dict for cluster cid, or {}."""
            idx = br_index.get(int(cid))
            return (boundary_result.grouped_boundary_phases_id[idx]
                    if idx is not None else {})

        # Per child cluster: set of parent-phase neighbours
        child_parents = {}
        for cid in child_ids:
            nb_phases = _boundary_phases(cid)
            child_parents[cid] = {
                nb for nb, ph in nb_phases.items()
                if ph == parent_phase_id and nb != -1
            }

        # ── Angular tolerance (precompute cosine) ─────────────────────────────
        cos_tol = np.cos(np.radians(angle_tol_deg))

        # ── Collect candidate pairs ────────────────────────────────────────────
        # Candidate = direct neighbours OR share a common parent AND are close
        candidate_pairs = set()

        for i, cid_a in enumerate(child_ids):
            bnd_a      = _boundaries(cid_a)
            parents_a  = child_parents[cid_a]

            for nb_id, coords in bnd_a.items():
                # Direct boundary neighbour that is also a child cluster
                if nb_id in child_id_set and nb_id > cid_a:
                    candidate_pairs.add((cid_a, nb_id))

            for cid_b in child_ids[i + 1:]:
                if (cid_a, cid_b) in candidate_pairs:
                    continue
                parents_b = child_parents[cid_b]
                shared_clusters = parents_a & parents_b
                if not shared_clusters:
                    continue  # no shared austenite neighbour at all

                if cluster_to_phys_grain is not None:
                    # Strict check: shared clusters must map to same physical grain
                    phys_a = {cluster_to_phys_grain.get(int(c), int(c))
                            for c in parents_a}
                    phys_b = {cluster_to_phys_grain.get(int(c), int(c))
                            for c in parents_b}
                    if not (phys_a & phys_b):
                        continue  # different physical grains — skip
                
                candidate_pairs.add((cid_a, cid_b))

        # ── Pairwise tests: geometry + disorientation ─────────────────────────
        disor_matrix = {}
        merge_pairs  = []   # pairs that pass ALL tests

        for cid_a, cid_b in sorted(candidate_pairs):
            e_a   = elongation[cid_a]
            e_b   = elongation[cid_b]

            d_a   = np.array(e_a['principal_direction'], dtype=float)
            d_b   = np.array(e_b['principal_direction'], dtype=float)
            c_a   = np.array(e_a['centroid'],            dtype=float)
            c_b   = np.array(e_b['centroid'],            dtype=float)
            ep1_a = np.array(e_a['end_point_1'],         dtype=float)
            ep2_a = np.array(e_a['end_point_2'],         dtype=float)
            ep1_b = np.array(e_b['end_point_1'],         dtype=float)
            ep2_b = np.array(e_b['end_point_2'],         dtype=float)
            w_a   = float(e_a['width'])
            w_b   = float(e_b['width'])
            mean_w = (w_a + w_b) / 2.0

            # ── Test 1: co-directional ─────────────────────────────────────
            dot = abs(np.dot(d_a, d_b))
            if dot < cos_tol:
                continue

            # Sign-align d_b to d_a for subsequent geometry
            d_ref = d_a
            d_b_aligned = d_b if np.dot(d_a, d_b) >= 0 else -d_b

            # ── Test 2: collinear ──────────────────────────────────────────
            delta_c  = c_b - c_a
            perp     = delta_c - np.dot(delta_c, d_ref) * d_ref
            if np.linalg.norm(perp) > collinearity_tol * mean_w:
                continue

            # ── Test 3: gap along principal direction ──────────────────────
            proj = lambda ep: float(np.dot(ep - c_a, d_ref))
            p1a, p2a = proj(ep1_a), proj(ep2_a)
            p1b, p2b = proj(ep1_b), proj(ep2_b)
            min_a, max_a = min(p1a, p2a), max(p1a, p2a)
            min_b, max_b = min(p1b, p2b), max(p1b, p2b)
            gap = max(0.0, max(min_a, min_b) - min(max_a, max_b))
            if gap > gap_tol_factor * mean_w:
                continue

            # ── Test 4: disorientation ─────────────────────────────────────
            G_a = self.avg_orientations.get(cid_a)
            G_b = self.avg_orientations.get(cid_b)
            if G_a is None or G_b is None:
                continue

            R_ab = G_b @ G_a.T
            R_eq = np.einsum('sij,jk->sik', symops, R_ab)
            q    = _R.from_matrix(R_eq).as_quat()
            w    = np.clip(np.abs(q[:, 3]), -1.0, 1.0)
            misori = float(np.degrees(2 * np.arccos(w)).min())

            disor_matrix[(cid_a, cid_b)] = misori

            if misori > threshold_deg:
                continue

            merge_pairs.append((cid_a, cid_b))

        # ── Union-Find ────────────────────────────────────────────────────────
        uf = {cid: cid for cid in child_ids}

        def _find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        for cid_a, cid_b in merge_pairs:
            ra, rb = _find(cid_a), _find(cid_b)
            if ra != rb:
                uf[rb] = ra

        lamella_groups = defaultdict(list)
        for cid in child_ids:
            lamella_groups[_find(cid)].append(cid)
        self.child_lamella_groups = dict(lamella_groups)  # store in self for access by export_results
        self.lamella_disor_matrix = disor_matrix  # store in self for access by export_results
        return dict(lamella_groups), disor_matrix
    # ============================================================================
    # ROI SELECTION AROUND LAMELLAR CLUSTER
    # ============================================================================

    def create_lamellar_roi(self, cluster_id, n_layers=1, 
                       longitudinal_shrinkage=0.1,
                       layer_thickness_relative=None,
                       pixel_size=None, use_hexagonal=None,
                       elongation_cache=None):
        """
        Create a rectangular ROI around a lamellar cluster for interface analysis.
        
        The ROI is aligned with the principal axes of the lamella. By default, it's
        SHORTENED along the length (to exclude tip neighbors) and EXTENDED in width
        (to include neighbor layers from both long sides).
        
        This is designed for analyzing the long interfaces of the lamella with neighbors.
        
        Parameters
        ----------
        cluster_id : int
            Lamellar cluster ID
        n_layers : int or tuple, optional
            Number of neighbor layers to include perpendicular to long interfaces.
            If int: same number on both sides.
            If tuple (n_side1, n_side2): different for each side.
            Default is 1 (includes 1 layer of neighbors on each long side).
        longitudinal_shrinkage : float, optional
            Fraction to shrink along the long axis (to exclude tip neighbors).
            Default is 0.1 (remove 10% from each end).
        layer_thickness_relative : float, optional
            Thickness of one layer as fraction of cluster width.
            If None, uses cluster width as estimate.
            Default is None (uses width).
        pixel_size : float, optional
            Pixel size for physical units. Ignored if elongation_cache provides units.
        use_hexagonal : bool, optional
            Use hexagonal grid. Ignored if elongation_cache provides units.
        elongation_cache : dict or list, optional
            Pre-computed elongation results from identify_lamellar_clusters.
            If provided, avoids recomputing PCA.
            Can be dict {cluster_id: data} or list of cluster data.
        
        Returns
        -------
        roi_info : dict
            Dictionary containing:
            - 'vertices': Full ROI vertices (4, 2)
            - 'semi_roi_1_vertices': Half ROI on side 1 (4, 2)
            - 'semi_roi_2_vertices': Half ROI on side 2 (4, 2)
            - Other ROI geometry and cluster information
        
        Examples
        --------
        >>> all_clusters = clustering_result.identify_lamellar_clusters(
        ...     phase_name='Ferrite'
        ... )
        >>> roi = clustering_result.create_lamellar_roi(
        ...     cluster_id=42,
        ...     n_layers=2,
        ...     elongation_cache=all_clusters
        ... )
        >>> # Full ROI
        >>> full_vertices = roi['vertices']
        >>> # Half ROIs (split at cluster centerline)
        >>> semi1_vertices = roi['semi_roi_1_vertices']
        >>> semi2_vertices = roi['semi_roi_2_vertices']
        """
        # Get elongation data from cache or compute
        elongation_data = None
        
        if elongation_cache is not None:
            if isinstance(elongation_cache, dict):
                elongation_data = elongation_cache.get(cluster_id)
            elif isinstance(elongation_cache, list):
                for cluster_data in elongation_cache:
                    if cluster_data['cluster_id'] == cluster_id:
                        elongation_data = cluster_data
                        break
        #print(elongation_data)
        if elongation_data is None:
            elongation = self.analyze_cluster_elongation(cluster_id, use_sklearn=False)
            if elongation is None:
                raise ValueError(f"Cluster {cluster_id} not found or too small")
            
            phase_id = self.cluster_phases_id[cluster_id]
            
            elongation_data = {
                'cluster_id': cluster_id,
                'phase_id': phase_id,
                'phase_name': self.data.phase_names[phase_id],
                'aspect_ratio': elongation['aspect_ratio'],
                'length': elongation['length'],
                'width': elongation['width'],
                'orientation_angle': elongation['principal_angle'],
                'principal_direction': elongation['principal_direction'],
                'perpendicular_direction': elongation['perpendicular_direction'],
                'centroid': elongation['centroid'],
                'end_point_1': elongation['end_point_1'],
                'end_point_2': elongation['end_point_2'],
                'area': elongation['area'],
                'n_pixels': elongation['n_pixels'],
                'eigenvalues': elongation['eigenvalues'],
                'pca_method': elongation['pca_method'],
            }
        
        aspect_ratio = elongation_data['aspect_ratio']
        if aspect_ratio < 2.5:
            print(f"Warning: Cluster {cluster_id} has aspect ratio {aspect_ratio:.2f}, "
                f"which may not be lamellar (< 2.5)")
        
        if longitudinal_shrinkage < 0 or longitudinal_shrinkage >= 0.5:
            raise ValueError(f"longitudinal_shrinkage must be in [0, 0.5), got {longitudinal_shrinkage}")
        
        # Get elongation parameters
        centroid = elongation_data['centroid']
        principal_dir = elongation_data['principal_direction']
        perpendicular_dir = elongation_data['perpendicular_direction']
        length = elongation_data['length']
        width = elongation_data['width']
        
        # Parse n_layers
        if isinstance(n_layers, int):
            n_layers_side1 = n_layers
            n_layers_side2 = n_layers
        else:
            n_layers_side1, n_layers_side2 = n_layers
        
        # Determine layer thickness
        if layer_thickness_relative is None:
            layer_thickness = width
        else:
            layer_thickness = width * layer_thickness_relative
        
        # Calculate ROI dimensions
        length_roi = length * (1.0 - 2.0 * longitudinal_shrinkage)
        
        if length_roi <= 0:
            raise ValueError(f"longitudinal_shrinkage={longitudinal_shrinkage} is too large")
        
        width_side1 = n_layers_side1 * layer_thickness
        width_side2 = n_layers_side2 * layer_thickness
        width_roi = width + width_side1 + width_side2
        
        # Calculate center offset
        width_offset = (width_side1 - width_side2) / 2
        center_roi = centroid + perpendicular_dir * width_offset
        
        # Define corners in local coordinate system
        half_length = length_roi / 2
        half_width = width_roi / 2
        
        # ========== FULL ROI VERTICES ==========
        corners_local = np.array([
            [-half_length, -half_width],  # Corner 0: bottom-left
            [-half_length, +half_width],  # Corner 1: top-left
            [+half_length, +half_width],  # Corner 2: top-right
            [+half_length, -half_width]   # Corner 3: bottom-right
        ])
        
        rotation_matrix = np.column_stack([principal_dir, perpendicular_dir])
        
        vertices = np.zeros((4, 2))
        for i in range(4):
            vertices[i] = center_roi + rotation_matrix @ corners_local[i]
        
        # ========== SEMI-ROI 1 VERTICES (Side 1, positive perpendicular direction) ==========
        # This semi-ROI goes from centerline to the outer edge on side 1
        # Width: from centerline (0) to +half_width
        semi1_half_width = half_width / 2  # Half of the full ROI width, centered on side 1
        semi1_center_offset = half_width / 2  # Shift center toward side 1
        
        semi1_corners_local = np.array([
            [-half_length, 0],                    # Corner 0: centerline start
            [-half_length, +half_width],          # Corner 1: outer edge start
            [+half_length, +half_width],          # Corner 2: outer edge end
            [+half_length, 0]                     # Corner 3: centerline end
        ])
        
        semi_roi_1_vertices = np.zeros((4, 2))
        for i in range(4):
            semi_roi_1_vertices[i] = center_roi + rotation_matrix @ semi1_corners_local[i]
        
        # ========== SEMI-ROI 2 VERTICES (Side 2, negative perpendicular direction) ==========
        # This semi-ROI goes from centerline to the outer edge on side 2
        # Width: from -half_width to centerline (0)
        semi2_corners_local = np.array([
            [-half_length, -half_width],          # Corner 0: outer edge start
            [-half_length, 0],                    # Corner 1: centerline start
            [+half_length, 0],                    # Corner 2: centerline end
            [+half_length, -half_width]           # Corner 3: outer edge end
        ])
        
        semi_roi_2_vertices = np.zeros((4, 2))
        for i in range(4):
            semi_roi_2_vertices[i] = center_roi + rotation_matrix @ semi2_corners_local[i]
        
        # ========================================
        
        # Calculate the two long edges (full ROI)
        long_edge_1 = (vertices[1], vertices[2])  # Side 1 (top edge)
        long_edge_2 = (vertices[0], vertices[3])  # Side 2 (bottom edge)
        
        # Calculate the centerline (dividing the two semi-ROIs)
        centerline_start = center_roi + rotation_matrix @ np.array([-half_length, 0])
        centerline_end = center_roi + rotation_matrix @ np.array([+half_length, 0])
        centerline = (centerline_start, centerline_end)
        
        removed_length_per_end = length * longitudinal_shrinkage
        
        return {
            # Full ROI
            'vertices': vertices,
            'center': center_roi,
            'width_roi': width_roi,
            'length_roi': length_roi,
            
            # Semi-ROIs (half thickness each)
            'semi_roi_1_vertices': semi_roi_1_vertices,
            'semi_roi_2_vertices': semi_roi_2_vertices,
            'semi_roi_1_width': half_width,
            'semi_roi_2_width': half_width,
            'centerline': centerline,
            
            # Orientation
            'orientation_angle': elongation_data['orientation_angle'],
            'principal_direction': principal_dir,
            'perpendicular_direction': perpendicular_dir,
            
            # Cluster info
            'cluster_id': cluster_id,
            'n_layers': (n_layers_side1, n_layers_side2),
            'longitudinal_shrinkage': longitudinal_shrinkage,
            'removed_length_per_end': removed_length_per_end,
            'layer_thickness': layer_thickness,
            'cluster_width': width,
            'cluster_length': length,
            
            # Edges
            'long_edge_1': long_edge_1,
            'long_edge_2': long_edge_2,
            'width_side1': width_side1,
            'width_side2': width_side2,
            
            # Data
            'cluster_info': elongation_data
        }
    
    def create_lamellar_roi_mask(self, cluster_id, n_layers=1,
                             longitudinal_shrinkage=0.1,
                             layer_thickness_relative=None,
                                elongation_cache=None):
        """
        Create boolean masks for the ROI around a lamellar cluster.
        
        Creates masks for the full ROI and both semi-ROIs.
        
        Parameters
        ----------
        cluster_id : int
            Lamellar cluster ID
        n_layers : int or tuple, optional
            Number of layers on each side. Default is 1.
        longitudinal_shrinkage : float, optional
            Fraction to remove from each end. Default is 0.1.
        layer_thickness_relative : float, optional
            Layer thickness as fraction of width. Default is None.
        elongation_cache : dict or list, optional
            Pre-computed elongation results. Avoids recomputing PCA.
        
        Returns
        -------
        roi_masks : dict
            Dictionary containing:
            - 'full_roi': Boolean mask for full ROI
            - 'semi_roi_1': Boolean mask for semi-ROI 1
            - 'semi_roi_2': Boolean mask for semi-ROI 2
        roi_info : dict
            ROI information from create_lamellar_roi
        
        Examples
        --------
        >>> all_clusters = clustering_result.identify_lamellar_clusters(
        ...     phase_name='Ferrite'
        ... )
        >>> roi_masks, roi_info = clustering_result.create_lamellar_roi_mask(
        ...     cluster_id=42,
        ...     n_layers=2,
        ...     elongation_cache=all_clusters
        ... )
        >>> 
        >>> # Access masks
        >>> full_mask = roi_masks['full_roi']
        >>> semi1_mask = roi_masks['semi_roi_1']
        >>> semi2_mask = roi_masks['semi_roi_2']
        >>> 
        >>> print(f"Full ROI: {np.sum(full_mask)} points")
        >>> print(f"Semi-ROI 1: {np.sum(semi1_mask)} points")
        >>> print(f"Semi-ROI 2: {np.sum(semi2_mask)} points")
        """
        # Get ROI vertices (using cache)
        roi_info = self.create_lamellar_roi(
            cluster_id, n_layers,
            longitudinal_shrinkage,
            layer_thickness_relative,
            elongation_cache=elongation_cache
        )
        
        # Create mask using point-in-polygon test
        from matplotlib.path import Path
        
        # All data points
        points = np.column_stack([self.data.X, self.data.Y])
        
        # Full ROI mask
        full_vertices = roi_info['vertices']
        full_path = Path(full_vertices)
        full_roi_mask = full_path.contains_points(points)
        
        # Semi-ROI 1 mask
        semi1_vertices = roi_info['semi_roi_1_vertices']
        semi1_path = Path(semi1_vertices)
        semi_roi_1_mask = semi1_path.contains_points(points)
        
        # Semi-ROI 2 mask
        semi2_vertices = roi_info['semi_roi_2_vertices']
        semi2_path = Path(semi2_vertices)
        semi_roi_2_mask = semi2_path.contains_points(points)
        
        roi_masks = {
            'full_roi': full_roi_mask,
            'semi_roi_1': semi_roi_1_mask,
            'semi_roi_2': semi_roi_2_mask
        }
        
        return roi_masks, roi_info

    def extract_roi_data(self, cluster_id, n_layers=1,
                        longitudinal_shrinkage=0.1,
                        layer_thickness_relative=None,
                        elongation_cache=None):
        """
        Extract all EBSD data within the ROI around a lamellar cluster.
        
        Extracts data for the full ROI and both semi-ROIs.
        
        Parameters
        ----------
        cluster_id : int
            Lamellar cluster ID
        n_layers : int or tuple, optional
            Number of layers. Default is 1.
        longitudinal_shrinkage : float, optional
            Fraction to remove from each end. Default is 0.1.
        layer_thickness_relative : float, optional
            Layer thickness. Default is None.
        elongation_cache : dict or list, optional
            Pre-computed elongation results. Avoids recomputing PCA.
        
        Returns
        -------
        roi_data : dict
            Dictionary containing data for full ROI and semi-ROIs:
            - 'full_roi': Data for complete ROI
            - 'semi_roi_1': Data for semi-ROI 1
            - 'semi_roi_2': Data for semi-ROI 2
            - 'roi_info': ROI geometry information
        
        Examples
        --------
        >>> all_clusters = clustering_result.identify_lamellar_clusters(
        ...     phase_name='Ferrite'
        ... )
        >>> roi_data = clustering_result.extract_roi_data(
        ...     cluster_id=42,
        ...     n_layers=2,
        ...     elongation_cache=all_clusters
        ... )
        >>> 
        >>> # Access full ROI data
        >>> full = roi_data['full_roi']
        >>> print(f"Full ROI: {len(full['X'])} points")
        >>> print(f"Clusters in full ROI: {full['unique_clusters']}")
        >>> 
        >>> # Access semi-ROI 1 data
        >>> semi1 = roi_data['semi_roi_1']
        >>> print(f"Semi-ROI 1: {len(semi1['X'])} points")
        >>> print(f"Clusters in semi-ROI 1: {semi1['unique_clusters']}")
        >>> 
        >>> # Access semi-ROI 2 data
        >>> semi2 = roi_data['semi_roi_2']
        >>> print(f"Semi-ROI 2: {len(semi2['X'])} points")
        """

        # Get ROI masks (using cache)
        roi_masks, roi_info = self.create_lamellar_roi_mask(
            cluster_id, n_layers,
            longitudinal_shrinkage,
            layer_thickness_relative,
            elongation_cache=elongation_cache
        )
        
        full_mask = roi_masks['full_roi']
        semi1_mask = roi_masks['semi_roi_1']
        semi2_mask = roi_masks['semi_roi_2']
        
        # Extract data for full ROI
        full_roi_data = {
            'mask': full_mask,
            'X': self.data.X[full_mask],
            'Y': self.data.Y[full_mask],
            'labels': self.labels[full_mask],
            'phases_id': self.data.phases_id[full_mask],
            'n_points': np.sum(full_mask)
        }
        
        # Add quaternions if available
        if hasattr(self.data, 'quaternions') and self.data.quaternions is not None:
            full_roi_data['quaternions'] = self.data.quaternions[full_mask]
        
        # Get unique clusters
        full_roi_data['unique_clusters'] = np.unique(full_roi_data['labels'][full_roi_data['labels'] > 0])
        
        # Extract data for semi-ROI 1
        semi_roi_1_data = {
            'mask': semi1_mask,
            'X': self.data.X[semi1_mask],
            'Y': self.data.Y[semi1_mask],
            'labels': self.labels[semi1_mask],
            'phases_id': self.data.phases_id[semi1_mask],
            'n_points': np.sum(semi1_mask)
        }
        
        if hasattr(self.data, 'quaternions') and self.data.quaternions is not None:
            semi_roi_1_data['quaternions'] = self.data.quaternions[semi1_mask]
        
        semi_roi_1_data['unique_clusters'] = np.unique(semi_roi_1_data['labels'][semi_roi_1_data['labels'] > 0])
        
        # Extract data for semi-ROI 2
        semi_roi_2_data = {
            'mask': semi2_mask,
            'X': self.data.X[semi2_mask],
            'Y': self.data.Y[semi2_mask],
            'labels': self.labels[semi2_mask],
            'phases_id': self.data.phases_id[semi2_mask],
            'n_points': np.sum(semi2_mask)
        }
        
        if hasattr(self.data, 'quaternions') and self.data.quaternions is not None:
            semi_roi_2_data['quaternions'] = self.data.quaternions[semi2_mask]
        
        semi_roi_2_data['unique_clusters'] = np.unique(semi_roi_2_data['labels'][semi_roi_2_data['labels'] > 0])
        
        return {
            'cluster_id': cluster_id,
            'full_roi': full_roi_data,
            'semi_roi_1': semi_roi_1_data,
            'semi_roi_2': semi_roi_2_data,
            'roi_info': roi_info
        }
    
    def visualize_lamellar_roi(self, cluster_id, n_layers=1,
                            longitudinal_shrinkage=0.1,
                            layer_thickness_relative=None,
                            show_neighbors=True,
                            show_long_edges=True,
                            show_removed_regions=True,
                            show_semi_rois=True,
                            elongation_cache=None,invert_y_axis=True,
                            figsize=(12, 10)):
        """
        Visualize a lamellar cluster with its ROI for interface analysis.
        
        Parameters
        ----------
        cluster_id : int
            Lamellar cluster ID
        n_layers : int or tuple, optional
            Number of layers perpendicular to long interfaces. Default is 1.
        longitudinal_shrinkage : float, optional
            Fraction to remove from each end. Default is 0.1.
        layer_thickness_relative : float, optional
            Layer thickness. Default is None.
        show_neighbors : bool, optional
            Show neighboring clusters. Default is True.
        show_long_edges : bool, optional
            Highlight the two long edges. Default is True.
        show_removed_regions : bool, optional
            Show the regions removed from tips. Default is True.
        show_semi_rois : bool, optional
            Show the two half-thickness semi-ROIs. Default is True.
        elongation_cache : dict or list, optional
            Pre-computed elongation results from identify_lamellar_clusters.
            Avoids recomputing PCA.
        figsize : tuple, optional
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib figure and axis
        roi_info : dict
            ROI information
        
        Examples
        --------
        >>> # Without cache
        >>> fig, ax, roi = clustering_result.visualize_lamellar_roi(
        ...     cluster_id=42, n_layers=2
        ... )
        >>> 
        >>> # With cache (efficient for multiple visualizations)
        >>> all_clusters = clustering_result.identify_lamellar_clusters(
        ...     phase_name='Ferrite'
        ... )
        >>> fig, ax, roi = clustering_result.visualize_lamellar_roi(
        ...     cluster_id=42,
        ...     n_layers=2,
        ...     show_semi_rois=True,
        ...     elongation_cache=all_clusters
        ... )
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        
        # Get ROI masks and info (using cache if provided)
        roi_masks, roi_info = self.create_lamellar_roi_mask(
            cluster_id, n_layers,
            longitudinal_shrinkage,
            layer_thickness_relative,
            elongation_cache=elongation_cache
        )
        
        # Use full ROI mask for visualization
        roi_mask = roi_masks['full_roi']
        
        info = roi_info['cluster_info']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all clusters in ROI
        if show_neighbors:
            roi_coords = np.column_stack([self.data.X[roi_mask], self.data.Y[roi_mask]])
            roi_labels = self.labels[roi_mask]
            
            unique_roi_clusters = np.unique(roi_labels[roi_labels > 0])
            
            import matplotlib.cm as cm
            colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_roi_clusters)))
            
            for i, c in enumerate(unique_roi_clusters):
                cluster_mask_roi = roi_labels == c
                coords = roi_coords[cluster_mask_roi]
                
                if c == cluster_id:
                    ax.plot(coords[:, 0], coords[:, 1], 'o', 
                        color='red', markersize=4, alpha=0.8,
                        label=f'Cluster {c} (lamellar)')
                else:
                    ax.plot(coords[:, 0], coords[:, 1], 'o',
                        color=colors[i], markersize=3, alpha=0.6,
                        label=f'Cluster {c}' if i < 5 else '')
        else:
            cluster_mask = self.labels == cluster_id
            coords = np.column_stack([self.data.X[cluster_mask], self.data.Y[cluster_mask]])
            ax.plot(coords[:, 0], coords[:, 1], 'ro', markersize=4, 
                label=f'Cluster {cluster_id} (lamellar)')
        
        # Show removed regions (tips that are excluded)
        if show_removed_regions and longitudinal_shrinkage > 0:
            centroid = info['centroid']
            principal_dir = roi_info['principal_direction']
            perpendicular_dir = roi_info['perpendicular_direction']
            cluster_length = roi_info['cluster_length']
            cluster_width = roi_info['cluster_width']
            removed_length = roi_info['removed_length_per_end']
            
            # Create rectangles for removed tip regions
            rotation_matrix = np.column_stack([principal_dir, perpendicular_dir])
            
            # Tip 1 (positive end)
            tip1_corners_local = np.array([
                [cluster_length/2 - removed_length, -cluster_width/2],
                [cluster_length/2, -cluster_width/2],
                [cluster_length/2, cluster_width/2],
                [cluster_length/2 - removed_length, cluster_width/2]
            ])
            tip1_corners = np.array([centroid + rotation_matrix @ c for c in tip1_corners_local])
            
            # Tip 2 (negative end)
            tip2_corners_local = np.array([
                [-cluster_length/2, -cluster_width/2],
                [-cluster_length/2 + removed_length, -cluster_width/2],
                [-cluster_length/2 + removed_length, cluster_width/2],
                [-cluster_length/2, cluster_width/2]
            ])
            tip2_corners = np.array([centroid + rotation_matrix @ c for c in tip2_corners_local])
            
            # Plot removed regions
            tip1_poly = Polygon(tip1_corners, fill=True, facecolor='yellow', 
                            edgecolor='orange', linewidth=2, alpha=0.3,
                            label=f'Removed tips ({longitudinal_shrinkage*100:.0f}% each)')
            tip2_poly = Polygon(tip2_corners, fill=True, facecolor='yellow', 
                            edgecolor='orange', linewidth=2, alpha=0.3)
            ax.add_patch(tip1_poly)
            ax.add_patch(tip2_poly)
        
        # Plot semi-ROIs (half-width ROIs)
        if show_semi_rois:
            # Semi-ROI 1 (side 1)
            semi1_vertices = roi_info['semi_roi_1_vertices']
            semi1_poly = Polygon(semi1_vertices, fill=False, edgecolor='cyan', 
                            linewidth=2, linestyle=':',
                            label=f'Semi-ROI 1 (width={roi_info["semi_roi_1_width"]:.1f})')
            ax.add_patch(semi1_poly)
            
            # Semi-ROI 2 (side 2)
            semi2_vertices = roi_info['semi_roi_2_vertices']
            semi2_poly = Polygon(semi2_vertices, fill=False, edgecolor='magenta', 
                            linewidth=2, linestyle=':',
                            label=f'Semi-ROI 2 (width={roi_info["semi_roi_2_width"]:.1f})')
            ax.add_patch(semi2_poly)
            
            # Plot centerline dividing the two semi-ROIs
            centerline = roi_info['centerline']
            ax.plot([centerline[0][0], centerline[1][0]], 
                [centerline[0][1], centerline[1][1]],
                'k--', linewidth=1.5, alpha=0.6, label='Centerline')
        
        # Plot full ROI rectangle
        vertices = roi_info['vertices']
        polygon = Polygon(vertices, fill=False, edgecolor='blue', 
                        linewidth=3, linestyle='--',
                        label='Full ROI')
        ax.add_patch(polygon)
        
        # Highlight long edges if requested
        if show_long_edges:
            long_edge_1 = roi_info['long_edge_1']
            long_edge_2 = roi_info['long_edge_2']
            
            # Long edge 1 (side 1)
            ax.plot([long_edge_1[0][0], long_edge_1[1][0]], 
                [long_edge_1[0][1], long_edge_1[1][1]],
                'g-', linewidth=5, alpha=0.8, 
                label=f'Long edge 1 ({roi_info["n_layers"][0]} layers)')
            
            # Long edge 2 (side 2)
            ax.plot([long_edge_2[0][0], long_edge_2[1][0]], 
                [long_edge_2[0][1], long_edge_2[1][1]],
                'm-', linewidth=5, alpha=0.8,
                label=f'Long edge 2 ({roi_info["n_layers"][1]} layers)')
        
        # Plot cluster centroid
        centroid = info['centroid']
        ax.plot(centroid[0], centroid[1], 'r*', markersize=15,
            label='Cluster centroid')
        
        # Plot ROI center
        center = roi_info['center']
        ax.plot(center[0], center[1], 'b^', markersize=12, 
            label='ROI center')
        
        # Plot principal axes
        principal_dir = info['principal_direction']
        perpendicular_dir = info['perpendicular_direction']
        length = info['length']
        width = info['width']
        
        # Principal axis (long) - along length
        scale = length * 0.6
        ax.arrow(centroid[0], centroid[1],
                principal_dir[0] * scale, principal_dir[1] * scale,
                color='darkred', width=1.0, head_width=4, head_length=3,
                alpha=0.7)
        ax.arrow(centroid[0], centroid[1],
                -principal_dir[0] * scale, -principal_dir[1] * scale,
                color='darkred', width=1.0, head_width=4, head_length=3,
                alpha=0.7, label='Long axis')
        
        # Perpendicular axis (short) - across width
        scale = width * 0.6
        ax.arrow(centroid[0], centroid[1],
                perpendicular_dir[0] * scale, perpendicular_dir[1] * scale,
                color='darkblue', width=1.0, head_width=4, head_length=3,
                alpha=0.7)
        ax.arrow(centroid[0], centroid[1],
                -perpendicular_dir[0] * scale, -perpendicular_dir[1] * scale,
                color='darkblue', width=1.0, head_width=4, head_length=3,
                alpha=0.7, label='Short axis')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Build title
        title = f'Lamellar Cluster {cluster_id} - Interface Analysis ROI\n'
        title += f'Aspect ratio: {info["aspect_ratio"]:.2f}, '
        title += f'Angle: {info["orientation_angle"]:.1f}°\n'
        title += f'Cluster: {info["length"]:.1f} (L) × {info["width"]:.1f} (W), '
        title += f'ROI: {roi_info["length_roi"]:.1f} (L) × {roi_info["width_roi"]:.1f} (W)\n'
        title += f'Shrinkage: {longitudinal_shrinkage*100:.0f}% each end ({roi_info["removed_length_per_end"]:.1f} removed), '
        title += f'Layers: {roi_info["n_layers"]}'
        if show_semi_rois:
            title += f'\nSemi-ROIs: Each {roi_info["semi_roi_1_width"]:.1f} width (half of full ROI)'
        
        ax.set_title(title, fontsize=10)
        ax.legend(loc='best', fontsize=8, ncol=2)  # Two columns for better layout
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        if invert_y_axis:
            ax.invert_yaxis()
        plt.tight_layout()
        
        return fig, ax, roi_info
    
# ============================================================================
# MISORIENTATION ANALYZER
# ============================================================================
    def compute_pixel_neighbor_misorientations(self, phase, distance=1, perimeteronly=True,
                                                distance_convention="OIM", roi=None):
        """
        Vectorized per-pixel, per-neighbor-slot misorientation angles (deg),
        via batched orientation-matrix rotation composition (same approach
        as compute_cluster_boundary_misorientations_fast), looping only over
        neighbor slots (typically 6-12) rather than over pixels or pixel
        pairs individually.

        Shared core for:
        - get_pixel_boundary_misorientations (pairs straddling an actual
        cluster boundary, i.e. self.labels differs)
        - get_pixel_max_neighbor_misorientation (per-pixel max across all
        neighbors, matching the legacy KAM-style
        "mis = np.max(misang, axis=1)" pattern used for map coloring)

        Parameters
        ----------
        phase : str
            Phase to restrict to. Only pixel pairs where BOTH pixels belong
            to this phase are considered; other entries stay -1 (invalid).
        distance, perimeteronly, distance_convention, roi :
            Passed through to self.data.compute_neighbors().

        Returns
        -------
        misang : (N, n_neighbors) ndarray
            Misorientation angle (deg), pixel i to its k-th neighbor.
            -1 where the neighbor is missing, outside ROI, or outside phase.
        neighbors : (N, n_neighbors) ndarray
            The neighbor index array used (from self.data.compute_neighbors).
        phase_mask : (N,) bool ndarray
            The phase+ROI mask used.
        """
        from scipy.spatial.transform import Rotation as _R

        phase_id = self.data.phase_ids[phase]
        symops_all = np.array(self.data.phases[phase]['symops'])
        symops = symops_all[np.array([round(np.linalg.det(s)) == 1 for s in symops_all])]

        neighbors = self.data.compute_neighbors(
            distance=distance, perimeteronly=perimeteronly,
            distance_convention=distance_convention, roi=roi
        )

        if roi is None:
            sel = self.data.rois.masks[0]
        else:
            sel = self.data.rois.masks[roi]

        phase_mask = (self.data.phases_id == phase_id) & sel

        N = self.data.N
        n_neighbors = neighbors.shape[1]
        misang = np.full((N, n_neighbors), -1.0, dtype=float)

        # Convert quaternions -> matrices once, only for pixels in this
        # phase/ROI (not the whole map).
        idxs = np.where(phase_mask)[0]
        M_all = np.full((N, 3, 3), np.nan)
        M_all[idxs] = np.array([quat_to_mat(q) for q in self.data.quaternions[idxs]])

        for k in range(n_neighbors):
            j = neighbors[:, k]
            valid = phase_mask & (j >= 0)
            valid[valid] &= phase_mask[j[valid]]
            if not np.any(valid):
                continue

            i_idx = np.where(valid)[0]
            j_idx = j[i_idx]

            A = M_all[i_idx]  # (P,3,3)
            B = M_all[j_idx]  # (P,3,3)

            R_ab = np.einsum("pij,pkj->pik", B, A, optimize=True)
            R_eq = np.einsum("sij,pjk->spik", symops, R_ab, optimize=True).reshape(-1, 3, 3)

            q = _R.from_matrix(R_eq).as_quat().reshape(len(symops), len(i_idx), 4)
            w = np.clip(np.abs(q[..., 3]), -1.0, 1.0)
            ang = np.degrees(2 * np.arccos(w))
            min_ang = np.min(ang, axis=0)

            misang[i_idx, k] = min_ang

        return misang, neighbors, phase_mask


    def get_pixel_boundary_misorientations(self, phase, distance=1, perimeteronly=True,
                                            distance_convention="OIM", roi=None):
        """
        Misorientation angles (deg) for pixel-pairs straddling an actual
        cluster boundary (self.labels differs), built from
        compute_pixel_neighbor_misorientations.

        Returns
        -------
        angles : (M,) ndarray
            One value per unique boundary pixel-pair (each physical pair
            counted once).
        pair_cluster_ids : (M, 2) ndarray
            The two (sorted) cluster labels for each pair.
        """
        misang, neighbors, phase_mask = self.compute_pixel_neighbor_misorientations(
            phase, distance=distance, perimeteronly=perimeteronly,
            distance_convention=distance_convention, roi=roi
        )

        labels = self.labels
        n_neighbors = misang.shape[1]

        angles_list = []
        pairs_list = []
        seen = set()

        for k in range(n_neighbors):
            j = neighbors[:, k]
            valid = misang[:, k] >= 0
            if not np.any(valid):
                continue
            i_idx = np.where(valid)[0]
            j_idx = j[i_idx]

            li = labels[i_idx]
            lj = labels[j_idx]
            boundary = (li != lj) & (li > 0) & (lj > 0)
            if not np.any(boundary):
                continue

            i_b = i_idx[boundary]
            j_b = j_idx[boundary]
            ang_b = misang[i_b, k]

            for ii, jj, aa in zip(i_b, j_b, ang_b):
                key = (ii, jj) if ii < jj else (jj, ii)
                if key in seen:
                    continue
                seen.add(key)
                angles_list.append(aa)
                pairs_list.append(tuple(sorted((int(labels[ii]), int(labels[jj])))))

        return np.array(angles_list), np.array(pairs_list)


    def get_pixel_max_neighbor_misorientation(self, phase, distance=1, perimeteronly=True,
                                            distance_convention="OIM", roi=None):
        """
        Per-pixel maximum misorientation to its neighbors (deg) -- vectorized
        equivalent of the legacy KAM-style "mis = np.max(misang, axis=1)"
        pattern, used to build a color array highlighting grain boundaries
        on a map.

        Returns
        -------
        mis : (N,) ndarray
            Per-pixel max neighbor misorientation. -1 outside phase/ROI or
            for pixels with zero valid neighbors.
        """
        misang, neighbors, phase_mask = self.compute_pixel_neighbor_misorientations(
            phase, distance=distance, perimeteronly=perimeteronly,
            distance_convention=distance_convention, roi=roi
        )
        return np.max(misang, axis=1)

    

# ============================================================================
# BOUNDARY ANALYZER
# ============================================================================
class BoundaryResult:
    """
    Container for boundary analysis results.
    
    Stores all boundary information with convenient access methods.
    """
    
    def __init__(self, clustering_result: ClusteringResult):
        self.clustering = clustering_result
        self.data = clustering_result.data
        
        # Raw boundary data
        self.clusters = None
        self.flat_neighbors = None
        self.flat_lengths = None
        self.slices = None
        
        # Boundary pixels
        self.boundary_x = None
        self.boundary_y = None
        self.boundary_nb = None
        self.boundary_nb_phase_id = None
        self.boundary_slices = None
        
        # Grouped boundaries
        self.grouped_boundaries = None
        self.grouped_boundary_phases_id = None
        self.cluster_phases_id = None
        
        # Representative points
        self.rep_points = None
        self.rep_point_info = None
    
    def get_cluster_boundaries(self, cluster_label):
        """Get all boundaries for a specific cluster."""
        cluster_idx = np.where(self.clusters == cluster_label)[0][0]
        return self.grouped_boundaries[cluster_idx]
    
    def get_interphase_neighbors(self, cluster_label):
        """Get inter-phase neighbors of a cluster."""
        cluster_idx = np.where(self.clusters == cluster_label)[0][0]
        cluster_phase_id = self.cluster_phases_id[cluster_idx]
        
        boundaries = self.grouped_boundaries[cluster_idx]
        boundary_phases_id = self.grouped_boundary_phases_id[cluster_idx]
        
        interphase = {}
        for nb_label, coords in boundaries.items():
            if nb_label == -1:
                continue
            nb_phase_id = boundary_phases_id[nb_label]
            if nb_phase_id != cluster_phase_id:
                interphase[nb_label] = {
                    'phase_id': nb_phase_id,
                    'coords': coords,
                    'n_pixels': len(coords)
                }
        
        return interphase
    
    def list_phase_pairs(self):
        """
        List all phase pairs that have inter-phase boundaries.
        """
        phase_pairs = set()
        
        for i, phases_dict in enumerate(self.grouped_boundary_phases_id):
            cluster_phase = self.cluster_phases_id[i]
            
            for neighbor_label, neighbor_phase in phases_dict.items():
                if neighbor_phase == -1:  # Skip ROI
                    continue
                if neighbor_phase != cluster_phase:
                    # Store as sorted tuple to avoid duplicates
                    pair = tuple(sorted([cluster_phase, neighbor_phase]))
                    phase_pairs.add(pair)
        
        print("Available phase pairs with inter-phase boundaries:")
        for p1, p2 in sorted(phase_pairs):
            print(f"  Phase {p1} ↔ Phase {p2}")
        
        return list(phase_pairs)


    def get_phase_boundaries(self, phase_id_1, phase_id_2):
        """Get all boundaries between two phases."""
        boundaries_p1_to_p2 = []
        boundaries_p2_to_p1 = []
        
        for i, cluster_label in enumerate(self.clusters):
            cluster_phase_id = self.cluster_phases_id[i]
            
            if cluster_phase_id == phase_id_1:
                boundaries = self.grouped_boundaries[i]
                boundary_phases_id = self.grouped_boundary_phases_id[i]
                
                for nb_label, coords in boundaries.items():
                    if nb_label == -1:
                        continue
                    if boundary_phases_id[nb_label] == phase_id_2:
                        boundaries_p1_to_p2.append({
                            'cluster': cluster_label,
                            'neighbor': nb_label,
                            'coords': coords
                        })
        
        return boundaries_p1_to_p2, boundaries_p2_to_p1
    
    def get_enclosed_clusters(self, cluster_id, target_phase_id=None):
        """
        Find all clusters that have their longest interface with the given cluster.
        
        A cluster is considered "enclosed" if:
        1. It shares a boundary with cluster_id
        2. Its longest boundary is with cluster_id (compared to all other boundaries)
        
        Parameters
        ----------
        cluster_id : int
            Cluster label to analyze (the "container" cluster)
        target_phase_id : int, optional
            Phase ID of clusters to search for.
            If None, searches for all phases different from the container cluster's phase.
        
        Returns
        -------
        enclosed_clusters : dict
            Dictionary with structure:
            {
                'cluster_id': cluster_id,
                'cluster_phase_id': phase of container cluster,
                'target_phase_id': target_phase_id,
                'enclosed': [list of cluster IDs],
                'n_enclosed': number of enclosed clusters,
                'boundary_lengths': {enclosed_cluster_id: boundary_length}
            }
        """
        # Find cluster index
        try:
            cluster_idx = np.where(self.clusters == cluster_id)[0][0]
        except IndexError:
            raise ValueError(f"Cluster {cluster_id} not found!")
        
        # Get phase of this cluster
        cluster_phase_id = self.cluster_phases_id[cluster_idx]
        
        # Determine target phase
        if target_phase_id is None:
            # Find all other phases
            all_phases = np.unique(list(self.cluster_phases_id))
            target_phases = [p for p in all_phases if p != cluster_phase_id]
        else:
            target_phases = [target_phase_id]
        
        # Get boundaries of this cluster
        boundaries = self.grouped_boundaries[cluster_idx]
        boundary_phases = self.grouped_boundary_phases_id[cluster_idx]
        
        # Find all clusters that touch cluster_id and are in target phase
        candidate_neighbors = {}
        
        for neighbor_label, coords in boundaries.items():
            if neighbor_label == -1:
                continue  # Skip ROI
            
            neighbor_phase = boundary_phases[neighbor_label]
            
            # Check if neighbor is in target phase
            if neighbor_phase in target_phases:
                # Store boundary length with cluster_id
                boundary_length = len(coords)
                candidate_neighbors[neighbor_label] = boundary_length
        
        # Now check each candidate: is this the longest boundary for that cluster?
        enclosed_clusters = []
        boundary_lengths = {}
        
        for candidate_id, length_with_cluster_id in candidate_neighbors.items():
            # Find candidate's index
            try:
                candidate_idx = np.where(self.clusters == candidate_id)[0][0]
            except IndexError:
                continue
            
            # Get all boundaries of the candidate cluster
            candidate_boundaries = self.grouped_boundaries[candidate_idx]
            
            # Find longest boundary of candidate cluster
            max_boundary_length = 0
            longest_neighbor = None
            
            for nb_label, nb_coords in candidate_boundaries.items():
                if nb_label == -1:
                    continue  # Skip ROI
                
                nb_length = len(nb_coords)
                if nb_length > max_boundary_length:
                    max_boundary_length = nb_length
                    longest_neighbor = nb_label
            
            # Select candidate if its longest boundary is with cluster_id
            if longest_neighbor == cluster_id:
                enclosed_clusters.append(candidate_id)
                boundary_lengths[candidate_id] = length_with_cluster_id
        
        return {
            'cluster_id': cluster_id,
            'cluster_phase_id': cluster_phase_id,
            'target_phase_id': target_phases,
            'enclosed': enclosed_clusters,
            'n_enclosed': len(enclosed_clusters),
            'boundary_lengths': boundary_lengths
        }


    def find_all_enclosures(self, phase1, phase2):
        """
        Find all clusters of phase1 that contain clusters of phase2.
        
        A phase2 cluster is "contained" by a phase1 cluster if the phase1 cluster
        provides the longest boundary for the phase2 cluster.
        
        Parameters
        ----------
        phase1 : int
            Phase ID of container clusters
        phase2 : int
            Phase ID of enclosed clusters
        
        Returns
        -------
        enclosures : list of dict
            List of enclosure information for each phase1 cluster that contains phase2 clusters.
            Each dict contains:
            - 'container_cluster': cluster ID
            - 'enclosed_clusters': list of enclosed cluster IDs
            - 'n_enclosed': number of enclosed clusters
            - 'boundary_lengths': dict {enclosed_id: boundary_length}
            - 'total_boundary_length': sum of all boundaries with enclosed clusters
        """
        enclosures = []
        
        # Get all clusters in phase1
        phase1_clusters = [
            cluster_id for i, cluster_id in enumerate(self.clusters)
            if self.cluster_phases_id[i] == phase1
        ]
        
        print(f"Analyzing {len(phase1_clusters)} clusters of phase {phase1}...")
        
        for cluster_id in phase1_clusters:
            result = self.get_enclosed_clusters(cluster_id, target_phase_id=phase2)
            
            if result['n_enclosed'] > 0:
                total_boundary_length = sum(result['boundary_lengths'].values())
                enclosures.append({
                    'container_cluster': cluster_id,
                    'enclosed_clusters': result['enclosed'],
                    'n_enclosed': result['n_enclosed'],
                    'boundary_lengths': result['boundary_lengths'],
                    'total_boundary_length': total_boundary_length
                })
        
        return enclosures


    def get_cluster_topology_info(self, cluster_id):
        """
        Get complete topological information about a cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster to analyze
        
        Returns
        -------
        topology : dict
            Dictionary containing:
            - 'cluster_id': cluster ID
            - 'phase_id': phase of this cluster
            - 'same_phase_neighbors': dict {neighbor_id: boundary_length}
            - 'inter_phase_neighbors': dict {phase_id: {neighbor_id: boundary_length}}
            - 'enclosed_clusters': dict {phase_id: [cluster_ids]}
            - 'longest_neighbor': ID of neighbor with longest boundary
            - 'longest_neighbor_length': length of longest boundary
            - 'n_pixels': number of pixels in cluster
            - 'touches_roi': whether cluster touches ROI boundary
            - 'roi_boundary_length': length of ROI boundary
        """
        # Find cluster index
        try:
            cluster_idx = np.where(self.clusters == cluster_id)[0][0]
        except IndexError:
            raise ValueError(f"Cluster {cluster_id} not found!")
        
        cluster_phase_id = self.cluster_phases_id[cluster_idx]
        
        # Get boundaries
        boundaries = self.grouped_boundaries[cluster_idx]
        boundary_phases = self.grouped_boundary_phases_id[cluster_idx]
        
        # Analyze neighbors with boundary lengths
        same_phase_neighbors = {}
        inter_phase_neighbors = {}
        touches_roi = False
        roi_boundary_length = 0
        
        longest_neighbor = None
        longest_neighbor_length = 0
        
        for neighbor_label, coords in boundaries.items():
            boundary_length = len(coords)
            
            if neighbor_label == -1:
                touches_roi = True
                roi_boundary_length = boundary_length
                continue
            
            neighbor_phase = boundary_phases[neighbor_label]
            
            # Track longest neighbor
            if boundary_length > longest_neighbor_length:
                longest_neighbor_length = boundary_length
                longest_neighbor = neighbor_label
            
            if neighbor_phase == cluster_phase_id:
                same_phase_neighbors[neighbor_label] = boundary_length
            else:
                if neighbor_phase not in inter_phase_neighbors:
                    inter_phase_neighbors[neighbor_phase] = {}
                inter_phase_neighbors[neighbor_phase][neighbor_label] = boundary_length
        
        # Find enclosed clusters (those whose longest boundary is with this cluster)
        enclosed_info = self.get_enclosed_clusters(cluster_id)
        
        # Organize enclosed by phase
        enclosed_by_phase = {}
        for enclosed_id in enclosed_info['enclosed']:
            enc_idx = np.where(self.clusters == enclosed_id)[0][0]
            enc_phase = self.cluster_phases_id[enc_idx]
            if enc_phase not in enclosed_by_phase:
                enclosed_by_phase[enc_phase] = []
            enclosed_by_phase[enc_phase].append(enclosed_id)
        
        # Count pixels
        n_pixels = np.sum(self.clustering.labels == cluster_id)
        
        return {
            'cluster_id': cluster_id,
            'phase_id': cluster_phase_id,
            'same_phase_neighbors': same_phase_neighbors,
            'inter_phase_neighbors': inter_phase_neighbors,
            'enclosed_clusters': enclosed_by_phase,
            'longest_neighbor': longest_neighbor,
            'longest_neighbor_length': longest_neighbor_length,
            'n_pixels': n_pixels,
            'touches_roi': touches_roi,
            'roi_boundary_length': roi_boundary_length
        }


    def get_cluster_parent(self, cluster_id):
        """
        Find the "parent" cluster - the neighbor with the longest boundary.
        
        Parameters
        ----------
        cluster_id : int
            Cluster to analyze
        
        Returns
        -------
        parent_info : dict
            Dictionary containing:
            - 'cluster_id': this cluster
            - 'parent_id': cluster with longest boundary (None if ROI is longest)
            - 'parent_phase_id': phase of parent
            - 'boundary_length': length of boundary with parent
            - 'boundary_fraction': fraction of total boundary
        """
        topology = self.get_cluster_topology_info(cluster_id)
        
        parent_id = topology['longest_neighbor']
        boundary_length = topology['longest_neighbor_length']
        
        # Calculate total boundary length
        total_boundary = boundary_length + topology.get('roi_boundary_length', 0)
        for neighbors in topology['same_phase_neighbors'].values():
            if isinstance(neighbors, dict):
                total_boundary += sum(neighbors.values())
            else:
                total_boundary += neighbors
        for phase_neighbors in topology['inter_phase_neighbors'].values():
            total_boundary += sum(phase_neighbors.values())
        
        if parent_id is not None:
            parent_idx = np.where(self.clusters == parent_id)[0][0]
            parent_phase = self.cluster_phases_id[parent_idx]
        else:
            parent_phase = None
        
        boundary_fraction = boundary_length / total_boundary if total_boundary > 0 else 0
        
        return {
            'cluster_id': cluster_id,
            'parent_id': parent_id,
            'parent_phase_id': parent_phase,
            'boundary_length': boundary_length,
            'boundary_fraction': boundary_fraction
        }
    def get_boundary_indices(self, cluster_id, neighbor_id=None):
        """
        Get original array indices for boundary pixels.
        
        Parameters
        ----------
        cluster_id : int
            Cluster label
        neighbor_id : int, optional
            Neighbor label. If None, returns all boundaries.
            Use -1 for ROI boundary.
        
        Returns
        -------
        indices : array (M,)
            Original array indices where data.X[indices], data.Y[indices]
            are the boundary coordinates. -1 for coordinates not found.
        coords : array (M, 2)
            Corresponding (x, y) coordinates
        """
        # Find cluster index
        try:
            cluster_idx = np.where(self.clusters == cluster_id)[0][0]
        except IndexError:
            raise ValueError(f"Cluster {cluster_id} not found!")
        
        # Get boundaries
        boundaries = self.grouped_boundaries[cluster_idx]
        
        if neighbor_id is not None:
            # Get specific boundary
            if neighbor_id not in boundaries:
                raise ValueError(f"Cluster {cluster_id} has no boundary with neighbor {neighbor_id}")
            coords = boundaries[neighbor_id]
        else:
            # Get all boundaries
            all_coords = []
            for nb_id, coords_nb in boundaries.items():
                all_coords.append(coords_nb)
            coords = np.vstack(all_coords) if all_coords else np.zeros((0, 2))
        
        # Convert coordinates to indices
        indices = self.data.coords_to_indices(coords)
        
        return indices, coords
    
    def get_all_boundary_indices_by_type(self, cluster_id):
        """
        Get boundary indices organized by type.
        
        Parameters
        ----------
        cluster_id : int
            Cluster label
        
        Returns
        -------
        boundary_indices : dict
            Dictionary with structure:
            {
                'roi': {'indices': array, 'coords': array},
                'same_phase': {neighbor_id: {'indices': array, 'coords': array}},
                'inter_phase': {neighbor_id: {'indices': array, 'coords': array}}
            }
        """
        # Find cluster index
        try:
            cluster_idx = np.where(self.clusters == cluster_id)[0][0]
        except IndexError:
            raise ValueError(f"Cluster {cluster_id} not found!")
        
        cluster_phase_id = self.cluster_phases_id[cluster_idx]
        boundaries = self.grouped_boundaries[cluster_idx]
        boundary_phases = self.grouped_boundary_phases_id[cluster_idx]
        
        result = {
            'roi': {'indices': None, 'coords': None},
            'same_phase': {},
            'inter_phase': {}
        }
        
        for neighbor_id, coords in boundaries.items():
            indices = self.data.coords_to_indices(coords)
            
            if neighbor_id == -1:
                # ROI boundary
                result['roi']['indices'] = indices
                result['roi']['coords'] = coords
            else:
                neighbor_phase = boundary_phases[neighbor_id]
                
                if neighbor_phase == cluster_phase_id:
                    # Same phase boundary
                    result['same_phase'][neighbor_id] = {
                        'indices': indices,
                        'coords': coords
                    }
                else:
                    # Inter-phase boundary
                    result['inter_phase'][neighbor_id] = {
                        'indices': indices,
                        'coords': coords
                    }
        
        return result
    
    def get_interphase_boundary_data(self, cluster_id, neighbor_id):
        """
        Get complete data for pixels on an inter-phase boundary.
        
        Parameters
        ----------
        cluster_id : int
            Cluster label
        neighbor_id : int
            Neighbor cluster label
        
        Returns
        -------
        boundary_data : dict
            Dictionary containing:
            - 'indices': original array indices
            - 'coords': (x, y) coordinates
            - 'X', 'Y': coordinates
            - 'orientations': orientation data
            - 'phases_id': phase labels
            - 'quality': quality values
            - 'n_pixels': number of boundary pixels
        """
        indices, coords = self.get_boundary_indices(cluster_id, neighbor_id)
        
        # Filter out invalid indices
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        valid_coords = coords[valid_mask]
        
        if len(valid_indices) == 0:
            return {
                'indices': np.array([], dtype=np.int32),
                'coords': np.zeros((0, 2)),
                'X': np.array([]),
                'Y': np.array([]),
                'orientations': np.zeros((0, 4)),
                'phases_id': np.array([], dtype=np.int32),
                'quality': np.array([]),
                'n_pixels': 0
            }
        
        return {
            'indices': valid_indices,
            'coords': valid_coords,
            'X': self.data.X[valid_indices],
            'Y': self.data.Y[valid_indices],
            'orientations': self.data.orientations[valid_indices],
            'phases_id': self.data.phases_id[valid_indices],
            'quality': self.data.quality[valid_indices],
            'n_pixels': len(valid_indices)
        }
    
    def get_parent_child_hierarchy_with_boundaries(self, parent_phase_id, child_phase_id,
                                                ref_idx=0, max_iter=10, tol=1e-6, 
                                                parent_ids_out=False,
                                                filter_lamellar_to_semi_rois=False,
                                                clustering_result=None,
                                                semi_roi_params=None,
                                                elongation_cache=None,
                                                min_aspect_ratio=2.5,
                                                include_touching_parents=True):
        """
        Get complete hierarchical structure of parent-child relationships with boundary details.
        
        For each parent cluster (phase 1), lists all enclosed children (phase 2),
        and for each parent-child pair, provides detailed boundary information including
        original array indices from both sides.
        
        Optionally filters lamellar child interfaces into semi-ROI 1 and semi-ROI 2.
        
        Parameters
        ----------
        parent_phase_id : int
            Phase ID of parent clusters
        child_phase_id : int
            Phase ID of child clusters
        ref_idx : int, optional
            Index of the referential orientation for calculation of similar orientations 
            prior their averaging. Default is 0.
        max_iter : int, optional
            Maximum iterations for averaging. Default is 10.
        tol : float, optional
            Tolerance for averaging. Default is 1e-6.
        parent_ids_out : bool, optional
            If True, also returns list of parent IDs. Default is False.
        filter_lamellar_to_semi_rois : bool, optional
            If True, identifies lamellar children and splits their interfaces into
            semi-ROI 1 and semi-ROI 2. Default is False (backward compatible).
        clustering_result : ClusteringResult, optional
            Required if filter_lamellar_to_semi_rois=True.
        semi_roi_params : dict, optional
            Parameters for semi-ROI creation:
            - 'n_layers': int (default 1)
            - 'longitudinal_shrinkage': float (default 0.1)
            - 'layer_thickness_relative': float (default None)
        elongation_cache : dict or list, optional
            Pre-computed elongation results to avoid recomputing PCA.
        min_aspect_ratio : float, optional
            Minimum aspect ratio to consider a cluster lamellar. Default is 2.5.
        include_touching_parents : bool, optional
            If True, includes children under ALL parent clusters they touch,
            not just the enclosing parent. This captures interfaces at borders
            between parent clusters. Default is False (backward compatible).
            When True, adds 'relationship' field to each child: 'enclosed' or 'touching'.
        
        Returns
        -------
        hierarchy : list of dict
            List of parent cluster information. Each dict contains:
            {
                'parent_cluster_id': int,
                'parent_phase_id': int,
                'parent_n_pixels': int,
                'n_children': int,
                'parent_boundary_indices_all': array,
                'child_boundary_indices_all': array,
                'parent_boundary_mask': array(bool),
                'child_boundary_mask': array(bool),
                'parent_boundary_avgori_all': array,
                'child_boundary_avgori_all': array,
                'children': [
                    {
                        'child_cluster_id': int,
                        'child_phase_id': int,
                        'child_n_pixels': int,
                        'relationship': str,  # 'enclosed' or 'touching' (if include_touching_parents=True)
                        'enclosing_parent_id': int,  # ID of parent that encloses this child (if relationship='touching')
                        'is_lamellar': bool,  # (only if filter_lamellar_to_semi_rois=True)
                        'aspect_ratio': float,  # (only if is_lamellar=True)
                        'interface': {...},
                        'semi_roi_1': {...},  # (only if lamellar)
                        'semi_roi_2': {...},  # (only if lamellar)
                        'roi_info': dict  # (only if lamellar)
                    },
                    ...
                ]
            }
        parent_ids : list (optional)
            Returned only if parent_ids_out=True
        
        Examples
        --------
        >>> # Standard usage (backward compatible)
        >>> hierarchy = boundary_result.get_parent_child_hierarchy_with_boundaries(1, 2)
        >>> 
        >>> # With touching parents (captures border interfaces)
        >>> hierarchy = boundary_result.get_parent_child_hierarchy_with_boundaries(
        ...     parent_phase_id=1,
        ...     child_phase_id=2,
        ...     filter_lamellar_to_semi_rois=True,
        ...     clustering_result=clustering_result,
        ...     elongation_cache=all_clusters,
        ...     include_touching_parents=True  # NEW
        ... )
        >>> 
        >>> # Check relationships
        >>> for parent in hierarchy:
        ...     for child in parent['children']:
        ...         if child['relationship'] == 'touching':
        ...             print(f"Child {child['child_cluster_id']} touches Parent {parent['parent_cluster_id']}")
        ...             print(f"  (enclosed by Parent {child['enclosing_parent_id']})")
        """
        from matplotlib.path import Path
        
        # Validate parameters
        if filter_lamellar_to_semi_rois and clustering_result is None:
            raise ValueError("filter_lamellar_to_semi_rois requires clustering_result parameter")
        
        if include_touching_parents and clustering_result is None:
            raise ValueError("include_touching_parents requires clustering_result parameter")
        
        # Set default semi_roi_params
        if semi_roi_params is None:
            semi_roi_params = {
                'n_layers': 1,
                'longitudinal_shrinkage': 0.1,
                'layer_thickness_relative': None
            }
        
        # Get all parent clusters that have children
        enclosures = self.find_all_enclosures(parent_phase_id, child_phase_id)
        parent_symops = np.array(self.data.phases[self.data.phase_names[parent_phase_id]]['symops'])
        child_symops = np.array(self.data.phases[self.data.phase_names[child_phase_id]]['symops'])
        hierarchy = []
        
        print(f"Building hierarchy for Phase {parent_phase_id} → Phase {child_phase_id}...")
        if filter_lamellar_to_semi_rois:
            print(f"  Filtering lamellar children (AR >= {min_aspect_ratio}) to semi-ROIs...")
        if include_touching_parents:
            print(f"  Including touching parents (captures border interfaces)...")
        
        parent_ids = []
        parent_boundary_avgori_all = self.data.orientations.copy()
        child_boundary_avgori_all = self.data.orientations.copy()
        
        # Helper function to create child info
        def create_child_info(parent_id, child_id, relationship='enclosed', enclosing_parent_id=None):
            """Create child info dictionary with interface and optional semi-ROI data."""
            
            # Get child cluster info
            child_n_pixels = np.sum(self.clustering.labels == child_id)
            
            # Get boundary data from parent's perspective
            parent_boundary_indices, parent_boundary_coords = \
                self.get_boundary_indices(parent_id, neighbor_id=child_id)
            
            # Get boundary data from child's perspective
            child_boundary_indices, child_boundary_coords = \
                self.get_boundary_indices(child_id, neighbor_id=parent_id)
            
            # Filter out invalid indices
            parent_valid_mask = parent_boundary_indices >= 0
            child_valid_mask = child_boundary_indices >= 0
            
            parent_valid_indices = parent_boundary_indices[parent_valid_mask]
            parent_valid_coords = parent_boundary_coords[parent_valid_mask]
            
            parent_X = self.data.X[parent_valid_indices]
            parent_Y = self.data.Y[parent_valid_indices]
            
            child_valid_indices = child_boundary_indices[child_valid_mask]
            child_valid_coords = child_boundary_coords[child_valid_mask]
            child_X = self.data.X[child_valid_indices]
            child_Y = self.data.Y[child_valid_indices]
            
            # Total boundary pixels (both sides)
            n_boundary_pixels = len(parent_valid_indices) + len(child_valid_indices)
            
            # Get reference orientations
            q_ref_parent = None
            if clustering_result is not None:
                q_ref_parent = clustering_result.avg_quats[parent_id]
            
            q_ref_child = None
            if clustering_result is not None:
                q_ref_child = clustering_result.avg_quats[child_id]
            
            # Calculate average orientations
            parent_q_mean, parent_M_mean, parent_M_best_cluster, parent_q_best_cluster = get_avg_orientations(
                self.data.quaternions[parent_valid_indices], parent_symops, 
                ref_idx=ref_idx, max_iter=max_iter, tol=tol, q_ref=q_ref_parent
            )
            child_q_mean, child_M_mean, child_M_best_cluster, child_q_best_cluster = get_avg_orientations(
                self.data.quaternions[child_valid_indices], child_symops, 
                ref_idx=ref_idx, max_iter=max_iter, tol=tol, q_ref=q_ref_child
            )
            
            # Build basic child info
            child_info = {
                'child_cluster_id': child_id,
                'child_phase_id': child_phase_id,
                'child_n_pixels': child_n_pixels,
                'relationship': relationship,
                'interface': {
                    'n_boundary_pixels': n_boundary_pixels,
                    'parent_side': {
                        'indices': parent_valid_indices,
                        'coords': parent_valid_coords,
                        'X': parent_X,
                        'Y': parent_Y,
                        'q_mean': parent_q_mean,
                        'M_mean': parent_M_mean,
                        'q_best_cluster': parent_q_best_cluster,
                        'M_best_cluster': parent_M_best_cluster,
                        'n_pixels': len(parent_valid_indices)
                    },
                    'child_side': {
                        'indices': child_valid_indices,
                        'coords': child_valid_coords,
                        'X': child_X,
                        'Y': child_Y,
                        'q_mean': child_q_mean,
                        'M_mean': child_M_mean,
                        'q_best_cluster': child_q_best_cluster,
                        'M_best_cluster': child_M_best_cluster,
                        'n_pixels': len(child_valid_indices)
                    }
                }
            }
            
            # Add enclosing parent ID if this is a touching relationship
            if relationship == 'touching' and enclosing_parent_id is not None:
                child_info['enclosing_parent_id'] = enclosing_parent_id
            
            # ========== ADD SEMI-ROI FILTERING FOR LAMELLAR CHILDREN ==========
            if filter_lamellar_to_semi_rois:
                # Check if this child is lamellar
                is_lamellar = False
                aspect_ratio = None
                
                # Get or compute elongation info
                elongation_info = None
                if elongation_cache is not None:
                    if isinstance(elongation_cache, dict):
                        elongation_info = elongation_cache.get(child_id)
                    elif isinstance(elongation_cache, list):
                        for cluster_data in elongation_cache:
                            if cluster_data['cluster_id'] == child_id:
                                elongation_info = cluster_data
                                break
                
                if elongation_info is None:
                    # Compute elongation
                    elongation_info = clustering_result.analyze_cluster_elongation(
                        child_id, use_sklearn=False
                    )
                
                if elongation_info is not None:
                    aspect_ratio = elongation_info.get('aspect_ratio', 0)
                    is_lamellar = aspect_ratio >= min_aspect_ratio
                
                child_info['is_lamellar'] = is_lamellar
                
                if is_lamellar:
                    child_info['aspect_ratio'] = aspect_ratio
                    
                    # Get ROI info (only compute once, reuse for touching parents)
                    roi_info = clustering_result.create_lamellar_roi(
                        cluster_id=child_id,
                        n_layers=semi_roi_params['n_layers'],
                        longitudinal_shrinkage=semi_roi_params['longitudinal_shrinkage'],
                        layer_thickness_relative=semi_roi_params['layer_thickness_relative'],
                        elongation_cache=elongation_cache
                    )
                    
                    child_info['roi_info'] = roi_info
                    
                    # Get semi-ROI vertices
                    semi1_vertices = roi_info['semi_roi_1_vertices']
                    semi2_vertices = roi_info['semi_roi_2_vertices']
                    
                    semi1_path = Path(semi1_vertices)
                    semi2_path = Path(semi2_vertices)
                    
                    # ========== FILTER PARENT SIDE ==========
                    # Semi-ROI 1
                    parent_semi1_mask = semi1_path.contains_points(parent_valid_coords)
                    parent_semi1_indices = parent_valid_indices[parent_semi1_mask]
                    parent_semi1_coords = parent_valid_coords[parent_semi1_mask]
                    
                    if len(parent_semi1_indices) > 0:
                        parent_semi1_q_mean, parent_semi1_M_mean, parent_semi1_M_best_sym, parent_semi1_q_best_sym = get_avg_orientations(
                            self.data.quaternions[parent_semi1_indices], parent_symops,
                            ref_idx=ref_idx, max_iter=max_iter, tol=tol, q_ref=q_ref_parent
                        )
                    else:
                        parent_semi1_q_mean = None
                        parent_semi1_M_mean = None
                        parent_semi1_M_best_sym = None
                        parent_semi1_q_best_sym = None
                    
                    # Semi-ROI 2
                    parent_semi2_mask = semi2_path.contains_points(parent_valid_coords)
                    parent_semi2_indices = parent_valid_indices[parent_semi2_mask]
                    parent_semi2_coords = parent_valid_coords[parent_semi2_mask]
                    
                    if len(parent_semi2_indices) > 0:
                        parent_semi2_q_mean, parent_semi2_M_mean, parent_semi2_M_best_sym, parent_semi2_q_best_sym = get_avg_orientations(
                            self.data.quaternions[parent_semi2_indices], parent_symops,
                            ref_idx=ref_idx, max_iter=max_iter, tol=tol, q_ref=q_ref_parent
                        )
                    else:
                        parent_semi2_q_mean = None
                        parent_semi2_M_best_sym = None
                        parent_semi2_q_best_sym = None
                        parent_semi2_M_mean = None
                    
                    # ========== FILTER CHILD SIDE ==========
                    # Semi-ROI 1
                    child_semi1_mask = semi1_path.contains_points(child_valid_coords)
                    child_semi1_indices = child_valid_indices[child_semi1_mask]
                    child_semi1_coords = child_valid_coords[child_semi1_mask]
                    
                    if len(child_semi1_indices) > 0:
                        child_semi1_q_mean, child_semi1_M_mean, child_semi1_M_best_sym, child_semi1_q_best_sym = get_avg_orientations(
                            self.data.quaternions[child_semi1_indices], child_symops,
                            ref_idx=ref_idx, max_iter=max_iter, tol=tol, q_ref=q_ref_child
                        )
                    else:
                        child_semi1_q_mean = None
                        child_semi1_M_best_sym = None
                        child_semi1_q_best_sym = None
                        child_semi1_M_mean = None
                    
                    # Semi-ROI 2
                    child_semi2_mask = semi2_path.contains_points(child_valid_coords)
                    child_semi2_indices = child_valid_indices[child_semi2_mask]
                    child_semi2_coords = child_valid_coords[child_semi2_mask]
                    
                    if len(child_semi2_indices) > 0:
                        child_semi2_q_mean, child_semi2_M_mean, child_semi2_M_best_sym, child_semi2_q_best_sym = get_avg_orientations(
                            self.data.quaternions[child_semi2_indices], child_symops,
                            ref_idx=ref_idx, max_iter=max_iter, tol=tol, q_ref=q_ref_child
                        )
                    else:
                        child_semi2_q_mean = None
                        child_semi2_M_best_sym = None
                        child_semi2_q_best_sym = None
                        child_semi2_M_mean = None
                    
                    # ========== ADD SEMI-ROI DATA TO CHILD INFO ==========
                    if len(parent_semi1_indices) != 0:
                        no_parent_side = False
                    else:
                        no_parent_side = True
                    child_info['semi_roi_1'] = {
                        'n_boundary_pixels': len(parent_semi1_indices) + len(child_semi1_indices),
                        'no_parent_side': no_parent_side,
                        'parent_side': {
                            'indices': parent_semi1_indices,
                            'coords': parent_semi1_coords,
                            'X': self.data.X[parent_semi1_indices] if len(parent_semi1_indices) > 0 else np.array([]),
                            'Y': self.data.Y[parent_semi1_indices] if len(parent_semi1_indices) > 0 else np.array([]),
                            'q_mean': parent_semi1_q_mean,
                            'M_mean': parent_semi1_M_mean,
                            'q_best_sym': parent_semi1_q_best_sym,
                            'M_best_sym': parent_semi1_M_best_sym,
                            'n_pixels': len(parent_semi1_indices)
                        },
                        'child_side': {
                            'indices': child_semi1_indices,
                            'coords': child_semi1_coords,
                            'X': self.data.X[child_semi1_indices] if len(child_semi1_indices) > 0 else np.array([]),
                            'Y': self.data.Y[child_semi1_indices] if len(child_semi1_indices) > 0 else np.array([]),
                            'q_mean': child_semi1_q_mean,
                            'M_mean': child_semi1_M_mean,
                            'q_best_sym': child_semi1_q_best_sym,
                            'M_best_sym': child_semi1_M_best_sym,
                            'n_pixels': len(child_semi1_indices)
                        }
                    }
                    if len(parent_semi2_indices) != 0:
                        no_parent_side = False
                    else:
                        no_parent_side = True
                    child_info['semi_roi_2'] = {
                        'n_boundary_pixels': len(parent_semi2_indices) + len(child_semi2_indices),
                        'no_parent_side': no_parent_side,
                        'parent_side': {
                            'indices': parent_semi2_indices,
                            'coords': parent_semi2_coords,
                            'X': self.data.X[parent_semi2_indices] if len(parent_semi2_indices) > 0 else np.array([]),
                            'Y': self.data.Y[parent_semi2_indices] if len(parent_semi2_indices) > 0 else np.array([]),
                            'q_mean': parent_semi2_q_mean,
                            'M_mean': parent_semi2_M_mean,
                            'q_best_sym': parent_semi2_q_best_sym,
                            'M_best_sym': parent_semi2_M_best_sym,
                            'n_pixels': len(parent_semi2_indices),
                        },
                        'child_side': {
                            'indices': child_semi2_indices,
                            'coords': child_semi2_coords,
                            'X': self.data.X[child_semi2_indices] if len(child_semi2_indices) > 0 else np.array([]),
                            'Y': self.data.Y[child_semi2_indices] if len(child_semi2_indices) > 0 else np.array([]),
                            'q_mean': child_semi2_q_mean,
                            'M_mean': child_semi2_M_mean,
                            'q_best_sym': child_semi2_q_best_sym,
                            'M_best_sym': child_semi2_M_best_sym,
                            'n_pixels': len(child_semi2_indices)
                        }
                    }
            # ==================================================================
            
            return child_info, parent_valid_indices, child_valid_indices
        
        # ========== FIRST PASS: BUILD HIERARCHY WITH ENCLOSED CHILDREN ==========
        # Track which children are enclosed by which parents
        enclosed_children_map = {}  # {child_id: enclosing_parent_id}
        
        for enc in enclosures:
            parent_id = enc['container_cluster']
            parent_ids.append(parent_id)
            child_ids = enc['enclosed_clusters']
            
            # Track enclosures
            for child_id in child_ids:
                enclosed_children_map[child_id] = parent_id
            
            # Get parent cluster info
            parent_n_pixels = np.sum(self.clustering.labels == parent_id)
            
            # Build children list with interface details
            children_list = []
            
            parent_boundary_indices_all = np.array([])
            child_boundary_indices_all = np.array([])
            
            for child_id in child_ids:
                child_info, parent_indices, child_indices = create_child_info(
                    parent_id, child_id, relationship='enclosed'
                )
                
                parent_boundary_indices_all = np.append(parent_boundary_indices_all, parent_indices)
                child_boundary_indices_all = np.append(child_boundary_indices_all, child_indices)
                
                # Update global orientation arrays
                if len(parent_indices) > 0:
                    parent_boundary_avgori_all[parent_indices, :, :] = child_info['interface']['parent_side']['M_mean']
                if len(child_indices) > 0:
                    child_boundary_avgori_all[child_indices] = child_info['interface']['child_side']['M_mean']
                
                children_list.append(child_info)
            
            parent_boundary_indices_all = parent_boundary_indices_all.astype(int)
            child_boundary_indices_all = child_boundary_indices_all.astype(int)
            parent_boundary_mask = (self.data.X * 0).astype(bool)
            parent_boundary_mask[parent_boundary_indices_all] = True
            child_boundary_mask = (self.data.X * 0).astype(bool)
            child_boundary_mask[child_boundary_indices_all] = True
            
            parent_info = {
                'parent_cluster_id': parent_id,
                'parent_phase_id': parent_phase_id,
                'parent_n_pixels': parent_n_pixels,
                'n_children': len(children_list),
                'children': children_list,
                'parent_boundary_indices_all': parent_boundary_indices_all,
                'child_boundary_indices_all': child_boundary_indices_all,
                'parent_boundary_mask': parent_boundary_mask,
                'child_boundary_mask': child_boundary_mask,
                'parent_boundary_avgori_all': parent_boundary_avgori_all,
                'child_boundary_avgori_all': child_boundary_avgori_all
            }
            
            hierarchy.append(parent_info)
        
        # ========== SECOND PASS: ADD TOUCHING CHILDREN (if requested) ==========
        if include_touching_parents:
            print(f"  Checking for touching relationships...")
            
            touching_count = 0
            
            # For each parent, check if any children from other parents touch it
            for parent_info in hierarchy:
                parent_id = parent_info['parent_cluster_id']
                
                # Find parent's index in boundary result
                parent_idx = np.where(self.clusters == parent_id)[0]
                
                if len(parent_idx) == 0:
                    continue
                
                parent_idx = parent_idx[0]
                
                # Get all neighbors of this parent from grouped_boundaries
                parent_boundaries = self.grouped_boundaries[parent_idx]
                parent_boundary_phases = self.grouped_boundary_phases_id[parent_idx]
                
                # Iterate through all neighbors (keys in boundaries dict)
                for neighbor_id in parent_boundaries.keys():
                    if neighbor_id == -1:
                        continue  # Skip ROI boundary
                    
                    # Get neighbor phase
                    neighbor_phase = parent_boundary_phases[neighbor_id]
                    
                    # Is it a child phase?
                    if neighbor_phase == child_phase_id:
                        # Is it NOT already in this parent's children (i.e., not enclosed)?
                        if neighbor_id not in [c['child_cluster_id'] for c in parent_info['children']]:
                            # This is a touching child!
                            enclosing_parent_id = enclosed_children_map.get(neighbor_id)
                            
                            if enclosing_parent_id is not None:
                                # Create child info for touching relationship
                                touching_child_info, parent_indices, child_indices = create_child_info(
                                    parent_id, neighbor_id, 
                                    relationship='touching',
                                    enclosing_parent_id=enclosing_parent_id
                                )
                                
                                # Add to this parent's children
                                parent_info['children'].append(touching_child_info)
                                parent_info['n_children'] += 1
                                
                                # Update boundary indices (append to existing)
                                parent_info['parent_boundary_indices_all'] = np.append(
                                    parent_info['parent_boundary_indices_all'], parent_indices
                                ).astype(int)
                                parent_info['child_boundary_indices_all'] = np.append(
                                    parent_info['child_boundary_indices_all'], child_indices
                                ).astype(int)
                                
                                # Update masks
                                if len(parent_indices) > 0:
                                    parent_info['parent_boundary_mask'][parent_indices] = True
                                if len(child_indices) > 0:
                                    parent_info['child_boundary_mask'][child_indices] = True
                                
                                touching_count += 1
            
            if touching_count > 0:
                print(f"  Found {touching_count} touching relationships")
            else:
                print(f"  No touching relationships found")
        
        # ========== SUMMARY ==========
        print(f"Found {len(hierarchy)} parent clusters with children")
        
        if filter_lamellar_to_semi_rois:
            # Count lamellar children (unique, not double-counting touching ones)
            all_lamellar_children = set()
            for parent in hierarchy:
                for child in parent['children']:
                    if child.get('is_lamellar', False):
                        all_lamellar_children.add(child['child_cluster_id'])
            print(f"  {len(all_lamellar_children)} lamellar children identified and filtered to semi-ROIs")
        
        print("""Hierarchy structure is as follows:
        hierarchy : list of dict
            List of parent cluster information. Each dict contains:
            {
                'parent_cluster_id': int,
                'parent_phase_id': int,
                'parent_n_pixels': int,
                'n_children': int,
                'parent_boundary_indices_all': array,
                'child_boundary_indices_all': array,
                'parent_boundary_mask': array(bool),
                'child_boundary_mask': array(bool),
                'parent_boundary_avgori_all': array,
                'child_boundary_avgori_all': array,
                'children': [
                    {
                        'child_cluster_id': int,
                        'child_phase_id': int,
                        'child_n_pixels': int,
                        'relationship': 'enclosed' or 'touching',
                        'enclosing_parent_id': int (if relationship='touching'),
                        'interface': {...},
                        # If filter_lamellar_to_semi_rois=True:
                        'is_lamellar': bool,
                        'aspect_ratio': float (if lamellar),
                        'semi_roi_1': {...} (if lamellar),
                        'semi_roi_2': {...} (if lamellar),
                        'roi_info': dict (if lamellar)
                    },
                    ...
                ]
            }
        """)
        
        if parent_ids_out:
            return hierarchy, parent_ids
        else:
            return hierarchy
    
    def print_parent_child_hierarchy(self, parent_phase_id, child_phase_id, 
                                 parent_cluster_id=None,
                                 max_parents=None, max_children=None):
        """
        Print formatted hierarchy of parent-child relationships.
        
        Parameters
        ----------
        parent_phase_id : int
            Phase ID of parent clusters
        child_phase_id : int
            Phase ID of child clusters
        parent_cluster_id : int, optional
            Specific parent cluster ID to print. If None, prints all parents.
        max_parents : int, optional
            Maximum number of parent clusters to print (ignored if parent_cluster_id is specified)
        max_children : int, optional
            Maximum number of children per parent to print
        """
        hierarchy = self.get_parent_child_hierarchy_with_boundaries(
            parent_phase_id, child_phase_id
        )
        
        # Filter for specific parent if requested
        if parent_cluster_id is not None:
            hierarchy = [p for p in hierarchy if p['parent_cluster_id'] == parent_cluster_id]
            
            if len(hierarchy) == 0:
                print(f"\nParent cluster {parent_cluster_id} not found or has no children of phase {child_phase_id}")
                return
        
        print(f"\n{'='*80}")
        if parent_cluster_id is not None:
            print(f"Parent-Child Hierarchy for Cluster {parent_cluster_id}: "
                f"Phase {parent_phase_id} → Phase {child_phase_id}")
        else:
            print(f"Parent-Child Hierarchy: Phase {parent_phase_id} → Phase {child_phase_id}")
        print(f"{'='*80}\n")
        
        for i, parent in enumerate(hierarchy):
            if parent_cluster_id is None and max_parents is not None and i >= max_parents:
                print(f"\n... and {len(hierarchy) - max_parents} more parent clusters")
                break
            
            print(f"Parent Cluster {parent['parent_cluster_id']} "
                f"(Phase {parent['parent_phase_id']}, {parent['parent_n_pixels']} pixels)")
            print(f"  └─ {parent['n_children']} children:")
            
            for j, child in enumerate(parent['children']):
                if max_children is not None and j >= max_children:
                    print(f"     ... and {len(parent['children']) - max_children} more children")
                    break
                
                print(f"\n     Child Cluster {child['child_cluster_id']} "
                    f"(Phase {child['child_phase_id']}, {child['child_n_pixels']} pixels)")
                
                interface = child['interface']
                print(f"       Interface: {interface['n_boundary_pixels']} total boundary pixels")
                print(f"         Parent side: {interface['parent_side']['n_pixels']} pixels")
                print(f"           Indices: {interface['parent_side']['indices'][:10]}...")
                print(f"         Child side:  {interface['child_side']['n_pixels']} pixels")
                print(f"           Indices: {interface['child_side']['indices'][:10]}...")
            
            print()

    


    def export_hierarchy_to_dict(self, parent_phase_id, child_phase_id):
        """
        Export hierarchy in a format suitable for JSON/analysis.
        
        Parameters
        ----------
        parent_phase_id : int
            Phase ID of parent clusters
        child_phase_id : int
            Phase ID of child clusters
        
        Returns
        -------
        export_data : dict
            Dictionary with simplified structure (arrays converted to lists)
        """
        hierarchy = self.get_parent_child_hierarchy_with_boundaries(
            parent_phase_id, child_phase_id
        )
        
        export_data = {
            'parent_phase_id': parent_phase_id,
            'child_phase_id': child_phase_id,
            'n_parents': len(hierarchy),
            'parents': []
        }
        
        for parent in hierarchy:
            parent_export = {
                'parent_cluster_id': int(parent['parent_cluster_id']),
                'parent_n_pixels': int(parent['parent_n_pixels']),
                'n_children': int(parent['n_children']),
                'children': []
            }
            
            for child in parent['children']:
                child_export = {
                    'child_cluster_id': int(child['child_cluster_id']),
                    'child_n_pixels': int(child['child_n_pixels']),
                    'interface': {
                        'n_boundary_pixels': int(child['interface']['n_boundary_pixels']),
                        'parent_side': {
                            'n_pixels': int(child['interface']['parent_side']['n_pixels']),
                            'indices': child['interface']['parent_side']['indices'].tolist()
                        },
                        'child_side': {
                            'n_pixels': int(child['interface']['child_side']['n_pixels']),
                            'indices': child['interface']['child_side']['indices'].tolist()
                        }
                    }
                }
                parent_export['children'].append(child_export)
            
            export_data['parents'].append(parent_export)
        
        return export_data


    def get_interface_data_for_hierarchy(self, parent_phase_id, child_phase_id):
        """
        Get complete EBSD data for all interfaces in the hierarchy.
        
        Parameters
        ----------
        parent_phase_id : int
            Phase ID of parent clusters
        child_phase_id : int
            Phase ID of child clusters
        
        Returns
        -------
        interface_data : list of dict
            List with one entry per parent-child interface:
            {
                'parent_cluster_id': int,
                'child_cluster_id': int,
                'parent_data': {
                    'indices': array,
                    'X': array, 'Y': array,
                    'orientations': array,
                    'phases_id': array,
                    'quality': array
                },
                'child_data': {
                    'indices': array,
                    'X': array, 'Y': array,
                    'orientations': array,
                    'phases_id': array,
                    'quality': array
                }
            }
        """
        hierarchy = self.get_parent_child_hierarchy_with_boundaries(
            parent_phase_id, child_phase_id
        )
        
        interface_data = []
        
        for parent in hierarchy:
            parent_id = parent['parent_cluster_id']
            
            for child in parent['children']:
                child_id = child['child_cluster_id']
                
                # Get indices
                parent_indices = child['interface']['parent_side']['indices']
                child_indices = child['interface']['child_side']['indices']
                
                # Extract full data
                entry = {
                    'parent_cluster_id': parent_id,
                    'child_cluster_id': child_id,
                    'parent_data': {
                        'indices': parent_indices,
                        'X': self.data.X[parent_indices],
                        'Y': self.data.Y[parent_indices],
                        'orientations': self.data.orientations[parent_indices],
                        'phases_id': self.data.phases_id[parent_indices],
                        'quality': self.data.quality[parent_indices]
                    },
                    'child_data': {
                        'indices': child_indices,
                        'X': self.data.X[child_indices],
                        'Y': self.data.Y[child_indices],
                        'orientations': self.data.orientations[child_indices],
                        'phases_id': self.data.phases_id[child_indices],
                        'quality': self.data.quality[child_indices]
                    }
                }
                
                interface_data.append(entry)
        
        return interface_data


    def save_hierarchy_to_hdf5(self, filename, parent_phase_id, child_phase_id):
        """
        Save complete hierarchy data to HDF5 file for later analysis.
        
        Parameters
        ----------
        filename : str
            Output HDF5 filename
        parent_phase_id : int
            Phase ID of parent clusters
        child_phase_id : int
            Phase ID of child clusters
        """
        import h5py
        
        hierarchy = self.get_parent_child_hierarchy_with_boundaries(
            parent_phase_id, child_phase_id
        )
        
        with h5py.File(filename, 'w') as f:
            # Store metadata
            f.attrs['parent_phase_id'] = parent_phase_id
            f.attrs['child_phase_id'] = child_phase_id
            f.attrs['n_parents'] = len(hierarchy)
            
            # Create groups for each parent
            for parent in hierarchy:
                parent_id = parent['parent_cluster_id']
                parent_group = f.create_group(f'parent_{parent_id}')
                
                parent_group.attrs['parent_cluster_id'] = parent_id
                parent_group.attrs['parent_n_pixels'] = parent['parent_n_pixels']
                parent_group.attrs['n_children'] = parent['n_children']
                
                # Create subgroups for each child
                for child in parent['children']:
                    child_id = child['child_cluster_id']
                    child_group = parent_group.create_group(f'child_{child_id}')
                    
                    child_group.attrs['child_cluster_id'] = child_id
                    child_group.attrs['child_n_pixels'] = child['child_n_pixels']
                    child_group.attrs['n_boundary_pixels'] = child['interface']['n_boundary_pixels']
                    
                    # Save parent side boundary data
                    parent_side = child_group.create_group('parent_side')
                    parent_indices = child['interface']['parent_side']['indices']
                    parent_side.create_dataset('indices', data=parent_indices)
                    parent_side.create_dataset('X', data=self.data.X[parent_indices])
                    parent_side.create_dataset('Y', data=self.data.Y[parent_indices])
                    parent_side.create_dataset('orientations', data=self.data.orientations[parent_indices])
                    parent_side.create_dataset('phases_id', data=self.data.phases_id[parent_indices])
                    parent_side.create_dataset('quality', data=self.data.quality[parent_indices])
                    
                    # Save child side boundary data
                    child_side = child_group.create_group('child_side')
                    child_indices = child['interface']['child_side']['indices']
                    child_side.create_dataset('indices', data=child_indices)
                    child_side.create_dataset('X', data=self.data.X[child_indices])
                    child_side.create_dataset('Y', data=self.data.Y[child_indices])
                    child_side.create_dataset('orientations', data=self.data.orientations[child_indices])
                    child_side.create_dataset('phases_id', data=self.data.phases_id[child_indices])
                    child_side.create_dataset('quality', data=self.data.quality[child_indices])
        
        print(f"Hierarchy data saved to {filename}")    
    def touches_scan_boundary(self, cluster_id, min_pixels=1):
        """
        Check whether a cluster touches the scan boundary.

        In boundary_result, scan-edge pixels are stored with neighbor_id == -1.

        Parameters
        ----------
        cluster_id : int
            Cluster label to check.
        min_pixels : int
            Minimum number of scan-edge boundary pixels to consider the
            cluster as touching the boundary. Default 1.

        Returns
        -------
        bool
        """
        try:
            cluster_idx = np.where(self.clusters == cluster_id)[0][0]
        except IndexError:
            return False
        boundaries = self.grouped_boundaries[cluster_idx]
        edge_coords = boundaries.get(-1, [])
        return len(edge_coords) >= min_pixels


    def detect_boundary_clusters(self, cluster_ids, min_pixels=1):
        """
        Identify which clusters in a given set touch the scan boundary.

        Parameters
        ----------
        cluster_ids : iterable of int
            Cluster IDs to check (e.g. all parent or all child IDs).
        min_pixels : int
            Minimum scan-edge pixels to count as touching. Default 1.

        Returns
        -------
        boundary_set : set of int
            Subset of cluster_ids that touch the scan edge.

        Example
        -------
        >>> parent_ids = set(clustering_result.labels_by_phase['A'])
        >>> child_ids  = set(clustering_result.labels_by_phase['M'])
        >>> boundary_parents  = boundary_result.detect_boundary_clusters(parent_ids)
        >>> boundary_children = boundary_result.detect_boundary_clusters(child_ids)
        """
        return {cid for cid in cluster_ids
                if self.touches_scan_boundary(cid, min_pixels)}


    def detect_boundary_meeting_pairs(self, parent_child_hierarchy,
                                    lamellar_child_clusters,
                                    parent_phase_id,
                                    child_phase_id,
                                    tip_distance_threshold=15.0,
                                    shared_boundary_min_pixels=3,
                                    verbose=False):
        """
        Automatically detect pairs of lamellae from neighbouring austenite
        grains whose tips point toward the same grain boundary.

        For each austenite-austenite boundary pair (pid_A, pid_B):
        - Find lamellar children of pid_A whose nearest tip is within
            tip_distance_threshold pixels of the shared boundary pixels.
        - Same for pid_B.
        - Cross-pair all candidates → boundary-meeting pairs.

        White (unindexed) lamella continuations are implicitly handled because
        the ROI tip (end_point_1/2 from identify_lamellar_clusters) extends
        to the actual cluster boundary, not the indexed pixels only.

        Parameters
        ----------
        parent_child_hierarchy : list of dict
            Output of self.get_parent_child_hierarchy_with_boundaries().
            Each dict: {'parent_cluster_id': int, 'children': [...]}
        lamellar_child_clusters : list of dict
            Output of clustering_result.identify_lamellar_clusters() or
            clustering_result.merge_lamellar_fragments().
            Each dict must contain 'cluster_id', 'end_point_1', 'end_point_2'.
        parent_phase_id : int
            Phase ID of the austenite (parent) phase.
        child_phase_id : int
            Phase ID of the martensite (child) phase.
        tip_distance_threshold : float
            Maximum distance in pixels from a lamella tip to the shared
            boundary pixels for it to be a meeting candidate. Default 15.
            Increase if lamellae have long unindexed regions near boundaries.
        shared_boundary_min_pixels : int
            Minimum shared boundary pixels between two parent grains for the
            boundary to be considered real (filters segmentation noise).
            Default 3.
        verbose : bool
            Print detected pairs. Default False.

        Returns
        -------
        meeting_pairs : list of tuples
            Each tuple: (lam_A, parent_A, lam_B, parent_B, label)
            where label is 'g{parent_A}-g{parent_B}'.
            Same format as config.BOUNDARY_MEETING_PAIRS — drop-in replacement.

        Example
        -------
        >>> parent_phase_id = clustering_result.data.phase_ids['A']
        >>> child_phase_id  = clustering_result.data.phase_ids['M']
        >>> meeting_pairs = boundary_result.detect_boundary_meeting_pairs(
        ...     parent_child_hierarchy      = parent_child_hierarchy_with_boundaries,
        ...     lamellar_child_clusters     = lamellar_child_clusters,
        ...     parent_phase_id             = parent_phase_id,
        ...     child_phase_id              = child_phase_id,
        ...     tip_distance_threshold      = 15.0,
        ...     verbose                     = True,
        ... )
        """
        # ── Build index: cluster_id → position in self.clusters ───────────────
        br_index = {int(cid): i for i, cid in enumerate(self.clusters)}

        def _get_boundaries(cid):
            idx = br_index.get(int(cid))
            return self.grouped_boundaries[idx] if idx is not None else {}

        def _get_boundary_phases(cid):
            idx = br_index.get(int(cid))
            return (self.grouped_boundary_phases_id[idx]
                    if idx is not None else {})

        # ── Build lookup: child_id → elongation data ──────────────────────────
        elongation_by_id = {
            item['cluster_id']: item
            for item in lamellar_child_clusters
        }

        # ── Build lookup: parent_id → lamellar child IDs ──────────────────────
        parent_to_children = {}
        for hier in parent_child_hierarchy:
            pid      = hier['parent_cluster_id']
            children = [c['child_cluster_id']
                        for c in hier.get('children', [])
                        if c['child_cluster_id'] in elongation_by_id]
            parent_to_children[pid] = children

        all_parents = list(parent_to_children.keys())

        # ── Find all austenite-austenite boundary pairs ────────────────────────
        au_au_boundaries = []
        for pid_a in all_parents:
            boundaries_a = _get_boundaries(pid_a)
            phases_a     = _get_boundary_phases(pid_a)
            for nb_id, coords in boundaries_a.items():
                if nb_id == -1:
                    continue                            # scan edge
                if nb_id not in parent_to_children:
                    continue                            # not an analysed parent
                if phases_a.get(nb_id, -1) != parent_phase_id:
                    continue                            # not austenite-austenite
                if nb_id <= pid_a:
                    continue                            # avoid duplicates
                if len(coords) < shared_boundary_min_pixels:
                    continue
                au_au_boundaries.append((pid_a, nb_id, coords))

        if verbose:
            print(f"Found {len(au_au_boundaries)} "
                f"austenite-austenite boundary pairs")

        # ── For each Au-Au boundary: find lamellae with tips pointing to it ───
        meeting_pairs = []
        seen          = set()

        for pid_a, pid_b, bnd_coords_ab in au_au_boundaries:

            # B→A boundary coords (may differ slightly due to pixel sampling)
            bnd_coords_ba = _get_boundaries(pid_b).get(pid_a, bnd_coords_ab)
            bnd_coords_ba = np.array(bnd_coords_ba, dtype=float)
            bnd_coords_ab = np.array(bnd_coords_ab, dtype=float)

            def _min_dist(tip, coords):
                diffs = coords - np.array(tip)[np.newaxis, :]
                return float(np.sqrt((diffs ** 2).sum(axis=1)).min())

            # Lamellae from pid_a pointing toward boundary with pid_b
            cands_a = []
            for cid in parent_to_children.get(pid_a, []):
                e   = elongation_by_id[cid]
                d1  = _min_dist(e['end_point_1'], bnd_coords_ab)
                d2  = _min_dist(e['end_point_2'], bnd_coords_ab)
                best = min(d1, d2)
                if best <= tip_distance_threshold:
                    cands_a.append((cid, best))

            # Lamellae from pid_b pointing toward boundary with pid_a
            cands_b = []
            for cid in parent_to_children.get(pid_b, []):
                e   = elongation_by_id[cid]
                d1  = _min_dist(e['end_point_1'], bnd_coords_ba)
                d2  = _min_dist(e['end_point_2'], bnd_coords_ba)
                best = min(d1, d2)
                if best <= tip_distance_threshold:
                    cands_b.append((cid, best))

            if verbose and (cands_a or cands_b):
                print(f"  Boundary g{pid_a}-g{pid_b}: "
                    f"{len(cands_a)} from g{pid_a}, "
                    f"{len(cands_b)} from g{pid_b}")

            # Cross-pair candidates
            label = f'g{pid_a}-g{pid_b}'
            for cid_a, da in cands_a:
                for cid_b, db in cands_b:
                    key = (min(cid_a, cid_b), max(cid_a, cid_b),
                        min(pid_a, pid_b), max(pid_a, pid_b))
                    if key in seen:
                        continue
                    seen.add(key)
                    meeting_pairs.append(
                        (cid_a, pid_a, cid_b, pid_b, label))
                    if verbose:
                        print(f"    Pair: lam {cid_a}(p{pid_a}) ↔ "
                            f"lam {cid_b}(p{pid_b})  "
                            f"tip_dist: {da:.1f}, {db:.1f} px")
        self.meeting_pairs = meeting_pairs  # Store in data for later use
        return meeting_pairs
class BoundaryAnalyzer:
    """
    Main class for grain boundary detection and analysis.
    
    Design: Facade pattern
    - Simplifies complex boundary analysis operations
    - Coordinates between different analysis steps
    """
    
    def __init__(self):
        self.result = None
    
    def analyze(self, clustering_result: ClusteringResult) -> BoundaryResult:
        """
        Perform complete boundary analysis.
        
        Parameters
        ----------
        clustering_result : ClusteringResult
            Clustering results to analyze
        
        Returns
        -------
        boundary_result : BoundaryResult
            Complete boundary analysis
        """
        result = BoundaryResult(clustering_result)
        
        # Step 1: Update 2D grid
        clustering_result.update_grid_2d()
        
        # Step 2: Find neighbors and boundaries
        self._find_neighbors_and_boundaries(result)
        
        # Step 3: Extract grouped boundaries
        self._extract_grouped_boundaries(result)
        
        # Step 4: Compute representative points
        self._compute_representative_points(result)
        
        self.result = result
        return result
    
    def _find_neighbors_and_boundaries(self, result):
        """Find cluster neighbors and boundary pixels."""
        grid = result.data.get_grid_2d()
        labels_2d = grid['labels_2d']
        phase_2d = grid['phase_2d']
        inside_mask = result.data.get_inside_mask2d()
        
        # Call your function
        (result.clusters, result.flat_neighbors, result.flat_lengths, 
         result.slices, result.boundary_y, result.boundary_x, 
         result.boundary_nb, result.boundary_nb_phase_id, 
         result.boundary_slices) = \
            find_cluster_neighbors_with_lengths_and_boundaries_numba_roi(
                labels_2d, inside_mask, phase_2d
            )
    
    def _extract_grouped_boundaries(self, result):
        """Extract boundaries grouped by neighbor."""
        grid = result.data.get_grid_2d()
        xs, ys = grid['xs'], grid['ys']
        
        (result.grouped_boundaries, 
         result.grouped_boundary_phases_id, 
         result.cluster_phases_id) = \
            extract_boundaries_grouped_by_neighbor(
                xs, ys,
                result.clusters,
                result.boundary_slices,
                result.boundary_x,
                result.boundary_y,
                result.boundary_nb,
                boundary_nb_phase=result.boundary_nb_phase_id,
                phase_2d=grid['phase_2d']
            )
    
    def _compute_representative_points(self, result):
        """Compute representative points for clusters."""
        (result.rep_points, 
         result.rep_point_info) = \
            representative_points_from_grouped_boundaries(
                result.grouped_boundaries,
                grouped_boundary_phases=result.grouped_boundary_phases_id,
                cluster_phases=result.cluster_phases_id
            )



# ============================================================================
# KAM and GND Analyzer
# ============================================================================

class KamGNDResult:
    """Container for clustering results with analysis methods."""
    
    def __init__(self, labels, algorithm, data, parameters=None, 
                 cluster_phases_id=None, com=None):
        """
        Parameters
        ----------
        labels : array (N,)
            Cluster labels
        algorithm : ClusteringAlgorithm
            Algorithm that produced these results
        data : EBSDData
            Original EBSD data
        parameters : dict, optional
            Algorithm parameters used
        cluster_phases_id : dict, optional
            Pre-computed mapping {cluster_label: phase_id}
            If provided, skips recomputation
        com : array (n_clusters, 2), optional
            Pre-computed cluster centers of mass
            If provided, skips recomputation
        """
        self.labels = labels

        self.algorithm = algorithm
        self.data = data
        self.parameters = parameters or {}
        
        # Cached analyses - can be provided or computed later
        self._clusters_unique = None
        self._cluster_sizes = None
        self._cluster_phases_id = cluster_phases_id  # ← Pre-computed if provided
        self._com = com  # ← Pre-computed if provided
        self._average_orientations = None


class KamGND:
    """
    Computation of kernel average misorientations and GND
    [1] C. Moussa, M. Bernacki, R. Besnard, N. Bozzolo, Ultramicroscopy 179 (2017) 63-72.
    """
    def __init__(self,burgersv = 3.015e-10, nneighbors=5, distance=1,perimeteronly=True,maxmis=None,distance_convention="OIM",roi=None,cluster_id=None, phase=None, out="deg"):
        self.distance = distance
        self.perimeteronly = perimeteronly
        self.maxmis = maxmis
        self.roi = roi
        self.cluster_id = cluster_id
        self.phase = phase
        self.out = out
        self.distance_convention = distance_convention
        self.nneighbors = nneighbors
        self.burgersv=burgersv
        
    def computeGND(self, clustering_result: ClusteringResult):

        distance = []  # real distance in (normally um) to n-th nearest neighbor
        for d in range(1, self.nneighbors + 1):
                distance.append(
                    clustering_result.data.get_distance_neighbors(distance=d, distance_convention=self.distance_convention)
                )
        # Converts lists to numpy array
        distance = np.array(distance)    
        for pi, phase in enumerate(clustering_result.unique_phases):
            phase_name = clustering_result.data.phase_names[phase]
            symops = np.array(clustering_result.data.phases[phase_name]['symops'])
            Sel = clustering_result.data.rois.masks_by_phase[0][phase_name]
            
            kam = []  # list with KAM values for every pixel for each distance
            for d in range(1, self.nneighbors + 1):
                neighbors = clustering_result.data.compute_neighbors(d, True, self.distance_convention, roi=None, sel= Sel)
                kam.append(self._get_KAM(clustering_result.data._ebsdData.M,neighbors,symops, Sel))                   
            
            kam = np.array(kam).T  # more convenient to work with the transposed array
            # Average KAM values for each DISTANCE
            kamavg = np.nan_to_num(kam).mean(axis=0)  # nan values become 0 first

            # Linear fit of KAM vs distance
            # Slope m
            m = self.nneighbors * (kam * distance).sum(axis=1) - kam.sum(axis=1) * distance.sum()
            m /= self.nneighbors * (distance**2).sum() - (distance.sum()) ** 2

            
            grad_mag = m*np.pi/180*1e6 # conversion from deg/um to rad/m
            if pi==0:
                GND = grad_mag / self.burgersv
            else:
                GND[Sel] = grad_mag[Sel] / self.burgersv

            if False:
                # Intercept b
                b = kam.sum(axis=1) - m * distance.sum()
                b /= nneighbors

                # Fitted KAM values
                kamfit = m.reshape(-1, 1) * distance + b.reshape(-1, 1)
                # Standard deviation
                sd = ((kam - kamfit) ** 2).sum(axis=1)
                sd /= nneighbors
                sd **= 0.5

                kammean = kam.mean(axis=1).reshape(-1, 1)
                SStot = ((kam - kammean) ** 2.0).sum(axis=1)
                SStot[SStot == 0.0] = 1.0
                SSres = ((kam - kamfit) ** 2.0).sum(axis=1)
                # R squared
                Rsquared = 1.0 - SSres / SStot    
        
        clustering_result.data.GND=GND
        return clustering_result


    def computeKAM(self, clustering_result: ClusteringResult):
        for pi, phase in enumerate(clustering_result.unique_phases):
            phase_name = clustering_result.data.phase_names[phase]
            symops = np.array(clustering_result.data.phases[phase_name]['symops'])
            Sel = clustering_result.data.rois.masks_by_phase[0][phase_name]
            #print(np.where(Sel)[0].shape[0])
            #print(self.phase)
            neighbors = clustering_result.data.compute_neighbors(
                self.distance, self.perimeteronly, self.distance_convention, roi=None, sel= Sel
            )
            if pi == 0:
                KAM = self._get_KAM(clustering_result.data._ebsdData.M,neighbors,symops, Sel)
            else:
                KAM[Sel] = self._get_KAM(clustering_result.data._ebsdData.M,neighbors,symops, Sel)[Sel]
        clustering_result.data.kam=KAM
        return clustering_result
    

    def computeKAM_single(self, clustering_result: ClusteringResult):
        """Perform KAM, GND analysis."""
        # Get mask
        if False:
            if self.roi is None:
                roimask = clustering_result.data.rois.masks[0]
            else:
                roimask = clustering_result.data.rois.masks[self.roi]
            if self.cluster_id is None:
                clustermask = clustering_result.data.rois.masks[0]
            else:
                if self.phase is None:
                    self.phase = clustering_result.data.phase_names[clustering_result.cluster_phases_id[self.cluster_id]]
                clustermask = clustering_result.get_cluster_mask(self.cluster_id)
            if self.phase is None:
                phasemask = clustering_result.data.rois.masks[0]
            else:
                phasemask =clustering_result.data.rois.masks_by_phase[0][self.phase]
            if self.phase is None:
                if clustering_result.unique_phases.shape[0]==0:
                    self.phase = clustering_result.data.phase_names[clustering_result.unique_phases[0]]

        Sel, self.phase =  clustering_result._getMask(roi=self.roi,cluster_id=self.cluster_id, phase=self.phase)
        if self.phase is None:
            print(f'Unknown phase')
        else:
            print(f'Calculating KAM for phase {self.phase}')
            symops = np.array(clustering_result.data.phases[self.phase]['symops'])
            #Sel = roimask*clustermask*phasemask
            #print(np.where(Sel)[0].shape[0])
            #print(self.phase)
            neighbors = clustering_result.data.compute_neighbors(
                self.distance, self.perimeteronly, self.distance_convention, roi=None, sel= Sel
            )
            KAM = self._get_KAM(clustering_result.data._ebsdData.M,neighbors,symops, Sel)
            
            return KAM

    def _get_KAM(self,M, neighbors,symops, sel):
        """
        Returns Kernel average misorientation map

        Parameters
        ----------
        distance : int (optional)
            Distance (in neighbor indexes) to the kernel
            Default: 1
        perimeteronly : bool (optional)
            If True, KAM is calculated using only pixels in the perimeter,
            else uses inner pixels as well
            Default: True
        maxmis : float (optional)
            Maximum misorientation angle (in degrees) accounted in the
            calculation of KAM
            Default: None
        sel : bool numpy 1D array (optional)
            Boolean array indicating which data points should be plotted
            Default: None

        Returns
        -------
        KAM : numpy ndarray shape(N) with KAM values in degrees
        """



        
        return self.kernel_average_misorientation(M, neighbors, symops, sel=sel, maxmis=self.maxmis, out=self.out)


    def kernel_average_misorientation(
        self, M, neighbors, symops, sel=None, maxmis=None, out="deg"
    ):
        """
        Calculates the Kernel Average Misorientation (KAM)

        M : numpy ndarray shape(N, 3, 3)
            List of rotation matrices describing the rotation from the sample
            coordinate frame to the crystal coordinate frame
        neighbors : numpy ndarray shape(N, K) - K being the number of neighbors
            Indices of the neighboring pixels
        sel : bool numpy 1D array (optional)
            Boolean array indicating data points calculations should be
            performed
            Default: None
        out : str (optional)
            Unit of the output. Possible values are:
            'deg': angle(s) in degrees
            'rad': angle(s) in radians
            Default: 'deg'
        **kwargs :
            verbose : bool (optional)
                If True, prints computation time
                Default: True

        Returns
        -------
        KAM : numpy ndarray shape(N) - M being the number of neighbors
            KAM : numpy ndarray shape(N) with KAM values
        """
        misang = self.misorientation_neighbors4kam(M, neighbors,symops, sel=sel, out=out)
        
        outliers = misang < 0  # filter out negative values
        if maxmis is not None:
            outliers |= misang > maxmis  # and values > maxmis

        misang[outliers] = 0.0
        nneighbors = np.count_nonzero(~outliers, axis=1)

        noneighbors = nneighbors == 0
        nneighbors[noneighbors] = 1  # to prevent division by 0

        KAM = np.sum(misang, axis=1) / nneighbors
        KAM[noneighbors] = np.nan  # invalid KAM when nneighbors is 0

        return KAM
    
    def misorientation_neighbors4kam(self, M, neighbors, C, sel=None, out="deg", phase=None,verbose=True):
        
        """
        Calculates the misorientation angle of every data point with respective
        orientation matrix provided in 'M' with respect to an arbitrary number
        of neighbors, whose indices are provided in the 'neighbors' argument.

        Parameters
        ----------
        M : numpy ndarray shape(N, 3, 3)
            List of rotation matrices describing the rotation from the sample
            coordinate frame to the crystal coordinate frame
        neighbors : numpy ndarray shape(N, K) - K being the number of neighbors
            Indices of the neighboring pixels
        sel : bool numpy 1D array (optional)
            Boolean array indicating data points calculations should be
            performed
            Default: None
        out : str (optional)
            Unit of the output. Possible values are:
            'deg': angle(s) in degrees
            'rad': angle(s) in radians
            Default: 'deg'
        **kwargs :
            verbose : bool (optional)
                If True, prints computation time
                Default: True

        Returns
        -------
        misang : numpy ndarray shape(N, K) - K being the number of neighbors
            KAM : numpy ndarray shape(N) with KAM values
        """
        N = M.shape[0]
        nneighbors = neighbors.shape[1]

        #if phase is None:
        #    key1=list(self.Phases.keys())[0]
        #else:
        #    key1=phase
        #print(f'Calculated for phase {key1}')
        #C = np.array(self.phases[key1]['symops'])

        # 2D array to store trace values initialized as -2 (trace values are
        # always in the [-1, 3] interval)
        tr = np.full((N, nneighbors), -2.0, dtype=float)
        # 2D array to store the misorientation angles in degrees
        misang = np.full((N, nneighbors), -1.0, dtype=float)

        if not isinstance(sel, np.ndarray):
            sel = np.full(N, True, dtype=bool)

        #verbose = kwargs.pop("verbose", True)
        if verbose:
            t0 = time.time()
            sys.stdout.write(
                "Calculating misorientations for {} points for {} neighbors".format(
                    np.count_nonzero(sel), nneighbors
                )
            )
            sys.stdout.write(" [")
            sys.stdout.flush()

        for k in range(nneighbors):
            # valid points, i.e., those part of the selection and with valid neighrbor index (> 0)
            ok = (neighbors[:, k] >= 0) & sel & sel[neighbors[:, k]]
            # Rotation from M[ok] to M[neighbors[ok, k]]
            # Equivalent to np.matmul(M[neighbors[ok,k]], M[ok].transpose([0,2,1]))
            T = np.einsum("ijk,imk->ijm", M[neighbors[ok, k]], M[ok])

            for m in range(len(C)):
                # Smart way to calculate the trace using einsum.
                # Equivalent to np.matmul(C[m], T).trace(axis1=1, axis2=2)
                a, b = C[m].nonzero()
                ttr = np.einsum("j,ij->i", C[m, a, b], T[:, a, b])
                tr[ok, k] = np.max(np.vstack([tr[ok, k], ttr]), axis=0)

            if verbose:
                if k > 0 and k < nneighbors:
                    sys.stdout.write(", ")
                sys.stdout.write("{}".format(k + 1))
                sys.stdout.flush()

        del T, ttr

        if verbose:
            sys.stdout.write("] in {:.2f} s\n".format(time.time() - t0))
            sys.stdout.flush()

        # Take care of tr > 3. that might happend due to rounding errors
        tr[tr > 3.0] = 3.0

        # Filter out invalid trace values
        ok = tr >= -1.0
        misang[ok] = trace_to_angle(tr[ok], out)
        return misang
# ============================================================================
# VISUALIZER
# ============================================================================

class EBSDVisualizer:
    """
    Handles all visualization tasks.
    
    Design: Strategy pattern for different plot types
    - Separates visualization from analysis logic
    - Easy to extend with new plot types
    """
    
    def __init__(self, figsize=(10, 10), dpi=100,invert_y_axis=True):
        self.figsize = figsize
        self.dpi = dpi
        self.invert_y_axis = invert_y_axis  # Default y-axis inversion
        self.fig = None
        self.axes = None
    def _getMask(self, clustering_result, roi=None,cluster_id=None, phase=None):
        if roi is None:
            roimask = clustering_result.data.rois.masks[0]
        else:
            roimask = clustering_result.data.rois.masks[roi]
        if cluster_id is None:
            clustermask = clustering_result.data.rois.masks[0]
        else:
            if phase is None:
                phase = clustering_result.data.phase_names[clustering_result.cluster_phases_id[cluster_id]]
            clustermask = clustering_result.get_cluster_mask(cluster_id)
        if phase is None:
            phasemask = clustering_result.data.rois.masks[0]
        else:
            phasemask =clustering_result.data.rois.masks_by_phase[0][phase]
        if phase is None:
            if clustering_result.unique_phases.shape[0]==0:
                phase = clustering_result.data.phase_names[clustering_result.unique_phases[0]]

        return roimask*clustermask*phasemask, phase
    def getColors(self,clusters_unique, labels, cmap='jet'):
        # Create mapping array where index = old label, value = new label
        max_label = clusters_unique.max()
        mapping = np.arange(max_label + 1)  # Initialize with identity mapping

        # Shuffle only the non-zero labels

        shuffled = np.random.permutation(clusters_unique)
        mapping[clusters_unique] = shuffled

        # Apply mapping
        renumbered_labels = mapping[labels]
        Colors = np.zeros((labels.shape[0], 4))
        Colors = cluster_colors(renumbered_labels,cmap_name=cmap)
        return Colors.astype(int)
    def plotClusters(self, clustering_result: ClusteringResult, d=[1,0,0], cluster_id=None, 
                    orientations=None, color_by='cluster', cmap='jet', tiling=None, 
                    scalebar=True, globalScale=False, roi=None, phase=None, color=None, 
                    data=None, mask=None, vmin=None, vmax=None, fig=None, ax=None,
                    show_labels=False, label_phase=None, label_clusters=None, 
                    label_fontsize=10, label_color='white', label_bbox=True,
                    label_bbox_style='round', clusters_unique=None, labels=None, **kwargs):
        """
        Plot clusters with optional cluster ID labels.
        
        Parameters
        ----------
        clustering_result : ClusteringResult
            Clustering result object
        d : list, optional
            Direction for IPF coloring. Default is [1,0,0].
        cluster_id : int or list, optional
            Specific cluster(s) to plot
        orientations : array, optional
            Custom orientations
        color_by : str, optional
            Coloring scheme: 'cluster', 'ipf', 'avgipf', 'data', 'kam', 'gnd'
        cmap : str, optional
            Colormap name. Default is 'jet'.
        tiling : optional
            Tiling parameter
        scalebar : bool, optional
            Show scalebar. Default is True.
        globalScale : bool, optional
            Use global scale. Default is False.
        roi : optional
            Region of interest
        phase : int or str, optional
            Phase to plot
        color : array, optional
            Custom colors
        data : array, optional
            Custom data for coloring
        mask : array, optional
            Mask for plotting
        vmin, vmax : float, optional
            Value range for data coloring
        fig, ax : matplotlib objects, optional
            Figure and axis to plot on
        show_labels : bool or str, optional
            Whether to show cluster ID labels:
            - False: No labels (default)
            - True or 'all': Show all cluster labels
            - 'phase': Show labels for clusters in specified phase (requires label_phase)
            - 'selected': Show labels only for specified clusters (requires label_clusters)
        label_phase : int or str, optional
            Phase ID or name for which to show labels (when show_labels='phase')
        label_clusters : int or list of int, optional
            Specific cluster ID(s) to label (when show_labels='selected')
        label_fontsize : int, optional
            Font size for labels. Default is 10.
        label_color : str, optional
            Text color for labels. Default is 'white'.
        label_bbox : bool, optional
            Whether to show background box for labels. Default is True.
        label_bbox_style : str, optional
            Style of label box: 'round', 'square', etc. Default is 'round'.
        **kwargs : optional
            Additional arguments passed to plotting functions
        
        Returns
        -------
        fig, ax : matplotlib figure and axis
        
        Examples
        --------
        >>> # Plot without labels
        >>> fig, ax = vis.plotClusters(clustering_result)
        >>> 
        >>> # Show all cluster labels
        >>> fig, ax = vis.plotClusters(clustering_result, show_labels=True)
        >>> 
        >>> # Show labels only for Ferrite phase
        >>> fig, ax = vis.plotClusters(clustering_result, 
        ...                           show_labels='phase', 
        ...                           label_phase='Ferrite')
        >>> 
        >>> # Show labels only for specific clusters
        >>> fig, ax = vis.plotClusters(clustering_result, 
        ...                           show_labels='selected', 
        ...                           label_clusters=[5, 12, 18, 23])
        """        
        import matplotlib.pyplot as plt
        plot = True
        
        mask2, phase = clustering_result._getMask(cluster_id=cluster_id, roi=roi, phase=phase)
        if mask is not None:
            mask = mask * mask2
        else:
            mask = mask2
        
        if np.where(mask)[0].shape[0] < 2:
            print('Data with only 1 pixel or less cannot be plotted')
            plot = False
            return None, None  # ADD THIS LINE - explicit return when not plotting
        
        if plot:
            if color_by == 'data' or color_by.lower() == 'kam' or color_by.lower() == 'gnd':
                if color_by.lower() == 'kam':
                    data = clustering_result.data.kam
                    #print(data)
                if color_by.lower() == 'gnd':
                    data = clustering_result.data.GND
                if vmin is None:
                    vmin = np.nanmin(data)
                if vmax is None:
                    vmax = np.nanmax(data)
                #print(data)
                #print(vmax)
                norm = plt.Normalize(vmin, vmax)
                cmap_obj = plt.get_cmap(cmap)
                Colors = cmap_obj(norm(data))
                Colors = Colors * 255
                Colors = Colors.astype(int)
                Colors[:, 3] = 255
                
                fig, ax = clustering_result.data.plot_colmap(
                    d=d, tiling=tiling, scalebar=scalebar, globalScale=globalScale, 
                    color=Colors, mask=mask, fig=fig, ax=ax, **kwargs
                )
            
            if color_by == 'cluster':
                if clusters_unique is None or labels is None:
                    clusters_unique = clustering_result._clusters_unique
                    labels = clustering_result.labels
                Colors = self.getColors(clusters_unique, labels, cmap=cmap)
                if color is None:
                    color = Colors
                if roi is None:
                    roi = clustering_result.parameters['roi']
                fig, ax = clustering_result.data.plot_colmap(
                    d=d, tiling=tiling, scalebar=scalebar, globalScale=globalScale, 
                    roi=roi, phase=phase, color=color, mask=mask, fig=fig, ax=ax, **kwargs
                )
            elif color_by == 'color':
                clusters_unique = clustering_result._clusters_unique
                labels = clustering_result.labels
                color = (np.zeros((labels.shape[0], 4))+color).astype(int)
                if roi is None:
                    roi = clustering_result.parameters['roi']
                fig, ax = clustering_result.data.plot_colmap(
                    d=d, tiling=tiling, scalebar=scalebar, globalScale=globalScale, 
                    roi=roi, phase=phase, color=color, mask=mask, fig=fig, ax=ax, **kwargs
                )
            
            elif color_by == 'avgipf':
                if orientations is not None:
                    Mavg = orientations
                else:
                    labels = clustering_result.labels
                    avg_orientations = clustering_result.avg_orientations
                    Mavg = copy.deepcopy(clustering_result.data.orientations)
                    for label in avg_orientations.keys():
                        Mavg[labels == label, :, :] = avg_orientations[label]
                if roi is None:
                    roi = clustering_result.parameters['roi']
                fig, ax = clustering_result.data.plot_IPF(  # REMOVE any assignment if this returns None
                    d, tiling=tiling, scalebar=scalebar, globalScale=globalScale, 
                    roi=roi, phase=phase, orientations=Mavg, mask=mask, fig=fig, ax=ax, **kwargs
                )
            
            elif color_by == 'ipf':
                if roi is None:
                    roi = clustering_result.parameters['roi']
                fig, ax = clustering_result.data.plot_IPF(  # REMOVE any assignment if this returns None
                    d, tiling=tiling, scalebar=scalebar, globalScale=globalScale, 
                    roi=roi, phase=phase, orientations=None, mask=mask, fig=fig, ax=ax, **kwargs
                )
            
            # ========== ADD CLUSTER LABELS ==========
            if show_labels:
                import matplotlib.pyplot as plt
                
                # Determine which clusters to label
                clusters_to_label = []
                
                if show_labels == True or show_labels == 'all':
                    # Label all clusters
                    clusters_to_label = clustering_result._clusters_unique.tolist()
                
                elif show_labels == 'phase':
                    # Label clusters in specified phase
                    if label_phase is None:
                        raise ValueError("show_labels='phase' requires label_phase parameter")
                    
                    # Get phase ID
                    if isinstance(label_phase, str):
                        # Find phase ID from name
                        phase_id = None
                        for pid, pname in clustering_result.data.phase_names.items():
                            if pname == label_phase:
                                phase_id = pid
                                break
                        if phase_id is None:
                            raise ValueError(f"Phase '{label_phase}' not found")
                    else:
                        phase_id = label_phase
                    
                    # Get clusters for this phase
                    if hasattr(clustering_result, 'labels_by_phase'):
                        phase_name = clustering_result.data.phase_names[phase_id]
                        if phase_name in clustering_result.labels_by_phase:
                            clusters_to_label = clustering_result.labels_by_phase[phase_name]
                    else:
                        # Fallback: find clusters manually
                        labels = clustering_result.labels
                        phases_id = clustering_result.data.phases_id
                        for cluster_label in clustering_result._clusters_unique:
                            cluster_mask = labels == cluster_label
                            cluster_phase = phases_id[cluster_mask][0]  # Get phase of first pixel
                            if cluster_phase == phase_id:
                                clusters_to_label.append(cluster_label)
                
                elif show_labels == 'selected':
                    # Label specific clusters
                    if label_clusters is None:
                        raise ValueError("show_labels='selected' requires label_clusters parameter")
                    
                    if isinstance(label_clusters, int):
                        clusters_to_label = [label_clusters]
                    else:
                        clusters_to_label = list(label_clusters)
                
                else:
                    raise ValueError(f"Invalid show_labels value: {show_labels}. "
                                f"Use False, True, 'all', 'phase', or 'selected'")
                
                # Get data coordinates and labels
                X = clustering_result.data.X
                Y = clustering_result.data.Y
                labels = clustering_result.labels
                
                # Apply mask if present
                if mask is not None:
                    X_masked = X[mask]
                    Y_masked = Y[mask]
                    labels_masked = labels[mask]
                else:
                    X_masked = X
                    Y_masked = Y
                    labels_masked = labels
                
                # Plot labels for each cluster
                for cluster_label in clusters_to_label:
                    # Get pixels in this cluster
                    cluster_pixels = labels_masked == cluster_label
                    
                    if np.sum(cluster_pixels) > 0:
                        # Calculate centroid
                        x_center = np.mean(X_masked[cluster_pixels])
                        y_center = np.mean(Y_masked[cluster_pixels])
                        
                        # Create label text
                        label_text = f'{cluster_label}'
                        
                        # Prepare bbox properties
                        if label_bbox:
                            bbox_props = dict(
                                boxstyle=label_bbox_style,
                                facecolor='black',
                                alpha=0.7,
                                edgecolor='white',
                                linewidth=1
                            )
                        else:
                            bbox_props = None
                        
                        # Add text label
                        ax.text(
                            x_center, y_center, label_text,
                            ha='center', va='center',
                            fontsize=label_fontsize,
                            color=label_color,
                            fontweight='bold',
                            bbox=bbox_props,
                            zorder=1000  # Ensure labels are on top
                        )
                
                # REMOVE THIS PRINT - it might be causing issues
                # print(f"Added labels for {len(clusters_to_label)} clusters")
            # ========================================
        
        return fig, ax
#clustering_result.get_cluster_mask(1)

    def plotAvgOriIPF(self,d, tiling=None, scalebar=True,globalScale=False, roi=None, phase=None, fig=None, ax=None,  **kwargs):
        Mavg=copy.deepcopy(self.data.orientations)
        for label in self.avg_orientations.keys():
            Mavg[self.labels==label,:,:]=self.avg_orientations[label] 
        if roi is None:
            roi = self.parameters['roi']
            
        self.data.plot_IPF(d, tiling=tiling, scalebar=scalebar,globalScale=globalScale, roi=roi, phase=phase, orientations=Mavg,fig=fig, ax=ax, **kwargs)

    def plot_clustering(self, clustering_result: ClusteringResult, 
                       color_by='cluster',invert_y_axis=None, **kwargs):
        """
        Plot clustering results.
        
        Parameters
        ----------
        clustering_result : ClusteringResult
            Results to plot
        color_by : str
            'cluster', 'phase', or 'quality'
        invert_y_axis : bool, optional
            Whether to invert y-axis. If None, uses instance default.
        """

        if invert_y_axis is None:
            invert_y_axis = self.invert_y_axis

        data = clustering_result.data
        labels = clustering_result.labels
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if color_by == 'cluster':
            colors = self._get_cluster_colors(labels)
        elif color_by == 'phase':
            colors = self._get_phase_colors(data._ebsdData.phase)
        elif color_by == 'quality':
            colors = data.quality
        
        scatter = ax.scatter(data.X, data.Y, c=colors, s=1, **kwargs)
        #ax.yaxis.set_inverted(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Clustering Result - colored by {color_by}')
        ax.axis('equal')
        if invert_y_axis:
            ax.invert_yaxis()
        if color_by in ['phase', 'quality']:
            plt.colorbar(scatter, ax=ax)
        
        self.fig = fig
        self.axes = [ax]
        
        return fig, ax
    
    def plot_boundaries(self, boundary_result: BoundaryResult,
                       boundary_type='all',invert_y_axis=None, **kwargs):
        """
        Plot grain boundaries.
        
        Parameters
        ----------
        boundary_result : BoundaryResult
            Boundary analysis results
        boundary_type : str
            'all', 'interphase', 'same_phase', 'roi'
        invert_y_axis : bool, optional
            Whether to invert y-axis. If None, uses instance default.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for i, boundaries in enumerate(boundary_result.grouped_boundaries):
            cluster_phase = boundary_result.cluster_phases_id[i]
            boundary_phases = boundary_result.grouped_boundary_phases_id[i]
            
            for nb_label, coords in boundaries.items():
                # Determine if this boundary should be plotted
                if nb_label == -1:
                    if boundary_type in ['all', 'roi']:
                        ax.plot(coords[:, 0], coords[:, 1], 'ko', 
                               markersize=2, alpha=0.5)
                else:
                    nb_phase = boundary_phases[nb_label]
                    is_interphase = (nb_phase != cluster_phase)
                    
                    if boundary_type == 'all':
                        color = 'red' if is_interphase else 'blue'
                        alpha = 0.8 if is_interphase else 0.3
                        ax.plot(coords[:, 0], coords[:, 1], 'o',
                               color=color, markersize=2, alpha=alpha)
                    elif boundary_type == 'interphase' and is_interphase:
                        ax.plot(coords[:, 0], coords[:, 1], 'ro',
                               markersize=2, alpha=0.8)
                    elif boundary_type == 'same_phase' and not is_interphase:
                        ax.plot(coords[:, 0], coords[:, 1], 'bo',
                               markersize=2, alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Grain Boundaries - {boundary_type}')
        if invert_y_axis:
            ax.invert_yaxis()
        #ax.yaxis.set_inverted(True)
        ax.axis('equal')
        
        return fig, ax
    
    def plot_single_cluster(self, boundary_result: BoundaryResult,
                           cluster_label: int, **kwargs):
        """Plot boundaries for a single cluster."""
        # Implementation from your plot_single_cluster_interphase_boundary
        pass

    def plot_parent_child_interface(self, boundary_result, parent_cluster_id, 
                                    child_cluster_id=None, side='both', 
                                    parent_phase_id=None, child_phase_id=None,
                                    ax=None, show_labels=True, invert_y_axis=None,
                                    show_parent_same_phase_boundaries=False,
                                    filter_to_semi_roi=None,
                                    semi_roi_params=None,
                                    clustering_result=None,
                                    elongation_cache=None,
                                    only_lamellar=False,
                                    min_aspect_ratio=2.5,
                                    plot_as_lines=False,
                                    line_width=2,
                                    show_legend=True,
                                    include_touching_parents=False,
                                    hierarchy=None):  # NEW PARAMETER
        """
        Plot parent-child interface(s) with flexible display options.
        
        Parameters
        ----------
        boundary_result : BoundaryResult
            Boundary analysis results
        parent_cluster_id : int
            Parent cluster ID to plot
        child_cluster_id : int, optional
            Specific child cluster ID to plot. If None, plots all children.
        side : str, optional
            Which side of interface to plot: 'both', 'parent', 'child'
            Default is 'both'
        parent_phase_id : int, optional
            Parent phase ID. If None, inferred from parent_cluster_id.
        child_phase_id : int, optional
            Child phase ID. If None, inferred from children.
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates new figure.
        show_labels : bool, optional
            Whether to show cluster ID text labels on plot. Default is True.
        invert_y_axis : bool, optional
            Whether to invert y-axis. If None, uses instance default.
        show_parent_same_phase_boundaries : bool, optional
            Whether to plot parent cluster boundaries with same-phase neighbors.
            Default is False.
        filter_to_semi_roi : int or str, optional
            Filter interface(s) to semi-ROI(s):
            - None: No filtering (default, shows full interface)
            - 1, 'side1', 'semi_roi_1': Filter to semi-ROI 1
            - 2, 'side2', 'semi_roi_2': Filter to semi-ROI 2
            - 'full', 'full_roi': Filter to full ROI
        semi_roi_params : dict, optional
            Parameters for semi-ROI creation.
        clustering_result : ClusteringResult, optional
            Clustering result object.
        elongation_cache : dict or list, optional
            Pre-computed elongation results.
        only_lamellar : bool, optional
            If True and child_cluster_id=None, plots only lamellar children.
            Default is False.
        min_aspect_ratio : float, optional
            Minimum aspect ratio to consider lamellar. Default is 2.5.
        plot_as_lines : bool, optional
            If True, plots boundaries as connected lines instead of scatter points.
            Uses nearest-neighbor ordering to connect points. Default is False.
        line_width : float, optional
            Line width when plot_as_lines=True. Default is 2.
        show_legend : bool, optional
            Whether to show legend. Default is True.
        include_touching_parents : bool, optional
            Whether to include touching children in the hierarchy.
            Default is False (backward compatible).
        hierarchy : list of dict, optional
            Pre-computed hierarchy from get_parent_child_hierarchy_with_boundaries.
            If provided, skips hierarchy computation.
            Default is None (computes hierarchy).
        
        Returns
        -------
        fig, ax : matplotlib figure and axis
        
        Examples
        --------
        >>> # Standard usage (computes hierarchy)
        >>> fig, ax = vis.plot_parent_child_interface(
        ...     boundary_result, parent_cluster_id=5, child_cluster_id=12
        ... )
        >>> 
        >>> # With pre-computed hierarchy (efficient for multiple plots)
        >>> hierarchy = boundary_result.get_parent_child_hierarchy_with_boundaries(
        ...     parent_phase_id=1,
        ...     child_phase_id=2,
        ...     filter_lamellar_to_semi_rois=True,
        ...     clustering_result=clustering_result,
        ...     elongation_cache=all_clusters,
        ...     min_aspect_ratio=2.5
        ... )
        >>> 
        >>> # Plot multiple parents using same hierarchy
        >>> for parent_id in [5, 10, 15]:
        ...     fig, ax = vis.plot_parent_child_interface(
        ...         boundary_result,
        ...         parent_cluster_id=parent_id,
        ...         hierarchy=hierarchy  # Reuse hierarchy
        ...     )
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.patches import Polygon
        from matplotlib.path import Path
        from scipy.spatial.distance import cdist
        
        def order_boundary_points(coords):
            """
            Order boundary points to form a continuous line using nearest-neighbor.
            
            Parameters
            ----------
            coords : array (N, 2)
                Unordered boundary coordinates
            
            Returns
            -------
            ordered_coords : array (N, 2)
                Ordered coordinates forming a continuous path
            """
            if len(coords) <= 1:
                return coords
            
            # Start with first point
            ordered = [coords[0]]
            remaining = list(range(1, len(coords)))
            current_idx = 0
            
            while remaining:
                # Find nearest unvisited point
                current_point = coords[current_idx:current_idx+1]
                remaining_points = coords[remaining]
                
                # Calculate distances
                distances = cdist(current_point, remaining_points)[0]
                nearest_idx = np.argmin(distances)
                
                # Add nearest point
                next_idx = remaining[nearest_idx]
                ordered.append(coords[next_idx])
                current_idx = next_idx
                remaining.pop(nearest_idx)
            
            return np.array(ordered)
        
        if invert_y_axis is None:
            invert_y_axis = self.invert_y_axis
        
        # Validate side parameter
        if side not in ['both', 'parent', 'child']:
            raise ValueError("side must be 'both', 'parent', or 'child'")
        
        # Validate filter_to_semi_roi
        if filter_to_semi_roi is not None:
            if clustering_result is None and hierarchy is None:
                raise ValueError("filter_to_semi_roi requires clustering_result parameter "
                            "(or pre-computed hierarchy with ROI info)")
        
        # Validate only_lamellar
        if only_lamellar:
            if child_cluster_id is not None:
                raise ValueError("only_lamellar only works when child_cluster_id=None (multiple children)")
            if clustering_result is None and hierarchy is None:
                raise ValueError("only_lamellar requires clustering_result parameter "
                            "(or pre-computed hierarchy with lamellar info)")
            if clustering_result is not None and elongation_cache is None and hierarchy is None:
                print("Warning: only_lamellar without elongation_cache will be slow (computing PCA for each child)")
        
        # Set default semi_roi_params
        if semi_roi_params is None:
            semi_roi_params = {
                'n_layers': 1,
                'longitudinal_shrinkage': 0.1,
                'layer_thickness_relative': None
            }
        
        # Find parent cluster index and phase
        try:
            parent_idx = np.where(boundary_result.clusters == parent_cluster_id)[0][0]
        except IndexError:
            raise ValueError(f"Parent cluster {parent_cluster_id} not found!")
        
        if parent_phase_id is None:
            parent_phase_id = boundary_result.cluster_phases_id[parent_idx]
        
        # ========== GET OR USE HIERARCHY ==========
        if hierarchy is None:
            # Need to compute hierarchy
            print("Computing hierarchy...")
            
            # Determine if we need lamellar filtering in hierarchy
            need_lamellar_filtering = (filter_to_semi_roi is not None) or only_lamellar
            
            # Build hierarchy call parameters
            hierarchy_params = {
                'parent_phase_id': parent_phase_id,
                'child_phase_id': child_phase_id if child_phase_id is not None else None,
            }
            
            # Add clustering_result if available (needed for include_touching_parents)
            if clustering_result is not None:
                hierarchy_params['clustering_result'] = clustering_result
                
                # Add include_touching_parents if requested
                if include_touching_parents:
                    hierarchy_params['include_touching_parents'] = True
                
                # Add lamellar filtering if needed
                if need_lamellar_filtering:
                    hierarchy_params['filter_lamellar_to_semi_rois'] = True
                    hierarchy_params['semi_roi_params'] = semi_roi_params
                    hierarchy_params['elongation_cache'] = elongation_cache
                    hierarchy_params['min_aspect_ratio'] = min_aspect_ratio
            elif include_touching_parents:
                # Warn user that touching parents requires clustering_result
                print("Warning: include_touching_parents=True requires clustering_result. "
                    "Plotting without touching children.")
            
            # Call hierarchy with appropriate parameters
            hierarchy = boundary_result.get_parent_child_hierarchy_with_boundaries(**hierarchy_params)
        else:
            # Use provided hierarchy
            print("Using pre-computed hierarchy")
        # ==========================================
        
        # Find this parent in hierarchy
        parent_data = None
        for p in hierarchy:
            if p['parent_cluster_id'] == parent_cluster_id:
                parent_data = p
                break
        
        if parent_data is None:
            raise ValueError(f"Parent cluster {parent_cluster_id} not found in hierarchy "
                            f"or has no children")
        
        # ========== FILTER CHILDREN ==========
        if child_cluster_id is not None:
            # Single child
            children_to_plot = [c for c in parent_data['children'] 
                            if c['child_cluster_id'] == child_cluster_id]
            if len(children_to_plot) == 0:
                raise ValueError(f"Child cluster {child_cluster_id} not found as child "
                            f"of parent {parent_cluster_id}")
            
            # Check if this is a touching relationship
            relationship = children_to_plot[0].get('relationship', 'enclosed')
            
            if filter_to_semi_roi is not None:
                if filter_to_semi_roi in [1, 'side1', 'semi_roi_1']:
                    roi_str = 'Semi-ROI 1'
                elif filter_to_semi_roi in [2, 'side2', 'semi_roi_2']:
                    roi_str = 'Semi-ROI 2'
                elif filter_to_semi_roi in ['full', 'full_roi']:
                    roi_str = 'Full ROI'
                else:
                    roi_str = str(filter_to_semi_roi)
                plot_title = (f'Interface: Parent {parent_cluster_id} → '
                            f'Child {child_cluster_id} ({relationship}, side: {side}, filtered to {roi_str})')
            else:
                plot_title = (f'Interface: Parent {parent_cluster_id} → '
                            f'Child {child_cluster_id} ({relationship}, side: {side})')
        else:
            # Multiple children
            if only_lamellar:
                # Filter for lamellar children using 'is_lamellar' flag from hierarchy
                children_to_plot = [c for c in parent_data['children'] 
                                if c.get('is_lamellar', False)]
                
                if len(children_to_plot) == 0:
                    # ========== BETTER ERROR MESSAGE ==========
                    total_children = len(parent_data['children'])
                    
                    # Check if any children have aspect ratio info
                    children_with_ar = [c for c in parent_data['children'] 
                                    if 'aspect_ratio' in c]
                    
                    if len(children_with_ar) > 0:
                        max_ar = max(c['aspect_ratio'] for c in children_with_ar)
                        error_msg = (f"Parent {parent_cluster_id} has {total_children} children "
                                f"but none are lamellar (max aspect ratio: {max_ar:.2f}, "
                                f"threshold: {min_aspect_ratio})")
                    else:
                        error_msg = (f"Parent {parent_cluster_id} has {total_children} children "
                                f"but none meet lamellar criteria (aspect ratio >= {min_aspect_ratio})")
                    
                    # Provide helpful suggestion
                    print(f"\n{error_msg}")
                    print(f"Suggestions:")
                    print(f"  - Use only_lamellar=False to plot all children")
                    print(f"  - Lower min_aspect_ratio (currently {min_aspect_ratio})")
                    
                    # Show children info
                    if total_children <= 5:
                        print(f"\nChildren of parent {parent_cluster_id}:")
                        for c in parent_data['children']:
                            child_id = c['child_cluster_id']
                            ar = c.get('aspect_ratio', 'N/A')
                            is_lam = c.get('is_lamellar', False)
                            rel = c.get('relationship', 'enclosed')
                            ar_str = f"{ar:.2f}" if isinstance(ar, (int, float)) else ar
                            print(f"  Child {child_id}: AR={ar_str}, lamellar={is_lam}, {rel}")
                    
                    raise ValueError(error_msg)
                    # ==========================================
                
                if filter_to_semi_roi is not None:
                    if filter_to_semi_roi in [1, 'side1', 'semi_roi_1']:
                        roi_str = 'Semi-ROI 1'
                    elif filter_to_semi_roi in [2, 'side2', 'semi_roi_2']:
                        roi_str = 'Semi-ROI 2'
                    elif filter_to_semi_roi in ['full', 'full_roi']:
                        roi_str = 'Full ROI'
                    else:
                        roi_str = str(filter_to_semi_roi)
                    plot_title = (f'Parent {parent_cluster_id} with {len(children_to_plot)} '
                                f'LAMELLAR children (AR >= {min_aspect_ratio}, filtered to {roi_str}, side: {side})')
                else:
                    plot_title = (f'Parent {parent_cluster_id} with {len(children_to_plot)} '
                                f'LAMELLAR children (AR >= {min_aspect_ratio}, side: {side})')
            else:
                # All children
                children_to_plot = parent_data['children']
                
                if filter_to_semi_roi is not None:
                    if filter_to_semi_roi in [1, 'side1', 'semi_roi_1']:
                        roi_str = 'Semi-ROI 1'
                    elif filter_to_semi_roi in [2, 'side2', 'semi_roi_2']:
                        roi_str = 'Semi-ROI 2'
                    elif filter_to_semi_roi in ['full', 'full_roi']:
                        roi_str = 'Full ROI'
                    else:
                        roi_str = str(filter_to_semi_roi)
                    plot_title = (f'Parent {parent_cluster_id} with {len(children_to_plot)} children '
                                f'(lamellar filtered to {roi_str}, side: {side})')
                else:
                    plot_title = (f'Parent {parent_cluster_id} with {len(children_to_plot)} children '
                                f'(side: {side})')
        # =====================================
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False
        
        # Plot parent cluster's same-phase boundaries if requested
        if show_parent_same_phase_boundaries:
            parent_boundaries = boundary_result.grouped_boundaries[parent_idx]
            parent_boundary_phases = boundary_result.grouped_boundary_phases_id[parent_idx]
            
            for neighbor_label, coords in parent_boundaries.items():
                if neighbor_label == -1:
                    continue
                
                neighbor_phase = parent_boundary_phases[neighbor_label]
                
                if neighbor_phase == parent_phase_id:
                    label = 'Parent same-phase boundaries' if (show_legend and neighbor_label == list(parent_boundaries.keys())[1]) else ''
                    
                    if plot_as_lines:
                        ordered_coords = order_boundary_points(coords)
                        ax.plot(ordered_coords[:, 0], ordered_coords[:, 1],
                            color='lightgray', linewidth=line_width*0.5, alpha=0.4,
                            label=label,
                            zorder=1)
                    else:
                        ax.scatter(
                            coords[:, 0], coords[:, 1],
                            c='lightgray', s=30, alpha=0.4,
                            marker='o', edgecolors='gray', linewidths=0.3,
                            label=label,
                            zorder=1
                        )
        
        # Generate colors for children
        n_children = len(children_to_plot)
        if n_children > 1:
            child_colors = cm.get_cmap('tab10')(np.linspace(0, 1, n_children))
        else:
            child_colors = ['red']
        
        # ========== PROCESS ROI INFO FOR MULTIPLE CHILDREN ==========
        roi_infos = {}

        if filter_to_semi_roi is not None:
            roi_key_map = {
                1: 'semi_roi_1_vertices', 'side1': 'semi_roi_1_vertices', 'semi_roi_1': 'semi_roi_1_vertices',
                2: 'semi_roi_2_vertices', 'side2': 'semi_roi_2_vertices', 'semi_roi_2': 'semi_roi_2_vertices',
                'full': 'vertices', 'full_roi': 'vertices'
            }
            vertices_key = roi_key_map.get(filter_to_semi_roi)
            
            if vertices_key is None:
                raise ValueError(f"Invalid filter_to_semi_roi: {filter_to_semi_roi}")
            
            # ========== USE ROI INFO FROM HIERARCHY IF AVAILABLE ==========
            for child in children_to_plot:
                child_id = child['child_cluster_id']
                
                # Check if ROI info already exists in hierarchy
                if 'roi_info' in child:
                    # Use pre-computed ROI from hierarchy
                    roi_infos[child_id] = child['roi_info']
                else:
                    # Need to compute ROI (child is not lamellar or hierarchy wasn't called with filtering)
                    if clustering_result is None:
                        print(f"Warning: Cannot compute ROI for child {child_id} (no clustering_result provided)")
                        continue
                    
                    # Check if this child is lamellar
                    is_lamellar = child.get('is_lamellar', False)
                    
                    if not is_lamellar:
                        # ========== IMPROVED: Better cache handling ==========
                        # Try to get from cache
                        elongation_info = None
                        if elongation_cache is not None:
                            if isinstance(elongation_cache, dict):
                                elongation_info = elongation_cache.get(child_id)
                            elif isinstance(elongation_cache, list):
                                for cluster_data in elongation_cache:
                                    if cluster_data['cluster_id'] == child_id:
                                        elongation_info = cluster_data
                                        break
                        
                        # If not in cache, skip (don't compute)
                        if elongation_info is None:
                            print(f"Warning: Child {child_id} not in elongation_cache. Skipping ROI computation.")
                            continue
                        
                        # Check aspect ratio from cache
                        aspect_ratio = elongation_info.get('aspect_ratio', 0)
                        if aspect_ratio < min_aspect_ratio:
                            continue  # Skip non-lamellar
                        # ====================================================
                    
                    # Compute ROI (we know it's lamellar or should be)
                    try:
                        roi_info = clustering_result.create_lamellar_roi(
                            cluster_id=child_id,
                            n_layers=semi_roi_params['n_layers'],
                            longitudinal_shrinkage=semi_roi_params['longitudinal_shrinkage'],
                            layer_thickness_relative=semi_roi_params['layer_thickness_relative'],
                            elongation_cache=elongation_cache  # Pass cache to avoid recomputation
                        )
                        roi_infos[child_id] = roi_info
                    except Exception as e:
                        print(f"Warning: Could not create ROI for child {child_id}: {e}")
                        continue
                
                # Plot ROI boundaries (only if we successfully got ROI info)
                if child_id in roi_infos:
                    if child_cluster_id is not None or len(children_to_plot) <= 3:
                        roi_info = roi_infos[child_id]
                        full_roi_vertices = roi_info['vertices']
                        roi_label = f'ROI {child_id}' if (show_legend and len(children_to_plot) > 1) else ('' if not show_legend else 'Full ROI')
                        full_roi_poly = Polygon(full_roi_vertices, fill=False, edgecolor='blue',
                                            linewidth=1.5, linestyle='--', alpha=0.5,
                                            label=roi_label,
                                            zorder=2)
                        ax.add_patch(full_roi_poly)
                        
                        if child_cluster_id is not None:
                            semi1_vertices = roi_info['semi_roi_1_vertices']
                            semi1_poly = Polygon(semi1_vertices, fill=False, edgecolor='cyan',
                                                linewidth=2, linestyle=':', alpha=0.7,
                                                label='Semi-ROI 1' if show_legend else '',
                                                zorder=2)
                            ax.add_patch(semi1_poly)
                            
                            semi2_vertices = roi_info['semi_roi_2_vertices']
                            semi2_poly = Polygon(semi2_vertices, fill=False, edgecolor='magenta',
                                                linewidth=2, linestyle=':', alpha=0.7,
                                                label='Semi-ROI 2' if show_legend else '',
                                                zorder=2)
                            ax.add_patch(semi2_poly)
                            
                            centerline = roi_info['centerline']
                            ax.plot([centerline[0][0], centerline[1][0]],
                                [centerline[0][1], centerline[1][1]],
                                'k--', linewidth=1.5, alpha=0.5, 
                                label='Centerline' if show_legend else '', 
                                zorder=2)
        # ============================================================
        
        # Plot each child interface
        for i, child in enumerate(children_to_plot):
            child_id = child['child_cluster_id']
            interface = child['interface']
            
            # Choose color
            if n_children > 1:
                color = child_colors[i]
            else:
                color = 'red'
            
            # Get coordinates
            parent_coords = interface['parent_side']['coords']
            child_coords = interface['child_side']['coords']
            parent_indices = interface['parent_side']['indices']
            child_indices = interface['child_side']['indices']
            
            # ========== APPLY SEMI-ROI FILTERING ==========
            if filter_to_semi_roi is not None and child_id in roi_infos:
                roi_info = roi_infos[child_id]
                roi_vertices = roi_info[vertices_key]
                roi_path = Path(roi_vertices)
                
                parent_mask = roi_path.contains_points(parent_coords)
                parent_coords_filtered = parent_coords[parent_mask]
                parent_indices_filtered = parent_indices[parent_mask]
                
                child_mask = roi_path.contains_points(child_coords)
                child_coords_filtered = child_coords[child_mask]
                child_indices_filtered = child_indices[child_mask]
                
                parent_coords_plot = parent_coords_filtered
                child_coords_plot = child_coords_filtered
                parent_indices_plot = parent_indices_filtered
                child_indices_plot = child_indices_filtered
                
                # Plot filtered-out points
                if child_cluster_id is not None and not plot_as_lines and show_legend:
                    if len(parent_coords) > 0:
                        parent_mask_out = ~parent_mask
                        if np.any(parent_mask_out):
                            ax.scatter(
                                parent_coords[parent_mask_out, 0],
                                parent_coords[parent_mask_out, 1],
                                c='lightgray', s=20, alpha=0.3,
                                marker='o', edgecolors='none',
                                label='Filtered out (parent)' if i == 0 else '',
                                zorder=1
                            )
                    
                    if len(child_coords) > 0:
                        child_mask_out = ~child_mask
                        if np.any(child_mask_out):
                            ax.scatter(
                                child_coords[child_mask_out, 0],
                                child_coords[child_mask_out, 1],
                                c='lightgray', s=20, alpha=0.3,
                                marker='s', edgecolors='none',
                                label='Filtered out (child)' if i == 0 else '',
                                zorder=1
                            )
            else:
                parent_coords_plot = parent_coords
                child_coords_plot = child_coords
                parent_indices_plot = parent_indices
                child_indices_plot = child_indices
            # ==============================================
            
            # ========== PLOT BOUNDARIES (LINES OR POINTS) ==========
            # Plot parent side
            if side in ['both', 'parent'] and len(parent_coords_plot) > 0:
                if n_children == 1:
                    parent_color = 'blue'
                    parent_label = f'Parent {parent_cluster_id} side ({len(parent_indices_plot)} pixels)' if show_legend else ''
                else:
                    parent_color = color
                    parent_label = f'Child {child_id}: parent side ({len(parent_indices_plot)} px)' if show_legend else ''
                
                if plot_as_lines:
                    ordered_parent = order_boundary_points(parent_coords_plot)
                    ax.plot(ordered_parent[:, 0], ordered_parent[:, 1],
                        color=parent_color, linewidth=line_width, alpha=0.8,
                        label=parent_label, zorder=5)
                else:
                    ax.scatter(
                        parent_coords_plot[:, 0], parent_coords_plot[:, 1], 
                        c=[parent_color], s=50, alpha=0.7,
                        marker='o', edgecolors='black', linewidths=0.5,
                        label=parent_label,
                        zorder=5
                    )
            
            # Plot child side
            if side in ['both', 'child'] and len(child_coords_plot) > 0:
                if n_children == 1:
                    child_color = 'red'
                    child_label = f'Child {child_id} side ({len(child_indices_plot)} pixels)' if show_legend else ''
                    child_marker = 'o'
                    child_linestyle = '-'
                else:
                    child_color = color
                    child_label = f'Child {child_id}: child side ({len(child_indices_plot)} px)' if show_legend else ''
                    child_marker = 's'
                    child_linestyle = '--'
                
                if plot_as_lines:
                    ordered_child = order_boundary_points(child_coords_plot)
                    ax.plot(ordered_child[:, 0], ordered_child[:, 1],
                        color=child_color, linewidth=line_width, alpha=0.8,
                        linestyle=child_linestyle,
                        label=child_label, zorder=5)
                else:
                    ax.scatter(
                        child_coords_plot[:, 0], child_coords_plot[:, 1],
                        c=[child_color], s=50, alpha=0.7,
                        marker=child_marker, edgecolors='black', linewidths=0.5,
                        label=child_label,
                        zorder=5
                    )
            # =======================================================
            
            # Add text labels if requested
            if show_labels and (len(parent_coords_plot) > 0 or len(child_coords_plot) > 0):
                if side == 'parent' and len(parent_coords_plot) > 0:
                    centroid_x = np.mean(parent_coords_plot[:, 0])
                    centroid_y = np.mean(parent_coords_plot[:, 1])
                elif side == 'child' and len(child_coords_plot) > 0:
                    centroid_x = np.mean(child_coords_plot[:, 0])
                    centroid_y = np.mean(child_coords_plot[:, 1])
                else:
                    all_coords = []
                    if len(parent_coords_plot) > 0:
                        all_coords.append(parent_coords_plot)
                    if len(child_coords_plot) > 0:
                        all_coords.append(child_coords_plot)
                    
                    if len(all_coords) > 0:
                        all_coords = np.vstack(all_coords)
                        centroid_x = np.mean(all_coords[:, 0])
                        centroid_y = np.mean(all_coords[:, 1])
                    else:
                        continue
                
                ax.text(centroid_x, centroid_y, f'{child_id}',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round', facecolor=color if n_children > 1 else 'red', 
                                alpha=0.8, edgecolor='black', linewidth=1.5),
                    zorder=10)
        
        # Add parent cluster representative point
        rep_point = boundary_result.rep_points[parent_idx]
        ax.plot(rep_point[0], rep_point[1], '*', 
            color='gold', markersize=25, markeredgecolor='black', markeredgewidth=2,
            label=f'Parent {parent_cluster_id} center' if show_legend else '', 
            zorder=100)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        
        # Handle legend - only if show_legend=True
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                    bbox_to_anchor=(1.05, 1), loc='upper left',
                    fontsize=9, framealpha=0.9, borderaxespad=0)
        
        ax.axis('equal')
        
        if invert_y_axis:
            ax.invert_yaxis()
        
        ax.grid(True, alpha=0.3)
        
        self.fig = fig
        self.axes = [ax] if not isinstance(self.axes, list) else self.axes
        
        return fig, ax

    def plot_parent_child_hierarchy_overview(self, boundary_result, parent_cluster_id, 
                                             parent_phase_id=None, child_phase_id=None,
                                             figsize=None, invert_y_axis=None,
                                             show_parent_same_phase_boundaries=False):
        """
        Create overview plot showing parent with all children in separate subplots.
        
        Parameters
        ----------
        boundary_result : BoundaryResult
            Boundary analysis results
        parent_cluster_id : int
            Parent cluster ID to plot
        parent_phase_id : int, optional
            Parent phase ID. If None, inferred from parent_cluster_id.
        child_phase_id : int, optional
            Child phase ID. If None, uses all child phases.
        figsize : tuple, optional
            Figure size. If None, uses (20, 5).
        invert_y_axis : bool, optional
            Whether to invert y-axis. If None, uses instance default.
        show_parent_same_phase_boundaries : bool, optional
            Whether to plot parent cluster boundaries with same-phase neighbors.
            Default is False.
        
        Returns
        -------
        fig, axes : matplotlib figure and axes array
        """
        import matplotlib.pyplot as plt
        
        if invert_y_axis is None:
            invert_y_axis = self.invert_y_axis
        
        if figsize is None:
            figsize = (24, 5)  # Increased width for legend space
        
        # Find parent cluster index and phase
        try:
            parent_idx = np.where(boundary_result.clusters == parent_cluster_id)[0][0]
        except IndexError:
            raise ValueError(f"Parent cluster {parent_cluster_id} not found!")
        
        if parent_phase_id is None:
            parent_phase_id = boundary_result.cluster_phases_id[parent_idx]
        
        # Get hierarchy
        hierarchy = boundary_result.get_parent_child_hierarchy_with_boundaries(
            parent_phase_id,
            child_phase_id if child_phase_id is not None else None
        )
        
        # Find this parent
        parent_data = None
        for p in hierarchy:
            if p['parent_cluster_id'] == parent_cluster_id:
                parent_data = p
                break
        
        if parent_data is None or parent_data['n_children'] == 0:
            raise ValueError(f"Parent cluster {parent_cluster_id} has no children")
        
        n_children = parent_data['n_children']
        
        # Create subplots: first shows all, rest show individual children
        n_plots = min(n_children + 1, 6)  # Limit to 6 total plots
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, constrained_layout=True)
        
        if n_plots == 1:
            axes = [axes]
        
        # First subplot: all children together
        self.plot_parent_child_interface(
            boundary_result=boundary_result,
            parent_cluster_id=parent_cluster_id,
            child_cluster_id=None,  # All children
            side='both',
            parent_phase_id=parent_phase_id,
            child_phase_id=child_phase_id,
            ax=axes[0],
            show_labels=True,
            invert_y_axis=invert_y_axis,
            show_parent_same_phase_boundaries=show_parent_same_phase_boundaries
        )
        axes[0].set_title(f'All {n_children} children', fontweight='bold')
        
        # Remaining subplots: individual children
        for i in range(1, min(n_plots, n_children + 1)):
            child = parent_data['children'][i - 1]
            child_id = child['child_cluster_id']
            
            self.plot_parent_child_interface(
                boundary_result=boundary_result,
                parent_cluster_id=parent_cluster_id,
                child_cluster_id=child_id,
                side='both',
                parent_phase_id=parent_phase_id,
                child_phase_id=child_phase_id,
                ax=axes[i],
                show_labels=True,
                invert_y_axis=invert_y_axis,
                show_parent_same_phase_boundaries=show_parent_same_phase_boundaries
            )
            axes[i].set_title(f'Child {child_id}', fontweight='bold')
        
        # Hide extra axes if we limited plots
        if n_plots < len(axes):
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
        
        self.fig = fig
        self.axes = axes
        
        return fig, axes
    
    def plot_enclosures_comparison(self, boundary_result, parent_cluster_id,
                                   parent_phase_id=None, child_phase_id=None,
                                   figsize=None, invert_y_axis=None,
                                   show_parent_same_phase_boundaries=False):
        """
        Create three-panel comparison: parent side, child side, and both.
        
        Parameters
        ----------
        boundary_result : BoundaryResult
            Boundary analysis results
        parent_cluster_id : int
            Parent cluster ID to plot
        parent_phase_id : int, optional
            Parent phase ID
        child_phase_id : int, optional
            Child phase ID
        figsize : tuple, optional
            Figure size. If None, uses (28, 8).
        invert_y_axis : bool, optional
            Whether to invert y-axis. If None, uses instance default.
        show_parent_same_phase_boundaries : bool, optional
            Whether to plot parent cluster boundaries with same-phase neighbors.
            Default is False.
        
        Returns
        -------
        fig, axes : matplotlib figure and axes array
        """
        import matplotlib.pyplot as plt
        
        if invert_y_axis is None:
            invert_y_axis = self.invert_y_axis
        
        if figsize is None:
            figsize = (28, 8)  # Increased width for legend space
        
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        
        sides = ['parent', 'child', 'both']
        titles = ['Parent Side Only', 'Child Side Only', 'Both Sides']
        
        for ax, side, title in zip(axes, sides, titles):
            self.plot_parent_child_interface(
                boundary_result=boundary_result,
                parent_cluster_id=parent_cluster_id,
                child_cluster_id=None,  # All children
                side=side,
                parent_phase_id=parent_phase_id,
                child_phase_id=child_phase_id,
                ax=ax,
                invert_y_axis=invert_y_axis,
                show_parent_same_phase_boundaries=show_parent_same_phase_boundaries
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        self.fig = fig
        self.axes = axes
        
        return fig, axes
    
    def plot_multiple_parents_grid(self, boundary_result, parent_phase_id, child_phase_id,
                                   max_parents=4, figsize=None, invert_y_axis=None,
                                   show_parent_same_phase_boundaries=False):
        """
        Plot interfaces for multiple parents in a grid layout.
        
        Parameters
        ----------
        boundary_result : BoundaryResult
            Boundary analysis results
        parent_phase_id : int
            Parent phase ID
        child_phase_id : int
            Child phase ID
        max_parents : int, optional
            Maximum number of parents to plot. Default is 4.
        figsize : tuple, optional
            Figure size. If None, uses (24, 20).
        invert_y_axis : bool, optional
            Whether to invert y-axis. If None, uses instance default.
        show_parent_same_phase_boundaries : bool, optional
            Whether to plot parent cluster boundaries with same-phase neighbors.
            Default is False.
        
        Returns
        -------
        fig, axes : matplotlib figure and axes array
        """
        import matplotlib.pyplot as plt
        
        if invert_y_axis is None:
            invert_y_axis = self.invert_y_axis
        
        if figsize is None:
            figsize = (24, 20)  # Increased width for legend space
        
        # Get hierarchy
        hierarchy = boundary_result.get_parent_child_hierarchy_with_boundaries(
            parent_phase_id, child_phase_id
        )
        
        # Limit number of parents
        n_parents = min(len(hierarchy), max_parents)
        
        # Determine grid size
        n_cols = 2
        n_rows = (n_parents + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                                constrained_layout=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes = axes.flatten()
        
        for i, parent in enumerate(hierarchy[:n_parents]):
            parent_id = parent['parent_cluster_id']
            
            self.plot_parent_child_interface(
                boundary_result=boundary_result,
                parent_cluster_id=parent_id,
                child_cluster_id=None,  # All children
                side='both',
                parent_phase_id=parent_phase_id,
                child_phase_id=child_phase_id,
                ax=axes[i],
                invert_y_axis=invert_y_axis,
                show_parent_same_phase_boundaries=show_parent_same_phase_boundaries
            )
        
        # Hide unused axes
        for i in range(n_parents, len(axes)):
            axes[i].set_visible(False)
        
        self.fig = fig
        self.axes = axes
        
        return fig, axes
        
    def _get_phase_colors(self, phases_id):
        """Generate colors for phases."""
        import matplotlib.cm as cm
        unique_phases_id = np.unique(phases_id)
        cmap = cm.get_cmap('Set1', len(unique_phases_id))
        
        colors = np.zeros((len(phases_id), 4))
        for i, phase_id in enumerate(unique_phases_id):
            colors[phases_id == phase_id] = cmap(i)
        
        return colors
    
    
    def _get_cluster_colors(self, labels):
        """Generate colors for clusters."""
        import matplotlib.cm as cm
        unique_labels = np.unique(labels[labels > 0])
        n_clusters = len(unique_labels)
        cmap = cm.get_cmap('tab20', n_clusters)
        
        colors = np.zeros((len(labels), 4))
        for i, label in enumerate(unique_labels):
            colors[labels == label] = cmap(i)
        
        return colors
    
    #def _get_phase_colors(self, phases):
    #    """Generate colors for phases."""
    #    import matplotlib.cm as cm
    #    unique_phases = np.unique(phases)
    #    cmap = cm.get_cmap('Set1', len(unique_phases))
    #    
    #    colors = np.zeros((len(phases), 4))
    #    for i, phase in enumerate(unique_phases):
    #        colors[phases == phase] = cmap(i)
    #    
    #    return colors


class EbsdHPAnalyzer():
    def __init__(self, hierarchy, clustering_result: ClusteringResult, boundary_result: BoundaryResult,
                 maxmillerindex=4, maxdevfrom90deg=4, maxnormaldev=2, maxdirdev=2):
            self.hierarchy = hierarchy
            self.clustering_result = clustering_result
            self.boundary_result = boundary_result
            self.data = clustering_result.data
            self.maxdevfrom90deg = maxdevfrom90deg
            self.maxnormaldev = maxnormaldev
            self.maxdirdev = maxdirdev
            self.maxmillerindex=maxmillerindex
            self.maxdevfrom90deg=maxdevfrom90deg
            for hi in self.hierarchy:
                if len([ch['is_lamellar'] for ch in hi['children'] if ch['is_lamellar']])>0:
                    parent_phase_id = hi['parent_phase_id']
                    #print(f"Parent cluster {hi['parent_cluster_id']} has following lamellar children:")
                    if 'children' in hi:
                        for ch in hi['children']:
                            if ch['is_lamellar']:
                                child_phase_id = ch['child_phase_id']
                                break
            #try:
            if True:
                print(f"Setting HP analyzer phases: parent {parent_phase_id}, child {child_phase_id}")
                self.keyau = self.clustering_result.data.phase_names[parent_phase_id]
                self.keyma = self.clustering_result.data.phase_names[child_phase_id]
                self.setPhases()
                self.austenite = self.clustering_result.data.austenite
                self.martensite = self.clustering_result.data.martensite
                self.PhaseNames = {self.austenite:'Austenite',self.martensite:'Martensite'}
                self.PhaseCols = {self.austenite:['r','m'],self.martensite:['b','c']}
                self.Symmetry = {self.austenite:self.clustering_result.data.phases[self.keyau]['symmetry'],self.martensite:self.clustering_result.data.phases[self.keyma]['symmetry']}
                self.T_AM = self.clustering_result.data.defGrad['NiTi']['A']['T_AM']
                self.F_AM = self.clustering_result.data.defGrad['NiTi']['A']['F_AM']
                self.CP = {self.austenite:self.clustering_result.data.OR['NiTi']['CIp'],self.martensite: self.clustering_result.data.OR['NiTi']['Cp']}
                self.CD = {self.austenite:self.clustering_result.data.OR['NiTi']['CId'],self.martensite: self.clustering_result.data.OR['NiTi']['Cd']}

                self.CrystalRef2Spatial=np.array([[0,1,0],[1,0,0],[0,0,1]])
                self.phase4HPguess = 'A'
                self.scan = self.clustering_result.data._ebsdData
                self.Sels=[]
                self.PhaseKeys=[]
                self.G2Cryst={}
                self.G2Sampl={}
                self.dirkey={}
                for key in self.Phases.keys():
                    self.PhaseKeys.append(key)
                    self.PhaseKeys
                    self.G2Cryst[key] = Orientation.from_matrix(self.clustering_result.data._ebsdData.M,symmetry=self.Symmetry[key])
                    self.G2Sampl[key] = np.transpose(self.G2Cryst[key], axes=(0,2,1))
                    self.dirkey[key] = plot.DirectionColorKeyTSL(self.Symmetry[key])
                self.setAttributes( phase4HPguess=self.austenite) 
                self.setAttributes(PhaseNames={self.austenite:'Austenite',self.martensite:'Martensite'})
                self.setAttributes(PhaseCols={self.austenite:['r','m'],self.martensite:['b','c']})
                self.setHP()
                self.hklsDict = {}
                for key in [self.keyau,self.keyma]:
                    self.hklsDict[key]={}
                    if self.phase4HPguess!=key:
                        fac=3.
                    else:
                        fac=1.
                    hkls, hkls2, fam = generate_hkls(int(fac*self.maxmillerindex),self.clustering_result.data.phases[key]['symops'])
                    for key2 in hkls2.keys():
                        for hkl in hkls2[key2]:
                            self.hklsDict[key][hkl]=key2


            #except:
            #    print("Failed to setup HP analyzer")
    def setAttributes(self,**kwargs):    
        """
        Set attributes for the EBSD analyzer.
        
        Args:
            **kwargs: Key-value pairs of attributes to set
        """
        self.__dict__.update(kwargs)
    
    def remap_hp_ids(self, id_map):
        """
        Update self.HP keys after merge_physical_grains() relabelling.
        Call once after each merge_physical_grains() call if singleHP
        was already run before the merge.

        Parameters
        ----------
        id_map : dict {old_id: new_id}
            Output of clustering_result.merge_physical_grains().
        """
        if not id_map:
            return
        new_HP = {}
        for pid, children in self.HP.items():
            new_pid = id_map.get(pid, pid)
            new_children = {}
            for cid, data in children.items():
                # Remap child cluster IDs but preserve meta keys
                # (parent_idx, child_idx are integers too — but they
                # are values not cluster IDs, so only remap actual
                # child cluster IDs that appear in id_map)
                if isinstance(cid, (int, np.integer)) and cid in id_map:
                    new_cid = id_map[cid]
                else:
                    new_cid = cid
                new_children[new_cid] = data
            new_HP[new_pid] = new_children
        self.HP = new_HP
    
    
    def listLamellarParents(self):
        for hi in self.hierarchy:
            if len([ch['is_lamellar'] for ch in hi['children'] if ch['is_lamellar']])>0:
                print(f"Parent cluster {hi['parent_cluster_id']} has following lamellar children:")
                if 'children' in hi:
                    for ch in hi['children']:
                        if ch['is_lamellar']:
                            print(f"{ch['child_cluster_id']}",end=', ')
                print() # newline for readability
            else:
                print(f"Parent cluster {hi['parent_cluster_id']} has no lamellar children:")

    def setHP(self):
        self.HP={}
        for hidx, hi in enumerate(self.hierarchy):
            if len([ch['is_lamellar'] for ch in hi['children'] if ch['is_lamellar']])>0:
                self.HP[hi['parent_cluster_id']]={}
                #print(f"Parent cluster {hi['parent_cluster_id']} has following lamellar children:")
                if 'children' in hi:
                    for chidx, ch in enumerate(hi['children']):
                        if ch['is_lamellar']:
                            self.HP[hi['parent_cluster_id']][ch['child_cluster_id']] = {}
                            self.HP[hi['parent_cluster_id']][ch['child_cluster_id']]['parent_idx']  = hidx
                            self.HP[hi['parent_cluster_id']][ch['child_cluster_id']]['child_idx']  = chidx
                            #print(f"{ch['child_cluster_id']}",end=', ')
                #print() # newline for readability


    def setPhases(self):
        """
        Set phase definitions for analysis.
        
        Args:
            phases: Phase definitions
        """
        self.Phases = self.clustering_result.data.phase_ids

    def singleHP(self, roi, parent_id, child_id,printout = False, nodirs=False, bestscore=False):  
        parent_idx = self.HP[parent_id][child_id]['parent_idx']
        child_idx = self.HP[parent_id][child_id]['child_idx']       
        if not self.hierarchy[parent_idx]['children'][child_idx][roi]['no_parent_side']:
            self.getOri(roi, parent_id, child_id)
            self.getInterface(roi,parent_id, child_id)
            self.getHBmatches(roi,parent_id, child_id)
            if printout:
                self.printHBmatches(roi,parent_id, child_id, nodirs=nodirs, bestscore=bestscore)
        else:
            print(f"No interface between parent_id {parent_id} and child_id {child_id} within roi {roi}")
    def getOri(self, roi, parent_id, child_id):
        self.Ln = {}
        for key in self.Phases.keys():
            self.Ln[key] = np.array([la/np.linalg.norm(la) for la in self.clustering_result.data.phases[key]['L'].T]).T
        parent_idx = self.HP[parent_id][child_id]['parent_idx']
        child_idx = self.HP[parent_id][child_id]['child_idx']       
        #for roi in ['semi_roi_1','semi_roi_2']:
        titles = 'G2Cryst_red G2Cryst_red_avg G2Sampl_red G2Sampl_red_avg <100>PF <100>PF_avg'
        self.HP[parent_id][child_id][roi] = {}
        for title in titles.split():
            self.HP[parent_id][child_id][roi][title] = {}
        for key in self.Phases.keys():
            if key == self.austenite:
                keyname = 'parent_side'
            else:
                keyname = 'child_side'
            self.HP[parent_id][child_id][roi]['G2Cryst_red'][key] = self.hierarchy[parent_idx]['children'][child_idx][roi][keyname]['M_best_sym']
            self.HP[parent_id][child_id][roi]['G2Sampl_red'][key]=np.transpose(self.HP[parent_id][child_id][roi]['G2Cryst_red'][key],axes=(0,2,1))
            self.HP[parent_id][child_id][roi]['G2Cryst_red_avg'][key]=self.hierarchy[parent_idx]['children'][child_idx][roi][keyname]['M_mean']
            self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][key]=self.HP[parent_id][child_id][roi]['G2Cryst_red_avg'][key].T
            

            pfi=[]
            for di in self.Ln[key].T:
                pfi.append(orilistMult(self.HP[parent_id][child_id][roi]['G2Sampl_red'][key],di))            
            self.HP[parent_id][child_id][roi]['<100>PF'][key] = np.hstack((pfi[0],pfi[1],pfi[2]))
            self.HP[parent_id][child_id][roi]['<100>PF_avg'][key] = self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][key].dot(self.Ln[key])

        self.getClosestOR(roi, parent_id, child_id)
        
        ORi=self.HP[parent_id][child_id][roi]['G2Cryst_red_avg'][self.keyma].dot(self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][self.keyau])
        axis_angle = Rotation.from_matrix(ORi).to_axes_angles()
        self.HP[parent_id][child_id][roi]['OR'] = { f'OR:{self.keyau}-{self.keyma}':ORi,'axis':np.array([axis_angle.axis.x[0],axis_angle.axis.y[0],axis_angle.axis.z[0]]),'angle':np.rad2deg(axis_angle.angle)[0]}

    def getClosestOR(self, roi, parent_id, child_id):
        """
        Find martensite variant with orientation relationship closest to EBSD data.
        
        Args:
            seli: Selection index
            
        Compares experimental OR with theoretical variants to find best match.
        """
    
        self.G2Cryst_red_ma_avg_alleq = np.tensordot(self.clustering_result.data.phases[self.keyma]['symops'], [self.HP[parent_id][child_id][roi]['G2Cryst_red_avg'][self.keyma]], axes=[[-1], [-2]]).transpose([2, 0, 1, 3])[0,:,:,:]
        self.HP[parent_id][child_id][roi]['T_AM_ebsd'] = np.tensordot(self.G2Cryst_red_ma_avg_alleq, [self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][self.keyau]], axes=[[-1], [-2]]).transpose([2, 0, 1, 3])[0,:,:,:]
        T_AM_T = np.array(self.T_AM).T
        D = np.tensordot(self.HP[parent_id][child_id][roi]['T_AM_ebsd'],T_AM_T, axes=[[-1], [-2]]).transpose([2, 0, 1, 3])
        tr = np.trace(D, axis1=2, axis2=3)
        neg = tr < -1.0
        tr[neg] = -tr[neg]
        idxs=np.unravel_index(np.argmax(tr),tr.shape)
        Cmp = self.compareEbsdTheory(roi,parent_id,child_id,idxs[0],idxs[1])
        self.HP[parent_id][child_id][roi]['Closest Variant'] = idxs[0]
        for key in Cmp.keys():
            self.HP[parent_id][child_id][roi][key] = copy.deepcopy(Cmp[key])
            self.HP[parent_id][child_id][roi][key+'_allvars'] = [] 
        for vari in range(self.T_AM.shape[2]):
            D = np.tensordot(self.HP[parent_id][child_id][roi]['T_AM_ebsd'],T_AM_T[vari:vari+1,:,:], axes=[[-1], [-2]]).transpose([2, 0, 1, 3])
            tr = np.trace(D, axis1=2, axis2=3)
            neg = tr < -1.0
            tr[neg] = -tr[neg]
            idxs=np.unravel_index(np.argmax(tr),tr.shape)
            Cmp = self.compareEbsdTheory(roi,parent_id,child_id,vari,idxs[1])
            for key in Cmp.keys():
                self.HP[parent_id][child_id][roi][key+'_allvars'].append(copy.deepcopy(Cmp[key]))

    def compareEbsdTheory(self, roi, parent_id, child_id, varIdx, equivalIdx):
        """
        Compare experimental EBSD data with theoretical predictions.
        
        Args:
            roi: Region of interest index
            parent_id: Parent cluster ID
            child_id: Child cluster ID
            varIdx: Variant index
            equivalIdx: Equivalent orientation index
            
        Returns:
            dict: Comparison results including misorientation and strains
        """
        Cmp={}
        titles = ['Closes equivalent G2Cryst','Theory G2Cryst','Theory <100>PF_avg','Theory <100>PF_avg_symop']
        for title in titles:
            Cmp[title]={}
            
        Cmp['Martensite symop'] = self.clustering_result.data.phases[self.keyma]['symops'][equivalIdx]
        Cmp['Closes equivalent G2Cryst'][self.keyma]=self.G2Cryst_red_ma_avg_alleq[equivalIdx,:,:]
        Cmp['Closest Theory OR'] = self.T_AM[:,:,varIdx]
        Cmp['Exp OR'] = self.HP[parent_id][child_id][roi]['T_AM_ebsd'][equivalIdx,:,:]
        Cmp['Theory G2Cryst'][self.keyma] = self.T_AM[:,:,varIdx].dot(self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][self.keyau])
        Cmp['Theory <100>PF_avg'][self.keyma] = Cmp['Theory G2Cryst'][self.keyma].T.dot(self.Ln[self.keyma])
        Cmp['Theory <100>PF_avg_symop'][self.keyma] = Cmp['Theory G2Cryst'][self.keyma].T.dot(Cmp['Martensite symop'].dot(self.Ln[self.keyma]))
        Cmp['dR OR'] = Rotation.from_matrix(Cmp['Theory G2Cryst'][self.keyma].dot(Cmp['Closes equivalent G2Cryst'][self.keyma].T))
        Cmp['Misori OR'] = np.rad2deg(Cmp['dR OR'].to_axes_angles().angle)[0]
        Cmp['Transformation strain']=[]
        for ldiri in np.eye(3):
            ldir=self.HP[parent_id][child_id][roi]['G2Cryst_red_avg'][self.keyau].dot(ldiri)
            Cmp['Transformation strain'].append({f'along {ldiri} [-]':np.sqrt(ldir.dot(self.F_AM[:,:,varIdx].T.dot(self.F_AM[:,:,varIdx].dot(ldir))))-1})
        return Cmp
    def getHBmatches(self,roi, parent_id, child_id):
        """
        Find matching habit plane candidates between phases.
        
        Args:
            roi: Region of interest index
            parent_id: Parent cluster ID
            child_id: Child cluster ID
            
        Matches crystallographic planes and directions that are consistent
        with observed interface traces in both phases.
        """
              
        HP_guess={}
        for phase in self.Phases.keys():
            interface_trace=self.HP[parent_id][child_id][roi]['Interfaces']['interface_trace'][phase]
            interfacenorm_trace=self.HP[parent_id][child_id][roi]['Interfaces']['interfacenorm_trace'][phase]
            G2Sampl = self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][phase]
            LrI = self.clustering_result.data.phases[phase]['LrI']
            Lr = self.clustering_result.data.phases[phase]['Lr']
            LI = self.clustering_result.data.phases[phase]['LI']
            L = self.clustering_result.data.phases[phase]['L']
            
            HP_guess[phase] = self.getNormals(interface_trace, interfacenorm_trace,LrI,Lr,G2Sampl,self.hklsDict[phase])
            idxs = np.argsort(HP_guess[phase]['HPvsTrace_angle'])[::-1]
            for key in HP_guess[phase].keys():
                HP_guess[phase][key]=HP_guess[phase][key][idxs]
        self.HP[parent_id][child_id][roi]['Interfaces']['HP_guess']=HP_guess
        
        HP_matches = {}
        HP_matches['Score']={}
        for title in ['low index habit plane', 'mean misalignment', 'number of corresponding directions','overall', 'fitcorresp']:
            HP_matches['Score'][title] = []
        for phase in self.Phases.keys():
            HP_matches[phase]={}
            HP_matches[phase]['Habit plane normal']={}
            HP_matches[phase]['Habit plane normal']['misalign']=[]
            HP_matches[phase]['Habit plane direction']={}
            HP_matches[phase]['Habit plane direction']['misalign']=[]
            for key in HP_matches[phase].keys():
                for key2 in HP_guess[self.keyau].keys():
                    HP_matches[phase][key][key2]=[]
        for n1,n_vec in enumerate(HP_guess[self.keyau]['n_miller_normvec_sampl']):
            for n2,n_vec2 in enumerate(HP_guess[self.keyma]['n_miller_normvec_sampl']):
                hb_misalign = np.arccos(abs(n_vec.dot(n_vec2)))*180/np.pi
                if hb_misalign <= self.maxnormaldev:
                    DP_guess={}
                    for phase,n in zip(self.Phases.keys(),[n1,n2]):
                        interface_trace=HP_guess[phase]['n_miller_normvec'][n]
                        interfacenorm_trace=perpendicular_vector(interface_trace)
                        G2Sampl = self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][phase]
                        LI = self.clustering_result.data.phases[phase]['LI']
                        L = self.clustering_result.data.phases[phase]['L']                
                        DP_guess[phase] = self.getNormals(interface_trace, interfacenorm_trace,LI,L,G2Sampl,self.hklsDict[phase],maxdevfrom90deg=0.)
                    v1=DP_guess[self.keyau]
                    v2=DP_guess[self.keyma]
                    dpmatch=[]
                    for d1,d_vec in enumerate(v1['n_miller_normvec_sampl']):
                        for d2,d_vec2 in enumerate(v2['n_miller_normvec_sampl']):
                            dp_misalign = np.arccos(abs(d_vec.dot(d_vec2)))*180/np.pi
                            if dp_misalign <= self.maxdirdev:
                                isin=False
                                for isinidx,idxs in enumerate(dpmatch):
                                    if idxs[0]==d1:
                                        isin=True
                                        break
                                if not isin:
                                    dpmatch.append((d1,d2,dp_misalign))
                                else:
                                    if dp_misalign<idxs[2]:
                                        dpmatch[isinidx]=(d1,d2,dp_misalign)
                    if len(dpmatch)>0:
                        for key in HP_guess[phase]:
                            HP_matches[self.keyau]['Habit plane normal'][key].append(HP_guess[self.keyau][key][n1])
                            HP_matches[self.keyma]['Habit plane normal'][key].append(HP_guess[self.keyma][key][n2])
                            HP_matches[self.keyau]['Habit plane direction'][key].append(v1[key][[idxs[0] for idxs in dpmatch]])
                            HP_matches[self.keyma]['Habit plane direction'][key].append(v2[key][[idxs[1] for idxs in dpmatch]])
                        
                        fitcorresp=True
                        vari=self.HP[parent_id][child_id][roi]['Closest Variant']
                        for idxs in dpmatch:
                            dp_ma = v2['n_miller'][idxs[1]]
                            dp_au = v1['n_miller'][idxs[0]]
                            dp_ma2 = vector2miller(self.HP[parent_id][child_id][roi]['Martensite symop'].dot(self.CD[self.keyau][:,:,vari].dot(dp_au)))
                            if not ((dp_ma2==dp_ma).all() or (-1*dp_ma2==dp_ma).all()):                                      
                                fitcorresp = False
                        hb_au = HP_guess[self.keyau]['n_miller'][n1]
                        hb_ma = HP_guess[self.keyma]['n_miller'][n2]
                        hb_ma2 = vector2miller(self.HP[parent_id][child_id][roi]['Martensite symop'].dot(self.CP[self.keyau][:,:,vari].dot(hb_au)))
                        if not ((hb_ma2==hb_ma).all() or (-1*hb_ma2==hb_ma).all()):
                            fitcorresp = False
                            
                        HP_matches[self.keyma]['Habit plane normal']['misalign'].append(hb_misalign)
                        HP_matches[self.keyma]['Habit plane direction']['misalign'].append([idxs[2] for idxs in dpmatch])
                        HP_matches[self.keyau]['Habit plane normal']['misalign'].append(hb_misalign)
                        HP_matches[self.keyau]['Habit plane direction']['misalign'].append([idxs[2] for idxs in dpmatch])
                        HP_matches['Score']['low index habit plane'].append(np.sum(np.abs(HP_guess[self.keyau]['n_miller'][n1]))+np.sum(np.abs(HP_guess[self.keyma]['n_miller'][n2])))
                        allmisalign = [abs(idxs[2]) for idxs in dpmatch]
                        allmisalign.append(abs(HP_guess[self.keyau]['HPvsTrace_angle'][n1]-90))
                        allmisalign.append(abs(HP_guess[self.keyma]['HPvsTrace_angle'][n2]-90))
                        allmisalign.append(hb_misalign)
                        HP_matches['Score']['mean misalignment'].append(np.mean(allmisalign))
                        HP_matches['Score']['number of corresponding directions'].append(len(dpmatch))
                        HP_matches['Score']['overall'].append(HP_matches['Score']['low index habit plane'][-1]/HP_matches['Score']['number of corresponding directions'][-1]+HP_matches['Score']['mean misalignment'][-1])
                        if fitcorresp:
                            HP_matches['Score']['fitcorresp'].append('yes')
                        else:
                            HP_matches['Score']['fitcorresp'].append('no')
        if True:
            idxs = np.argsort(HP_matches['Score']['overall'])
            for phase in self.Phases.keys():
                for key1 in HP_matches[phase].keys():
                    for key in HP_matches[phase][key1].keys():
                        HP_matches[phase][key1][key]=[HP_matches[phase][key1][key][idx] for idx in idxs]
            for key in HP_matches['Score'].keys():
                HP_matches['Score'][key]=[HP_matches['Score'][key][idx] for idx in idxs]            
        self.HP[parent_id][child_id][roi]['Interfaces']['HP_matches'] = HP_matches    
    def getNormals(self,interface_trace, interfacenorm_trace, LrI, Lr, G2Sampl, hklsDict, angles = np.linspace(0,180,361),maxdevfrom90deg=None, maxmillerindex=None):
        """
        Generate candidate plane normals perpendicular to interface trace.
        
        Args:
            interface_trace: Interface trace direction
            interfacenorm_trace: Normal to interface trace
            LrI: Inverse reciprocal lattice matrix
            Lr: Reciprocal lattice matrix
            G2Sampl: Crystal to sample transformation
            angles: Rotation angles to test
            maxdevfrom90deg: Maximum deviation from 90 degrees
            maxmillerindex: Maximum Miller index
            
        Returns:
            dict: Candidate plane normals meeting criteria
        """
        N_guess={}
        titles = 'n_vec n_vec_sampl n_miller n_miller_family n_miller_normvec n_miller_normvec_sampl HPvsTrace_angle'.split()
        for title in titles:
            N_guess[f'{title}'] =[]

        if maxdevfrom90deg is None:
            maxdevfrom90deg=self.maxdevfrom90deg
            
        if maxmillerindex is None:
            maxmillerindex=self.maxmillerindex
        for an in angles:
            var = {}
            var['n_vec'] = Rotation.from_axes_angles(interface_trace, an, degrees=True).to_matrix().dot(interfacenorm_trace)[0,:]
            var['n_vec_sampl'] = G2Sampl.dot(var['n_vec'])
            var['n_miller'] = np.round(vector2millerround(LrI.dot(var['n_vec'])))           
            if np.abs(var['n_miller']).max()>self.maxmillerindex:
                var['n_miller'] = np.round(vector2millerround(LrI.dot(var['n_vec']),MIN=False))
            var['n_miller_normvec'] = Lr.dot(var['n_miller'])
            var['n_miller_normvec'] /= np.linalg.norm(var['n_miller_normvec'])
            var['n_miller_normvec_sampl'] = G2Sampl.dot(var['n_miller_normvec'])
            var['HPvsTrace_angle'] = np.arccos(abs(var['n_miller_normvec'].dot(interface_trace)))*180/np.pi
            if np.abs(var['n_miller']).max()<=maxmillerindex and abs(var['HPvsTrace_angle']-90)<=maxdevfrom90deg:
                var['n_miller_family'] = hklsDict[tuple(var['n_miller'])]
                isin=False
                for nm in N_guess['n_miller']:
                    if (nm==var['n_miller']).all() or (nm==-1*var['n_miller']).all():
                        isin=True
                if not isin:
                    for title in titles:
                        N_guess[f'{title}'].append(var[f'{title}'])

        for key in N_guess.keys():
            N_guess[key]=np.array(N_guess[key])
        return N_guess

    def getInterface(self,roi, parent_id, child_id):
        parent_idx = self.HP[parent_id][child_id]['parent_idx']
        child_idx = self.HP[parent_id][child_id]['child_idx']       

        Xv=np.hstack((self.hierarchy[parent_idx]['children'][child_idx][roi]['parent_side']['X'], self.hierarchy[parent_idx]['children'][child_idx][roi]['child_side']['X']))
        Yv=np.hstack((self.hierarchy[parent_idx]['children'][child_idx][roi]['parent_side']['Y'], self.hierarchy[parent_idx]['children'][child_idx][roi]['child_side']['Y']))
        
        linfit = np.polyfit(Xv, Yv, 1)
        my=Yv.mean()
        mx=Xv.mean()
        interface_trace = np.array([1,linfit[0],0])
        interface_trace/=np.sqrt(interface_trace.dot(interface_trace))
        interfacenorm_trace = np.array([1,-1/linfit[0],0])
        interfacenorm_trace/=np.sqrt(interfacenorm_trace.dot(interfacenorm_trace))
        self.HP[parent_id][child_id][roi]['Interfaces']={'x':Xv,'y':Yv,
                                                    'linfit':linfit,'linfitnorm':np.array([-1/linfit[0],my+mx/linfit[0]]),
                                                    'interface_trace_sample':interface_trace,'interfacenorm_trace_sample':interfacenorm_trace}
        self.HP[parent_id][child_id][roi]['Interfaces']['interface_trace']={}
        self.HP[parent_id][child_id][roi]['Interfaces']['interfacenorm_trace']={}
        for key in self.Phases.keys():
            self.HP[parent_id][child_id][roi]['Interfaces']['interface_trace'][key]=self.HP[parent_id][child_id][roi]['G2Cryst_red_avg'][key].dot(self.CrystalRef2Spatial.dot(interface_trace))
            self.HP[parent_id][child_id][roi]['Interfaces']['interfacenorm_trace'][key]=self.HP[parent_id][child_id][roi]['G2Cryst_red_avg'][key].dot(self.CrystalRef2Spatial.dot(interfacenorm_trace))

    def printHBmatches(self,roi, parent_id, child_id, nodirs=False, bestscore=False):
        """
        Print habit plane matching results.
        
        Args:
            roi: Region of interest index
            parent_id: Parent cluster ID
            child_id: Child cluster ID
            nodirs (bool): Skip direction information
            bestscore (bool): Only show best match
        """
        print('---------------------------------------------------------')
        print(f'Interface between parent cluster {parent_id} and child cluster {child_id} within ROI {roi}:')
        print('---------------------------------------------------------')
        HP_matches = self.HP[parent_id][child_id][roi]['Interfaces']['HP_matches']
        if not bestscore:
            idxs=range(len(HP_matches[self.keyau]['Habit plane normal']['n_vec']))
        else:
            idxs=[0]
        for idx in idxs:
            key = 'Habit plane normal'
            an = HP_matches[self.keyau][key]['HPvsTrace_angle'][idx]
            an2 = HP_matches[self.keyma][key]['HPvsTrace_angle'][idx]
            n_miller = HP_matches[self.keyau][key]['n_miller'][idx]
            n_miller_fam = HP_matches[self.keyau][key]['n_miller_family'][idx]
            n_miller2 = HP_matches[self.keyma][key]['n_miller'][idx]
            n_miller2_fam = HP_matches[self.keyma][key]['n_miller_family'][idx]
            vdv = HP_matches[self.keyma][key]['misalign'][idx]
            CV = self.HP[parent_id][child_id][roi]['Closest Variant']
            TR = np.round(self.HP[parent_id][child_id][roi]['Transformation strain_allvars'][CV][0]['along [1. 0. 0.] [-]'],decimals=4)
            print(f"Fitting Closest LCV {CV}:{HP_matches['Score']['fitcorresp'][idx]}, Score:{np.round(HP_matches['Score']['overall'][idx],decimals=2)},  mean misalignment:{np.round(HP_matches['Score']['mean misalignment'][idx],decimals=2)}")
            print(f"Transformation strain along sample [100] from the closest LCV {CV}: {TR}")
            print(f"Normals: misalignment:{np.round(vdv,decimals=2)}, {n_miller}_A|{{{n_miller_fam}}}_A ({np.round(an,decimals=2)})/{n_miller2}_M|{{{n_miller2_fam}}}_M ({np.round(an2,decimals=2)})")
            if not nodirs:
                for idd in range(len(HP_matches[self.keyau]['Habit plane direction']['n_miller'][idx])):
                    key = 'Habit plane direction'
                    an = HP_matches[self.keyau][key]['HPvsTrace_angle'][idx][idd]
                    an2 = HP_matches[self.keyma][key]['HPvsTrace_angle'][idx][idd]
                    n_miller = HP_matches[self.keyau][key]['n_miller'][idx][idd]
                    n_miller_fam = HP_matches[self.keyau][key]['n_miller_family'][idx][idd]
                    n_miller2 = HP_matches[self.keyma][key]['n_miller'][idx][idd]
                    n_miller2_fam = HP_matches[self.keyma][key]['n_miller_family'][idx][idd]
                    vdv = HP_matches[self.keyma][key]['misalign'][idx][idd]
                    print(f"Directions: misalignment:{np.round(vdv,decimals=2)},{n_miller}_A|<{n_miller_fam}>_A ({np.round(an,decimals=2)})/{n_miller2}_M|<{n_miller2_fam}>_M ({np.round(an2,decimals=2)})")
            print("====================================================================================")

    def getHB(self,roi, parent_id, child_id,VarIdx=None,name='NiTi'):
        """
        Determine habit plane candidates for interfaces.
        
        Args:
            roi: Region of interest index
            parent_id: Parent cluster ID
            child_id: Child cluster ID
            VarIdx: Variant index
            name (str): Name identifier
            
        Finds crystallographic planes that could correspond to observed
        interface traces in both phases.
        """
        interface_trace=self.HP[parent_id][child_id][roi]['Interfaces']['interface_trace'][self.phase4HPguess]
        interfacenorm_trace=self.HP[parent_id][child_id][roi]['Interfaces']['interfacenorm_trace'][self.phase4HPguess]
        G2Sampl = self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][self.phase4HPguess]
        if self.phase4HPguess==self.keyau:
            interface_trace2=self.HP[parent_id][child_id][roi]['Interfaces']['interface_trace'][self.keyma]
            interfacenorm_trace2=self.HP[parent_id][child_id][roi]['Interfaces']['interfacenorm_trace'][self.keyma]
            G2Sampl2 = self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][self.keyma]
        else:
            interface_trace2=self.HP[parent_id][child_id][roi]['Interfaces']['interface_trace'][self.keyau]
            interfacenorm_trace2=self.HP[parent_id][child_id][roi]['Interfaces']['interfacenorm_trace'][self.keyau]
            G2Sampl2 = self.HP[parent_id][child_id][roi]['G2Sampl_red_avg'][self.keyau]
            
        if VarIdx is None:
            VarIdx = self.HP[parent_id][child_id][roi]['Closest Variant']
        if self.phase4HPguess == self.keyau:
            self.phase2ndHPguess = self.keyma
            Lr2 = self.clustering_result.data.phases[self.keyma]['Lr'] 
            L2 = self.clustering_result.data.phases[self.keyma]['L'] 
        else:
            self.phase2ndHPguess = self.keyau
            Lr2 = self.clustering_result.data.phases[self.keyau]['Lr'] 
            L2 = self.clustering_result.data.phases[self.keyau]['L'] 
            
        #get guesses in one phase
        HP_guess = self.getNormals(interface_trace, interfacenorm_trace,
                                    self.clustering_result.data.phases[self.phase4HPguess]['LrI'],self.clustering_result.data.phases[self.phase4HPguess]['Lr'],G2Sampl,self.hklsDict[self.phase4HPguess])
        #get corresponding guesses in the other phase for the variant providing closest math with the experimental OR
        HP_guess2ndphase, HP_guess2ndphase_allvars = self.getCorrespNormals(HP_guess, interface_trace2, 
                                                    self.CP[self.phase4HPguess][:,:,self.HP[parent_id][child_id][roi]['Closest Variant']],Lr2,LCall=self.CP[self.phase4HPguess])

        sortidxs = np.argsort(HP_guess['HPvsTrace_angle']+np.array(HP_guess2ndphase['HPvsTrace_angle']))[::-1]

        for key in HP_guess.keys():
            HP_guess[key] = (HP_guess[key])[sortidxs]
        for key in HP_guess2ndphase.keys():
            HP_guess2ndphase[key] = np.array(HP_guess2ndphase[key])[sortidxs]
        for key in HP_guess2ndphase_allvars.keys():
            HP_guess2ndphase_allvars[key] = np.array(HP_guess2ndphase_allvars[key])[sortidxs]
            
        self.HP[parent_id][child_id][roi]['Interfaces']['HP_guess']={}
        self.HP[parent_id][child_id][roi]['Interfaces']['HP_guess'][self.phase4HPguess]=HP_guess
        self.HP[parent_id][child_id][roi]['Interfaces']['HP_guess'][self.phase2ndHPguess]={}
        self.HP[parent_id][child_id][roi]['Interfaces']['HP_guess'][self.phase2ndHPguess]['Closest Variant'] = HP_guess2ndphase
        self.HP[parent_id][child_id][roi]['Interfaces']['HP_guess'][self.phase2ndHPguess]['allvars'] = HP_guess2ndphase_allvars

        DP_guess = self.getNormals(interfacenorm_trace,interface_trace,
                                    self.clustering_result.data.phases[self.phase4HPguess]['LI'], self.clustering_result.data.phases[self.phase4HPguess]['L'],G2Sampl,self.hklsDict[self.phase4HPguess])
        
        DP_guess2ndphase, DP_guess2ndphase_allvars = self.getCorrespNormals(DP_guess, interfacenorm_trace2, 
                                                    self.CD[self.phase4HPguess][:,:,self.HP[parent_id][child_id][roi]['Closest Variant']],L2,LCall=self.CD[self.phase4HPguess],hklsDict=self.hklsDict[self.phase2ndHPguess])

        sortidxs = np.argsort(DP_guess['HPvsTrace_angle']+np.array(DP_guess2ndphase['HPvsTrace_angle']))[::-1]

        for key in DP_guess.keys():
            DP_guess[key] = (DP_guess[key])[sortidxs]
        for key in DP_guess2ndphase.keys():
            DP_guess2ndphase[key] = np.array(DP_guess2ndphase[key])[sortidxs]
        for key in DP_guess2ndphase_allvars.keys():
            DP_guess2ndphase_allvars[key] = np.array(DP_guess2ndphase_allvars[key])[sortidxs]
            
        self.HP[parent_id][child_id][roi]['Interfaces']['DP_guess']={}
        self.HP[parent_id][child_id][roi]['Interfaces'][self.phase4HPguess]=DP_guess
        self.HP[parent_id][child_id][roi]['Interfaces'][self.phase2ndHPguess]={}
        self.HP[parent_id][child_id][roi]['Interfaces'][self.phase2ndHPguess]['Closest Variant'] = DP_guess2ndphase
        self.HP[parent_id][child_id][roi]['Interfaces'][self.phase2ndHPguess]['allvars'] = DP_guess2ndphase_allvars
    def getCorrespNormals(self,N_guess, interface_trace2, LC, Lr2, hklsDict, LCall=None):
        """
        Get corresponding normals in second phase using orientation relationship.
        
        Args:
            N_guess: Candidate normals in first phase
            interface_trace2: Interface trace in second phase
            LC: Lattice correspondence matrix
            Lr2: Reciprocal lattice matrix for second phase
            LCall: All variant correspondence matrices
            
        Returns:
            tuple: Corresponding normals for closest variant and all variants
        """
        titles = 'n_miller n_miller_normvec HPvsTrace_angle'.split()
        N_guess2ndphase={}
        N_guess2ndphase_allvars={}
        for title in titles:
            N_guess2ndphase[f'{title}'] =[]
            N_guess2ndphase_allvars[f'{title}'] =[]
        for n_miller in N_guess['n_miller']:
            N_guess2ndphase['n_miller'].append(LC.dot(n_miller))
            N_guess2ndphase['n_miller_family'].append(hklsDict[tuple(N_guess2ndphase['n_miller'][-1])])
            N_guess2ndphase['n_miller_normvec'].append(Lr2.dot(N_guess2ndphase['n_miller'][-1]))
            N_guess2ndphase['n_miller_normvec'][-1]=N_guess2ndphase['n_miller_normvec'][-1]/np.linalg.norm(N_guess2ndphase['n_miller_normvec'][-1])
            N_guess2ndphase['HPvsTrace_angle'].append(np.arccos(abs(N_guess2ndphase['n_miller_normvec'][-1].dot(interface_trace2)))*180/np.pi)
            if LCall is not None:
                N_guess2ndphase_vari={}
                for title in titles:
                    N_guess2ndphase_vari[f'{title}'] =[]
                for vari in range(self.T_AM .shape[2]):
                    N_guess2ndphase_vari['n_miller'].append(LCall[:,:,vari].dot(n_miller))
                    N_guess2ndphase_vari['n_miller_family'].append(hklsDict[tuple(N_guess2ndphase_vari['n_miller'][-1])])
                    N_guess2ndphase_vari['n_miller_normvec'].append(Lr2.dot(N_guess2ndphase_vari['n_miller'][-1]))
                    N_guess2ndphase_vari['n_miller_normvec'][-1]=N_guess2ndphase_vari['n_miller_normvec'][-1]/np.linalg.norm(N_guess2ndphase_vari['n_miller_normvec'][-1])
                    N_guess2ndphase_vari['HPvsTrace_angle'].append(np.arccos(abs(N_guess2ndphase_vari['n_miller_normvec'][-1].dot(interface_trace2)))*180/np.pi)

                for title in titles:
                   N_guess2ndphase_allvars[f'{title}'].append(N_guess2ndphase_vari[title])
                
        if LCall is not None:
            return N_guess2ndphase,N_guess2ndphase_allvars
        else:
            return N_guess2ndphase
        
    def analyse_or_preservation(self, meeting_pairs,
                                threshold_deg=3.0,
                                verbose=False):
        """
        Test whether the orientation relationship (OR) is preserved across
        grain boundaries for boundary-meeting lamella pairs.

        Mathematical basis (exact, no approximation):
            OR1 = OR2  ↔  M_ma = M_au
            where M_au = G_au2 · G_au1^T
                M_ma = G_ma2 · G_ma1^T

        The residual angle quantifies deviation from perfect OR preservation:
            residual = min over (S_ma, S_au) of
                    angle( S_ma · M_ma · M_au^T · S_au^T )

        Near zero  → same OR in both grains, crystallographically coupled.
        Large      → different OR, independent nucleation.

        Parameters
        ----------
        meeting_pairs : list of tuples
            Each tuple: (lam_A, parent_A, lam_B, parent_B, label).
            Output of boundary_result.detect_boundary_meeting_pairs().
        threshold_deg : float
            Residual angle below which OR is considered preserved. Default 3.0.
        verbose : bool
            Print results per pair. Default False.

        Returns
        -------
        results : list of dict
            One dict per valid pair with keys:
                lam_A, parent_A, lam_B, parent_B, boundary,
                au_disor      : austenite disorientation (deg),
                ma_disor      : martensite disorientation (deg),
                or_residual   : OR preservation residual angle (deg),
                or_preserved  : bool,
                high_angle    : bool (au_disor > 15 deg),
                score_A, score_B : best overall scores for each lamella,
                fit_A, fit_B  : lattice correspondence fit for each lamella.

        Notes
        -----
        LCV numbers are grain-specific (indexed relative to whichever of the
        48 symmetry-equivalent austenite orientations EBSD selected per grain).
        Cross-grain LCV comparisons are INVALID — use or_residual instead.
        """
        from scipy.spatial.transform import Rotation as _R

        # ── Symmetry operations (proper only, det=+1) ─────────────────────────
        def _proper_symops(phase_key):
            s = np.array(self.clustering_result.data.phases[phase_key]['symops'])
            return s[np.array([round(np.linalg.det(si)) == 1 for si in s])]

        symops_au = _proper_symops(self.keyau)
        symops_ma = _proper_symops(self.keyma)

        # ── Disorientation ────────────────────────────────────────────────────
        def _disor(G1, G2, symops):
            R_ab = G2 @ G1.T
            R_eq = np.einsum('sij,jk->sik', symops, R_ab)
            q    = _R.from_matrix(R_eq).as_quat()
            w    = np.clip(np.abs(q[:, 3]), -1.0, 1.0)
            return float(np.degrees(2 * np.arccos(w)).min())

        # ── OR residual ───────────────────────────────────────────────────────
        def _or_residual(G_au1, G_ma1, G_au2, G_ma2):
            M_ma = G_ma2 @ G_ma1.T
            M_au = G_au2 @ G_au1.T
            D    = M_ma @ M_au.T
            angles = []
            for S_ma in symops_ma:
                for S_au in symops_au:
                    M_sym = S_ma @ D @ S_au.T
                    if abs(np.linalg.det(M_sym) - 1.0) > 0.05:
                        continue
                    q = _R.from_matrix(M_sym).as_quat()
                    w = np.clip(abs(q[3]), -1.0, 1.0)
                    angles.append(np.degrees(2 * np.arccos(w)))
            return float(min(angles)) if angles else np.nan

        # ── Get best HP result for a (parent, child) pair ────────────────────
        def _get_best_hp(pid, cid):
            """
            Return orientation matrices and best score for (pid, cid).
            Searches both ROIs and takes the one with the lowest overall score.
            """
            if pid not in self.HP:
                return None
            if cid not in self.HP[pid]:
                return None

            best_score = np.inf
            best       = None

            for roi in ('semi_roi_1', 'semi_roi_2'):
                if roi not in self.HP[pid][cid]:
                    continue
                roi_data = self.HP[pid][cid][roi]

                # Orientation matrices
                G_au = roi_data.get('G2Cryst_red_avg', {}).get(self.keyau)
                G_ma = roi_data.get('G2Cryst_red_avg', {}).get(self.keyma)
                if G_au is None or G_ma is None:
                    continue

                # Score is stored in HP_matches['Score']['overall']
                hp_matches = roi_data.get('Interfaces', {}).get('HP_matches')
                if hp_matches is None:
                    continue
                overall_scores = hp_matches.get('Score', {}).get('overall', [])
                if len(overall_scores) == 0:
                    continue

                score    = float(overall_scores[0])   # rank-1 (already sorted)
                fitcorr  = hp_matches['Score'].get('fitcorresp', ['n/a'])[0]

                if score < best_score:
                    best_score = score
                    best = {
                        'G_au':   G_au,
                        'G_ma':   G_ma,
                        'score':  score,
                        'fit':    fitcorr,
                        'roi':    roi,
                    }

            return best

        # ── Process pairs ─────────────────────────────────────────────────────
        results = []
        for lam_a, pid_a, lam_b, pid_b, label in meeting_pairs:

            # Skip self-pairs (same lamella detected on both sides of boundary)
            if lam_a == lam_b:
                if verbose:
                    print(f"  Skipping {label} lam {lam_a}↔{lam_b}: "
                        f"same lamella")
                continue

            hp_a = _get_best_hp(pid_a, lam_a)
            hp_b = _get_best_hp(pid_b, lam_b)

            if hp_a is None:
                if verbose:
                    print(f"  Skipping {label} lam {lam_a}↔{lam_b}: "
                        f"HP data missing for lam {lam_a} (parent {pid_a})")
                continue
            if hp_b is None:
                if verbose:
                    print(f"  Skipping {label} lam {lam_a}↔{lam_b}: "
                        f"HP data missing for lam {lam_b} (parent {pid_b})")
                continue

            G_au_a = hp_a['G_au']
            G_ma_a = hp_a['G_ma']
            G_au_b = hp_b['G_au']
            G_ma_b = hp_b['G_ma']

            au_d = _disor(G_au_a, G_au_b, symops_au)
            ma_d = _disor(G_ma_a, G_ma_b, symops_ma)
            or_r = _or_residual(G_au_a, G_ma_a, G_au_b, G_ma_b)

            entry = {
                'lam_A':       lam_a,
                'parent_A':    pid_a,
                'lam_B':       lam_b,
                'parent_B':    pid_b,
                'boundary':    label,
                'au_disor':    au_d,
                'ma_disor':    ma_d,
                'disor_delta':  abs(ma_d - au_d),        # ← add
                'same_macro_variant': abs(ma_d - au_d) < threshold_deg,  # ← add
                'or_residual': or_r,
                'or_preserved':or_r < threshold_deg,
                'high_angle':  au_d > 15.0,
                'score_A':     hp_a['score'],
                'score_B':     hp_b['score'],
                'fit_A':       hp_a['fit'],
                'fit_B':       hp_b['fit'],
            }
            results.append(entry)

            if verbose:
                print(f"  {label}  "
                    f"lam {lam_a}(p{pid_a}) ↔ lam {lam_b}(p{pid_b})  "
                    f"au={au_d:.1f}°  ma={ma_d:.1f}°  "
                    f"OR_res={or_r:.1f}°  "
                    f"{'PRESERVED' if entry['or_preserved'] else 'not preserved'}")

        if verbose:
            n_pres = sum(r['or_preserved'] for r in results)
            la = [r for r in results if not r['high_angle']]
            ha = [r for r in results if r['high_angle']]
            print(f"\nSummary: {n_pres}/{len(results)} OR preserved")
            if la:
                print(f"  Low-angle boundaries (<15°):  "
                    f"{sum(r['or_preserved'] for r in la)}/{len(la)}")
            if ha:
                print(f"  High-angle boundaries (≥15°): "
                    f"{sum(r['or_preserved'] for r in ha)}/{len(ha)}")
        self.or_results=results
        return results
    def plot_or_histogram(self, results, threshold_deg=3.0,
                        bins=30, figsize=(12, 5)):
        """
        Plot histogram of OR residual angles from analyse_or_preservation()
        to identify the natural gap between preserved and not-preserved cases.

        Parameters
        ----------
        results : list of dict
            Output of analyse_or_preservation().
        threshold_deg : float
            Current threshold shown as vertical line. Default 3.0.
        bins : int
            Number of histogram bins. Default 30.
        figsize : tuple
            Figure size. Default (12, 5).

        Returns
        -------
        fig, axes : matplotlib Figure and Axes array (1×2).
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if not results:
            print("No results to plot.")
            return None, None

        import numpy as np
        or_res  = np.array([r['or_residual']  for r in results])
        au_d    = np.array([r['au_disor']     for r in results])
        ha_mask = np.array([r['high_angle']   for r in results])
        labels  = np.array([r['boundary']     for r in results])

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # ── Panel 1: OR residual histogram split by LA / HA ──────────────────
        ax1 = fig.add_subplot(gs[0])
        or_la = or_res[~ha_mask]
        or_ha = or_res[ha_mask]
        bin_edges = np.linspace(0, max(or_res.max(), 1), bins + 1)
        ax1.hist(or_ha, bins=bin_edges, color='steelblue', alpha=0.7,
                edgecolor='white', label=f'High-angle Au (≥15°, n={len(or_ha)})')
        ax1.hist(or_la, bins=bin_edges, color='orangered', alpha=0.85,
                edgecolor='white', label=f'Low-angle Au (<15°, n={len(or_la)})')
        ax1.axvline(threshold_deg, color='black', lw=2, ls='--',
                    label=f'Threshold ({threshold_deg}°)')
        ax1.set_xlabel('OR residual angle (deg)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('OR Residual Distribution\n(natural gap = threshold)',
                    fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # Annotate suggested threshold if a gap is visible
        # Find first local minimum in low-angle histogram for suggestion
        if len(or_la) >= 3:
            la_counts, la_edges = np.histogram(or_la, bins=bin_edges)
            # Simple gap detection: find bin where count drops to 0 or near 0
            # after the first peak
            peak_idx   = np.argmax(la_counts)
            gap_thresh = None
            for i in range(peak_idx + 1, len(la_counts)):
                if la_counts[i] == 0:
                    gap_thresh = la_edges[i]
                    break
                if la_counts[i] <= 1 and i > peak_idx + 1:
                    gap_thresh = la_edges[i]
                    break
            if gap_thresh is not None and abs(gap_thresh - threshold_deg) > 0.5:
                ax1.axvline(gap_thresh, color='green', lw=1.5, ls=':',
                            label=f'Suggested gap ({gap_thresh:.1f}°)')
                ax1.legend(fontsize=9)

        # ── Panel 2: OR residual vs Au disorientation scatter ────────────────
        ax2 = fig.add_subplot(gs[1])
        sc_la = ax2.scatter(au_d[~ha_mask], or_res[~ha_mask],
                            c='orangered', alpha=0.7, s=30, edgecolors='none',
                            label='Low-angle Au (<15°)')
        sc_ha = ax2.scatter(au_d[ha_mask],  or_res[ha_mask],
                            c='steelblue',  alpha=0.5, s=20, edgecolors='none',
                            label='High-angle Au (≥15°)')
        ax2.axhline(threshold_deg, color='black', lw=1.5, ls='--',
                    label=f'Threshold ({threshold_deg}°)')
        # 1:1 reference line
        lim = max(au_d.max(), or_res.max()) * 1.05
        ax2.plot([0, lim], [0, lim], 'k-', lw=0.8, alpha=0.3,
                label='OR_res = Au_disor')
        ax2.set_xlabel('Austenite disorientation (deg)', fontsize=11)
        ax2.set_ylabel('OR residual angle (deg)', fontsize=11)
        ax2.set_title('OR Residual vs Austenite Disorientation', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.25)

        # Annotate preserved pairs
        pres_mask = or_res < threshold_deg
        for i, r in enumerate(results):
            if pres_mask[i]:
                ax2.annotate(
                    f"{r['lam_A']}↔{r['lam_B']}",
                    (au_d[i], or_res[i]),
                    fontsize=6.5, alpha=0.8,
                    xytext=(4, 4), textcoords='offset points')

        plt.suptitle('OR Preservation Analysis', fontsize=13, fontweight='bold')
        plt.tight_layout()

        # Panel 3: disor_delta histogram
        ax3 = fig.add_subplot(gs[2])
        deltas_la = [r['disor_delta'] for r in results if not r['high_angle']]
        deltas_ha = [r['disor_delta'] for r in results if r['high_angle']]
        ax3.hist(deltas_ha, bins=bin_edges, color='steelblue', alpha=0.7,
                label=f'High-angle (n={len(deltas_ha)})')
        ax3.hist(deltas_la, bins=bin_edges, color='orangered', alpha=0.85,
                label=f'Low-angle (n={len(deltas_la)})')
        ax3.axvline(threshold_deg, color='black', lw=2, ls='--',
                    label=f'Threshold ({threshold_deg}°)')
        ax3.set_xlabel('|ma_disor - au_disor| (deg)')
        ax3.set_ylabel('Count')
        ax3.set_title('Macroscopic Variant Compatibility\n(|M_ma - M_au| angle)')
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        return fig, (ax1, ax2, ax3)
    
    def export_results(self, output_path,
                        parent_grain_groups=None,
                        parent_disor_matrix=None,
                        child_lamella_groups=None,
                        lamellar_child_clusters=None,
                        meeting_pairs=None,
                        or_results=None,
                        boundary_parents=None,
                        boundary_children=None,
                        params=None,
                        export_xlsx=None):
        """
        Collect all analysis results and write to results.json for
        interpretation in Claude. Optionally also exports hpstat.xlsx.

        All parameters are optional. If not provided explicitly, the method
        falls back to results embedded on clustering_result, boundary_result,
        and self by the pipeline methods.
        Explicit parameters always take priority over embedded attributes.

        Parameters
        ----------
        output_path : str or Path
            Where to write results.json.
        parent_grain_groups : dict, optional
            {root_id: [fragment_ids]} from reconstruct_physical_grains().
            Falls back to clustering_result.parent_grain_groups.
        parent_disor_matrix : dict, optional
            {(p1,p2): angle_deg} from reconstruct_physical_grains().
            Falls back to clustering_result.parent_disor_matrix.
        child_lamella_groups : dict, optional
            {root_id: [fragment_ids]} from reconstruct_physical_lamellas().
            Falls back to clustering_result.child_lamella_groups.
        lamellar_child_clusters : list, optional
            From identify_lamellar_clusters().
            Falls back to clustering_result.lamellar_child_clusters.
        meeting_pairs : list, optional
            From boundary_result.detect_boundary_meeting_pairs().
            Falls back to boundary_result.meeting_pairs, then self.meeting_pairs.
        or_results : list, optional
            From analyse_or_preservation().
            Falls back to self.or_results.
        boundary_parents : set, optional
            From boundary_result.detect_boundary_clusters(parent_ids).
            Falls back to boundary_result.boundary_parent_clusters.
        boundary_children : set, optional
            From boundary_result.detect_boundary_clusters(child_ids).
            Falls back to boundary_result.boundary_child_clusters.
        params : dict, optional
            Analysis parameters. Falls back to self.params.
        export_xlsx : str or Path, optional
            If provided, also writes hpstat.xlsx.

        Returns
        -------
        results : dict
            Assembled results dict (also written to output_path).

        Example
        -------
        >>> # Minimal call — all data embedded on objects
        >>> hpa.export_results(output_path='results.json')

        >>> # Explicit call — overrides embedded data
        >>> hpa.export_results(
        ...     output_path          = 'results.json',
        ...     parent_grain_groups  = parent_grain_groups,
        ...     parent_disor_matrix  = parent_disor_matrix,
        ...     child_lamella_groups = child_lamella_groups,
        ...     params               = params,
        ...     export_xlsx          = 'hpstat.xlsx',
        ... )
        """
        import json
        from pathlib import Path as _Path
        from collections import defaultdict

        # ── Resolve: explicit > embedded on objects > default ─────────────
        def _resolve(explicit, *fallbacks, default=None):
            if explicit is not None:
                return explicit
            for fb in fallbacks:
                if fb is not None:
                    return fb
            return default

        parent_grain_groups = _resolve(
            parent_grain_groups,
            getattr(self.clustering_result, 'parent_grain_groups', None),
            default={})

        parent_disor_matrix = _resolve(
            parent_disor_matrix,
            getattr(self.clustering_result, 'parent_disor_matrix', None),
            default={})

        child_lamella_groups = _resolve(
            child_lamella_groups,
            getattr(self.clustering_result, 'child_lamella_groups', None),
            default={})

        lamellar_child_clusters = _resolve(
            lamellar_child_clusters,
            getattr(self.clustering_result, 'lamellar_child_clusters', None),
            default=None)

        meeting_pairs = _resolve(
            meeting_pairs,
            getattr(self.boundary_result, 'meeting_pairs',    None),
            getattr(self,                 'meeting_pairs',    None),
            default=[])

        or_results = _resolve(
            or_results,
            getattr(self, 'or_results', None),
            default=[])

        boundary_parents = _resolve(
            boundary_parents,
            getattr(self.boundary_result, 'boundary_parent_clusters', None),
            default=set())

        boundary_children = _resolve(
            boundary_children,
            getattr(self.boundary_result, 'boundary_child_clusters', None),
            default=set())

        params = _resolve(
            params,
            getattr(self, 'params', None),
            default={})

        # ── JSON encoder ──────────────────────────────────────────────────
        class _Enc(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return None if np.isnan(obj) else float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (set, frozenset)):
                    return sorted(int(x) for x in obj)
                if isinstance(obj, _Path):
                    return str(obj)
                try:
                    import pandas as _pd
                    if isinstance(obj, _pd.DataFrame):
                        return obj.to_dict('records')
                except ImportError:
                    pass
                return super().default(obj)

        # ── Section: metadata ─────────────────────────────────────────────
        meta = {
            'n_scan_points': int(len(self.clustering_result.labels)),
            'phases':        list(self.clustering_result.data.phase_ids.keys()),
            'parent_phase':  self.keyau,
            'child_phase':   self.keyma,
            'params':        params,
        }

        # ── Section: clustering ───────────────────────────────────────────
        disor_vals = list(parent_disor_matrix.values())
        n_au_ids   = sum(len(v) for v in parent_grain_groups.values())
        n_ma_ids   = sum(len(v) for v in child_lamella_groups.values())

        clustering = {
            'austenite': {
                'n_cluster_ids':     n_au_ids,
                'n_physical_grains': len(parent_grain_groups),
                'n_multi_fragment':  sum(1 for v in parent_grain_groups.values()
                                        if len(v) > 1),
                'n_boundary_grains': len(boundary_parents),
                'boundary_grain_ids':sorted(int(x) for x in boundary_parents),
                'multi_fragment_groups': {
                                str(int(gid)): [int(f) for f in sorted(frags)]
                                for gid, frags in parent_grain_groups.items()
                                if len(frags) > 1
                },
                'pairwise_disor': {
                    'n_pairs':    len(disor_vals),
                    'mean_deg':   float(np.mean(disor_vals))   if disor_vals else None,
                    'median_deg': float(np.median(disor_vals)) if disor_vals else None,
                    'max_deg':    float(np.max(disor_vals))    if disor_vals else None,
                },
            },
            'martensite': {
                'n_cluster_ids':       n_ma_ids,
                'n_physical_lamellae': len(child_lamella_groups),
                'n_multi_fragment':    sum(1 for v in child_lamella_groups.values()
                                        if len(v) > 1),
                'n_boundary_lamellae': len(boundary_children),
            },
        }

        # ── Section: HP statistics ────────────────────────────────────────
        hp_stats = self._collect_hp_stats(
            lamellar_child_clusters = lamellar_child_clusters,
            export_xlsx             = export_xlsx,
        )

        # ── Section: Schmid-factor analysis ──────────────────────────────
        schmid_stats = self._collect_schmid_stats(
            grain_groups     = parent_grain_groups,
            disor_matrix     = parent_disor_matrix,
            boundary_parents = boundary_parents,
        )

        # ── Section: OR preservation ──────────────────────────────────────
        if not or_results:
            or_section = {
                'n_meeting_pairs': len(meeting_pairs) if meeting_pairs else 0,
                'note': 'Run hpa.analyse_or_preservation() first',
            }
        else:
            or_arr  = np.array([r['or_residual']  for r in or_results])
            au_arr  = np.array([r['au_disor']     for r in or_results])
            ha_arr  = np.array([r['high_angle']   for r in or_results])
            pres    = np.array([r['or_preserved'] for r in or_results])
            la_mask = ~ha_arr

            r_au_or, p_au_or = None, None
            if len(or_arr) > 4:
                try:
                    from scipy import stats as _st
                    r_au_or = float(_st.spearmanr(au_arr, or_arr)[0])
                    p_au_or = float(_st.spearmanr(au_arr, or_arr)[1])
                except Exception:
                    pass

            by_bnd = defaultdict(list)
            for r in or_results:
                by_bnd[r['boundary']].append(r)

            or_section = {
                'n_meeting_pairs_detected': len(meeting_pairs) if meeting_pairs else 0,
                'n_pairs_evaluated':        len(or_results),
                'threshold_deg':            params.get('or_preserved_deg', 3.0),
                'overall': {
                    'n_preserved':        int(pres.sum()),
                    'pct_preserved':      float(100 * pres.mean()),
                    'or_residual_mean':   float(or_arr.mean()),
                    'or_residual_median': float(np.median(or_arr)),
                },
                'low_angle_boundaries': {
                    'n_pairs':          int(la_mask.sum()),
                    'n_preserved':      int(pres[la_mask].sum()),
                    'pct_preserved':    float(100 * pres[la_mask].mean())
                                        if la_mask.sum() else 0.0,
                    'or_residual_mean': float(or_arr[la_mask].mean())
                                        if la_mask.sum() else None,
                },
                'high_angle_boundaries': {
                    'n_pairs':          int(ha_arr.sum()),
                    'n_preserved':      int(pres[ha_arr].sum()),
                    'pct_preserved':    float(100 * pres[ha_arr].mean())
                                        if ha_arr.sum() else 0.0,
                    'or_residual_mean': float(or_arr[ha_arr].mean())
                                        if ha_arr.sum() else None,
                },
                'au_disor_vs_or_residual': {
                    'spearman_r': r_au_or,
                    'spearman_p': p_au_or,
                },
                'by_boundary': {
                    bnd: {
                        'n_pairs':          len(rows),
                        'au_disor_deg':     round(rows[0]['au_disor'], 1),
                        'high_angle':       rows[0]['high_angle'],
                        'or_residual_mean': round(float(np.mean(
                            [r['or_residual'] for r in rows])), 2),
                        'or_residual_min':  round(float(np.min(
                            [r['or_residual'] for r in rows])), 2),
                        'n_preserved':      int(sum(
                            r['or_preserved'] for r in rows)),
                    }
                    for bnd, rows in sorted(by_bnd.items())
                },
                'all_pairs': or_results,
                'interpretation': {
                    'low_angle': (
                        'OR is preserved across low-angle boundaries when '
                        'residual ≈ 0: martensite inherits the small austenite '
                        'misorientation rigidly.'),
                    'high_angle': (
                        'OR is NOT preserved across high-angle boundaries: '
                        'lamellae nucleated independently.'),
                    'lcv_warning': (
                        'LCV same/different comparisons across grains are '
                        'INVALID. Use or_residual instead.'),
                },
                'macro_variant': {
                    'n_same_macro':  int(sum(
                        r.get('same_macro_variant', False) for r in or_results)),
                    'n_same_macro_or_preserved': int(sum(
                        r.get('same_macro_variant', False) and r['or_preserved']
                        for r in or_results)),
                    'n_same_macro_or_not': int(sum(
                        r.get('same_macro_variant', False) and not r['or_preserved']
                        for r in or_results)),
                    'disor_delta_mean': float(np.mean(
                        [r.get('disor_delta', 0) for r in or_results])),
                    'interpretation': (
                        'Same macroscopic variant (|ma_disor - au_disor| < threshold) '
                        'means both lamellae have the same absolute orientation in the '
                        'sample frame but does not guarantee the same lattice '
                        'correspondence. OR_res distinguishes these cases: '
                        'same macro variant + OR preserved = continuous growth possible; '
                        'same macro variant + OR NOT preserved = same habit plane '
                        'direction but different atom shuffle.'
                    ),
                },
            }

        # ── Section: variant compatibility ────────────────────────────────
        variant_section = getattr(self, 'variant_analysis', {})

        # ── Assemble ──────────────────────────────────────────────────────
        results = {
            'metadata':          meta,
            'clustering':        clustering,
            'hp_statistics':     hp_stats,
            'schmid_statistics': schmid_stats,
            'or_preservation':   or_section,
            'variant_analysis':  variant_section,
            'warnings': [
                'LCV numbers are grain-specific — cross-grain comparisons INVALID.',
                'Apparent grain size is 2D section only — size correlations '
                'may be stereological artefacts.',
                'Boundary-meeting pairs detected automatically via tip proximity.',
                "OR preservation uses proper rotations only for B19' 2/m: {E, 2(010)}.",
                'Merge martensite fragments before austenite.',
            ],
        }

        # ── Write JSON ────────────────────────────────────────────────────
        out = _Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, cls=_Enc)

        print(f"Results written to {out}")
        if export_xlsx is not None:
            print(f"hpstat written to {export_xlsx}")
        print(f"\nUpload {out.name} to Claude and ask e.g.:")
        print("  'Is the OR preserved across grain boundaries?'")
        print("  'Which HP families dominate — does that match theory?'")
        print("  'Compare these results to the previous dataset.'")
        print("  'Interpret the OR residual histogram — natural threshold?'")

        self.params  = params
        self.results = results
        return results    
    def _collect_hp_stats(self, lamellar_child_clusters=None,
                        export_xlsx=None):
        """
        Collect habit plane statistics directly from self.HP.

        Replaces the manual restable construction in the notebook.
        Returns a statistics summary dict and optionally writes hpstat.xlsx.

        Parameters
        ----------
        lamellar_child_clusters : list, optional
            Output of clustering_result.identify_lamellar_clusters() or
            merge_lamellar_fragments(). Used as elongation cache for
            parent/child cluster geometry. If None, geometry is computed
            on the fly via analyze_cluster_elongation().
        export_xlsx : str or Path, optional
            If provided, writes the full flat table to this Excel file.
            Same format as the manually constructed hpstat.xlsx.
            If None, no Excel file is written.

        Returns
        -------
        stats : dict
            Summary statistics for the hp_statistics section of
            export_results(). Keys: n_interfaces_rank1, score,
            habit_plane_families, oop, lcv_distribution, strain,
            lcv_warning.
        """
        import operator
        from functools import reduce
        from collections import defaultdict
        from pathlib import Path as _Path

        # ── Build elongation cache ─────────────────────────────────────────
        elong_cache = {}
        if lamellar_child_clusters is not None:
            for item in lamellar_child_clusters:
                elong_cache[item['cluster_id']] = item

        # ── Table columns (same as original notebook export) ──────────────
        restable = {
            'austenite parent cluster id':    [],
            'martensite child cluster id':    [],
            'Parent cluster size (px)':       [],
            'Parent cluster area [um]':       [],
            'Parent cluster average orientation': [],
            'Child cluster size (px)':        [],
            'Child cluster area [um]':        [],
            'Child cluster average orientation': [],
            'Martensite child lamella length':    [],
            'Martensite child lamella width':     [],
            'Martensite child lamella principal direction in sample coordinate system': [],
            'Martensite child lamella width perpendicular direction in the sample coordinate system': [],
            'Vector of the interface trace in the sample coordinate system':           [],
            'Vector of the normal of the interface trace in the sample coordinate system': [],
            'Average orientation matrix of Austenite phase at the interface':  [],
            'Average orientation matrix of Martensite phase at the interface': [],
            'roi (which side of the lamella)':                  [],
            'score order of the habit plane match':             [],
            'Closest theoretical martensite variant (LCV)':     [],
            'Transformation strain [-] along sample [100] from the closest LCV': [],
            'Score':                                            [],
            'Fit lattice correspondence':                       [],
            'Miller indices of the habit plane in Austenite phase':  [],
            'Unified symmetry equivalent Miller indices of the habit plane in Austenite phase': [],
            'Unit vector of the habit plane of Austenite phase in the crystal coordinate system': [],
            'Unit vector of the habit plane of Austenite phase in the sample coordinate system':  [],
            'Miller indices of the habit plane in Martensite phase': [],
            'Unified symmetry equivalent Miller indices of the habit plane in Martensite phase': [],
            'Unit vector of the habit plane of Martensite phase in the crystal coordinate system': [],
            'Unit vector of the habit plane of Martensite phase in the sample coordinate system':  [],
        }

        # ── Stats accumulators ─────────────────────────────────────────────
        scores_r1     = []
        scores_all    = []
        fit_r1        = []
        au_fam_r1     = defaultdict(int)
        ma_fam_r1     = defaultdict(int)
        oop_angles_r1 = []
        strains_r1    = []
        lcv_r1        = []
        n_candidates  = []

        # ── Main traversal ─────────────────────────────────────────────────
        for parent_id in self.HP.keys():
            for child_id in self.HP[parent_id].keys():
                for roi in ('semi_roi_1', 'semi_roi_2'):
                    if roi not in self.HP[parent_id][child_id]:
                        continue
                    rd = self.HP[parent_id][child_id][roi]
                    if 'Interfaces' not in rd:
                        continue
                    hp = rd['Interfaces'].get('HP_matches', {})
                    scores_list = hp.get('Score', {}).get('overall', [])
                    fit_list    = hp.get('Score', {}).get('fitcorresp', [])
                    if len(scores_list) == 0:
                        continue

                    n_candidates.append(len(scores_list))
                    CV = rd.get('Closest Variant')
                    TR = None
                    if CV is not None:
                        try:
                            TR = float(np.round(
                                rd['Transformation strain_allvars'][CV][0]['along [1. 0. 0.] [-]'],
                                decimals=4))
                        except (KeyError, IndexError, TypeError):
                            TR = None
                    # Cluster geometry
                    pdata = self.clustering_result.get_lamellar_cluster_info(
                        parent_id, elongation_cache=elong_cache)
                    cdata = self.clustering_result.get_lamellar_cluster_info(
                        child_id, elongation_cache=elong_cache)

                    # Interface geometry
                    iface  = rd['Interfaces']
                    G_au   = rd['G2Cryst_red_avg'].get(self.keyau)
                    G_ma   = rd['G2Cryst_red_avg'].get(self.keyma)
                    trace  = iface.get('interface_trace_sample')
                    ntrace = iface.get('interfacenorm_trace_sample')

                    # Per-candidate rows
                    for idx, score in enumerate(scores_list):
                        fit = fit_list[idx] if idx < len(fit_list) else 'n/a'

                        # ── HP normal data ─────────────────────────────────
                        def _get(phase_key, field):
                            vals = hp.get(phase_key, {}).get(
                                'Habit plane normal', {}).get(field, [])
                            return vals[idx] if idx < len(vals) else None

                        au_miller  = _get(self.keyau, 'n_miller')
                        au_fam_val = _get(self.keyau, 'n_miller_family')
                        au_nvec    = _get(self.keyau, 'n_vec')
                        au_nvec_s  = _get(self.keyau, 'n_vec_sampl')
                        ma_miller  = _get(self.keyma, 'n_miller')
                        ma_fam_val = _get(self.keyma, 'n_miller_family')
                        ma_nvec    = _get(self.keyma, 'n_vec')
                        ma_nvec_s  = _get(self.keyma, 'n_vec_sampl')

                        # ── Append to table ────────────────────────────────
                        restable['austenite parent cluster id'].append(parent_id)
                        restable['martensite child cluster id'].append(child_id)

                        for prefix, data in [('Parent', pdata), ('Child', cdata)]:
                            if data is not None:
                                restable[f'{prefix} cluster size (px)'].append(
                                    data.get('n_pixels'))
                                restable[f'{prefix} cluster area [um]'].append(
                                    data.get('area'))
                                restable[f'{prefix} cluster average orientation'].append(
                                    self.clustering_result.avg_orientations.get(
                                        parent_id if prefix=='Parent' else child_id))
                            else:
                                for k in [f'{prefix} cluster size (px)',
                                        f'{prefix} cluster area [um]',
                                        f'{prefix} cluster average orientation']:
                                    restable[k].append(None)

                        if cdata is not None:
                            pd2 = self.CrystalRef2Spatial[0:2, 0:2]
                            restable['Martensite child lamella length'].append(
                                cdata.get('length'))
                            restable['Martensite child lamella width'].append(
                                cdata.get('width'))
                            restable['Martensite child lamella principal direction in sample coordinate system'].append(
                                pd2.dot(cdata['principal_direction'])
                                if cdata.get('principal_direction') is not None else None)
                            restable['Martensite child lamella width perpendicular direction in the sample coordinate system'].append(
                                pd2.dot(cdata['perpendicular_direction'])
                                if cdata.get('perpendicular_direction') is not None else None)
                        else:
                            for k in ['Martensite child lamella length',
                                    'Martensite child lamella width',
                                    'Martensite child lamella principal direction in sample coordinate system',
                                    'Martensite child lamella width perpendicular direction in the sample coordinate system']:
                                restable[k].append(None)

                        restable['Vector of the interface trace in the sample coordinate system'].append(trace)
                        restable['Vector of the normal of the interface trace in the sample coordinate system'].append(ntrace)
                        restable['Average orientation matrix of Austenite phase at the interface'].append(G_au)
                        restable['Average orientation matrix of Martensite phase at the interface'].append(G_ma)
                        restable['roi (which side of the lamella)'].append(roi)
                        restable['score order of the habit plane match'].append(idx + 1)
                        restable['Closest theoretical martensite variant (LCV)'].append(CV)
                        restable['Transformation strain [-] along sample [100] from the closest LCV'].append(TR)
                        restable['Score'].append(float(score))
                        restable['Fit lattice correspondence'].append(fit)
                        restable['Miller indices of the habit plane in Austenite phase'].append(au_miller)
                        restable['Unified symmetry equivalent Miller indices of the habit plane in Austenite phase'].append(au_fam_val)
                        restable['Unit vector of the habit plane of Austenite phase in the crystal coordinate system'].append(au_nvec)
                        restable['Unit vector of the habit plane of Austenite phase in the sample coordinate system'].append(au_nvec_s)
                        restable['Miller indices of the habit plane in Martensite phase'].append(ma_miller)
                        restable['Unified symmetry equivalent Miller indices of the habit plane in Martensite phase'].append(ma_fam_val)
                        restable['Unit vector of the habit plane of Martensite phase in the crystal coordinate system'].append(ma_nvec)
                        restable['Unit vector of the habit plane of Martensite phase in the sample coordinate system'].append(ma_nvec_s)

                        # ── Stats (rank-1 only) ────────────────────────────
                        if idx == 0:
                            scores_r1.append(float(score))
                            fit_r1.append(fit)
                            if CV is not None:
                                lcv_r1.append(int(CV))
                            if TR is not None:
                                strains_r1.append(float(TR))
                            if au_fam_val is not None:
                                au_fam_r1[str(au_fam_val)] += 1
                            if ma_fam_val is not None:
                                ma_fam_r1[str(ma_fam_val)] += 1
                            if au_nvec_s is not None:
                                v = np.array(au_nvec_s)
                                n = np.linalg.norm(v)
                                if n > 1e-10:
                                    oop_angles_r1.append(float(
                                        np.degrees(np.arccos(
                                            np.clip(abs(v[2]/n), 0, 1)))))
                        scores_all.append(float(score))

        # ── Export to Excel ────────────────────────────────────────────────
        if export_xlsx is not None:
            import pandas as pd
            df = pd.DataFrame(restable)
            df.to_excel(str(export_xlsx), index=False)
            print(f"hpstat written to {export_xlsx}  "
                f"({len(df)} rows × {len(df.columns)} columns)")

        # ── Statistics summary ─────────────────────────────────────────────
        n_r1      = len(scores_r1)
        n_fit_yes = sum(1 for f in fit_r1 if f == 'yes')
        n_fit_no  = sum(1 for f in fit_r1 if f == 'no')
        oop_arr   = np.array(oop_angles_r1)
        oop_flag  = 20.0
        n_flagged = int((oop_arr <= oop_flag).sum()) if len(oop_arr) else 0
        lcv_counts = defaultdict(int)
        for v in lcv_r1:
            lcv_counts[int(v)] += 1

        return {
            'n_interfaces_rank1':   n_r1,
            'n_candidates_total':   len(scores_all),
            'candidates_per_interface': {
                'mean': float(np.mean(n_candidates)) if n_candidates else None,
                'max':  int(np.max(n_candidates))    if n_candidates else None,
            },
            'score': {
                'rank1_mean':  float(np.mean(scores_r1)) if scores_r1 else None,
                'rank1_std':   float(np.std(scores_r1))  if scores_r1 else None,
                'all_mean':    float(np.mean(scores_all)) if scores_all else None,
                'n_fit_yes':   n_fit_yes,
                'n_fit_no':    n_fit_no,
                'pct_fit_yes': float(100*n_fit_yes/n_r1) if n_r1 else None,
            },
            'habit_plane_families': {
                'austenite':   dict(sorted(
                    au_fam_r1.items(), key=lambda x: -x[1])),
                'martensite':  dict(sorted(
                    ma_fam_r1.items(), key=lambda x: -x[1])),
                'dominant_au': max(au_fam_r1, key=au_fam_r1.get)
                            if au_fam_r1 else None,
            },
            'oop': {
                'mean_deg':    float(oop_arr.mean())     if len(oop_arr) else None,
                'median_deg':  float(np.median(oop_arr)) if len(oop_arr) else None,
                'n_flagged':   n_flagged,
                'flag_threshold_deg': oop_flag,
                'pct_flagged': float(100*n_flagged/n_r1) if n_r1 else None,
            },
            'lcv_distribution':  dict(sorted(lcv_counts.items())),
            'strain': {
                'mean': float(np.mean(strains_r1)) if strains_r1 else None,
                'std':  float(np.std(strains_r1))  if strains_r1 else None,
            },
            'lcv_warning': (
                'LCV numbers are grain-specific — EBSD independently selects '
                'one of the 48 symmetry-equivalent austenite orientations per grain. '
                'Cross-grain LCV comparisons are INVALID. '
                'Use or_residual for cross-grain variant comparison.'
            ),
        }
    def _collect_schmid_stats(self, grain_groups, disor_matrix,
                                boundary_parents):
        """
        Per-physical-grain Schmid-factor analysis.

        Handles the case where self.HP was built before or after
        merge_physical_grains() by using id_map stored on
        clustering_result to translate between pre- and post-merge IDs.

        Parameters
        ----------
        grain_groups : dict {root_id: [fragment_ids]}
            Output of reconstruct_physical_grains().
            Keys are post-merge IDs.
        disor_matrix : dict {(p1,p2): angle_deg}
            Output of reconstruct_physical_grains().
        boundary_parents : set
            Output of boundary_result.detect_boundary_clusters().
            Contains post-merge IDs.

        Returns
        -------
        dict with keys:
            per_grain, correlations, stereological,
            n_physical_grains, n_interior, n_boundary, notes
        """
        try:
            from scipy import stats as _st
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        # ── ID translation ─────────────────────────────────────────────
        # id_map stored by merge_physical_grains: {old_id: new_id}
        # reverse gives new_id → old_id, needed to find HP entries
        # built with pre-merge IDs
        #hp_parent_ids     = set(self.HP.keys())
        hp_parent_ids = set(int(k) for k in self.HP.keys())
        grain_to_clusters = getattr(self.clustering_result, 'grain_to_clusters', None)
        grain_avg_ori     = getattr(self.clustering_result, 'grain_avg_orientations',
                                    self.clustering_result.avg_orientations)

        def _to_hp_ids(root_id, fragments):
            """Return HP parent IDs for a physical grain."""
            result = set()
            for cid in list(fragments) + [root_id]:
                if int(cid) in hp_parent_ids:
                    result.add(int(cid))
            return result
        # ── Helpers ────────────────────────────────────────────────────
        def _sp(xs, ys):
            pairs = [(x, y) for x, y in zip(xs, ys)
                     if x is not None and y is not None
                     and not (isinstance(x, float) and (
                         x != x or abs(x) == float('inf')))
                     and not (isinstance(y, float) and (
                         y != y or abs(y) == float('inf')))]
            if not _has_scipy or len(pairs) < 5:
                return None, None
            xs2, ys2 = zip(*pairs)
            r, p = _st.spearmanr(xs2, ys2)
            return float(r), float(p)

        def _sig(p):
            if p is None: return 'n/a'
            return ('***' if p < 0.001 else
                    '**'  if p < 0.01  else
                    '*'   if p < 0.05  else 'ns')

        cluster_sizes = self.clustering_result.cluster_sizes

        # ── Per-physical-grain aggregation ─────────────────────────────
        per_grain = []

        for root_id, fragments in sorted(grain_groups.items()):
            parent_ids_to_try = _to_hp_ids(root_id, fragments)
            #print(f"grain {root_id}: fragments={sorted(fragments)}, hp_ids={parent_ids_to_try}")
            #break  # just check first grain
            # ── Apparent size ──────────────────────────────────────────────────────────
            # Labels unchanged — cluster_sizes uses original IDs directly
            apparent_size = sum(cluster_sizes.get(int(f), 0) for f in fragments)

            # ── Max internal disorientation ────────────────────────────
            max_int_disor = max(
                (disor_matrix.get((min(a, b), max(a, b)), 0.0)
                 for a in fragments for b in fragments if a != b),
                default=0.0)

            # ── Is boundary grain ──────────────────────────────────────────────────
            # boundary_parents and grain_groups both use post-merge IDs
            is_boundary = (root_id in boundary_parents or
                        any(f in boundary_parents for f in fragments))
            # ── HP data lookup ─────────────────────────────────────────
            parent_ids_to_try = _to_hp_ids(root_id, fragments)

            if not parent_ids_to_try:
                per_grain.append({
                    'phys_grain_id':      root_id,
                    'fragments':          sorted(fragments),
                    'n_fragments':        len(fragments),
                    'max_internal_disor': round(float(max_int_disor), 3),
                    'apparent_size_px':   apparent_size,
                    'n_lamellae':         0,
                    'mean_strain':        None,
                    'std_strain':         None,
                    'lamella_density':    None,
                    'is_boundary':        is_boundary,
                })
                continue

            strains   = []
            n_lam_set = set()

            for parent_id in parent_ids_to_try:
                for child_id in self.HP[parent_id].keys():
                    # Skip non-cluster meta keys (parent_idx, child_idx)
                    if not isinstance(child_id, (int, np.integer)):
                        continue
                    n_lam_set.add(int(child_id))

                    # Best score across both ROIs
                    best_score  = np.inf
                    best_strain = None
                    for roi in ('semi_roi_1', 'semi_roi_2'):
                        if roi not in self.HP[parent_id][child_id]:
                            continue
                        rd   = self.HP[parent_id][child_id][roi]
                        hp_m = rd.get('Interfaces', {}).get(
                            'HP_matches', {})
                        scores = hp_m.get('Score', {}).get('overall', [])
                        if len(scores) == 0:
                            continue
                        score = float(scores[0])
                        if score < best_score:
                            best_score = score
                            CV = rd.get('Closest Variant')
                            if CV is not None:
                                try:
                                    s = float(np.round(
                                        rd['Transformation strain_allvars'
                                           ][CV][0]['along [1. 0. 0.] [-]'],
                                        decimals=4))
                                    best_strain = s
                                except (KeyError, IndexError,
                                        TypeError, ValueError):
                                    pass
                    if best_strain is not None:
                        strains.append(best_strain)

            n_lam       = len(n_lam_set)
            mean_strain = float(np.mean(strains))    if strains          else None
            std_strain  = float(np.std(strains))     if len(strains) > 1 else 0.0
            lam_density = (float(n_lam) / apparent_size * 1000
                           if apparent_size > 0 else None)

            per_grain.append({
                'phys_grain_id':      root_id,
                'fragments':          sorted(fragments),
                'n_fragments':        len(fragments),
                'max_internal_disor': round(float(max_int_disor), 3),
                'apparent_size_px':   apparent_size,
                'n_lamellae':         n_lam,
                'mean_strain':        mean_strain,
                'std_strain':         std_strain,
                'lamella_density':    lam_density,
                'is_boundary':        is_boundary,
            })

        # ── Early exit if no data ──────────────────────────────────────
        if not per_grain:
            return {
                'n_physical_grains': 0,
                'n_interior': 0,
                'n_boundary': 0,
                'per_grain': [],
                'correlations': [],
                'stereological': {},
                'notes': [
                    'No per-grain data — check that grain_groups keys '
                    'match self.HP parent IDs, or that id_map is stored '
                    'on clustering_result after merge_physical_grains().'
                ],
            }

        # ── Spearman correlations ──────────────────────────────────────
        sizes    = [g['apparent_size_px'] for g in per_grain]
        n_lams   = [g['n_lamellae']       for g in per_grain]
        m_strain = [g['mean_strain']      for g in per_grain]
        s_strain = [g['std_strain']       for g in per_grain]
        density  = [g['lamella_density']  for g in per_grain]

        interior = [g for g in per_grain if not g['is_boundary']]
        i_sizes  = [g['apparent_size_px'] for g in interior]
        i_nlams  = [g['n_lamellae']       for g in interior]
        i_strain = [g['mean_strain']      for g in interior]
        i_dens   = [g['lamella_density']  for g in interior]

        def _corr(a, b, label):
            r, p = _sp(a, b)
            n = sum(1 for x, y in zip(a, b)
                    if x is not None and y is not None)
            return {'label': label, 'r': r, 'p': p,
                    'sig': _sig(p), 'n': n}

        correlations = [
            _corr(sizes,    n_lams,
                  'size vs n_lamellae (ALL) — likely artefact'),
            _corr(density,  m_strain,
                  'lamella_density vs mean_strain (ALL)'),
            _corr(s_strain, n_lams,
                  'strain_std vs n_lamellae (ALL) — multi-variant'),
            _corr(i_sizes,  i_nlams,
                  'size vs n_lamellae (INTERIOR only)'),
            _corr(i_dens,   i_strain,
                  'lamella_density vs mean_strain (INTERIOR only)'),
            _corr(i_strain, i_nlams,
                  'mean_strain vs n_lamellae (INTERIOR only)'),
        ]

        # ── Stereological artefact check ───────────────────────────────
        r_raw,  p_raw  = _sp(sizes, n_lams)
        r_dens, p_dens = _sp(sizes, density)
        is_artefact    = (r_dens is not None and r_dens < 0.1 and
                          (p_dens is None or p_dens > 0.05))

        stereological = {
            'size_vs_count_r':     r_raw,
            'size_vs_count_p':     p_raw,
            'size_vs_count_sig':   _sig(p_raw),
            'size_vs_density_r':   r_dens,
            'size_vs_density_p':   p_dens,
            'size_vs_density_sig': _sig(p_dens),
            'is_pure_artefact':    is_artefact,
            'interpretation': (
                'Size-count correlation is a stereological artefact: '
                'larger apparent grains expose more lamellae to the '
                'scan window, not because they transform more.'
                if is_artefact else
                'Residual size effect may exist beyond stereological '
                'bias — verify with 3D reconstruction.'
            ),
        }

        n_interior = sum(1 for g in per_grain if not g['is_boundary'])
        n_boundary = sum(1 for g in per_grain if g['is_boundary'])

        return {
            'n_physical_grains': len(per_grain),
            'n_interior':        n_interior,
            'n_boundary':        n_boundary,
            'per_grain':         per_grain,
            'correlations':      correlations,
            'stereological':     stereological,
            'notes': [
                'LCV numbers are grain-specific — strain IS comparable '
                'across grains (sample-frame quantity).',
                'Strain diversity (std_strain) predicts multi-variant '
                'transformation better than mean strain.',
                'Use interior grains only for unbiased Schmid analysis.',
                'Lamella density (n_lam / apparent_size × 1000) corrects '
                'for the stereological size-count artefact.',
            ],
        }
    def analyse_variant_compatibility(self,
                                        grain_groups,
                                        disor_matrix,
                                        meeting_pairs=None,
                                        ma_threshold_deg=2.0,
                                        or_threshold_deg=3.0,
                                        verbose=False):
        """
        Variant clustering within grains and cross-boundary variant
        compatibility analysis.

        Step 1 — Unique variant groups within each grain
        -------------------------------------------------
        Within each physical austenite grain, lamellae are grouped into
        crystallographically distinct martensite variants by clustering
        their pairwise martensite-martensite disorientations. Two lamellae
        belong to the same variant group if their disorientation < 
        ma_threshold_deg (using B19' proper symops).

        Each variant group is represented by the lamella with the most
        pixels. Its transformation strain and average orientation are
        stored as group properties.

        Step 2 — Cross-boundary variant compatibility
        -----------------------------------------------
        For each pair of neighbouring austenite grains (from disor_matrix),
        for each combination of variant groups (one from each grain):
        compute the OR residual angle:

            residual = min over (S_ma, S_au) of
                    angle( S_ma · M_ma · M_au^T · S_au^T )

        where M_ma = G_ma2 · G_ma1^T  and  M_au = G_au2 · G_au1^T.

        Near zero → same OR in both grains (compatible variants).
        Large     → independent nucleation required (incompatible).

        Step 3 — Meeting vs non-meeting comparison
        -------------------------------------------
        For each grain boundary pair, record the minimum OR residual
        across all variant group combinations, and whether lamellae
        actually meet at that boundary (from meeting_pairs).

        Statistical test: Mann-Whitney comparing min OR residual for
        meeting vs non-meeting grain boundary pairs.

        Parameters
        ----------
        grain_groups : dict {root_id: [fragment_ids]}
            Output of reconstruct_physical_grains().
        disor_matrix : dict {(p1,p2): angle_deg}
            Output of reconstruct_physical_grains().
        meeting_pairs : list of tuples, optional
            Output of boundary_result.detect_boundary_meeting_pairs().
            If None, uses self.meeting_pairs if set.
        ma_threshold_deg : float
            Martensite disorientation below which two lamellae are
            considered the same variant. Default 2.0.
        or_threshold_deg : float
            OR residual below which two variant groups across a boundary
            are considered compatible. Default 3.0.
        verbose : bool
            Print progress. Default False.

        Returns
        -------
        dict with keys:
            per_grain          : list of per-grain variant group dicts
            boundary_pairs     : list of per-boundary-pair dicts
            meeting_stats      : comparison of meeting vs non-meeting pairs
            summary            : high-level summary statistics
        """
        from collections import defaultdict
        from scipy.spatial.transform import Rotation as _R

        if meeting_pairs is None:
            meeting_pairs = getattr(self, 'meeting_pairs', [])

        # ── Symmetry operations ───────────────────────────────────────────
        def _proper_symops(phase_key):
            s = np.array(self.clustering_result.data.phases[phase_key]['symops'])
            return s[np.array([round(np.linalg.det(si)) == 1 for si in s])]

        symops_au = _proper_symops(self.keyau)
        symops_ma = _proper_symops(self.keyma)

        # ── Disorientation helper ─────────────────────────────────────────
        def _disor(G1, G2, symops):
            R_ab = G2 @ G1.T
            R_eq = np.einsum('sij,jk->sik', symops, R_ab)
            q    = _R.from_matrix(R_eq).as_quat()
            w    = np.clip(np.abs(q[:, 3]), -1.0, 1.0)
            return float(np.degrees(2 * np.arccos(w)).min())

        # ── OR residual helper ────────────────────────────────────────────
        def _or_residual(G_au1, G_ma1, G_au2, G_ma2):
            M_ma = G_ma2 @ G_ma1.T
            M_au = G_au2 @ G_au1.T
            D    = M_ma @ M_au.T
            angles = []
            for S_ma in symops_ma:
                for S_au in symops_au:
                    M_sym = S_ma @ D @ S_au.T
                    if abs(np.linalg.det(M_sym) - 1.0) > 0.05:
                        continue
                    q = _R.from_matrix(M_sym).as_quat()
                    w = np.clip(abs(q[3]), -1.0, 1.0)
                    angles.append(np.degrees(2 * np.arccos(w)))
            return float(min(angles)) if angles else np.nan

        # ── ID translation ────────────────────────────────────────────────
        id_map         = getattr(self.clustering_result, 'id_map', {})
        reverse_id_map = {v: k for k, v in id_map.items()} if id_map else {}
        #hp_parent_ids  = set(self.HP.keys())
        hp_parent_ids = set(int(k) for k in self.HP.keys())
        def _hp_parents(fragments, root_id):
            result = set()
            for cid in list(fragments) + [root_id]:
                if cid in hp_parent_ids:
                    result.add(cid)
                orig = reverse_id_map.get(cid, cid)
                if orig in hp_parent_ids:
                    result.add(orig)
            return result

        # ── Get best HP data for a (parent_id, child_id) ─────────────────
        def _get_hp(pid, cid):
            """Return (G_au, G_ma, strain, score) for best ROI."""
            if pid not in self.HP or cid not in self.HP[pid]:
                return None
            best_score  = np.inf
            best        = None
            for roi in ('semi_roi_1', 'semi_roi_2'):
                if roi not in self.HP[pid][cid]:
                    continue
                rd   = self.HP[pid][cid][roi]
                hp_m = rd.get('Interfaces', {}).get('HP_matches', {})
                scores = hp_m.get('Score', {}).get('overall', [])
                if len(scores) == 0:
                    continue
                score = float(scores[0])
                if score < best_score:
                    best_score = score
                    G_au = rd.get('G2Cryst_red_avg', {}).get(self.keyau)
                    G_ma = rd.get('G2Cryst_red_avg', {}).get(self.keyma)
                    CV   = rd.get('Closest Variant')
                    strain = None
                    if CV is not None:
                        try:
                            strain = float(np.round(
                                rd['Transformation strain_allvars'][CV][0][
                                    'along [1. 0. 0.] [-]'], decimals=4))
                        except (KeyError, IndexError, TypeError, ValueError):
                            pass
                    n_px = self.clustering_result.cluster_sizes.get(cid, 0)
                    best = {'G_au': G_au, 'G_ma': G_ma,
                            'strain': strain, 'score': score,
                            'n_pixels': n_px}
            return best

        # ── Step 1: Variant groups within each grain ──────────────────────
        per_grain   = []
        grain_vgroups = {}   # {root_id: [{'rep_cid', 'G_au', 'G_ma',
                            #              'strain', 'members'}]}

        for root_id, fragments in sorted(grain_groups.items()):
            hp_pids = _hp_parents(fragments, root_id)
            if not hp_pids:
                per_grain.append({
                    'phys_grain_id': root_id,
                    'n_lamellae':    0,
                    'n_variants':    0,
                    'variant_groups': [],
                })
                grain_vgroups[root_id] = []
                continue

            # Collect all lamellae with HP data for this grain
            lamellae = []   # list of {'cid', 'pid', 'G_au', 'G_ma',
                            #          'strain', 'score', 'n_pixels'}
            seen_cids = set()
            for pid in hp_pids:
                for cid in self.HP[pid].keys():
                    if not isinstance(cid, (int, np.integer)):
                        continue
                    if cid in seen_cids:
                        continue
                    hp = _get_hp(pid, cid)
                    if hp is None or hp['G_ma'] is None:
                        continue
                    hp['cid'] = int(cid)
                    hp['pid'] = pid
                    lamellae.append(hp)
                    seen_cids.add(cid)

            if not lamellae:
                per_grain.append({
                    'phys_grain_id': root_id,
                    'n_lamellae':    0,
                    'n_variants':    0,
                    'variant_groups': [],
                })
                grain_vgroups[root_id] = []
                continue

            # Union-Find on martensite disorientation
            uf = {i: i for i in range(len(lamellae))}

            def _find(x):
                while uf[x] != x:
                    uf[x] = uf[uf[x]]; x = uf[x]
                return x

            def _union(x, y):
                rx, ry = _find(x), _find(y)
                if rx != ry: uf[ry] = rx

            for i in range(len(lamellae)):
                for j in range(i + 1, len(lamellae)):
                    G_ma_i = lamellae[i]['G_ma']
                    G_ma_j = lamellae[j]['G_ma']
                    if G_ma_i is None or G_ma_j is None:
                        continue
                    d = _disor(G_ma_i, G_ma_j, symops_ma)
                    if d < ma_threshold_deg:
                        _union(i, j)

            # Build variant groups
            groups = defaultdict(list)
            for i in range(len(lamellae)):
                groups[_find(i)].append(i)

            vgroups = []
            for groot, members in groups.items():
                # Representative = lamella with most pixels
                rep_idx = max(members,
                            key=lambda i: lamellae[i]['n_pixels'])
                rep     = lamellae[rep_idx]
                strains = [lamellae[i]['strain'] for i in members
                        if lamellae[i]['strain'] is not None]
                vgroups.append({
                    'rep_cid':      rep['cid'],
                    'rep_pid':      rep['pid'],
                    'G_au':         rep['G_au'],
                    'G_ma':         rep['G_ma'],
                    'mean_strain':  float(np.mean(strains)) if strains else None,
                    'std_strain':   float(np.std(strains))  if len(strains) > 1
                                    else 0.0,
                    'n_members':    len(members),
                    'member_cids':  [lamellae[i]['cid'] for i in members],
                })

            grain_vgroups[root_id] = vgroups
            per_grain.append({
                'phys_grain_id':  root_id,
                'n_lamellae':     len(lamellae),
                'n_variants':     len(vgroups),
                'variant_groups': [{k: v for k, v in vg.items()
                                    if k not in ('G_au', 'G_ma')}
                                    for vg in vgroups],
            })

            if verbose:
                print(f"  Grain {root_id}: {len(lamellae)} lamellae → "
                    f"{len(vgroups)} variant groups")

        # ── Build set of meeting grain pairs ─────────────────────────────
        # meeting_pairs: (lam_A, parent_A, lam_B, parent_B, label)
        # Map parent IDs → physical grain root IDs
        pid_to_root = {}
        for root_id, fragments in grain_groups.items():
            for frag in fragments:
                pid_to_root[frag] = root_id
            pid_to_root[root_id] = root_id
        # Also map pre-merge IDs
        for new_id, old_id in reverse_id_map.items():
            if new_id in pid_to_root:
                pid_to_root[old_id] = pid_to_root[new_id]

        meeting_grain_pairs = set()
        for lam_a, pid_a, lam_b, pid_b, _ in (meeting_pairs or []):
            root_a = pid_to_root.get(pid_a, pid_a)
            root_b = pid_to_root.get(pid_b, pid_b)
            if root_a != root_b:
                meeting_grain_pairs.add(
                    (min(root_a, root_b), max(root_a, root_b)))

        # ── Step 2: Cross-boundary variant compatibility ───────────────────
        all_grain_roots   = sorted(grain_vgroups.keys())
        boundary_pairs    = []
        min_or_meeting    = []   # min OR residual for pairs that meet
        min_or_nomeeting  = []   # min OR residual for pairs that don't meet

        for i, root_a in enumerate(all_grain_roots):
            vg_a = grain_vgroups[root_a]
            if not vg_a:
                continue
            for root_b in all_grain_roots[i + 1:]:
                vg_b = grain_vgroups[root_b]
                if not vg_b:
                    continue

                # Only analyse neighbouring grains
                au_d = disor_matrix.get(
                    (min(root_a, root_b), max(root_a, root_b)))
                if au_d is None:
                    continue

                do_meet = (min(root_a, root_b),
                        max(root_a, root_b)) in meeting_grain_pairs

                # All pairwise OR residuals between variant groups
                pair_results = []
                for vg_i in vg_a:
                    for vg_j in vg_b:
                        G_au_i = vg_i.get('G_au')
                        G_ma_i = vg_i.get('G_ma')
                        G_au_j = vg_j.get('G_au')
                        G_ma_j = vg_j.get('G_ma')
                        if any(G is None for G in
                            [G_au_i, G_ma_i, G_au_j, G_ma_j]):
                            continue
                        or_r = _or_residual(G_au_i, G_ma_i,
                                            G_au_j, G_ma_j)
                        strain_delta = (
                            abs(vg_i['mean_strain'] - vg_j['mean_strain'])
                            if (vg_i['mean_strain'] is not None and
                                vg_j['mean_strain'] is not None)
                            else None)
                        pair_results.append({
                            'vg_A_cid':      vg_i['rep_cid'],
                            'vg_B_cid':      vg_j['rep_cid'],
                            'or_residual':   or_r,
                            'compatible':    or_r < or_threshold_deg,
                            'strain_delta':  strain_delta,
                            'strain_A':      vg_i['mean_strain'],
                            'strain_B':      vg_j['mean_strain'],
                        })

                if not pair_results:
                    continue

                min_or = min(p['or_residual'] for p in pair_results)
                n_compat = sum(1 for p in pair_results
                            if p['compatible'])

                boundary_pairs.append({
                    'grain_A':        root_a,
                    'grain_B':        root_b,
                    'au_disor':       au_d,
                    'high_angle':     au_d > 15.0,
                    'do_meet':        do_meet,
                    'n_vg_A':         len(vg_a),
                    'n_vg_B':         len(vg_b),
                    'n_combinations': len(pair_results),
                    'min_or_residual':min_or,
                    'n_compatible':   n_compat,
                    'any_compatible': n_compat > 0,
                    'variant_pairs':  pair_results,
                })

                if do_meet:
                    min_or_meeting.append(min_or)
                else:
                    min_or_nomeeting.append(min_or)

        # ── Step 3: Statistical comparison ───────────────────────────────
        meeting_stats = {
            'n_meeting_pairs':    len(min_or_meeting),
            'n_nomeeting_pairs':  len(min_or_nomeeting),
            'meeting_min_or_mean':
                float(np.mean(min_or_meeting))   if min_or_meeting  else None,
            'meeting_min_or_median':
                float(np.median(min_or_meeting)) if min_or_meeting  else None,
            'nomeeting_min_or_mean':
                float(np.mean(min_or_nomeeting)) if min_or_nomeeting else None,
            'nomeeting_min_or_median':
                float(np.median(min_or_nomeeting)) if min_or_nomeeting else None,
            'mannwhitney_p':      None,
            'mannwhitney_u':      None,
            'interpretation':     'Insufficient data for statistical test',
        }

        if len(min_or_meeting) >= 3 and len(min_or_nomeeting) >= 3:
            try:
                from scipy import stats as _st
                u, p = _st.mannwhitneyu(
                    min_or_meeting, min_or_nomeeting,
                    alternative='less')   # meeting pairs have lower OR residual?
                meeting_stats['mannwhitney_p'] = float(p)
                meeting_stats['mannwhitney_u'] = float(u)
                if p < 0.05:
                    meeting_stats['interpretation'] = (
                        f'Meeting pairs have significantly lower min OR '
                        f'residual than non-meeting pairs (p={p:.4f}): '
                        f'variant compatibility is a driving force for '
                        f'cross-boundary lamella growth.')
                else:
                    meeting_stats['interpretation'] = (
                        f'No significant difference in min OR residual '
                        f'between meeting and non-meeting pairs (p={p:.4f}): '
                        f'variant compatibility does not predict whether '
                        f'lamellae meet at the boundary.')
            except Exception as e:
                meeting_stats['interpretation'] = f'Statistical test failed: {e}'

        # ── Summary ───────────────────────────────────────────────────────
        n_grains_with_hp  = sum(1 for g in per_grain if g['n_lamellae'] > 0)
        all_n_variants    = [g['n_variants'] for g in per_grain
                            if g['n_lamellae'] > 0]
        n_compat_bnd      = sum(1 for bp in boundary_pairs
                                if bp['any_compatible'])

        summary = {
            'n_physical_grains':       len(per_grain),
            'n_grains_with_hp_data':   n_grains_with_hp,
            'mean_variants_per_grain': float(np.mean(all_n_variants))
                                    if all_n_variants else None,
            'max_variants_per_grain':  int(max(all_n_variants))
                                    if all_n_variants else None,
            'n_boundary_pairs_analysed':len(boundary_pairs),
            'n_boundary_pairs_meeting': len(min_or_meeting),
            'n_compatible_boundaries':  n_compat_bnd,
            'pct_compatible_boundaries':float(100 * n_compat_bnd /
                                            len(boundary_pairs))
                                        if boundary_pairs else None,
            'ma_threshold_deg':         ma_threshold_deg,
            'or_threshold_deg':         or_threshold_deg,
        }

        result = {
            'per_grain':      per_grain,
            'boundary_pairs': [{k: v for k, v in bp.items()
                                if k != 'variant_pairs'}
                            for bp in boundary_pairs],
            'boundary_pairs_full': boundary_pairs,
            'meeting_stats':  meeting_stats,
            'summary':        summary,
        }

        # Store on self for use by export_results / export_pdf
        self.variant_analysis = result
        return result
    def export_pdf(self, output_path,
                    parent_grain_groups=None,
                    parent_disor_matrix=None,
                    child_lamella_groups=None,
                    lamellar_child_clusters=None,
                    meeting_pairs=None,
                    or_results=None,
                    boundary_parents=None,
                    boundary_children=None,
                    params=None,
                    results=None,
                    export_tex=True):
        """
        Generate a PDF report from the analysis results.

        All parameters are optional. If not provided explicitly, falls back
        to results embedded on clustering_result, boundary_result, and self.
        Explicit parameters always take priority over embedded attributes.

        If results is None and self.results is not set, calls export_results()
        internally to collect them (no xlsx written).

        Parameters
        ----------
        output_path : str or Path
            Where to write the PDF report.
        parent_grain_groups : dict, optional
            Falls back to clustering_result.parent_grain_groups.
        parent_disor_matrix : dict, optional
            Falls back to clustering_result.parent_disor_matrix.
        child_lamella_groups : dict, optional
            Falls back to clustering_result.child_lamella_groups.
        lamellar_child_clusters : list, optional
            Falls back to clustering_result.lamellar_child_clusters.
        meeting_pairs : list, optional
            Falls back to boundary_result.meeting_pairs.
        or_results : list, optional
            Falls back to self.or_results.
        boundary_parents : set, optional
            Falls back to boundary_result.boundary_parent_clusters.
        boundary_children : set, optional
            Falls back to boundary_result.boundary_child_clusters.
        params : dict, optional
            Falls back to self.params.
        results : dict, optional
            Pre-computed results from export_results().
            Falls back to self.results.
            If None and self.results is not set, export_results() is called.
        export_tex : bool, optional
            Also write .tex source alongside PDF. Default True.

        Returns
        -------
        Path : path to written PDF.

        Example
        -------
        >>> # Minimal — all data embedded on objects, results already computed
        >>> hpa.export_pdf(output_path='report.pdf')

        >>> # Explicit — overrides embedded data
        >>> hpa.export_pdf(
        ...     output_path         = 'report.pdf',
        ...     parent_grain_groups = parent_grain_groups,
        ...     params              = params,
        ... )
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch
        import os
        from pathlib import Path as _Path
        from collections import defaultdict

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                            Image as RLImage, Table, TableStyle,
                                            PageBreak, HRFlowable)
            from reportlab.lib.enums import TA_JUSTIFY
        except ImportError:
            raise ImportError("reportlab is required: pip install reportlab")

        # ── Resolve: explicit > embedded on objects > default ─────────────
        def _resolve(explicit, *fallbacks, default=None):
            if explicit is not None:
                return explicit
            for fb in fallbacks:
                if fb is not None:
                    return fb
            return default

        parent_grain_groups = _resolve(
            parent_grain_groups,
            getattr(self.clustering_result, 'parent_grain_groups', None),
            default={})

        parent_disor_matrix = _resolve(
            parent_disor_matrix,
            getattr(self.clustering_result, 'parent_disor_matrix', None),
            default={})

        child_lamella_groups = _resolve(
            child_lamella_groups,
            getattr(self.clustering_result, 'child_lamella_groups', None),
            default={})

        lamellar_child_clusters = _resolve(
            lamellar_child_clusters,
            getattr(self.clustering_result, 'lamellar_child_clusters', None),
            default=None)

        meeting_pairs = _resolve(
            meeting_pairs,
            getattr(self.boundary_result, 'meeting_pairs',  None),
            getattr(self,                 'meeting_pairs',  None),
            default=[])

        or_results = _resolve(
            or_results,
            getattr(self, 'or_results', None),
            default=[])

        boundary_parents = _resolve(
            boundary_parents,
            getattr(self.boundary_result, 'boundary_parent_clusters', None),
            default=set())

        boundary_children = _resolve(
            boundary_children,
            getattr(self.boundary_result, 'boundary_child_clusters', None),
            default=set())

        params = _resolve(
            params,
            getattr(self, 'params', None),
            default={})

        results = _resolve(
            results,
            getattr(self, 'results', None),
            default=None)

        or_threshold_deg = params.get('or_preserved_deg', 3.0)

        # ── Collect results if still None ─────────────────────────────────
        if results is None:
            results = self.export_results(
                output_path          = str(output_path).replace('.pdf', '.json'),
                parent_grain_groups  = parent_grain_groups,
                parent_disor_matrix  = parent_disor_matrix,
                child_lamella_groups = child_lamella_groups,
                lamellar_child_clusters = lamellar_child_clusters,
                meeting_pairs        = meeting_pairs,
                or_results           = or_results,
                boundary_parents     = boundary_parents,
                boundary_children    = boundary_children,
                params               = params,
            )

        # ── rest of export_pdf body unchanged from here ───────────────────
        # (COLORS, _savefig, fig_dir, all figure generation,
        #  PDF story building, _write_tex call — all identical)
        # Only internal references to grain_groups/disor_matrix/lamella_groups
        # need updating to parent_grain_groups/parent_disor_matrix/child_lamella_groups    

        if meeting_pairs is None:
            meeting_pairs = getattr(self, 'meeting_pairs', [])
        if or_results is None:
            or_results = getattr(self, 'or_results', [])
        if boundary_parents is None:
            boundary_parents = set()
        if params is None:
            params = {}
        or_threshold_deg = params.get('or_preserved_deg', 3.0)
        # ── Helpers ───────────────────────────────────────────────────────
        COLORS = {
            'primary':   '#2C5F8A',
            'secondary': '#E07B39',
            'green':     '#4A9B6F',
            'red':       '#C0392B',
            'purple':    '#7B5EA7',
            'grey':      '#7F8C8D',
        }
        #tmpdir   = tempfile.mkdtemp()

        # ── Figure output directory ───────────────────────────────────────────
        out     = _Path(output_path)
        fig_dir = out.parent / 'figs'
        fig_dir.mkdir(parents=True, exist_ok=True)
        figpaths = []
        def _fmt(v, decimals=3):
            if v is None: return '—'
            try:
                return f'{float(v):.{decimals}f}'
            except (TypeError, ValueError):
                return str(v)
        def _savefig(name, fig, dpi=150):
            p = os.path.join(fig_dir, f'{name}.png')
            fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            figpaths.append(p)
            return p

        # ── Figure 1: Grain reconstruction ────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle('Physical Grain and Lamella Reconstruction',
                    fontsize=13, fontweight='bold')

        # 1a: pairwise disorientation histogram
        ax = axes[0]
        disor_vals = list(parent_disor_matrix.values())
        ax.hist(disor_vals, bins=35, color=COLORS['primary'],
                edgecolor='white', alpha=0.85)
        ax.axvline(params.get('frag_merge_deg', 2.0),
                color=COLORS['red'], lw=2, ls='--',
                label=f'Merge threshold ({params.get("frag_merge_deg",2.0)}°)')
        ax.set_xlabel('Disorientation (deg)'); ax.set_ylabel('Count')
        ax.set_title(
            f'Pairwise Au Disorientation\n'
            f'{results["clustering"]["austenite"]["n_cluster_ids"]} IDs → '
            f'{results["clustering"]["austenite"]["n_physical_grains"]} physical grains')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

        # 1b: fragment group sizes
        ax = axes[1]
        frag_sizes = [len(v) for v in parent_grain_groups.values()]
        from collections import Counter
        cnt = Counter(frag_sizes)
        ax.bar([str(k) for k in sorted(cnt)],
            [cnt[k] for k in sorted(cnt)],
            color=COLORS['secondary'], edgecolor='white', alpha=0.85)
        ax.set_xlabel('Fragments per physical grain')
        ax.set_ylabel('N physical grains')
        ax.set_title('Grain Fragmentation')
        ax.grid(axis='y', alpha=0.3)

        # 1c: lamella fragmentation
        ax = axes[2]
        lam_sizes = [len(v) for v in child_lamella_groups.values()]
        cnt2 = Counter(lam_sizes)
        ax.bar([str(k) for k in sorted(cnt2)],
            [cnt2[k] for k in sorted(cnt2)],
            color=COLORS['green'], edgecolor='white', alpha=0.85)
        ax.set_xlabel('Fragments per physical lamella')
        ax.set_ylabel('N physical lamellae')
        ax.set_title('Lamella Fragmentation')
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        fig1 = _savefig('fig1_reconstruction', fig)

        # ── Figure 2: HP statistics ───────────────────────────────────────
        hp = results.get('hp_statistics', {})
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle('Habit Plane Statistics', fontsize=13, fontweight='bold')

        # 2a: HP family distribution
        ax = axes[0]
        au_fam = hp.get('habit_plane_families', {}).get('austenite', {})
        if au_fam:
            fams = list(au_fam.keys())[:8]
            vals = [au_fam[f] for f in fams]
            bars = ax.bar(fams, vals, color=COLORS['primary'],
                        edgecolor='white', alpha=0.85)
            ax.tick_params(axis='x', rotation=35)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.3, str(val),
                        ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('Count'); ax.set_xlabel('HP Family (Austenite)')
        ax.set_title('Habit Plane Families (rank-1)')
        ax.grid(axis='y', alpha=0.3)

        # 2b: OOP angle distribution
        ax = axes[1]
        oop_data = hp.get('oop', {})
        # Reconstruct from self.HP
        oop_vals = []
        for pid in self.HP:
            for cid in self.HP[pid]:
                for roi in ('semi_roi_1','semi_roi_2'):
                    if roi not in self.HP[pid][cid]: continue
                    rd = self.HP[pid][cid][roi]
                    hp_m = rd.get('Interfaces',{}).get('HP_matches',{})
                    nvecs = hp_m.get(self.keyau,{}).get(
                        'Habit plane normal',{}).get('n_vec_sampl',[])
                    if len(nvecs) > 0:
                        v = np.array(nvecs[0])
                        n = np.linalg.norm(v)
                        if n > 1e-10:
                            oop_vals.append(float(np.degrees(
                                np.arccos(np.clip(abs(v[2]/n),0,1)))))
                        break
        if oop_vals:
            ax.hist(oop_vals, bins=30, color=COLORS['purple'],
                    edgecolor='white', alpha=0.85)
            ax.axvline(params.get('oop_flag_deg', 20.0),
                    color=COLORS['red'], lw=2, ls='--',
                    label=f'Flag threshold ({params.get("oop_flag_deg",20.0)}°)')
            ax.legend(fontsize=8)
        ax.set_xlabel('OOP angle (deg)'); ax.set_ylabel('Count')
        ax.set_title(
            f'Out-of-Plane Angle\n'
            f'(n_flagged={oop_data.get("n_flagged","?")})')
        ax.grid(axis='y', alpha=0.3)

        # 2c: Score distribution
        ax = axes[2]
        sc_r1  = hp.get('score', {})
        all_scores, r1_scores = [], []
        for pid in self.HP:
            for cid in self.HP[pid]:
                for roi in ('semi_roi_1','semi_roi_2'):
                    if roi not in self.HP[pid][cid]: continue
                    rd  = self.HP[pid][cid][roi]
                    scs = rd.get('Interfaces',{}).get(
                        'HP_matches',{}).get('Score',{}).get('overall',[])
                    all_scores.extend([float(s) for s in scs])
                    if scs: r1_scores.append(float(scs[0]))
                    break
        if all_scores:
            bins = np.linspace(0, max(all_scores), 35)
            ax.hist(all_scores, bins=bins, color=COLORS['grey'],
                    alpha=0.6, edgecolor='white', label='All candidates')
            ax.hist(r1_scores, bins=bins, color=COLORS['primary'],
                    alpha=0.85, edgecolor='white', label='Rank-1')
        ax.set_xlabel('Score'); ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig2 = _savefig('fig2_hp_stats', fig)

        # ── Figure 3: Schmid-factor analysis ─────────────────────────────
        sc = results.get('schmid_analysis', {})
        pg = sc.get('per_grain', [])
        fig = plt.figure(figsize=(14, 9))
        gs3 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)
        fig.suptitle('Per-Grain Schmid-Factor Analysis',
                    fontsize=13, fontweight='bold')

        def _scatter(ax, xs, ys, xlabel, ylabel, title,
                    color=COLORS['primary'], label=None):
            xs = [x for x,y in zip(xs,ys) if x is not None and y is not None]
            ys = [y for x,y in zip(xs,ys) if x is not None and y is not None]
            if not xs: return
            ax.scatter(xs, ys, s=35, alpha=0.7, color=color,
                    edgecolors='none', label=label)
            try:
                from scipy import stats as _st
                r, p = _st.spearmanr(xs, ys)
                sig = ('***' if p<0.001 else '**' if p<0.01
                    else '*' if p<0.05 else 'ns')
                ax.set_title(f'{title}\nr_s={r:.3f} {sig}', fontsize=10)
            except Exception:
                ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(alpha=0.25)

        sizes   = [g['apparent_size_px'] for g in pg]
        nlams   = [g['n_lamellae']       for g in pg]
        mstrain = [g['mean_strain']      for g in pg]
        sstrain = [g['std_strain']       for g in pg]
        dens    = [g['lamella_density']  for g in pg]
        is_bnd  = [g['is_boundary']      for g in pg]
        colors_g= [COLORS['secondary'] if b else COLORS['primary']
                    for b in is_bnd]

        ax = fig.add_subplot(gs3[0,0])
        if pg:
            ax.scatter(sizes, nlams, s=35, c=colors_g, alpha=0.75,
                    edgecolors='none')
            try:
                from scipy import stats as _st
                r,p = _st.spearmanr(sizes, nlams)
                sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
                ax.set_title(f'Size vs N Lamellae\nr_s={r:.3f} {sig} ⚠ likely artefact',
                            fontsize=10)
            except Exception:
                ax.set_title('Size vs N Lamellae', fontsize=10)
            ax.legend(handles=[
                Patch(color=COLORS['secondary'], label='Boundary grain'),
                Patch(color=COLORS['primary'],   label='Interior grain')],
                fontsize=7)
        ax.set_xlabel('Apparent size (px)'); ax.set_ylabel('N lamellae')
        ax.grid(alpha=0.25)

        ax = fig.add_subplot(gs3[0,1])
        _scatter(ax, sizes, dens,
                'Apparent size (px)', 'Lamella density (n/1000px)',
                'Size vs Density\n(flat/neg = pure artefact)')

        ax = fig.add_subplot(gs3[0,2])
        _scatter(ax, mstrain, nlams,
                'Mean transformation strain', 'N lamellae',
                'Mean Strain vs N Lamellae')

        ax = fig.add_subplot(gs3[1,0])
        _scatter(ax, sstrain, nlams,
                'Strain std dev within grain', 'N lamellae',
                'Strain Diversity vs N Lamellae\n(multi-variant indicator)')

        ax = fig.add_subplot(gs3[1,1])
        _scatter(ax, mstrain, dens,
                'Mean transformation strain',
                'Lamella density (n/1000px)',
                'Strain vs Lamella Density\n(stereology-corrected)')

        ax = fig.add_subplot(gs3[1,2])
        # Stereotype check summary as text panel
        ax.axis('off')
        stereo = sc.get('stereological', {})
        corrs  = sc.get('correlations', [])
        lines  = ['Stereological artefact check:\n']
        lines.append(
            f"Size vs count:    r_s={stereo.get('size_vs_count_r','?')}"
            f" {'⚠ artefact' if stereo.get('is_pure_artefact') else '✓ real effect'}")
        lines.append(
            f"Size vs density:  r_s={stereo.get('size_vs_density_r','?')}\n")
        lines.append('Spearman correlations:')
        for c in corrs:
            if c['r'] is not None:
                lines.append(
                    f"  {c['label'][:40]}\n"
                    f"    r={c['r']:.3f}  {c['sig']}")
        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
                fontsize=7.5, va='top', ha='left',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#EBF5FB',
                        edgecolor='#2C5F8A', alpha=0.9))
        fig.tight_layout()
        fig3 = _savefig('fig3_schmid', fig)

        # ── Figure 4: OR preservation ─────────────────────────────────────
        if or_results:
            fig4, _ = self.plot_or_histogram(
                or_results,
                threshold_deg=params.get('or_preserved_deg', 3.0))
            fig4_path = _savefig('fig4_or', fig4)
        else:
            fig4_path = None

        # ── Figure 5: boundary meeting pairs ─────────────────────────────
        if or_results:
            or_arr  = np.array([r['or_residual'] for r in or_results])
            au_arr  = np.array([r['au_disor']    for r in or_results])
            ha_arr  = np.array([r['high_angle']  for r in or_results])
            pres    = np.array([r['or_preserved'] for r in or_results])

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('OR Preservation: Au vs Ma Disorientation',
                        fontsize=13, fontweight='bold')

            ax = axes[0]
            ax.scatter(au_arr[~ha_arr], or_arr[~ha_arr],
                    s=40, alpha=0.8, color=COLORS['primary'],
                    edgecolors='none', label='Low-angle (<15°)')
            ax.scatter(au_arr[ha_arr],  or_arr[ha_arr],
                    s=50, alpha=0.8, color=COLORS['red'],
                    marker='*', edgecolors='none',
                    label='High-angle (≥15°)')
            lim = max(au_arr.max(), or_arr.max()) * 1.05
            ax.plot([0,lim],[0,lim],'k--',lw=1,alpha=0.4,
                    label='OR_res = Au_disor')
            ax.axhline(params.get('or_preserved_deg',3.0),
                    color='black', lw=1.5, ls=':',
                    label=f'Threshold ({params.get("or_preserved_deg",3.0)}°)')
            ax.set_xlabel('Austenite disorientation (deg)')
            ax.set_ylabel('OR residual (deg)')
            n_pres = int(pres.sum())
            ax.set_title(
                f'OR Preserved: {n_pres}/{len(or_results)} pairs '
                f'({100*pres.mean():.0f}%)')
            ax.legend(fontsize=8); ax.grid(alpha=0.25)

            ax = axes[1]
            by_bnd = defaultdict(list)
            for r in or_results:
                by_bnd[r['boundary']].append(r)
            bnd_names  = sorted(by_bnd.keys())
            bnd_means  = [np.mean([r['or_residual'] for r in by_bnd[b]])
                        for b in bnd_names]
            bnd_colors = [COLORS['red']
                        if by_bnd[b][0]['high_angle']
                        else COLORS['primary']
                        for b in bnd_names]
            ax.bar(range(len(bnd_names)), bnd_means,
                color=bnd_colors, edgecolor='white', alpha=0.85)
            ax.axhline(params.get('or_preserved_deg',3.0),
                    color='black', lw=1.5, ls='--', alpha=0.7)
            ax.set_xticks(range(len(bnd_names)))
            ax.set_xticklabels(
                [b.replace('g','G').replace('-','\n')
                for b in bnd_names], fontsize=7.5)
            ax.set_ylabel('Mean OR residual (deg)')
            ax.set_title('OR Residual per Boundary')
            ax.legend(handles=[
                Patch(color=COLORS['red'],   label='High-angle'),
                Patch(color=COLORS['primary'],label='Low-angle')],
                fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            fig5_path = _savefig('fig5_or_by_boundary', fig)
        else:
            fig5_path = None

        # ── Figure 6: Variant compatibility ──────────────────────────────
        vc = results.get('variant_analysis', {})
        vc_summary  = vc.get('summary', {})
        vc_bnd      = vc.get('boundary_pairs', [])
        vc_meeting  = vc.get('meeting_stats', {})

        if vc and vc_bnd:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Variant Compatibility Analysis',
                        fontsize=13, fontweight='bold')

            # 6a: variants per grain histogram
            ax = axes[0]
            n_vars = [g['n_variants'] for g in vc.get('per_grain', [])
                    if g['n_lamellae'] > 0]
            if n_vars:
                from collections import Counter
                cnt = Counter(n_vars)
                ax.bar([str(k) for k in sorted(cnt)],
                    [cnt[k] for k in sorted(cnt)],
                    color=COLORS['primary'], edgecolor='white', alpha=0.85)
            ax.set_xlabel('Distinct variants per grain')
            ax.set_ylabel('N grains')
            ax.set_title(
                f'Variant Groups per Grain\n'
                f'mean={_fmt(vc_summary.get("mean_variants_per_grain"), 1)}  '
                f'max={vc_summary.get("max_variants_per_grain","?")}')
            ax.grid(axis='y', alpha=0.3)

            # 6b: min OR residual — meeting vs non-meeting
            ax = axes[1]
            meeting_or  = [bp['min_or_residual'] for bp in vc_bnd
                        if bp['do_meet']]
            nomeeting_or= [bp['min_or_residual'] for bp in vc_bnd
                        if not bp['do_meet']]
            bins = np.linspace(0, max(
                (max(meeting_or)  if meeting_or  else 0),
                (max(nomeeting_or) if nomeeting_or else 0),
                1), 25)
            if nomeeting_or:
                ax.hist(nomeeting_or, bins=bins,
                        color=COLORS['primary'], alpha=0.6,
                        edgecolor='white',
                        label=f'No meeting (n={len(nomeeting_or)})')
            if meeting_or:
                ax.hist(meeting_or, bins=bins,
                        color=COLORS['secondary'], alpha=0.85,
                        edgecolor='white',
                        label=f'Meeting (n={len(meeting_or)})')
            ax.axvline(or_threshold_deg, color='black',
                    lw=1.5, ls='--',
                    label=f'Threshold ({or_threshold_deg}°)')
            p_val = vc_meeting.get('mannwhitney_p')
            p_str = f'p={p_val:.3f}' if p_val is not None else 'p=n/a'
            ax.set_xlabel('Min OR residual (deg)')
            ax.set_ylabel('Count')
            ax.set_title(
                f'Min OR Residual: Meeting vs Non-meeting\n'
                f'Mann-Whitney {p_str}')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

            # 6c: min OR residual vs Au disorientation scatter
            ax = axes[2]
            for bp in vc_bnd:
                color  = COLORS['secondary'] if bp['do_meet'] \
                        else COLORS['primary']
                marker = 'D' if bp['do_meet'] else 'o'
                alpha  = 0.9 if bp['do_meet'] else 0.5
                size   = 50  if bp['do_meet'] else 25
                ax.scatter(bp['au_disor'], bp['min_or_residual'],
                        c=color, marker=marker,
                        alpha=alpha, s=size, edgecolors='none')
            ax.axhline(or_threshold_deg, color='black',
                    lw=1.5, ls='--', alpha=0.7)
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(color=COLORS['secondary'], label='Meeting pairs'),
                Patch(color=COLORS['primary'],   label='Non-meeting pairs')],
                fontsize=8)
            ax.set_xlabel('Austenite disorientation (deg)')
            ax.set_ylabel('Min OR residual across variant groups (deg)')
            ax.set_title('Variant Compatibility vs Boundary Character')
            ax.grid(alpha=0.25)

            fig.tight_layout()
            fig6_path = _savefig('fig6_variant_compatibility', fig)
        else:
            fig6_path = None
        # ── Build PDF ─────────────────────────────────────────────────────
        out     = _Path(output_path)
        doc     = SimpleDocTemplate(
            str(out), pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2.2*cm, bottomMargin=2.2*cm,
            title='EBSD Habit Plane Analysis Report')
        W, H    = A4
        TW      = W - 4*cm

        styles  = getSampleStyleSheet()
        s_title = ParagraphStyle('T', parent=styles['Title'],
                                fontSize=20, leading=26, spaceAfter=6)
        s_h1    = ParagraphStyle('H1', parent=styles['Heading1'],
                                fontSize=13, spaceBefore=14, spaceAfter=5,
                                textColor=colors.HexColor('#2C5F8A'))
        s_h2    = ParagraphStyle('H2', parent=styles['Heading2'],
                                fontSize=10.5, spaceBefore=8, spaceAfter=4,
                                textColor=colors.HexColor('#4A4A4A'))
        s_body  = ParagraphStyle('B', parent=styles['Normal'],
                                fontSize=10, leading=15, spaceAfter=6,
                                alignment=TA_JUSTIFY)
        s_note  = ParagraphStyle('N', parent=styles['Normal'],
                                fontSize=9, leading=13, spaceAfter=5,
                                textColor=colors.HexColor('#666666'),
                                alignment=TA_JUSTIFY)

        def _img(path, width=TW):
            if not path or not os.path.exists(path):
                return Spacer(1, 0.5*cm)
            from PIL import Image as PILImage
            im = PILImage.open(path)
            w0, h0 = im.size
            return RLImage(path, width=width, height=width*h0/w0)

        def _tbl(headers, rows, col_widths=None):
            data = [headers] + rows
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ('BACKGROUND',  (0,0),(-1,0), colors.HexColor('#2C5F8A')),
                ('TEXTCOLOR',   (0,0),(-1,0), colors.white),
                ('FONTNAME',    (0,0),(-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',    (0,0),(-1,-1), 9),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),
                [colors.white, colors.HexColor('#EBF5FB')]),
                ('GRID', (0,0),(-1,-1), 0.4, colors.HexColor('#BBBBBB')),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('TOPPADDING',(0,0),(-1,-1),4),
                ('BOTTOMPADDING',(0,0),(-1,-1),4),
                ('LEFTPADDING',(0,0),(-1,-1),6),
            ]))
            return t

        story = []

        # Cover
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph('EBSD Habit Plane Analysis', s_title))
        story.append(HRFlowable(width=TW, thickness=2,
                                color=colors.HexColor('#2C5F8A'),
                                spaceAfter=10))
        meta = results['metadata']
        story.append(Paragraph(
            f'Parent phase: <b>{meta["parent_phase"]}</b> | '
            f'Child phase: <b>{meta["child_phase"]}</b> | '
            f'Scan points: <b>{meta["n_scan_points"]:,}</b>',
            s_body))
        story.append(Spacer(1, 0.4*cm))

        # Dataset summary table
        clust = results['clustering']
        au    = clust['austenite']
        ma    = clust['martensite']
        sum_rows = [
            ['Austenite cluster IDs',      str(au['n_cluster_ids'])],
            ['Physical austenite grains',  str(au['n_physical_grains'])],
            ['Multi-fragment grains',      str(au['n_multi_fragment'])],
            ['Boundary grains (truncated)',str(au['n_boundary_grains'])],
            ['Martensite cluster IDs',     str(ma['n_cluster_ids'])],
            ['Physical lamellae',          str(ma['n_physical_lamellae'])],
            ['Multi-fragment lamellae',    str(ma['n_multi_fragment'])],
            ['Boundary lamellae (truncated)',str(ma['n_boundary_lamellae'])],
        ]
        if hp.get('subsets'):
            sum_rows += [
                ['HP interfaces (rank-1)',
                str(hp.get('n_interfaces_rank1','?'))],
                ['Fit lattice correspondence (rank-1)',
                f'{hp.get("score",{}).get("n_fit_yes","?")} '
                f'({hp.get("score",{}).get("pct_fit_yes","?"):.0f}%)'
                if isinstance(hp.get("score",{}).get("pct_fit_yes"), float)
                else '?'],
            ]
        story.append(_tbl(['Parameter','Value'], sum_rows,
                        col_widths=[10*cm, 6*cm]))
        story.append(PageBreak())

        # Section 1: Reconstruction
        story.append(Paragraph('1. Physical Grain and Lamella Reconstruction',s_h1))
        story.append(Paragraph(
            f'Using a fragment-merge threshold of '
            f'{params.get("frag_merge_deg",2.0)}° for austenite and '
            f'{params.get("lamella_merge_deg",2.0)}° for martensite, '
            f'{au["n_cluster_ids"]} austenite cluster IDs collapsed to '
            f'{au["n_physical_grains"]} physical grains '
            f'({au["n_multi_fragment"]} multi-fragment), and '
            f'{ma["n_cluster_ids"]} martensite IDs to '
            f'{ma["n_physical_lamellae"]} physical lamellae '
            f'({ma["n_multi_fragment"]} multi-fragment). '
            f'{au["n_boundary_grains"]} grains touch the scan boundary and '
            f'have underestimated lamella counts.', s_body))
        story.append(_img(fig1))
        if au.get('multi_fragment_groups'):
            frag_rows = [
                [str(gid), str(frags), str(len(frags))]
                for gid, frags in au['multi_fragment_groups'].items()]
            story.append(Paragraph('Multi-fragment austenite grains:', s_h2))
            story.append(_tbl(['Root ID','Fragment IDs','N fragments'],
                                frag_rows,
                                col_widths=[3*cm, 10*cm, 3*cm]))
        story.append(PageBreak())

        # Section 2: HP statistics
        story.append(Paragraph('2. Habit Plane Statistics', s_h1))
        sc_data = hp.get('score', {})
        story.append(Paragraph(
            f'Rank-1 candidates: <b>{hp.get("n_interfaces_rank1","?")}</b> '
            f'from {hp.get("n_candidates_total","?")} total. '
            f'Mean score (rank-1): '
            f'{sc_data.get("rank1_mean","?"):.2f} ± '
            f'{sc_data.get("rank1_std","?"):.2f}. '
            f'Fit=Yes: {sc_data.get("n_fit_yes","?")} '
            f'({sc_data.get("pct_fit_yes","?"):.0f}%). '
            f'OOP flagged (≤{params.get("oop_flag_deg",20.0)}°): '
            f'{hp.get("oop",{}).get("n_flagged","?")} '
            f'({hp.get("oop",{}).get("pct_flagged","?"):.0f}%).',
            s_body))
        story.append(_img(fig2))
        fam = hp.get('habit_plane_families', {}).get('austenite', {})
        if fam:
            n_r1 = hp.get('n_interfaces_rank1', 1) or 1
            fam_rows = [[f, str(c), f'{100*c/n_r1:.1f}%']
                        for f, c in list(fam.items())[:8]]
            story.append(_tbl(
                ['HP Family (Austenite)','Count','%'],
                fam_rows, col_widths=[6*cm, 4*cm, 4*cm]))
        story.append(Paragraph(
            'LCV numbers are grain-specific — cross-grain LCV comparisons '
            'are INVALID. Use or_residual for cross-grain variant comparison.',
            s_note))
        story.append(PageBreak())

        # Section 3: Schmid
        story.append(Paragraph('3. Per-Grain Schmid-Factor Analysis', s_h1))
        stereo = sc.get('stereological', {})
        story.append(Paragraph(
            f'Analysis covers {sc.get("n_physical_grains","?")} physical grains '
            f'({sc.get("n_interior","?")} interior, '
            f'{sc.get("n_boundary","?")} boundary-truncated). '
            f'Stereological check: size vs count '
            f'r_s={_fmt(stereo.get("size_vs_count_r"))}, '
            f'size vs density '
            f'r_s={_fmt(stereo.get("size_vs_density_r"))} — '
            f'{"<b>pure artefact</b>" if stereo.get("is_pure_artefact") else "<b>possible real effect</b>"}.',
            s_body))
        story.append(_img(fig3))
        if sc.get('correlations'):
            corr_rows = [
                [c['label'][:55],
                f'{c["r"]:.3f}' if c['r'] is not None else '—',
                c['sig'], str(c['n'])]
                for c in sc['correlations']]
            story.append(_tbl(
                ['Correlation', 'Spearman r', 'Sig.', 'N'],
                corr_rows, col_widths=[9*cm, 2.5*cm, 2*cm, 2.5*cm]))
        for note in sc.get('notes', []):
            story.append(Paragraph(f'• {note}', s_note))
        story.append(PageBreak())

        # Section 4: OR preservation
        story.append(Paragraph('4. OR Preservation Across Grain Boundaries',s_h1))
        or_sec = results.get('or_preservation', {})
        if not or_results:
            story.append(Paragraph(
                'OR preservation analysis not available — run '
                'hpa.analyse_or_preservation() first.', s_body))
        else:
            ov = or_sec.get('overall', {})
            la = or_sec.get('low_angle_boundaries', {})
            ha = or_sec.get('high_angle_boundaries', {})
            story.append(Paragraph(
                f'<b>{or_sec.get("n_meeting_pairs_detected","?")} '
                f'boundary-meeting pairs detected</b> '
                f'(tip distance ≤ {params.get("tip_distance_px",15.0):.0f}px). '
                f'{or_sec.get("n_pairs_evaluated","?")} evaluated. '
                f'OR preservation threshold: '
                f'{params.get("or_preserved_deg",3.0)}°.',
                s_body))
            story.append(Paragraph(
                'Same macroscopic variant (|ma_disor - au_disor| &lt; threshold) '
                'indicates both lamellae have the same absolute orientation in the '
                'sample frame — same habit plane direction — but does not guarantee '
                'the same lattice correspondence. OR_res distinguishes these cases.',
                s_note))
            mv   = or_sec.get('macro_variant', {})
            or_rows = [
                ['All pairs',
                str(ov.get('n_preserved', '?')),
                _fmt(ov.get('pct_preserved'), 0) + '%',
                _fmt(ov.get('or_residual_mean'), 1) + '°'],

                ['Low-angle boundaries (<15°)',
                str(la.get('n_preserved', '?')),
                _fmt(la.get('pct_preserved'), 0) + '%',
                _fmt(la.get('or_residual_mean'), 1) + '°'
                if la.get('or_residual_mean') else '—'],

                ['High-angle boundaries (≥15°)',
                str(ha.get('n_preserved', '?')),
                _fmt(ha.get('pct_preserved'), 0) + '%',
                _fmt(ha.get('or_residual_mean'), 1) + '°'
                if ha.get('or_residual_mean') else '—'],

                ['', '', '', ''],   # spacer row

                ['Same macroscopic variant (|ma-au| < thresh)',
                str(mv.get('n_same_macro', '?')),
                '—', '—'],

                ['  → OR also preserved (continuous growth)',
                str(mv.get('n_same_macro_or_preserved', '?')),
                '—', '—'],

                ['  → OR NOT preserved (same habit plane, diff. correspondence)',
                str(mv.get('n_same_macro_or_not', '?')),
                '—', '—'],
            ]
            story.append(_tbl(
                ['Group', 'N', '% preserved', 'Mean OR residual'],
                or_rows,
                col_widths=[8*cm, 2*cm, 2.5*cm, 3*cm]))
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(
                mv.get('interpretation', ''), s_note))
            if fig4_path:
                story.append(_img(fig4_path))
            if fig5_path:
                story.append(_img(fig5_path))
            # Per-boundary table
            by_bnd = or_sec.get('by_boundary', {})
            if by_bnd:
                bnd_rows = [
                    [bnd,
                    f'{v["au_disor_deg"]:.1f}°',
                    'Yes' if v['high_angle'] else 'No',
                    str(v['n_pairs']),
                    f'{v["or_residual_mean"]:.2f}°',
                    f'{v["or_residual_min"]:.2f}°',
                    str(v['n_preserved'])]
                    for bnd, v in sorted(by_bnd.items())]
                story.append(Paragraph('Results per boundary:', s_h2))
                story.append(_tbl(
                    ['Boundary','Au disor','HAB','N pairs',
                    'OR res mean','OR res min','N preserved'],
                    bnd_rows,
                    col_widths=[3*cm,2.2*cm,1.5*cm,2*cm,
                                2.5*cm,2.5*cm,2.5*cm]))

            interp = or_sec.get('interpretation', {})
            for key in ('low_angle','high_angle','lcv_warning'):
                if key in interp:
                    story.append(Paragraph(
                        f'• {interp[key]}', s_note))
        story.append(PageBreak())
        
        # Section 5: Variant compatibility
        story.append(Paragraph(
            '5. Variant Compatibility Analysis', s_h1))

        if not vc or not vc_bnd:
            story.append(Paragraph(
                'Variant compatibility analysis not available — '
                'run hpa.analyse_variant_compatibility() first.',
                s_body))
        else:
            story.append(Paragraph(
                f'Within each physical grain, lamellae were grouped into '
                f'crystallographically distinct martensite variants using a '
                f'{params.get("ma_threshold_deg", 2.0)}° disorientation '
                f'threshold. '
                f'Mean variants per grain: '
                f'<b>{_fmt(vc_summary.get("mean_variants_per_grain"), 1)}</b> '
                f'(max: {vc_summary.get("max_variants_per_grain","?")}). '
                f'{vc_summary.get("n_boundary_pairs_analysed","?")} '
                f'neighbouring grain pairs analysed. '
                f'Compatible variant combinations (OR residual &lt; '
                f'{params.get("or_preserved_deg", 3.0)}°) found at '
                f'{vc_summary.get("n_compatible_boundaries","?")} boundaries '
                f'({_fmt(vc_summary.get("pct_compatible_boundaries"), 0)}%).',
                s_body))
            story.append(_img(fig6_path))

            # Meeting vs non-meeting table
            vc_rows = [
                ['Meeting pairs',
                str(vc_meeting.get('n_meeting_pairs','?')),
                _fmt(vc_meeting.get('meeting_min_or_mean'), 1) + '°',
                _fmt(vc_meeting.get('meeting_min_or_median'), 1) + '°'],
                ['Non-meeting pairs',
                str(vc_meeting.get('n_nomeeting_pairs','?')),
                _fmt(vc_meeting.get('nomeeting_min_or_mean'), 1) + '°',
                _fmt(vc_meeting.get('nomeeting_min_or_median'), 1) + '°'],
            ]
            story.append(_tbl(
                ['Group', 'N pairs', 'Mean min OR res', 'Median min OR res'],
                vc_rows, col_widths=[5*cm, 2.5*cm, 3.5*cm, 3.5*cm]))
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(
                f'<b>Statistical test:</b> '
                f'{vc_meeting.get("interpretation", "—")}',
                s_body))

        story.append(PageBreak())

        # Section 6: warnings  (renumber from 5 to 6)
        story.append(Paragraph(
            '6. Methodological Notes and Warnings', s_h1))
        # Section 5: warnings
        story.append(Paragraph('Methodological Notes and Warnings', s_h1))
        for w in results.get('warnings', []):
            story.append(Paragraph(f'• {w}', s_note))

        # Build
        doc.build(story)

        

        print(f"PDF written to {out}")
        # After all figures are saved, collect paths
        fig_paths = {
            'fig1': fig1,
            'fig2': fig2,
            'fig3': fig3,
            'fig4': fig4_path,
            'fig5': fig5_path,
            'fig6': fig6_path,
        }

        # Write LaTeX if requested
        if export_tex:
            tex_path = _Path(str(output_path).replace('.pdf', '.tex'))
            self._write_tex(tex_path, results, fig_paths, params)
        return out

    def _write_tex(self, tex_path, results, fig_paths, params):
        """
        Write a LaTeX source file alongside the PDF.

        Uses the same figures and results dict as export_pdf().
        The tex file is self-contained with \\includegraphics for each figure.
        Figures are referenced by relative path — keep them in the same
        directory as the tex file.

        Parameters
        ----------
        tex_path : Path
            Output .tex file path.
        results : dict
            Output of export_results().
        fig_paths : dict
            {'fig1': path, 'fig2': path, ...} — figure PNG paths.
        params : dict
            Analysis parameters.
        """
        from pathlib import Path as _Path
        import os

        tex_path = _Path(tex_path)
        tex_dir  = tex_path.parent

        # Make figure paths relative to tex file
        def _relfig(p):
            if p is None: return None
            try:
                return os.path.relpath(p, tex_dir)
            except ValueError:
                return p  # Windows cross-drive fallback

        meta   = results.get('metadata', {})
        clust  = results.get('clustering', {})
        au     = clust.get('austenite', {})
        ma     = clust.get('martensite', {})
        hp     = results.get('hp_statistics', {})
        sc     = results.get('schmid_analysis', {})
        or_sec = results.get('or_preservation', {})
        stereo = sc.get('stereological', {})

        def _fmt(v, decimals=3):
            if v is None: return '—'
            if isinstance(v, float): return f'{v:.{decimals}f}'
            return str(v)

        def _sig(p):
            if p is None: return 'n/a'
            return ('***' if p<0.001 else '**' if p<0.01
                    else '*' if p<0.05 else 'ns')

        def _fig(path, caption, label, width=r'\textwidth'):
            if path is None: return ''
            rp = _relfig(path).replace('\\','/')
            return (
                f'\\begin{{figure}}[htbp]\n'
                f'  \\centering\n'
                f'  \\includegraphics[width={width}]{{{rp}}}\n'
                f'  \\caption{{{caption}}}\n'
                f'  \\label{{{label}}}\n'
                f'\\end{{figure}}\n')

        def _tbl_tex(header_cols, rows, col_fmt=None):
            """Build a simple LaTeX table."""
            n = len(header_cols)
            if col_fmt is None:
                col_fmt = 'l' + 'r' * (n - 1)
            lines = [
                f'\\begin{{tabular}}{{{col_fmt}}}',
                '  \\toprule',
                '  ' + ' & '.join(
                    f'\\textbf{{{h}}}' for h in header_cols) + ' \\\\',
                '  \\midrule',
            ]
            for row in rows:
                lines.append('  ' + ' & '.join(str(c) for c in row) + ' \\\\')
            lines += ['  \\bottomrule', '\\end{tabular}']
            return '\n'.join(lines)

        # ── Preamble ──────────────────────────────────────────────────────
        lines = [
            r'\documentclass[11pt,a4paper]{article}',
            r'\usepackage[utf8]{inputenc}',
            r'\usepackage[T1]{fontenc}',
            r'\usepackage{graphicx}',
            r'\usepackage{booktabs}',
            r'\usepackage{amsmath}',
            r'\usepackage{hyperref}',
            r'\usepackage{geometry}',
            r'\usepackage{xcolor}',
            r'\usepackage{float}',
            r'\definecolor{ebsdblue}{HTML}{2C5F8A}',
            r'\geometry{margin=2.5cm}',
            r'\hypersetup{colorlinks=true, linkcolor=ebsdblue, '
            r'urlcolor=ebsdblue, citecolor=ebsdblue}',
            r'\title{\textbf{EBSD Habit Plane Analysis Report}}',
            r'\author{}',
            r'\date{\today}',
            r'\begin{document}',
            r'\maketitle',
            r'\tableofcontents',
            r'\newpage',
            '',
        ]

        # ── Section 1: Dataset overview ───────────────────────────────────
        lines += [
            r'\section{Dataset Overview}',
            '',
            f'Parent phase: \\textbf{{{meta.get("parent_phase","?")}}} | '
            f'Child phase: \\textbf{{{meta.get("child_phase","?")}}} | '
            f'Scan points: \\textbf{{{meta.get("n_scan_points",0):,}}}.',
            '',
            r'\subsection*{Parameters}',
            r'\begin{itemize}',
        ]
        for k, v in (params or {}).items():
            lines.append(f'  \\item {k.replace("_","~")} = {v}')
        lines += [r'\end{itemize}', '']

        # Dataset summary table
        sum_rows = [
            ['Austenite cluster IDs',         au.get('n_cluster_ids','?')],
            ['Physical austenite grains',      au.get('n_physical_grains','?')],
            ['Multi-fragment grains',          au.get('n_multi_fragment','?')],
            ['Boundary grains (truncated)',    au.get('n_boundary_grains','?')],
            ['Martensite cluster IDs',         ma.get('n_cluster_ids','?')],
            ['Physical lamellae',              ma.get('n_physical_lamellae','?')],
            ['Multi-fragment lamellae',        ma.get('n_multi_fragment','?')],
            ['Boundary lamellae',              ma.get('n_boundary_lamellae','?')],
            ['HP interfaces (rank-1)',         hp.get('n_interfaces_rank1','?')],
        ]
        lines += [
            r'\begin{table}[htbp]',
            r'  \centering',
            r'  \caption{Dataset summary}',
            r'  \label{tab:summary}',
            '  ' + _tbl_tex(['Parameter','Value'], sum_rows, 'lr'),
            r'\end{table}',
            '',
        ]

        # ── Section 2: Reconstruction ─────────────────────────────────────
        lines += [
            r'\newpage',
            r'\section{Physical Grain and Lamella Reconstruction}',
            '',
            f'Using a fragment-merge threshold of '
            f'{params.get("frag_merge_deg", 2.0)}\\textdegree{{ }}for '
            f'austenite and '
            f'{params.get("lamella_merge_deg", 2.0)}\\textdegree{{ }}for '
            f'martensite, '
            f'{au.get("n_cluster_ids","?")} austenite cluster IDs collapsed to '
            f'{au.get("n_physical_grains","?")} physical grains '
            f'({au.get("n_multi_fragment","?")} multi-fragment), and '
            f'{ma.get("n_cluster_ids","?")} martensite IDs to '
            f'{ma.get("n_physical_lamellae","?")} physical lamellae '
            f'({ma.get("n_multi_fragment","?")} multi-fragment). '
            f'{au.get("n_boundary_grains","?")} grains touch the scan '
            f'boundary and have underestimated lamella counts.',
            '',
            _fig(fig_paths.get('fig1'),
                'Physical grain and lamella reconstruction.',
                'fig:reconstruction'),
        ]

        # Multi-fragment table
        mfg = au.get('multi_fragment_groups', {})
        if mfg:
            mfg_rows = [[gid, str(frags), str(len(frags))]
                        for gid, frags in mfg.items()]
            lines += [
                r'\begin{table}[htbp]',
                r'  \centering',
                r'  \caption{Multi-fragment austenite grains}',
                r'  \label{tab:fragments}',
                '  ' + _tbl_tex(
                    ['Root ID', 'Fragment IDs', 'N fragments'],
                    mfg_rows, 'lll'),
                r'\end{table}',
                '',
            ]

        # ── Section 3: HP statistics ──────────────────────────────────────
        sc_data = hp.get('score', {})
        oop     = hp.get('oop', {})
        lines += [
            r'\newpage',
            r'\section{Habit Plane Statistics}',
            '',
            f'Rank-1 candidates: \\textbf{{{hp.get("n_interfaces_rank1","?")}}} '
            f'from {hp.get("n_candidates_total","?")} total. '
            f'Mean score (rank-1): '
            f'{_fmt(sc_data.get("rank1_mean"), 2)} '
            f'$\\pm$ {_fmt(sc_data.get("rank1_std"), 2)}. '
            f'Fit\\,=\\,Yes: {sc_data.get("n_fit_yes","?")} '
            f'({_fmt(sc_data.get("pct_fit_yes"), 0)}\\%). '
            f'OOP flagged ($\\leq${params.get("oop_flag_deg",20.0)}\\textdegree): '
            f'{oop.get("n_flagged","?")} ({_fmt(oop.get("pct_flagged"),0)}\\%).',
            '',
            _fig(fig_paths.get('fig2'),
                'Habit plane families, OOP angle distribution, and '
                'score distribution.',
                'fig:hp_stats'),
        ]

        fam = hp.get('habit_plane_families', {}).get('austenite', {})
        if fam:
            n_r1 = hp.get('n_interfaces_rank1', 1) or 1
            fam_rows = [[f, c, f'{100*c/n_r1:.1f}\\%']
                        for f, c in list(fam.items())[:8]]
            lines += [
                r'\begin{table}[htbp]',
                r'  \centering',
                r'  \caption{Habit plane families (austenite, rank-1)}',
                r'  \label{tab:hp_families}',
                '  ' + _tbl_tex(
                    ['HP Family', 'Count', r'\%'],
                    fam_rows, 'lrr'),
                r'\end{table}',
                '',
            ]

        lines += [
            r'\begin{quote}\small\textit{',
            r'Note: LCV numbers are grain-specific --- cross-grain LCV '
            r'comparisons are \textbf{invalid}. '
            r'Use \texttt{or\_residual} for cross-grain variant comparison.}',
            r'\end{quote}',
            '',
        ]

        # ── Section 4: Schmid ─────────────────────────────────────────────
        lines += [
            r'\newpage',
            r'\section{Per-Grain Schmid-Factor Analysis}',
            '',
            f'Analysis covers {sc.get("n_physical_grains","?")} physical '
            f'grains ({sc.get("n_interior","?")} interior, '
            f'{sc.get("n_boundary","?")} boundary-truncated). '
            f'Stereological check: size vs count '
            f'$r_s$ = {_fmt(stereo.get("size_vs_count_r"),3)}, '
            f'size vs density '
            f'$r_s$ = {_fmt(stereo.get("size_vs_density_r"),3)} --- '
            f'{"\\textbf{pure stereological artefact}" if stereo.get("is_pure_artefact") else "possible real effect"}.',
            '',
            _fig(fig_paths.get('fig3'),
                'Per-grain Schmid-factor analysis: size, lamella count, '
                'density, and transformation strain correlations.',
                'fig:schmid'),
        ]

        if sc.get('correlations'):
            corr_rows = [
                [c['label'][:50].replace('_',r'\_'),
                _fmt(c['r'],3) if c['r'] is not None else '—',
                _sig(c['p']),
                str(c['n'])]
                for c in sc['correlations']]
            lines += [
                r'\begin{table}[htbp]',
                r'  \centering',
                r'  \caption{Spearman correlations --- Schmid analysis}',
                r'  \label{tab:schmid_corr}',
                '  ' + _tbl_tex(
                    ['Correlation', '$r_s$', 'Sig.', '$N$'],
                    corr_rows, 'lrrl'),
                r'\end{table}',
                '',
            ]

        for note in sc.get('notes', []):
            lines.append(f'\\textit{{\\small {note.replace("_", r"_")}}}\\\\')
        lines.append('')

        # ── Section 5: OR preservation ────────────────────────────────────
        or_sec  = results.get('or_preservation', {})
        ov      = or_sec.get('overall', {})
        la      = or_sec.get('low_angle_boundaries', {})
        ha      = or_sec.get('high_angle_boundaries', {})
        interp  = or_sec.get('interpretation', {})

        lines += [
            r'\newpage',
            r'\section{OR Preservation Across Grain Boundaries}',
            '',
        ]

        if not or_sec.get('n_pairs_evaluated'):
            lines.append(
                r'OR preservation analysis not available --- '
                r'run \texttt{hpa.analyse\_or\_preservation()} first.')
        else:
            lines += [
                f'\\textbf{{{or_sec.get("n_meeting_pairs_detected","?")} '
                f'boundary-meeting pairs detected}} '
                f'(tip distance $\\leq$ '
                f'{params.get("tip_distance_px",15.0):.0f}\\,px). '
                f'{or_sec.get("n_pairs_evaluated","?")} evaluated. '
                f'OR preservation threshold: '
                f'{params.get("or_preserved_deg",3.0)}\\textdegree.',
                '',
            ]

            or_rows = [
                ['All pairs',
                str(ov.get('n_preserved','?')),
                f'{_fmt(ov.get("pct_preserved"),0)}\\%',
                f'{_fmt(ov.get("or_residual_mean"),1)}\\textdegree'],
                ['Low-angle ($<$15\\textdegree)',
                str(la.get('n_preserved','?')),
                f'{_fmt(la.get("pct_preserved"),0)}\\%',
                f'{_fmt(la.get("or_residual_mean"),1)}\\textdegree'
                if la.get('or_residual_mean') else '---'],
                ['High-angle ($\\geq$15\\textdegree)',
                str(ha.get('n_preserved','?')),
                f'{_fmt(ha.get("pct_preserved"),0)}\\%',
                f'{_fmt(ha.get("or_residual_mean"),1)}\\textdegree'
                if ha.get('or_residual_mean') else '---'],
            ]
            lines += [
                r'\begin{table}[htbp]',
                r'  \centering',
                r'  \caption{OR preservation summary}',
                r'  \label{tab:or_summary}',
                '  ' + _tbl_tex(
                    ['Group', 'N preserved', r'\% preserved',
                    'Mean OR residual'],
                    or_rows, 'lrrr'),
                r'\end{table}',
                '',
                _fig(fig_paths.get('fig4'),
                    'OR residual histogram and scatter plot.',
                    'fig:or_hist'),
                _fig(fig_paths.get('fig5'),
                    'OR residual per boundary.',
                    'fig:or_boundary'),
            ]

            # Per-boundary table
            by_bnd = or_sec.get('by_boundary', {})
            if by_bnd:
                bnd_rows = [
                    [bnd.replace('_', r'\_'),
                    f'{v["au_disor_deg"]:.1f}\\textdegree',
                    'Yes' if v['high_angle'] else 'No',
                    str(v['n_pairs']),
                    f'{v["or_residual_mean"]:.2f}\\textdegree',
                    f'{v["or_residual_min"]:.2f}\\textdegree',
                    str(v['n_preserved'])]
                    for bnd, v in sorted(by_bnd.items())]
                lines += [
                    r'\begin{table}[htbp]',
                    r'  \centering',
                    r'  \caption{OR preservation per boundary}',
                    r'  \label{tab:or_by_boundary}',
                    '  ' + _tbl_tex(
                        ['Boundary', 'Au disor', 'HAB', 'N pairs',
                        'OR mean', 'OR min', 'N pres.'],
                        bnd_rows, 'lrclrrr'),
                    r'\end{table}',
                    '',
                ]

            for key, label in [
                    ('low_angle',  'Low-angle boundaries'),
                    ('high_angle', 'High-angle boundaries'),
                    ('lcv_warning','LCV note')]:
                if key in interp:
                    lines.append(
                        f'\\textbf{{{label}:}} '
                        f'\\textit{{{interp[key]}}}\\\\')
            lines.append('')

        # ── Section 5: Variant compatibility ─────────────────────────────
        vc      = results.get('variant_analysis', {})
        vc_bnd  = vc.get('boundary_pairs', [])
        vc_sum  = vc.get('summary', {})
        vc_meet = vc.get('meeting_stats', {})

        lines += [
            r'\newpage',
            r'\section{Variant Compatibility Analysis}',
            '',
        ]

        if not vc or not vc_bnd:
            lines.append(
                r'Variant compatibility analysis not available --- '
                r'run \texttt{hpa.analyse\_variant\_compatibility()} first.')
        else:
            lines += [
                f'Within each physical grain, lamellae were grouped into '
                f'crystallographically distinct martensite variants using a '
                f'{params.get("ma_threshold_deg", 2.0)}\\textdegree{{ }}'
                f'disorientation threshold. '
                f'Mean variants per grain: '
                f'\\textbf{{{_fmt(vc_sum.get("mean_variants_per_grain"),1)}}} '
                f'(max: {vc_sum.get("max_variants_per_grain","?")}). '
                f'{vc_sum.get("n_boundary_pairs_analysed","?")} neighbouring '
                f'grain pairs analysed. '
                f'Compatible boundaries: '
                f'{vc_sum.get("n_compatible_boundaries","?")} '
                f'({_fmt(vc_sum.get("pct_compatible_boundaries"),0)}\\%).',
                '',
                _fig(fig_paths.get('fig6'),
                    'Variant compatibility analysis: variants per grain, '
                    'min OR residual for meeting vs non-meeting boundary '
                    'pairs, and OR residual vs boundary character.',
                    'fig:variant'),
            ]

            vc_rows = [
                ['Meeting pairs',
                str(vc_meet.get('n_meeting_pairs','?')),
                f'{_fmt(vc_meet.get("meeting_min_or_mean"),1)}\\textdegree',
                f'{_fmt(vc_meet.get("meeting_min_or_median"),1)}\\textdegree'],
                ['Non-meeting pairs',
                str(vc_meet.get('n_nomeeting_pairs','?')),
                f'{_fmt(vc_meet.get("nomeeting_min_or_mean"),1)}\\textdegree',
                f'{_fmt(vc_meet.get("nomeeting_min_or_median"),1)}\\textdegree'],
            ]
            lines += [
                r'\begin{table}[htbp]',
                r'  \centering',
                r'  \caption{Variant compatibility: meeting vs non-meeting pairs}',
                r'  \label{tab:variant_compat}',
                '  ' + _tbl_tex(
                    ['Group', 'N pairs',
                    'Mean min OR res', 'Median min OR res'],
                    vc_rows, 'lrrr'),
                r'\end{table}',
                '',
                f'\\textbf{{Statistical test:}} '
                f'\\textit{{{vc_meet.get("interpretation","—")}}}',
                '',
            ]

        # ── Section 6: warnings ───────────────────────────────────────────
        lines += [
            r'\newpage',
            r'\section{Methodological Notes and Warnings}',
            r'\begin{itemize}',
        ]
        for w in results.get('warnings', []):
            lines.append(f'  \\item {w}')
        lines += [r'\end{itemize}', '']
        # ── Write ─────────────────────────────────────────────────────────
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"LaTeX source written to {tex_path}")
        print(f"Compile with: pdflatex {tex_path.name}")
        return tex_path
class EbsdInterfaceAnalyzer(getPhases):
    """
    EBSD-specific analyzer for interface characterization between phases.
    
    Extends getPhases with EBSD-specific functionality for analyzing
    phase boundaries, misorientations, and interface crystallography.
    """
    
    def __init__(self):
        """
        Initialize EBSD analyzer with default parameters.
        """
        getPhases.__init__(self)
        self.setAttributes(maxmillerindex=4, maxdevfrom90deg=4, phase4HPguess=self.austenite) 
        self.setAttributes(PhaseNames={self.austenite:'Austenite',self.martensite:'Martensite'})
        self.setAttributes(PhaseCols={self.austenite:['r','m'],self.martensite:['b','c']})
        
    def setAttributes(self,**kwargs):    
        """
        Set attributes for the EBSD analyzer.
        
        Args:
            **kwargs: Key-value pairs of attributes to set
        """
        self.__dict__.update(kwargs)

    def setPhases(self, phases):
        """
        Set phase definitions for analysis.
        
        Args:
            phases: Phase definitions
        """
        self.Phases = phases
        
    def get_neighbors_oim(self, distance):
        """
        Returns list of relative indices of neighboring pixels for hexagonal grid.
        
        Args:
            distance (int): Neighbor distance in pixels
            
        Returns:
            tuple: (j_list, i_list) of neighbor indices
            
        From pyebsd - implements OIM convention for hexagonal grids.
        """
        if self.scan.grid.lower() == "hexgrid":
            R60 = np.array(
                [[_COS60, -_SIN60], [_SIN60, _COS60]]
            )  # 60 degrees rotation matrix

            j_list = np.arange(-distance, distance, 2)
            i_list = np.full(j_list.shape, -distance)

            xy = np.vstack([j_list * _COS60, i_list * _SIN60])

            j_list, i_list = list(j_list), list(i_list)

            for r in range(1, 6):
                xy = np.dot(R60, xy)  # 60 degrees rotation
                j_list += list((xy[0] / _COS60).round(0).astype(int))
                i_list += list((xy[1] / _SIN60).round(0).astype(int))
        else:  # sqrgrid
            R90 = np.array([[0, -1], [1, 0]], dtype=int)  # 90 degrees rotation matrix
            xy = np.vstack(
                [
                    np.arange(-distance, distance, dtype=int),
                    np.full(2 * distance, -distance, dtype=int),
                ]
            )

            j_list, i_list = list(xy[0]), list(xy[1])

            for r in range(1, 4):
                xy = np.dot(R90, xy)
                j_list += list(xy[0])
                i_list += list(xy[1])

        return j_list, i_list

    def get_neighbors(
        self, distance=1, perimeteronly=True, distance_convention="OIM", sel=None
    ):
        """
        Get indices of neighboring pixels for every pixel at given distance.
        
        Args:
            distance (int): Neighbor distance
            perimeteronly (bool): Only include perimeter pixels
            distance_convention (str): 'OIM' or 'fixed' distance convention
            sel: Selection mask for pixels
            
        Returns:
            array: Neighbor indices array
            
        From pyebsd - calculates neighbor relationships for EBSD data.
        """
        if distance_convention.lower() == "oim":
            _get_neighbors = self.get_neighbors_oim
        else:
            raise Exception(
                'get_neighbors: unknown distance convention "{}"'.format(
                    distance_convention
                )
            )

        if perimeteronly:
            # only pixels in the perimeter
            j_shift, i_shift = _get_neighbors(distance)
        else:
            # including inner pixels
            j_shift, i_shift = [], []
            for d in range(1, distance + 1):
                j_sh, i_sh = _get_neighbors(d)
                j_shift += j_sh
                i_shift += i_sh

        n_neighbors = len(j_shift)
        if sel is None:
            sel = np.full(self.scan.N, True, dtype=bool)

        # x
        j_neighbors = np.full((self.scan.N, n_neighbors), -1, dtype=int)
        j_neighbors[sel] = np.add.outer(self.scan.j[sel], j_shift)
        # y
        i_neighbors = np.full((self.scan.N, n_neighbors), -1, dtype=int)
        i_neighbors[sel] = np.add.outer(self.scan.i[sel], i_shift)

        # i, j out of allowed range
        outliers = (
            (j_neighbors < 0)
            | (j_neighbors >= self.scan.ncols)
            | (i_neighbors < 0)
            | (i_neighbors >= self.scan.nrows)
        )

        neighbors_ind = np.full((self.scan.N, n_neighbors), -1, dtype=int)
        neighbors_ind[sel] = self.scan.ij_to_index(i_neighbors[sel], j_neighbors[sel])
        neighbors_ind[outliers] = -1

        return neighbors_ind.astype(int)
    
    def kernel_average_misorientation(
        self, M, neighbors, sel=None, maxmis=None, out="deg", **kwargs
    ):
        """
        Calculates the Kernel Average Misorientation (KAM)

        M : numpy ndarray shape(N, 3, 3)
            List of rotation matrices describing the rotation from the sample
            coordinate frame to the crystal coordinate frame
        neighbors : numpy ndarray shape(N, K) - K being the number of neighbors
            Indices of the neighboring pixels
        sel : bool numpy 1D array (optional)
            Boolean array indicating data points calculations should be
            performed
            Default: None
        out : str (optional)
            Unit of the output. Possible values are:
            'deg': angle(s) in degrees
            'rad': angle(s) in radians
            Default: 'deg'
        **kwargs :
            verbose : bool (optional)
                If True, prints computation time
                Default: True

        Returns
        -------
        KAM : numpy ndarray shape(N) - M being the number of neighbors
            KAM : numpy ndarray shape(N) with KAM values
        """
        misang = self.misorientation_neighbors4kam(self.scan.M, neighbors, sel=sel, out=out, **kwargs)
        
        outliers = misang < 0  # filter out negative values
        if maxmis is not None:
            outliers |= misang > maxmis  # and values > maxmis

        misang[outliers] = 0.0
        nneighbors = np.count_nonzero(~outliers, axis=1)

        noneighbors = nneighbors == 0
        nneighbors[noneighbors] = 1  # to prevent division by 0

        KAM = np.sum(misang, axis=1) / nneighbors
        KAM[noneighbors] = np.nan  # invalid KAM when nneighbors is 0

        return KAM
    
    def get_neighbors_fixed(self, distance):
        """
        Returns list of relative indices of the neighboring pixels for
        a given distance in pixels
        """


        neighbors_hexgrid_fixed = [
        # 1st neighbors
        [[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]],
        # 2nd neighbors
        [[3, 1], [0, 2], [-3, 1], [-3, -1], [0, -2], [3, -1]],
        # 3rd neighbors and so on...
        [[4, 0], [2, 2], [-2, 2], [-4, 0], [-2, -2], [2, -2]],
        [
            [5, 1],
            [4, 2],
            [1, 3],
            [-1, 3],
            [-4, 2],
            [-5, 1],
            [-5, -1],
            [-4, -2],
            [-1, -3],
            [1, -3],
            [4, -2],
            [5, -1],
        ],
        [[6, 0], [3, 3], [-3, 3], [-6, 0], [-3, -3], [3, -3]],
        [[6, 2], [0, 4], [-6, 2], [-6, -2], [0, -4], [6, -2]],
        [
            [7, 1],
            [5, 3],
            [2, 4],
            [-2, 4],
            [-5, 3],
            [-7, 1],
            [-7, -1],
            [-5, -3],
            [-2, -4],
            [2, -4],
            [5, -3],
            [7, -1],
        ],
        [[8, 0], [4, 4], [-4, 4], [-8, 0], [-4, -4], [4, -4]],
        [
            [8, 2],
            [7, 3],
            [1, 5],
            [-1, 5],
            [-7, 3],
            [-8, 2],
            [-8, -2],
            [-7, -3],
            [-1, -5],
            [1, -5],
            [7, -3],
            [8, -2],
        ],
        [
            [9, 1],
            [6, 4],
            [3, 5],
            [-3, 5],
            [-6, 4],
            [-9, 1],
            [-9, -1],
            [-6, -4],
            [-3, -5],
            [3, -5],
            [6, -4],
            [9, -1],
        ],
        [[10, 0], [5, 5], [-5, 5], [-10, 0], [-5, -5], [5, -5]],
        [[9, 3], [0, 6], [-9, 3], [-9, -3], [0, -6], [9, -3]],
        [
            [10, 2],
            [8, 4],
            [2, 6],
            [-2, 6],
            [-8, 4],
            [-10, 2],
            [-10, -2],
            [-8, -4],
            [-2, -6],
            [2, -6],
            [8, -4],
            [10, -2],
        ],
        [
            [11, 1],
            [7, 5],
            [4, 6],
            [-4, 6],
            [-7, 5],
            [-11, 1],
            [-11, -1],
            [-7, -5],
            [-4, -6],
            [4, -6],
            [7, -5],
            [11, -1],
        ],
        # 15th neighbors
        [[12, 0], [6, 6], [-6, 6], [-12, 0], [-6, -6], [6, -6]],
        ]
        self.neighbors_hexgrid_fixed = neighbors_hexgrid_fixed
        self._n_neighbors_hexgrid_fixed = len(neighbors_hexgrid_fixed)


        if self.scan.grid.lower() == "hexgrid":
            if distance > self._n_neighbors_hexgrid_fixed:
                raise Exception(
                    "get_neighbors_fixed not supported for distance > {}".format(
                        self._n_neighbors_hexgrid_fixed
                    )
                )
            j_list, i_list = list(zip(*self.neighbors_hexgrid_fixed[distance - 1]))
        else:
            raise Exception(
                "get_neighbors_fixed not yet supported for grid type {}".format(
                    self.scan.grid
                )
            )
        return list(j_list), list(i_list)

    def get_distance_neighbors(self, distance, distance_convention="OIM"):
        """
        Returns distance, in um, to the n-th (distance-th) neighbor

        Arguments
        ---------
        distance : int
            Distance with respect to the central pixel defined in terms of
            the nearest neighbor, i.e., distance = 3 represents the 3rd
            closest neighbor pixels
        distance_convention : str (optional)
            Distance convention used for selecting the neighboring pixels.
            Two possible values are allowed: 'OIM' or 'fixed'.
            The OIM convention is used by the TSL OIM software and is
            explained in its manual. 'fixed' stands for fixed distance,
            meaning that the neighbors are defined based on a fixed
            distance from the central pixel.
            Default : OIM

        Returns
        -------
        d : float
            Distance, in um, to the n-th (distance-th) neighbor
        """
        if distance_convention.lower() == "oim":
            j, i = self.get_neighbors_oim(distance)
        elif distance_convention.lower() == "fixed":
            j, i = self.get_neighbors_fixed(distance)
        else:
            raise Exception(
                ("get_distance_neighbors: unknown distance convention " '"{}"').format(
                    distance_convention
                )
            )
        #print(j)
        if self.scan.grid.lower() == "hexgrid":
            #d = 0.5 * ((self.scan.dx*np.array(j)) ** 2 + 3.0 * (self.scan.dy*np.array(i)) ** 2) ** 0.5
            d = self.scan.dx*0.5 * ((np.array(j)) ** 2 + 3.0 * (np.array(i)) ** 2) ** 0.5
        else:  # sqrgrid
            d = ((self.scan.dx*np.array(j)) ** 2 + (self.scan.dy*np.array(i)) ** 2) ** 0.5

        return d.mean()    
    def get_distance_neighbors_xy(self, distance, distance_convention="OIM"):
        """
        Returns distance, in um, to the n-th (distance-th) neighbor

        Arguments
        ---------
        distance : int
            Distance with respect to the central pixel defined in terms of
            the nearest neighbor, i.e., distance = 3 represents the 3rd
            closest neighbor pixels
        distance_convention : str (optional)
            Distance convention used for selecting the neighboring pixels.
            Two possible values are allowed: 'OIM' or 'fixed'.
            The OIM convention is used by the TSL OIM software and is
            explained in its manual. 'fixed' stands for fixed distance,
            meaning that the neighbors are defined based on a fixed
            distance from the central pixel.
            Default : OIM

        Returns
        -------
        d : float
            Distance, in um, to the n-th (distance-th) neighbor
        """
        if distance_convention.lower() == "oim":
            j, i = self.get_neighbors_oim(distance)
        elif distance_convention.lower() == "fixed":
            j, i = self.get_neighbors_fixed(distance)
        else:
            raise Exception(
                ("get_distance_neighbors: unknown distance convention " '"{}"').format(
                    distance_convention
                )
            )
        #print(j)
        if self.scan.grid.lower() == "hexgrid":
            #d = 0.5 * ((self.scan.dx*np.array(j)) ** 2 + 3.0 * (self.scan.dy*np.array(i)) ** 2) ** 0.5
            dx = self.scan.dx*np.array(j)
            dy = self.scan.dy*np.array(i)
        else:  # sqrgrid
            dx = self.scan.dx*np.array(j)
            dy = self.scan.dy*np.array(i)

        return dx,dy   
    def get_distance_neighbors_ij(self, distance, distance_convention="OIM"):
        """
        Returns distance, in um, to the n-th (distance-th) neighbor

        Arguments
        ---------
        distance : int
            Distance with respect to the central pixel defined in terms of
            the nearest neighbor, i.e., distance = 3 represents the 3rd
            closest neighbor pixels
        distance_convention : str (optional)
            Distance convention used for selecting the neighboring pixels.
            Two possible values are allowed: 'OIM' or 'fixed'.
            The OIM convention is used by the TSL OIM software and is
            explained in its manual. 'fixed' stands for fixed distance,
            meaning that the neighbors are defined based on a fixed
            distance from the central pixel.
            Default : OIM

        Returns
        -------
        d : float
            Distance, in um, to the n-th (distance-th) neighbor
        """
        if distance_convention.lower() == "oim":
            j, i = self.get_neighbors_oim(distance)
        elif distance_convention.lower() == "fixed":
            j, i = self.get_neighbors_fixed(distance)
        else:
            raise Exception(
                ("get_distance_neighbors: unknown distance convention " '"{}"').format(
                    distance_convention
                )
            )
        
        return np.array(i),np.array(j)    
    def ij_to_index(self, i, j):
            """
            i, j grid positions to pixel index (self.index)

            Parameters
            ----------
            i : int or numpy ndarray
                Column number (y coordinate) according to grid description below
            j : int or numpy ndarray
                Row number (x coordinate) according to grid description below

            Returns
            -------
            index : int or numpy ndarray
                Pixel index

            Grid description for HexGrid:
            -----------------------------
            o : ncols_odd
            c : ncols_odd + ncols_even
            r : nrows
            n : total number of pixels

            ===================================
                        index
            0     1     2       o-2   o-1
            *     *     *  ...   *     *
                o    o+1            c-1
                *     *     ...      *
            c    c+1   c+2     c+o-2 c+o-1
            *     *     *  ...   *     *
                            .
                            .
                            .      n-1
                *     *     ...      *

            ===================================
                        j, i
            0  1  2  3  4   j         m-1
            *     *     *  ...   *     *   0

                *     *     ...      *      1

            *     *     *  ...   *     *   2
                            .
                            .              i
                            .
                *     *     ...      *     r-1

            Grid description for SqrGrid
            ----------------------------
            c : ncols_odd = ncols_even
            r : nrows
            n : total number of pixels

            ===================================
                        index
            0     1     2       c-2   c-1
            *     *     *  ...   *     *
            c    c+1   c+2     2c-2  2c-1
            *     *     *  ...   *     *
                            .
                            .
                            .   n-2   n-1
            *     *     *  ...   *     *

            ===================================
                        j, i
            0     1     2   j   n-2   n-1
            *     *     *  ...   *     *   0

            *     *     *        *     *   1
                            .
                            .              i
                            .
            *     *     *  ...   *     *  r-1

            """
            if self.scan.grid.lower() == "hexgrid":
                index = (i // 2) * self.scan.ncols + (j // 2)
                # ncols_odd > ncols_even is the normal situation
                if self.scan.ncols_odd > self.scan.ncols_even:
                    index += (j % 2) * self.scan.ncols_odd
                    forbidden = i % 2 != j % 2  # forbidden i, j pairs
                else:
                    index += (1 - j % 2) * self.scan.ncols_odd
                    forbidden = i % 2 == j % 2
                # This turns negative every i, j pair where j > ncols
                index *= 1 - self.scan.N * (j // self.scan.ncols)
                # Turns forbidden values negative
                index = np.array(index)
                index[forbidden] = -1
                if index.ndim == 0:
                    index = int(index)
            else:
                index = i * self.scan.ncols + j
            return index
    def get_KAM(
        self,
        distance=1,
        perimeteronly=True,
        maxmis=None,
        distance_convention="OIM",
        sel=None,
        **kwargs
    ):
        """
        Returns Kernel average misorientation map

        Parameters
        ----------
        distance : int (optional)
            Distance (in neighbor indexes) to the kernel
            Default: 1
        perimeteronly : bool (optional)
            If True, KAM is calculated using only pixels in the perimeter,
            else uses inner pixels as well
            Default: True
        maxmis : float (optional)
            Maximum misorientation angle (in degrees) accounted in the
            calculation of KAM
            Default: None
        sel : bool numpy 1D array (optional)
            Boolean array indicating which data points should be plotted
            Default: None

        Returns
        -------
        KAM : numpy ndarray shape(N) with KAM values in degrees
        """
        neighbors = self.get_neighbors(
            distance, perimeteronly, distance_convention, sel
        )
        return self.kernel_average_misorientation(
            self.scan.M, neighbors, sel=sel, maxmis=maxmis, out="deg", **kwargs)
    
    def misorientation_neighbors4kam(self, M, neighbors, sel=None, out="deg", phase=None, **kwargs):
        
        """
        Calculates the misorientation angle of every data point with respective
        orientation matrix provided in 'M' with respect to an arbitrary number
        of neighbors, whose indices are provided in the 'neighbors' argument.

        Parameters
        ----------
        M : numpy ndarray shape(N, 3, 3)
            List of rotation matrices describing the rotation from the sample
            coordinate frame to the crystal coordinate frame
        neighbors : numpy ndarray shape(N, K) - K being the number of neighbors
            Indices of the neighboring pixels
        sel : bool numpy 1D array (optional)
            Boolean array indicating data points calculations should be
            performed
            Default: None
        out : str (optional)
            Unit of the output. Possible values are:
            'deg': angle(s) in degrees
            'rad': angle(s) in radians
            Default: 'deg'
        **kwargs :
            verbose : bool (optional)
                If True, prints computation time
                Default: True

        Returns
        -------
        misang : numpy ndarray shape(N, K) - K being the number of neighbors
            KAM : numpy ndarray shape(N) with KAM values
        """
        N = M.shape[0]
        nneighbors = neighbors.shape[1]

        if phase is None:
            key1=list(self.Phases.keys())[0]
        else:
            key1=phase
        print(f'Calculated for phase {key1}')
        C = np.array(self.phases[key1]['symops'])

        # 2D array to store trace values initialized as -2 (trace values are
        # always in the [-1, 3] interval)
        tr = np.full((N, nneighbors), -2.0, dtype=float)
        # 2D array to store the misorientation angles in degrees
        misang = np.full((N, nneighbors), -1.0, dtype=float)

        if not isinstance(sel, np.ndarray):
            sel = np.full(N, True, dtype=bool)

        verbose = kwargs.pop("verbose", True)
        if verbose:
            t0 = time.time()
            sys.stdout.write(
                "Calculating misorientations for {} points for {} neighbors".format(
                    np.count_nonzero(sel), nneighbors
                )
            )
            sys.stdout.write(" [")
            sys.stdout.flush()

        for k in range(nneighbors):
            # valid points, i.e., those part of the selection and with valid neighrbor index (> 0)
            ok = (neighbors[:, k] >= 0) & sel & sel[neighbors[:, k]]
            # Rotation from M[ok] to M[neighbors[ok, k]]
            # Equivalent to np.matmul(M[neighbors[ok,k]], M[ok].transpose([0,2,1]))
            T = np.einsum("ijk,imk->ijm", M[neighbors[ok, k]], M[ok])

            for m in range(len(C)):
                # Smart way to calculate the trace using einsum.
                # Equivalent to np.matmul(C[m], T).trace(axis1=1, axis2=2)
                a, b = C[m].nonzero()
                ttr = np.einsum("j,ij->i", C[m, a, b], T[:, a, b])
                tr[ok, k] = np.max(np.vstack([tr[ok, k], ttr]), axis=0)

            if verbose:
                if k > 0 and k < nneighbors:
                    sys.stdout.write(", ")
                sys.stdout.write("{}".format(k + 1))
                sys.stdout.flush()

        del T, ttr

        if verbose:
            sys.stdout.write("] in {:.2f} s\n".format(time.time() - t0))
            sys.stdout.flush()

        # Take care of tr > 3. that might happend due to rounding errors
        tr[tr > 3.0] = 3.0

        # Filter out invalid trace values
        ok = tr >= -1.0
        misang[ok] = trace_to_angle(tr[ok], out)
        return misang

    def misorientation_neighbors(self, neighbors=None, distance = 1, sel=None, out="deg", phase=None,  **kwargs):
        """
        Calculate misorientation angles between pixels and their neighbors.
        
        Args:
            neighbors: Precomputed neighbor indices
            distance (int): Neighbor distance
            sel: Pixel selection mask
            out (str): Output unit ('deg' or 'rad')
            phase: Phase to consider
            **kwargs: Additional arguments
            
        Calculates misorientation using crystal symmetry operations.
        From pyebsd - computes grain boundary misorientations.
        """
        N = self.scan.M.shape[0]
        if neighbors is None:
            neighbors = self.get_neighbors(distance=distance)
            
        nneighbors = neighbors.shape[1]
        if phase is None:
            key1=list(self.Phases.keys())[0]
        else:
            key1=phase
        C = np.array(self.phases[key1]['symops'])
        if sel is None:
            sel=list(range(0,len(self.SelPaths)))
        allpx=False
        
        if type(sel)==str:
            if sel=='all':
                allpx=True
                sel=[0]
        if (type(sel)!=list or not isinstance(np.array([1,0]), np.ndarray)):
            sel=[sel]
        if not allpx:
            self.misang=[]
        
        for seli in sel:
            if not allpx:
                verts = self.SelVerts[seli]
                path = self.SelPaths[seli]
                inside = path.contains_points(np.vstack((self.scan.x,self.scan.y)).T)
            else:
                inside = np.full(N, True, dtype=bool)
            # 2D array to store trace values initialized as -2 (trace values are
            # always in the [-1, 3] interval)
            tr = np.full((N, nneighbors), -2.0, dtype=float)
            # 2D array to store the misorientation angles in degrees
            misang = np.full((N, nneighbors), -1.0, dtype=float)
            
            for k in range(nneighbors):
                # valid points, i.e., those part of the selection and with valid neighrbor index (> 0)
                ok = (neighbors[:, k] >= 0) & inside & inside[neighbors[:, k]]
                # Rotation from M[ok] to M[neighbors[ok, k]]
                # Equivalent to np.matmul(M[neighbors[ok,k]], M[ok].transpose([0,2,1]))
                T = np.einsum("ijk,imk->ijm", self.scan.M[neighbors[ok, k]], self.scan.M[ok])
        
                for m in range(len(C)):
                    # Smart way to calculate the trace using einsum.
                    # Equivalent to np.matmul(C[m], T).trace(axis1=1, axis2=2)
                    a, b = C[m].nonzero()
                    ttr = np.einsum("j,ij->i", C[m, a, b], T[:, a, b])
                    tr[ok, k] = np.max(np.vstack([tr[ok, k], ttr]), axis=0)
        
            del T, ttr
        
            # Take care of tr > 3. that might happend due to rounding errors
            tr[tr > 3.0] = 3.0
        
            # Filter out invalid trace values
            ok = tr >= -1.0
            misang[ok] = trace_to_angle(tr[ok], out)
            if not allpx:
                self.misang.append(misang)
            else:
                self.allmisang = misang
    def grad_from_neighbors_least_squares01(self, misang, dx, dy, weights=None, reg=0.0):
        """
        Estimate local misorientation gradients (∂θ/∂x, ∂θ/∂y)
        from neighbor misorientations using weighted least squares.

        Parameters
        ----------
        misang : (N, M) array
            Misorientation between pixel i and its M neighbors.
            -1 where neighbor is missing.
        dx, dy : (N, M) arrays
            Displacements from pixel i to neighbor j (same units).
            -1 where neighbor is missing.
            or dx, dy : (M) if it is the same for all pixels
        weights : (N, M) array or None
            Optional per-neighbor weights (same shape as misang).
            If None, all valid neighbors are weighted equally.
        reg : float
            Small Tikhonov regularization term (added to diagonal)
            to stabilize ill-conditioned least squares fits.

        Returns
        -------
        gradx, grady : (N,) arrays
            Least-squares misorientation gradients.
        """
        N, M = misang.shape
        gradx = np.zeros(N)
        grady = np.zeros(N)

        if len(dx.shape)==1:
            same=True
        else:
            same=False
        if weights is None:
            weights = np.ones_like(misang)

        for i in range(N):
            # --- select valid neighbors ---
            mask = (misang[i] != -1)
            #mask = (misang[i] != 0) & (dx[i] != -1) & (dy[i] != -1)
            if not np.any(mask):
                gradx[i] = np.nan
                grady[i] = np.nan
                continue

            # --- extract valid values ---
            dθ = misang[i, mask]
            if same:
                X = np.stack([dx[ mask], dy[ mask]], axis=1)
            else:
                X = np.stack([dx[i, mask], dy[i, mask]], axis=1)
            W = np.diag(weights[i, mask])

            # --- weighted least squares solution ---
            # minimize ||W*(X·g - dθ)||² + reg*||g||²
            # where g = [∂θ/∂x, ∂θ/∂y]
            A = X.T @ W @ X + reg * np.eye(2)
            b = X.T @ W @ dθ

            try:
                g = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # fallback to pseudo-inverse if singular
                g = np.linalg.pinv(A) @ b

            gradx[i], grady[i] = g

        return gradx, grady
    def grad_from_neighbors_least_squares(self ,misang, dx, dy, weights=None, reg=0.0,maxmis=10):
        """
        misang: (Npix, M) signed small-angle differences (radians). NaN for missing neighbor.
        dx, dy: (Npix) physical offsets (meters) from center to neighbor.
        weights: optional (Npix, M) array of nonnegative weights (e.g., 1/dist). If None all ones.
        reg: Tikhonov regularization scalar (>=0). Small e.g. 1e-12 helps stability.
        Returns:
        phi_x, phi_y: arrays (Npix,) giving partial derivatives in radians/m.
        info: dict with 'rank' and 'cond' arrays of length Npix for diagnostics.

        example usage notes

        If your dx, dy are integer pixel offsets (e.g. ±1, ±1) multiplied by a step-size s (meters), you can pass actual physical offsets; don't forget the meter unit.

        Choose weights to prefer nearer neighbors or those with higher confidence. A typical choice: weights = 1.0 / (np.sqrt(dx**2 + dy**2) + eps) or weights = 1.0 / (distance**2 + eps).

        Use reg=1e-12 (or 1e-10) if you see cond exploding for some pixels, but don't over-regularize.
        """
        Npix, M = misang.shape
        phi_x = np.full(Npix, np.nan, dtype=float)
        phi_y = np.full(Npix, np.nan, dtype=float)
        rank = np.zeros(Npix, dtype=int)
        cond = np.full(Npix, np.nan, dtype=float)

        if weights is None:
            weights = np.ones_like(dx)
        # mask invalid neighbors
        valid = np.isfinite(misang) & np.isfinite(dx) & np.isfinite(dy) & (weights > 0) & (misang != -1) & (misang <= maxmis)

        for p in range(Npix):
            mask = valid[p, :]
            if np.count_nonzero(mask) < 2:
                # not enough info to determine 2 components
                continue

            vkx = dx[p,mask].astype(float)   # shape (m,)
            vky = dy[p,mask].astype(float)
            bk  = misang[p, mask].astype(float)
            wk  = weights[p,mask].astype(float)

            # Build weighted normal equations: (V^T W V + reg I) g = V^T W b
            # where V rows are [vkx, vky]
            W = np.diag(wk)
            V = np.vstack((vkx, vky)).T         # shape (m,2)
            VTW = V.T * wk                     # broadcasting, same as V.T @ W
            A = VTW @ V                        # 2x2
            bvec = VTW @ bk                    # length 2

            if reg > 0:
                A += reg * np.eye(2)

            # diagnostics
            try:
                # condition number for diagnostics
                cond[p] = np.linalg.cond(A)
                # Solve
                g = np.linalg.solve(A, bvec)
                phi_x[p], phi_y[p] = g[0], g[1]
                rank[p] = np.linalg.matrix_rank(A)
            except np.linalg.LinAlgError:
                # fallback to least-squares
                g, *_ = np.linalg.lstsq(V * np.sqrt(wk)[:,None], bk * np.sqrt(wk), rcond=None)
                phi_x[p], phi_y[p] = g[0], g[1]
                rank[p] = np.linalg.matrix_rank(V * np.sqrt(wk)[:,None])
                cond[p] = np.nan

        info = dict(rank=rank, cond=cond)
        return phi_x, phi_y, info

    def getMisori2phase(self,misorithreshold=2,sel=None, plot=False):
        """
        Identify phase boundaries based on misorientation threshold.
        
        Args:
            misorithreshold (float): Misorientation threshold in degrees
            sel: Selection mask
            plot (bool): Whether to plot results
            
        Uses misorientation to detect phase boundaries in EBSD data.
        """
        if sel is None:
            sel=list(range(0,len(self.SelPaths)))
        if type(sel)!=list or not isinstance(np.array([1,0]), np.ndarray):
            sel=[sel]
        for seli in sel:
            verts = self.SelVerts[seli]
            path = self.SelPaths[seli]
            inside = path.contains_points(np.vstack((self.scan.x,self.scan.y)).T)
            key1=list(self.Phases.keys())[0]
            umatsa = np.array(self.G2Cryst[key1][inside].to_matrix())
            
            disori = disorimat(umatsa,self.phases[key1]['symops'])
            phasein=copy.deepcopy(self.scan.phase[inside])*0
            idxs = np.where(disori[0,:]>misorithreshold)
            phasein[idxs[0]]=1
            self.scan.phase[inside]=phasein
            
            if plot:
                fig,ax = plt.subplots()
                pos=ax.matshow(disori)
                fig.colorbar(pos, ax=ax)
                fig,ax = plt.subplots()
                ax.plot(disori.flatten(),'o')

    def getMeanOris(self,sel=None,keyma=None, keyau=None, OR=True):
        """
        Calculate mean orientations for selected regions.
        
        Args:
            sel: Region selection
            keyma: Martensite phase key
            keyau: Austenite phase key
            OR (bool): Calculate orientation relationship
            
        Computes average orientations and pole figures for phase regions.
        """
        
        self.Ln = {}
        for key in self.phases.keys():
            self.Ln[key] = np.array([la/np.linalg.norm(la) for la in self.phases[key]['L'].T]).T

        if sel is None:
            sel=list(range(0,len(self.SelPaths)))
        if type(sel)!=list or not isinstance(np.array([1,0]), np.ndarray):
            sel=[sel]
        self.Sels=[None]*len(self.SelPaths)
        for seli in sel:
            self.Sels[seli]={}            
            verts = self.SelVerts[seli]
            path = self.SelPaths[seli]

            self.Sels[seli]['verts']=copy.deepcopy(np.array(self.SelVerts[seli]))
            self.Sels[seli]['path']=self.SelPaths[seli]
            self.Sels[seli]['inside']=path.contains_points(np.vstack((self.scan.x,self.scan.y)).T)
            self.Sels[seli]['all_px']=(self.Sels[seli]['inside'] == True)*0>1
            titles = 'px G2Cryst_red G2Cryst_red_avg G2Sampl_red G2Sampl_red_avg <100>PF <100>PF_avg'
            for title in titles.split():
                self.Sels[seli][title] = {}
        
            for key in self.Phases.keys():
                self.Sels[seli]['px'][key] = (self.Sels[seli]['inside'] == True)*(self.scan.phase==self.Phases[key])
                self.Sels[seli]['all_px']+=self.Sels[seli]['px'][key]  
                self.Sels[seli]['G2Cryst_red'][key] = symmetry_reduced_oris(np.array(self.G2Cryst[key][self.Sels[seli][f'px'][key]].to_matrix()),self.phases[key]['symops'])
                self.Sels[seli]['G2Sampl_red'][key]=np.transpose(self.Sels[seli]['G2Cryst_red'][key],axes=(0,2,1))
                self.Sels[seli]['G2Cryst_red_avg'][key]=Rotation.from_matrix(self.Sels[seli]['G2Cryst_red'][key]).mean().to_matrix()[0]
                self.Sels[seli]['G2Sampl_red_avg'][key]=self.Sels[seli]['G2Cryst_red_avg'][key].T
                
                pfi=[]
                for di in self.Ln[key].T:
                    pfi.append(orilistMult(self.Sels[seli]['G2Sampl_red'][key],di))            
                self.Sels[seli]['<100>PF'][key] = np.hstack((pfi[0],pfi[1],pfi[2]))
                self.Sels[seli]['<100>PF_avg'][key] = self.Sels[seli]['G2Sampl_red_avg'][key].dot(self.Ln[key])
                
            
            if keyma is None and keyau is None and OR:
                self.keyma=self.martensite
                self.keyau=self.austenite
                self.getClosestOR(seli)                
            elif keyma is not None and keyau is None and OR:
                self.keyma=keyma
                self.keyau=self.austenite
                self.getClosestOR(seli)
            elif keyau is not None and keyma is None and OR:
                self.keyma=self.martensite
                self.keyau=keyau
                self.getClosestOR(seli)
            elif keyau is not None and keyma is not None and OR:
                self.keyma=keyma
                self.keyau=keyau
                self.getClosestOR(seli)
            
            ORi=self.Sels[seli]['G2Cryst_red_avg'][self.keyma].dot(self.Sels[seli]['G2Sampl_red_avg'][self.keyau])
            axis_angle = Rotation.from_matrix(ORi).to_axes_angles()
            self.Sels[seli]['OR'] = { f'OR:{self.keyau}-{self.keyma}':OR,'axis':np.array([axis_angle.axis.x[0],axis_angle.axis.y[0],axis_angle.axis.z[0]]),'angle':np.rad2deg(axis_angle.angle)[0]}

    def getClosestOR(self,seli):
        """
        Find martensite variant with orientation relationship closest to EBSD data.
        
        Args:
            seli: Selection index
            
        Compares experimental OR with theoretical variants to find best match.
        """
        self.G2Cryst_red_ma_avg_alleq = np.tensordot(self.phases[self.keyma]['symops'], [self.Sels[seli]['G2Cryst_red_avg'][self.keyma]], axes=[[-1], [-2]]).transpose([2, 0, 1, 3])[0,:,:,:]
        self.T_AM_ebsd = np.tensordot(self.G2Cryst_red_ma_avg_alleq, [self.Sels[seli]['G2Sampl_red_avg'][self.keyau]], axes=[[-1], [-2]]).transpose([2, 0, 1, 3])[0,:,:,:]
        T_AM_T = np.array(self.T_AM).T
        D = np.tensordot(self.T_AM_ebsd,T_AM_T, axes=[[-1], [-2]]).transpose([2, 0, 1, 3])
        tr = np.trace(D, axis1=2, axis2=3)
        neg = tr < -1.0
        tr[neg] = -tr[neg]
        idxs=np.unravel_index(np.argmax(tr),tr.shape)
        Cmp = self.compareEbsdTheory(seli,idxs[0],idxs[1])
        self.Sels[seli]['Closest Variant'] = idxs[0]
        for key in Cmp.keys():
            self.Sels[seli][key] = copy.deepcopy(Cmp[key])
            self.Sels[seli][key+'_allvars'] = [] 
        for vari in range(self.T_AM.shape[2]):
            D = np.tensordot(self.T_AM_ebsd,T_AM_T[vari:vari+1,:,:], axes=[[-1], [-2]]).transpose([2, 0, 1, 3])
            tr = np.trace(D, axis1=2, axis2=3)
            neg = tr < -1.0
            tr[neg] = -tr[neg]
            idxs=np.unravel_index(np.argmax(tr),tr.shape)
            Cmp = self.compareEbsdTheory(seli,vari,idxs[1])
            for key in Cmp.keys():
                self.Sels[seli][key+'_allvars'].append(copy.deepcopy(Cmp[key]))

    def compareEbsdTheory(self,seli,varIdx,equivalIdx):
        """
        Compare experimental EBSD data with theoretical predictions.
        
        Args:
            seli: Selection index
            varIdx: Variant index
            equivalIdx: Equivalent orientation index
            
        Returns:
            dict: Comparison results including misorientation and strains
        """
        Cmp={}
        titles = ['Closes equivalent G2Cryst','Theory G2Cryst','Theory <100>PF_avg','Theory <100>PF_avg_symop']
        for title in titles:
            Cmp[title]={}
            
        Cmp['Martensite symop'] = self.phases[self.keyma]['symops'][equivalIdx]
        Cmp['Closes equivalent G2Cryst'][self.keyma]=self.G2Cryst_red_ma_avg_alleq[equivalIdx,:,:]
        Cmp['Closest Theory OR'] = self.T_AM[:,:,varIdx]
        Cmp['Exp OR'] = self.T_AM_ebsd[equivalIdx,:,:]
        Cmp['Theory G2Cryst'][self.keyma] = self.T_AM[:,:,varIdx].dot(self.Sels[seli]['G2Cryst_red_avg'][self.keyau])
        Cmp['Theory <100>PF_avg'][self.keyma] = Cmp['Theory G2Cryst'][self.keyma].T.dot(self.Ln[self.keyma])
        Cmp['Theory <100>PF_avg_symop'][self.keyma] = Cmp['Theory G2Cryst'][self.keyma].T.dot(Cmp['Martensite symop'].dot(self.Ln[self.keyma]))
        Cmp['dR OR'] = Rotation.from_matrix(Cmp['Theory G2Cryst'][self.keyma].dot(Cmp['Closes equivalent G2Cryst'][self.keyma].T))
        Cmp['Misori OR'] = np.rad2deg(Cmp['dR OR'].to_axes_angles().angle)[0]
        Cmp['Transformation strain']=[]
        for ldiri in np.eye(3):
            ldir=self.Sels[seli]['G2Cryst_red_avg'][self.keyau].dot(ldiri)
            Cmp['Transformation strain'].append({f'along {ldiri} [-]':np.sqrt(ldir.dot(self.F_AM[:,:,varIdx].T.dot(self.F_AM[:,:,varIdx].dot(ldir))))-1})
        return Cmp

    def getInterfaces(self):
        """
        Detect phase interfaces in all selections.
        
        Uses spatial gradient of phase numbers to identify boundaries.
        """
        for seli in range(len(self.Sels)):
            self.getInterface(seli)

    def getInterface(self,seli):
        """
        Detect phase interfaces for a specific selection.
        
        Args:
            seli: Selection index
            
        Identifies interface traces using phase number gradients and
        fits linear interfaces.
        """
        #get traces of the interfaces between the phase
        #it is based on the spatial gradient of phase numbers

        #get pixels of the selection
        ysel=self.scan.y[self.Sels[seli]['all_px']]
        xsel=self.scan.x[self.Sels[seli]['all_px']]
        #get pixel spacing
        dx = self.scan.dx
        dy = self.scan.dy

        #create rectangular grid, including the selection (this may be any polygon...)
        ymin=ysel.min()
        ymax=ysel.max()
        xmin=xsel.min()
        xmax=xsel.max()
        xv=np.linspace(xmin,xmin+int((xmax-xmin)/dx)*dx,int((xmax-xmin)/dx)+1)
        yv=np.linspace(ymin,ymin+int((ymax-ymin)/dy)*dy,int((ymax-ymin)/dy)+1)
        Xv, Yv = np.meshgrid(xv,yv, indexing='ij')

        #get indexes withing the rectangulat grid that do not lie in the selected polygon
        idxs = self.scan.xy_to_index(Xv.flatten(),Yv.flatten())
        #get indexes outside selection
        idxs2=[]
        idd=np.where(self.Sels[seli]['all_px'])[0]
        for ii,idx in enumerate(idxs):
            if idx not in idd:
                idxs2.append(ii)
        idxs2=np.array(idxs2)

        #get phase numbers in the selection
        Phase = self.scan.phase[idxs]

        #assign a different number to pixels lieing outside the selected polygon
        Phase[idxs2]=10

        #calculate the gradient of phase numbers
        Phase=Phase.reshape(Xv.shape)
        xx_black,yy_black = np.gradient(Phase)
        xx_black[np.abs(xx_black)!=0.5]=0
        yy_black[np.abs(yy_black)!=0.5]=0
        grad=np.sign(xx_black)*(np.abs(xx_black)+np.abs(yy_black))
        dI=self.Phases[self.keyma]-self.Phases[self.keyau]
        
        self.Sels[seli]['Interfaces']=[]
        for sgn in [1,-1]:
            Interfaces = copy.deepcopy(Phase*0)
            Idxs = np.argwhere(grad==sgn*dI)
            if Idxs.shape[0]>0:
                Interfaces[Idxs[:,0],Idxs[:,1]]=1
                linfit = np.polyfit(Xv[Idxs[:,0],Idxs[:,1]].flatten(), Yv[Idxs[:,0],Idxs[:,1]].flatten(), 1)
                my=Yv[Idxs[:,0],Idxs[:,1]].flatten().mean()
                mx=Xv[Idxs[:,0],Idxs[:,1]].flatten().mean()
                interface_trace = np.array([1,linfit[0],0])
                interface_trace/=np.sqrt(interface_trace.dot(interface_trace))
                interfacenorm_trace = np.array([1,-1/linfit[0],0])
                interfacenorm_trace/=np.sqrt(interfacenorm_trace.dot(interfacenorm_trace))
                self.Sels[seli]['Interfaces'].append({'x':Xv[Idxs[:,0],Idxs[:,1]].flatten(),'y':Yv[Idxs[:,0],Idxs[:,1]].flatten(),
                                                            'linfit':linfit,'linfitnorm':np.array([-1/linfit[0],my+mx/linfit[0]]),
                                                           'interface_trace_sample':interface_trace,'interfacenorm_trace_sample':interfacenorm_trace})
                self.Sels[seli]['Interfaces'][-1]['interface_trace']={}
                self.Sels[seli]['Interfaces'][-1]['interfacenorm_trace']={}
                for key in self.Phases.keys():
                    self.Sels[seli]['Interfaces'][-1]['interface_trace'][key]=self.Sels[seli]['G2Cryst_red_avg'][key].dot(self.CrystalRef2Spatial.dot(interface_trace))
                    self.Sels[seli]['Interfaces'][-1]['interfacenorm_trace'][key]=self.Sels[seli]['G2Cryst_red_avg'][key].dot(self.CrystalRef2Spatial.dot(interfacenorm_trace))

    def getHB(self,sel=None,ifaces=None,VarIdx=None,name='NiTi'):
        """
        Determine habit plane candidates for interfaces.
        
        Args:
            sel: Selection indices
            ifaces: Interface indices
            VarIdx: Variant index
            name (str): Name identifier
            
        Finds crystallographic planes that could correspond to observed
        interface traces in both phases.
        """
        if sel is None:
            sel=list(range(len(self.Sels)))
        for seli in sel:
            if ifaces is None:
                ifaces = list(range(len(self.Sels[seli]['Interfaces'])))
            for iface in ifaces:
                interface_trace=self.Sels[seli]['Interfaces'][iface]['interface_trace'][self.phase4HPguess]
                interfacenorm_trace=self.Sels[seli]['Interfaces'][iface]['interfacenorm_trace'][self.phase4HPguess]
                G2Sampl = self.Sels[seli]['G2Sampl_red_avg'][self.phase4HPguess]
                if self.phase4HPguess==self.keyau:
                    interface_trace2=self.Sels[seli]['Interfaces'][iface]['interface_trace'][self.keyma]
                    interfacenorm_trace2=self.Sels[seli]['Interfaces'][iface]['interfacenorm_trace'][self.keyma]
                    G2Sampl2 = self.Sels[seli]['G2Sampl_red_avg'][self.keyma]
                else:
                    interface_trace2=self.Sels[seli]['Interfaces'][iface]['interface_trace'][self.keyau]
                    interfacenorm_trace2=self.Sels[seli]['Interfaces'][iface]['interfacenorm_trace'][self.keyau]
                    G2Sampl2 = self.Sels[seli]['G2Sampl_red_avg'][self.keyau]
                    
                if VarIdx is None:
                    VarIdx = self.Sels[seli]['Closest Variant']
                if self.phase4HPguess == self.keyau:
                    self.phase2ndHPguess = self.keyma
                    Lr2 = self.phases[self.keyma]['Lr']
                    L2 = self.phases[self.keyma]['L']
                else:
                    self.phase2ndHPguess = self.keyau
                    Lr2 = self.phases[self.keyau]['Lr']
                    L2 = self.phases[self.keyau]['L']

                #get guesses in one phase
                HP_guess = self.getNormals(interface_trace, interfacenorm_trace,
                                           self.phases[self.phase4HPguess]['LrI'], self.phases[self.phase4HPguess]['Lr'],G2Sampl)
                #get corresponding guesses in the other phase for the variant providing closest math with the experimental OR
                HP_guess2ndphase, HP_guess2ndphase_allvars = self.getCorrespNormals(HP_guess, interface_trace2, 
                                                          self.CP[self.phase4HPguess][:,:,self.Sels[seli]['Closest Variant']],Lr2,LCall=self.CP[self.phase4HPguess])

                sortidxs = np.argsort(HP_guess['HPvsTrace_angle']+np.array(HP_guess2ndphase['HPvsTrace_angle']))[::-1]
        
                for key in HP_guess.keys():
                    HP_guess[key] = (HP_guess[key])[sortidxs]
                for key in HP_guess2ndphase.keys():
                    HP_guess2ndphase[key] = np.array(HP_guess2ndphase[key])[sortidxs]
                for key in HP_guess2ndphase_allvars.keys():
                    HP_guess2ndphase_allvars[key] = np.array(HP_guess2ndphase_allvars[key])[sortidxs]
                    
                self.Sels[seli]['Interfaces'][iface]['HP_guess']={}
                self.Sels[seli]['Interfaces'][iface]['HP_guess'][self.phase4HPguess]=HP_guess
                self.Sels[seli]['Interfaces'][iface]['HP_guess'][self.phase2ndHPguess]={}
                self.Sels[seli]['Interfaces'][iface]['HP_guess'][self.phase2ndHPguess]['Closest Variant'] = HP_guess2ndphase
                self.Sels[seli]['Interfaces'][iface]['HP_guess'][self.phase2ndHPguess]['allvars'] = HP_guess2ndphase_allvars

                DP_guess = self.getNormals(interfacenorm_trace,interface_trace,
                                           self.phases[self.phase4HPguess]['LI'], self.phases[self.phase4HPguess]['L'],G2Sampl)
                
                DP_guess2ndphase, DP_guess2ndphase_allvars = self.getCorrespNormals(DP_guess, interfacenorm_trace2, 
                                                          self.CD[self.phase4HPguess][:,:,self.Sels[seli]['Closest Variant']],L2,LCall=self.CD[self.phase4HPguess])

                sortidxs = np.argsort(DP_guess['HPvsTrace_angle']+np.array(DP_guess2ndphase['HPvsTrace_angle']))[::-1]
        
                for key in DP_guess.keys():
                    DP_guess[key] = (DP_guess[key])[sortidxs]
                for key in DP_guess2ndphase.keys():
                    DP_guess2ndphase[key] = np.array(DP_guess2ndphase[key])[sortidxs]
                for key in DP_guess2ndphase_allvars.keys():
                    DP_guess2ndphase_allvars[key] = np.array(DP_guess2ndphase_allvars[key])[sortidxs]
                    
                self.Sels[seli]['Interfaces'][iface]['DP_guess']={}
                self.Sels[seli]['Interfaces'][iface]['DP_guess'][self.phase4HPguess]=DP_guess
                self.Sels[seli]['Interfaces'][iface]['DP_guess'][self.phase2ndHPguess]={}
                self.Sels[seli]['Interfaces'][iface]['DP_guess'][self.phase2ndHPguess]['Closest Variant'] = DP_guess2ndphase
                self.Sels[seli]['Interfaces'][iface]['DP_guess'][self.phase2ndHPguess]['allvars'] = DP_guess2ndphase_allvars

    def getHBmatches(self,sel=None,ifaces=None):
        """
        Find matching habit plane candidates between phases.
        
        Args:
            sel: Selection indices
            ifaces: Interface indices
            
        Matches crystallographic planes and directions that are consistent
        with observed interface traces in both phases.
        """
        if sel is None:
            sel=list(range(len(self.Sels)))
        for seli in sel:
            if ifaces is None:
                ifaces = list(range(len(self.Sels[seli]['Interfaces'])))
            for iface in ifaces:
                
                HP_guess={}
                for phase in self.Phases.keys():
                    interface_trace=self.Sels[seli]['Interfaces'][iface]['interface_trace'][phase]
                    interfacenorm_trace=self.Sels[seli]['Interfaces'][iface]['interfacenorm_trace'][phase]
                    G2Sampl = self.Sels[seli]['G2Sampl_red_avg'][phase]
                    LrI = self.phases[phase]['LrI']
                    Lr = self.phases[phase]['Lr']
                    LI = self.phases[phase]['LI']
                    L = self.phases[phase]['L']
                    
                    HP_guess[phase] = self.getNormals(interface_trace, interfacenorm_trace,LrI,Lr,G2Sampl)
                    idxs = np.argsort(HP_guess[phase]['HPvsTrace_angle'])[::-1]
                    for key in HP_guess[phase].keys():
                        HP_guess[phase][key]=HP_guess[phase][key][idxs]
                self.Sels[seli]['Interfaces'][iface]['HP_guess']=HP_guess
                
                HP_matches = {}
                HP_matches['Score']={}
                for title in ['low index habit plane', 'mean misalignment', 'number of corresponding directions','overall', 'fitcorresp']:
                    HP_matches['Score'][title] = []
                for phase in self.Phases.keys():
                    HP_matches[phase]={}
                    HP_matches[phase]['Habit plane normal']={}
                    HP_matches[phase]['Habit plane normal']['misalign']=[]
                    HP_matches[phase]['Habit plane direction']={}
                    HP_matches[phase]['Habit plane direction']['misalign']=[]
                    for key in HP_matches[phase].keys():
                        for key2 in HP_guess[self.keyau].keys():
                             HP_matches[phase][key][key2]=[]
                
                for n1,n_vec in enumerate(HP_guess[self.keyau]['n_miller_normvec_sampl']):
                    for n2,n_vec2 in enumerate(HP_guess[self.keyma]['n_miller_normvec_sampl']):
                        hb_misalign = np.arccos(abs(n_vec.dot(n_vec2)))*180/np.pi
                        if hb_misalign <= self.maxnormaldev:
                            DP_guess={}
                            for phase,n in zip(self.Phases.keys(),[n1,n2]):
                                interface_trace=HP_guess[phase]['n_miller_normvec'][n]
                                interfacenorm_trace=perpendicular_vector(interface_trace)
                                G2Sampl = self.Sels[seli]['G2Sampl_red_avg'][phase]
                                LI = self.phases[phase]['LI']
                                L = self.phases[phase]['L']                
                                DP_guess[phase] = self.getNormals(interface_trace, interfacenorm_trace,LI,L,G2Sampl,maxdevfrom90deg=0.)
                            v1=DP_guess[self.keyau]
                            v2=DP_guess[self.keyma]
                            dpmatch=[]
                            for d1,d_vec in enumerate(v1['n_miller_normvec_sampl']):
                                for d2,d_vec2 in enumerate(v2['n_miller_normvec_sampl']):
                                    dp_misalign = np.arccos(abs(d_vec.dot(d_vec2)))*180/np.pi
                                    if dp_misalign <= self.maxdirdev:
                                        isin=False
                                        for isinidx,idxs in enumerate(dpmatch):
                                            if idxs[0]==d1:
                                                isin=True
                                                break
                                        if not isin:
                                            dpmatch.append((d1,d2,dp_misalign))
                                        else:
                                            if dp_misalign<idxs[2]:
                                                dpmatch[isinidx]=(d1,d2,dp_misalign)
                            if len(dpmatch)>0:
                                for key in HP_guess[phase]:
                                    HP_matches[self.keyau]['Habit plane normal'][key].append(HP_guess[self.keyau][key][n1])
                                    HP_matches[self.keyma]['Habit plane normal'][key].append(HP_guess[self.keyma][key][n2])
                                    HP_matches[self.keyau]['Habit plane direction'][key].append(v1[key][[idxs[0] for idxs in dpmatch]])
                                    HP_matches[self.keyma]['Habit plane direction'][key].append(v2[key][[idxs[1] for idxs in dpmatch]])
                                fitcorresp=True
                                vari=self.Sels[seli]['Closest Variant']
                                for idxs in dpmatch:
                                    dp_ma = v2['n_miller'][idxs[1]]
                                    dp_au = v1['n_miller'][idxs[0]]
                                    dp_ma2 = vector2miller(self.Sels[seli]['Martensite symop'].dot(self.CD[self.keyau][:,:,vari].dot(dp_au)))
                                    if not ((dp_ma2==dp_ma).all() or (-1*dp_ma2==dp_ma).all()):                                      
                                        fitcorresp = False
                                hb_au = HP_guess[self.keyau]['n_miller'][n1]
                                hb_ma = HP_guess[self.keyma]['n_miller'][n2]
                                hb_ma2 = vector2miller(self.Sels[seli]['Martensite symop'].dot(self.CP[self.keyau][:,:,vari].dot(hb_au)))
                                if not ((hb_ma2==hb_ma).all() or (-1*hb_ma2==hb_ma).all()):
                                    fitcorresp = False
                                    
                                HP_matches[self.keyma]['Habit plane normal']['misalign'].append(hb_misalign)
                                HP_matches[self.keyma]['Habit plane direction']['misalign'].append([idxs[2] for idxs in dpmatch])
                                HP_matches[self.keyau]['Habit plane normal']['misalign'].append(hb_misalign)
                                HP_matches[self.keyau]['Habit plane direction']['misalign'].append([idxs[2] for idxs in dpmatch])
                                HP_matches['Score']['low index habit plane'].append(np.sum(np.abs(HP_guess[self.keyau]['n_miller'][n1]))+np.sum(np.abs(HP_guess[self.keyma]['n_miller'][n2])))
                                allmisalign = [abs(idxs[2]) for idxs in dpmatch]
                                allmisalign.append(abs(HP_guess[self.keyau]['HPvsTrace_angle'][n1]-90))
                                allmisalign.append(abs(HP_guess[self.keyma]['HPvsTrace_angle'][n2]-90))
                                allmisalign.append(hb_misalign)
                                HP_matches['Score']['mean misalignment'].append(np.mean(allmisalign))
                                HP_matches['Score']['number of corresponding directions'].append(len(dpmatch))
                                HP_matches['Score']['overall'].append(HP_matches['Score']['low index habit plane'][-1]/HP_matches['Score']['number of corresponding directions'][-1]+HP_matches['Score']['mean misalignment'][-1])
                                if fitcorresp:
                                    HP_matches['Score']['fitcorresp'].append('yes')
                                else:
                                    HP_matches['Score']['fitcorresp'].append('no')
                if True:
                    idxs = np.argsort(HP_matches['Score']['overall'])
                    for phase in self.Phases.keys():
                        for key1 in HP_matches[phase].keys():
                            for key in HP_matches[phase][key1].keys():
                                HP_matches[phase][key1][key]=[HP_matches[phase][key1][key][idx] for idx in idxs]
                    for key in HP_matches['Score'].keys():
                        HP_matches['Score'][key]=[HP_matches['Score'][key][idx] for idx in idxs]            
                self.Sels[seli]['Interfaces'][iface]['HP_matches'] = HP_matches

    def printHBmatches(self,sel=None,ifaces=None, nodirs=False, bestscore=False):
        """
        Print habit plane matching results.
        
        Args:
            sel: Selection indices
            ifaces: Interface indices
            nodirs (bool): Skip direction information
            bestscore (bool): Only show best match
        """
        if sel is None:
            sel=list(range(len(self.Sels)))
        for seli in sel:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Selection # {seli+1} of {len(sel)}')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            if ifaces is None:
                ifaces = list(range(len(self.Sels[seli]['Interfaces'])))
            for iface in ifaces:
                print(f'Interface # {iface+1} of {len(ifaces)}')
                print('---------------------------------------------------------')
                HP_matches = self.Sels[seli]['Interfaces'][iface]['HP_matches']
                if not bestscore:
                    idxs=range(len(HP_matches[self.keyau]['Habit plane normal']['n_vec']))
                else:
                    idxs=[0]
                for idx in idxs:
                    key = 'Habit plane normal'
                    an = HP_matches[self.keyau][key]['HPvsTrace_angle'][idx]
                    an2 = HP_matches[self.keyma][key]['HPvsTrace_angle'][idx]
                    n_miller = HP_matches[self.keyau][key]['n_miller'][idx]
                    n_miller2 = HP_matches[self.keyma][key]['n_miller'][idx]
                    vdv = HP_matches[self.keyma][key]['misalign'][idx]
                    CV = self.Sels[seli]['Closest Variant']
                    TR = np.round(self.Sels[seli]['Transformation strain_allvars'][CV][0]['along [1. 0. 0.] [-]'],decimals=4)
                    print(f"Fitting Closest LCV {CV}:{HP_matches['Score']['fitcorresp'][idx]}, Score:{np.round(HP_matches['Score']['overall'][idx],decimals=2)},  mean misalignment:{np.round(HP_matches['Score']['mean misalignment'][idx],decimals=2)}")
                    print(f"Transformation strain along [100] from the closest LCV {CV}: {TR}")
                    print(f"Normals: misalignment:{np.round(vdv,decimals=2)}, {n_miller}_A ({np.round(an,decimals=2)})/{n_miller2}_M ({np.round(an2,decimals=2)})")
                    if not nodirs:
                        for idd in range(len(HP_matches[self.keyau]['Habit plane direction']['n_miller'][idx])):
                            key = 'Habit plane direction'
                            an = HP_matches[self.keyau][key]['HPvsTrace_angle'][idx][idd]
                            an2 = HP_matches[self.keyma][key]['HPvsTrace_angle'][idx][idd]
                            n_miller = HP_matches[self.keyau][key]['n_miller'][idx][idd]
                            n_miller2 = HP_matches[self.keyma][key]['n_miller'][idx][idd]
                            vdv = HP_matches[self.keyma][key]['misalign'][idx][idd]
                            print(f"Directions: misalignment:{np.round(vdv,decimals=2)},{n_miller}_A ({np.round(an,decimals=2)})/{n_miller2}_M ({np.round(an2,decimals=2)})")
                    print("====================================================================================")

    def printCorresp(self,sel=None,ifaces=None, printfor=None,printvars=None):
        """
        Print correspondence relationships between phases.
        
        Args:
            sel: Selection indices
            ifaces: Interface indices
            printfor: Phase to print correspondence for
            printvars: Variants to include
        """
        if printvars is None:
            printvars = range(self.CD[self.keyau].shape[2])
        if printfor is None:
            printfor = self.keyma
        if sel is None:
            sel=list(range(len(self.Sels)))
        for seli in sel:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Selection # {seli+1} of {len(sel)}')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            if ifaces is None:
                ifaces = list(range(len(self.Sels[seli]['Interfaces'])))
            if printvars=='closest':
                printvars=[self.Sels[seli]['Closest Variant']]
            for iface in ifaces:
                print(f'Interface # {iface+1} of {len(ifaces)}')
                print('---------------------------------------------------------')
                HP_matches = self.Sels[seli]['Interfaces'][iface]['HP_matches']
                for idx,vv in enumerate(HP_matches[self.keyau]['Habit plane normal']['n_vec']):
                    key = 'Habit plane normal'
                    hb_au = HP_matches[self.keyau]['Habit plane normal']['n_miller'][idx]
                    hb_ma = HP_matches[self.keyma]['Habit plane normal']['n_miller'][idx]
                    n_vec = HP_matches[self.keyau][key]['n_miller_normvec_sampl'][idx]
                    n_vec2 = HP_matches[self.keyma][key]['n_miller_normvec_sampl'][idx]
                    an = HP_matches[self.keyau][key]['HPvsTrace_angle'][idx]
                    an2 = HP_matches[self.keyma][key]['HPvsTrace_angle'][idx]
                    vdv = np.arccos(abs(n_vec.dot(n_vec2)))*180/np.pi
                    print(f"Habit plane: misalign.:{np.round(vdv,decimals=2)},{hb_au}_A ({np.round(an,decimals=2)})/{hb_ma}_M ({np.round(an2,decimals=2)})")
                    print(f'Best fitting variant is {self.Sels[seli]["Closest Variant"]}')
                    for idd in range(len(HP_matches[self.keyau]['Habit plane direction']['n_miller'][idx])):
                        dp_au = HP_matches[self.keyau]['Habit plane direction']['n_miller'][idx][idd]
                        dp_ma = HP_matches[self.keyma]['Habit plane direction']['n_miller'][idx][idd]
                        for vari in printvars:
                            hb_au2 = self.CP[self.keyma][:,:,vari].dot(hb_ma)
                            dp_au2 = self.CD[self.keyma][:,:,vari].dot(dp_ma)
                            if vari==self.Sels[seli]['Closest Variant']:
                                hb_ma2 = self.Sels[seli]['Martensite symop'].dot(self.CP[self.keyau][:,:,vari].dot(hb_au))
                                dp_ma2 = self.Sels[seli]['Martensite symop'].dot(self.CD[self.keyau][:,:,vari].dot(dp_au))
                            else:
                                hb_ma2 = self.CP[self.keyau][:,:,vari].dot(hb_au)
                                dp_ma2 = self.CD[self.keyau][:,:,vari].dot(dp_au)
                            if printfor == self.keyma:
                                print('Habit plane normal: ebsd vs. calculated from austenite/direction in habit plane: ebsd vs. calculated from austenite')
                                print(f'Variant {vari}: {hb_ma}_M vs. {vector2miller(hb_ma2)}_M/{dp_ma}_M vs. {vector2miller(dp_ma2)}_M')
                            else:
                                print('Habit plane normal: ebsd vs. calculated from martensite/direction in habit plane: ebsd vs. calculated from martensite')
                                print(f'Variant {vari}: {hb_au}_A vs. {vector2miller(hb_au2)}_A/{dp_au}_A vs. {vector2miller(dp_au2)}_A')
                            print('-----------------------------------------')
                        print("=====================================================")

    def getNormals(self,interface_trace, interfacenorm_trace, LrI, Lr, G2Sampl, angles = np.linspace(0,180,361),maxdevfrom90deg=None, maxmillerindex=None):
        """
        Generate candidate plane normals perpendicular to interface trace.
        
        Args:
            interface_trace: Interface trace direction
            interfacenorm_trace: Normal to interface trace
            LrI: Inverse reciprocal lattice matrix
            Lr: Reciprocal lattice matrix
            G2Sampl: Crystal to sample transformation
            angles: Rotation angles to test
            maxdevfrom90deg: Maximum deviation from 90 degrees
            maxmillerindex: Maximum Miller index
            
        Returns:
            dict: Candidate plane normals meeting criteria
        """
        N_guess={}
        titles = 'n_vec n_vec_sampl n_miller n_miller_normvec n_miller_normvec_sampl HPvsTrace_angle'.split()
        for title in titles:
            N_guess[f'{title}'] =[]

        if maxdevfrom90deg is None:
            maxdevfrom90deg=self.maxdevfrom90deg
            
        if maxmillerindex is None:
            maxmillerindex=self.maxmillerindex
        for an in angles:
            var = {}
            var['n_vec'] = Rotation.from_axes_angles(interface_trace, an, degrees=True).to_matrix().dot(interfacenorm_trace)[0,:]
            var['n_vec_sampl'] = G2Sampl.dot(var['n_vec'])
            var['n_miller'] = np.round(vector2millerround(LrI.dot(var['n_vec'])))
            if np.abs(var['n_miller']).max()>self.maxmillerindex:
                var['n_miller'] = np.round(vector2millerround(LrI.dot(var['n_vec']),MIN=False))
                
            var['n_miller_normvec'] = Lr.dot(var['n_miller'])
            var['n_miller_normvec'] /= np.linalg.norm(var['n_miller_normvec'])
            var['n_miller_normvec_sampl'] = G2Sampl.dot(var['n_miller_normvec'])
            var['HPvsTrace_angle'] = np.arccos(abs(var['n_miller_normvec'].dot(interface_trace)))*180/np.pi
            if np.abs(var['n_miller']).max()<=maxmillerindex and abs(var['HPvsTrace_angle']-90)<=maxdevfrom90deg:
                isin=False
                for nm in N_guess['n_miller']:
                    if (nm==var['n_miller']).all() or (nm==-1*var['n_miller']).all():
                        isin=True
                if not isin:
                    for title in titles:
                        N_guess[f'{title}'].append(var[f'{title}'])

        for key in N_guess.keys():
            N_guess[key]=np.array(N_guess[key])
        return N_guess

    def getCorrespNormals(self,N_guess, interface_trace2, LC, Lr2, LCall=None):
        """
        Get corresponding normals in second phase using orientation relationship.
        
        Args:
            N_guess: Candidate normals in first phase
            interface_trace2: Interface trace in second phase
            LC: Lattice correspondence matrix
            Lr2: Reciprocal lattice matrix for second phase
            LCall: All variant correspondence matrices
            
        Returns:
            tuple: Corresponding normals for closest variant and all variants
        """
        titles = 'n_miller n_miller_normvec HPvsTrace_angle'.split()
        N_guess2ndphase={}
        N_guess2ndphase_allvars={}
        for title in titles:
            N_guess2ndphase[f'{title}'] =[]
            N_guess2ndphase_allvars[f'{title}'] =[]
        for n_miller in N_guess['n_miller']:
            N_guess2ndphase['n_miller'].append(LC.dot(n_miller))
            N_guess2ndphase['n_miller_normvec'].append(Lr2.dot(N_guess2ndphase['n_miller'][-1]))
            N_guess2ndphase['n_miller_normvec'][-1]=N_guess2ndphase['n_miller_normvec'][-1]/np.linalg.norm(N_guess2ndphase['n_miller_normvec'][-1])
            N_guess2ndphase['HPvsTrace_angle'].append(np.arccos(abs(N_guess2ndphase['n_miller_normvec'][-1].dot(interface_trace2)))*180/np.pi)
            if LCall is not None:
                N_guess2ndphase_vari={}
                for title in titles:
                    N_guess2ndphase_vari[f'{title}'] =[]

                for vari in range(self.T_AM .shape[2]):
                    N_guess2ndphase_vari['n_miller'].append(LCall[:,:,vari].dot(n_miller))
                    N_guess2ndphase_vari['n_miller_normvec'].append(Lr2.dot(N_guess2ndphase_vari['n_miller'][-1]))
                    N_guess2ndphase_vari['n_miller_normvec'][-1]=N_guess2ndphase_vari['n_miller_normvec'][-1]/np.linalg.norm(N_guess2ndphase_vari['n_miller_normvec'][-1])
                    N_guess2ndphase_vari['HPvsTrace_angle'].append(np.arccos(abs(N_guess2ndphase_vari['n_miller_normvec'][-1].dot(interface_trace2)))*180/np.pi)

                for title in titles:
                   N_guess2ndphase_allvars[f'{title}'].append(N_guess2ndphase_vari[title])
                
        if LCall is not None:
            return N_guess2ndphase,N_guess2ndphase_allvars
        else:
            return N_guess2ndphase
