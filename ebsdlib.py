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

        for selPath in self.selector.selPaths:
            self.rois.masks.append(selPath.contains_points(np.vstack((self.X,self.Y)).T))
            mask_by_phase={}
            for phase_id in self.phase_names:
                mask_by_phase[self.phase_names[phase_id]] = (self.phases_id==phase_id)*self.rois.masks[-1]
            self.rois.masks_by_phase.append(mask_by_phase)
        self.selector.selPaths.insert(0,SelPaths)
        self.selector.selVerts.insert(0,SelVerts)
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
            neighbors, sel, data.phases_id,
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
                                    'distance_convention':self.distance_convention
                                    }, cluster_phases_id=cluster_phase_map, com=com)
        
        return result
    
    def _cluster_multiphase(self,X, Y, Q, sym_quats_dict, neighbors, Sel, phases_id,
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
        return np.unique(self.data.phases_id)
           
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
        for phase_id in np.unique(self.data.phases_id):
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
        
        return cluster_elongations
    
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

            b = 2.54e-10  # Burgers vector [m]
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
                    label_bbox_style='round', **kwargs):
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
                if color_by.lower() == 'gnd':
                    data = clustering_result.data.GND
                
                if vmin is None:
                    vmin = np.nanmin(data)
                if vmax is None:
                    vmax = np.nanmax(data)
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
