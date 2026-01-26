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
@njit
def find_best_symmetric_quat(q, q_ref, symops, max_iter=10, tol=1e-6):
    q_best = q.copy()
    M_best = quat_to_mat(q_best)
    min_ang = quat_misori_deg(q_best, q_ref)

    for _ in range(max_iter):
        improved = False
        for s in range(symops.shape[0]):
            q_sym = quat_mult(mat_to_quat(symops[s]), q)
            q_sym /= np.linalg.norm(q_sym)
            ang = quat_misori_deg(q_sym, q_ref)
            if ang + tol < min_ang:
                min_ang = ang
                q_best = q_sym.copy()
                M_best = quat_to_mat(q_best)
                improved = True
        if not improved:
            break
    return q_best, M_best, min_ang


# --- Numba-compatible average orientation ---
def average_orientations_numba(M, labels, symops, max_iter=10, tol=1e-6):
    N = M.shape[0]
    unique_labels = np.unique(labels)
    
    avg_q_dict = {}
    avg_M_dict = {}
    M_best_dict = {}  # new: stores all best-symmetric matrices per cluster

    for i in range(unique_labels.shape[0]):
        lab = unique_labels[i]
        if lab == 0: 
            continue

        # collect indices of current cluster
        idxs = np.where(labels == lab)[0]
        n_pix = idxs.shape[0]

        # reference quaternion = first in cluster
        q_ref = mat_to_quat(M[idxs[0]])
        #q_ref = mat_to_quat(np.eye(3))
        

        # store best-symmetric matrices for this cluster
        M_best_cluster = np.zeros((n_pix,3,3))
        q_sum = np.zeros(4)

        for j in range(n_pix):
            q_best, M_best, _ = find_best_symmetric_quat(mat_to_quat(M[idxs[j]]), q_ref, symops, max_iter, tol)
            if np.dot(q_best, q_ref) < 0:
                q_best *= -1.0
            q_sum += q_best
            M_best_cluster[j] = M_best

        # average quaternion
        q_mean = q_sum / np.linalg.norm(q_sum)
        M_mean = quat_to_mat(q_mean)

        avg_q_dict[lab] = q_mean
        avg_M_dict[lab] = M_mean
        M_best_dict[lab] = M_best_cluster

    return avg_M_dict, avg_q_dict, M_best_dict

def find_best_symmetric_oris(avg_M_dict, label_ref, symops, max_iter=10, tol=1e-6):
    if type(symops)==list:
        symops = np.array(symops)
    M_best_dict = {}  # new: stores all best-symmetric matrices per cluster

    q_ref = mat_to_quat(avg_M_dict[label_ref])

    for lab in avg_M_dict.keys():

        q_best, M_best, _ = find_best_symmetric_quat(mat_to_quat(avg_M_dict[lab]), q_ref, symops, max_iter, tol)
        M_best_dict[lab] = M_best

    return M_best_dict

# --- Numba-compatible average orientation ---
def average_orientations_numba_withref(M, labels, symops, labref=None, max_iter=10, tol=1e-6):
    ##Not working!!!!!!!!!!!!!!!!!!!!!!
    N = M.shape[0]
    avg_q_dict = {}
    avg_M_dict = {}
    M_best_dict = {}  # new: stores all best-symmetric matrices per cluster
    unique_labels = np.unique(labels[labels>0])
    if labref is None:
        #find largest subgrain
        numpx = []
        for lab in np.unique(labels[labels>0]):
            numpx.append((np.where(labels==lab)[0].shape[0]))
        idx=np.argmax(numpx)
        lab = np.unique(labels[labels>0])[idx]
        labmax = np.unique(labels[labels>0])[idx]
        labref = labmax
    else:
        lab = labref
        labmax = labref
    # collect indices of current cluster
    idxs = np.where(labels == lab)[0]
    
    # reference quaternion = first in cluster
    q_ref = mat_to_quat(M[idxs[0]])
    
    avg_q_dict = {}
    avg_M_dict = {}
    M_best_dict = {}  # new: stores all best-symmetric matrices per cluster

    for i in range(unique_labels.shape[0]):
        lab = unique_labels[i]
        if lab == 0: 
            continue

        # collect indices of current cluster
        idxs = np.where(labels == lab)[0]
        n_pix = idxs.shape[0]

                

        # store best-symmetric matrices for this cluster
        M_best_cluster = np.zeros((n_pix,3,3))
        q_sum = np.zeros(4)

        for j in range(n_pix):
            q_best, M_best, _ = find_best_symmetric_quat(mat_to_quat(M[idxs[j]]), q_ref, symops, max_iter, tol)
            if np.dot(q_best, q_ref) < 0:
                q_best *= -1.0
            q_sum += q_best
            M_best_cluster[j] = M_best

        # average quaternion
        q_mean = q_sum / np.linalg.norm(q_sum)
        M_mean = quat_to_mat(q_mean)

        avg_q_dict[lab] = q_mean
        avg_M_dict[lab] = M_mean
        M_best_dict[lab] = M_best_cluster

    return avg_M_dict, avg_q_dict, M_best_dict,labref




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
            
            self._grid_2d = {
                'xs': xs, 'ys': ys,
                'labels_2d': labels_2d,
                'phase_2d': phase_2d,
                'x_map': x_map, 'y_map': y_map,
                'shape': (ny, nx)
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
    def average_orientations(self, phase, max_iter=10, tol=1e-6):
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
            n_pix = idxs.shape[0]

            # reference quaternion = first in cluster
            q_ref = self.data.quaternions[idxs[0]]#mat_to_quat(M[idxs[0]])
            #q_ref = mat_to_quat(np.eye(3))
            

            # store best-symmetric matrices for this cluster
            M_best_cluster = np.zeros((n_pix,3,3))
            q_sum = np.zeros(4)

            for j in range(n_pix):
                q_best, M_best, _ = find_best_symmetric_quat(self.data.quaternions[idxs[j]], q_ref, symops, max_iter, tol)
                if np.dot(q_best, q_ref) < 0:
                    q_best *= -1.0
                q_sum += q_best
                M_best_cluster[j] = M_best

            # average quaternion
            q_mean = q_sum / np.linalg.norm(q_sum)
            M_mean = quat_to_mat(q_mean)

            avg_q_dict[lab] = q_mean
            avg_M_dict[lab] = M_mean
            M_best_dict[lab] = M_best_cluster

        return avg_M_dict, avg_q_dict, M_best_dict
    


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
    
    def __init__(self, figsize=(10, 10), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
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
    def plotClusters(self,clustering_result: ClusteringResult, d=[1,0,0], cluster_id=None, color_by='cluster', cmap='jet', tiling=None, scalebar=True,globalScale=False, roi=None, phase=None, color=None, data=None, vmin=None,vmax=None, fig=None,ax=None, **kwargs):
        plot=True
        #if cluster_id is None:
        #    mask=None
        #else:
        mask,phase =clustering_result._getMask(cluster_id=cluster_id, roi=roi, phase=phase)
        if np.where(mask)[0].shape[0]<2:
            print('Data with only 1 pixel or less cannot be plotted')
            plot=False
        if plot:
            if color_by=='data' or color_by.lower()=='kam' or color_by.lower()=='gnd':
                #mask, phase = self._getMask(clustering_result, roi, cluster_id, phase)
                if color_by.lower()=='kam':
                    data=clustering_result.data.kam
                if color_by.lower()=='gnd':
                    data=clustering_result.data.GND
                #print(color_by)
                if vmin is None:
                    vmin = np.nanmin(data)
                if vmax is None:
                    vmax = np.nanmax(data)
                norm = plt.Normalize(vmin,vmax)
                cmap = plt.get_cmap(cmap)
                Colors = cmap(norm(data))
                Colors=Colors*255
                Colors=Colors.astype(int)
                Colors[:,3]=255
                #print(mask)
                #print(np.where(mask)[0].shape)
                fig,ax = clustering_result.data.plot_colmap(d=d,tiling=tiling, scalebar=scalebar,globalScale=globalScale, color=Colors, mask=mask, fig=fig, ax=ax, **kwargs)
            if color_by=='cluster':
                #print(color_by)
                clusters_unique = clustering_result._clusters_unique
                labels = clustering_result.labels
                Colors = self.getColors(clusters_unique, labels, cmap=cmap)
                if color is None:
                    color = Colors
                if roi is None:
                    roi = clustering_result.parameters['roi']
                fig,ax = clustering_result.data.plot_colmap(d=d,tiling=tiling, scalebar=scalebar,globalScale=globalScale, roi=roi, phase=phase, color=color, mask=mask,fig=fig, ax=ax, **kwargs)
            elif color_by=='avgipf':
                #print(mask)
                labels = clustering_result.labels
                avg_orientations = clustering_result.avg_orientations
                Mavg=copy.deepcopy(clustering_result.data.orientations)
                for label in avg_orientations.keys():
                    Mavg[labels==label,:,:]=avg_orientations[label] 
                if roi is None:
                    roi = clustering_result.parameters['roi']
                clustering_result.data.plot_IPF(d, tiling=tiling, scalebar=scalebar,globalScale=globalScale, roi=roi, phase=phase, orientations=Mavg, mask=mask,fig=fig, ax=ax, **kwargs)
            elif color_by=='ipf':
                if roi is None:
                    roi = clustering_result.parameters['roi']
                clustering_result.data.plot_IPF(d, tiling=tiling, scalebar=scalebar,globalScale=globalScale, roi=roi, phase=phase, orientations=None, mask=mask,fig=fig, ax=ax, **kwargs)
        return fig,ax
#clustering_result.get_cluster_mask(1)

    def plotAvgOriIPF(self,d, tiling=None, scalebar=True,globalScale=False, roi=None, phase=None, fig=None, ax=None,  **kwargs):
        Mavg=copy.deepcopy(self.data.orientations)
        for label in self.avg_orientations.keys():
            Mavg[self.labels==label,:,:]=self.avg_orientations[label] 
        if roi is None:
            roi = self.parameters['roi']
            
        self.data.plot_IPF(d, tiling=tiling, scalebar=scalebar,globalScale=globalScale, roi=roi, phase=phase, orientations=Mavg,fig=fig, ax=ax, **kwargs)

    def plot_clustering(self, clustering_result: ClusteringResult, 
                       color_by='cluster', **kwargs):
        """
        Plot clustering results.
        
        Parameters
        ----------
        clustering_result : ClusteringResult
            Results to plot
        color_by : str
            'cluster', 'phase', or 'quality'
        """
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
        ax.yaxis.set_inverted(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Clustering Result - colored by {color_by}')
        ax.axis('equal')
        
        if color_by in ['phase', 'quality']:
            plt.colorbar(scatter, ax=ax)
        
        self.fig = fig
        self.axes = [ax]
        
        return fig, ax
    
    def plot_boundaries(self, boundary_result: BoundaryResult,
                       boundary_type='all', **kwargs):
        """
        Plot grain boundaries.
        
        Parameters
        ----------
        boundary_result : BoundaryResult
            Boundary analysis results
        boundary_type : str
            'all', 'interphase', 'same_phase', 'roi'
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
        ax.yaxis.set_inverted(True)
        ax.axis('equal')
        
        return fig, ax
    
    def plot_single_cluster(self, boundary_result: BoundaryResult,
                           cluster_label: int, **kwargs):
        """Plot boundaries for a single cluster."""
        # Implementation from your plot_single_cluster_interphase_boundary
        pass
    
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
    
    def _get_phase_colors(self, phases):
        """Generate colors for phases."""
        import matplotlib.cm as cm
        unique_phases = np.unique(phases)
        cmap = cm.get_cmap('Set1', len(unique_phases))
        
        colors = np.zeros((len(phases), 4))
        for i, phase in enumerate(unique_phases):
            colors[phases == phase] = cmap(i)
        
        return colors

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
