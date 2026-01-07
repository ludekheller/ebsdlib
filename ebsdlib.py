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
import copy
import numpy as np
from orilib import * # Orientations, quaternions, Euler angles
from projlib import * # Stereographic projections
from plotlib import * # Crystallographic plotting
from crystlib import * # Crystallographic calculations
from effelconst import  * #effective elastic constants calculations

from crystals import Crystal
from numpy import sqrt
from getphases import getPhases

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
def find_cluster_neighbors_with_lengths_and_boundaries_numba_roi(labels_2d, inside_mask):
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


import numpy as np

def prepare_boundaries_for_numba(grouped_boundaries):
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


@njit
def representative_points_from_grouped_boundaries_numba(
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



def representative_points_from_grouped_boundaries(grouped_boundaries, n_samples=1000):
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
