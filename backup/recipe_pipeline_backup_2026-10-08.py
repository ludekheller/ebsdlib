# =============================================================================
#  Recipe-driven EBSD-HP pipeline
# =============================================================================
#  A single `recipe` dict is the source of truth for BOTH:
#    - running a fresh analysis          -> run_pipeline(recipe)
#    - rebuilding dataEBSD on restore    -> hp_checkpoint.rebuild_dataEBSD(recipe)
#
#  Assumes the ebsdlib names are imported in this namespace, e.g.:
#     from ebsdlib import (EBSDData, MisorientationClustering, KamGND,
#                          BoundaryAnalyzer, EbsdHPAnalyzer)
#  (plus numpy as np)
# =============================================================================
import numpy as np


# -----------------------------------------------------------------------------
#  THE RECIPE  (edit here; everything downstream reads from it)
# -----------------------------------------------------------------------------
RECIPE = dict(
    # ---- data + crystallography (rebuilds dataEBSD; raw pixels re-read) ------
    filename = '/home/lheller/DATA/ebsd/SEM/tristan/S2-II-Area4_Rescan5_spherical255_NPAR_Tristan/S2-II-Area4_Rescan5_spherical255_NPAR_Tristan.oh5',  # source .oh5 scan
    cif_A    = '/home/lheller/Documents/diffraction/cifs/NiTi - pm3m.cif',    # austenite CIF (cubic Pm-3m)
    cif_M    = '/home/lheller/Documents/diffraction/cifs/NiTiB19p.cif',       # martensite CIF (B19' monoclinic)
    aA = 3.015,                          # austenite lattice parameter a [Å] (Bhattacharya 2003)
    aM = 2.889, bM = 4.12, cM = 4.622,   # martensite lattice parameters a,b,c [Å]
    betaM = 96.8 * np.pi / 180,          # martensite monoclinic angle beta [rad] (96.8°)
    phaseID = {'A': 1, 'M': 2},          # map phase name -> phase id in the .oh5 data
    or_d    = [1, 0, 0],                 # sample direction passed to select_rois(d=...)
    roiID   = 1,                         # which ROI (region of interest) to analyse
    roi_verts = None,                    # stored ROI polygon vertices (filled by
                                         # finalize_rois); None => select interactively once

    # ---- clustering (MisorientationClustering) ------------------------------
    clustering = dict(
        ang_thr=2.5,                     # grain-boundary misorientation threshold [deg]
        dmax=3,                          # max neighbour distance searched when clustering [px]
        minidxs=[10, 3],                 # min cluster size [px] per phase (e.g. [austenite, martensite])
        distance=1,                      # neighbour step used for pairwise misorientation [px]
        perimeteronly=[False, True],     # per phase: cluster on perimeter pixels only? (A=no, M=yes)
        distance_convention="OIM"),      # neighbour/distance convention ("OIM" grid convention)

    # ---- KAM / GND (KamGND) -------------------------------------------------
    kamgnd = dict(
        distance=2,                      # neighbour shell radius for KAM/GND [px]
        perimeteronly=False,             # use all pixels (not just cluster perimeters)
        maxmis=10),                      # max misorientation counted in KAM [deg] (caps outliers)

    # ---- lamellar-cluster identification (identify_lamellar_clusters) -------
    lamellar = dict(
        phase_id=2,                      # phase to test for lamellar shape (2 = martensite)
        min_aspect_ratio=10,             # min length/width for a cluster to count as a lamella
        use_hexagonal=True,              # treat scan grid as hexagonal in the shape analysis
        use_sklearn=True,                # use scikit-learn PCA for the elongation fit
        show_progress=True,              # print a progress bar
        min_width_threshold=0,           # min lamella width [px] (0 = no lower bound)
        min_pixesl_threshold=10),        # min pixel count for a cluster to be considered

    # ---- physical reconstruction (merge fragmented grains / lamellae) -------
    reconstruct_grains   = dict(
        threshold_deg=2.0),              # merge austenite fragments if disorientation < this [deg]
    reconstruct_lamellas = dict(
        threshold_deg=5.0,               # merge martensite fragments if disorientation < this [deg]
        angle_tol_deg=8.0,               # max angle between fragment principal directions [deg]
        collinearity_tol=1.0,            # max perpendicular offset, in units of mean lamella width
        gap_tol_factor=8.0),             # max end-to-end gap along the lamella, in units of mean width

    # ---- hierarchy + meeting pairs ------------------------------------------
    hierarchy = dict(
        filter_lamellar_to_semi_rois=True,  # split each lamella into its two long-side semi-ROIs
        semi_roi_params=None,            # semi-ROI geometry overrides (None = defaults)
        min_aspect_ratio=-0.1),          # aspect-ratio floor for inclusion (-0.1 = keep all)
    meeting   = dict(
        tip_distance_threshold=15.0),    # two lamella tips are a "meeting pair" if within this [px]

    # ---- downstream analysis params (report / OR / variant compatibility) ---
    params = dict(
        frag_merge_deg=2.0,              # austenite fragment-merge threshold used in the report [deg]
        lamella_merge_deg=2.0,           # martensite fragment-merge threshold used in the report [deg]
        oop_flag_deg=20.0,               # flag habit planes whose out-of-plane angle exceeds this [deg]
        or_preserved_deg=3.0,            # OR-residual threshold for "OR preserved" across a boundary [deg]
        tip_distance_px=15.0,            # tip-proximity threshold for boundary meeting pairs [px]
        ma_threshold_deg=2.0),           # martensite disorientation to group variants within a grain [deg]
)


# -----------------------------------------------------------------------------
#  ROI selection helpers (interactive once, then persisted in the recipe)
# -----------------------------------------------------------------------------
def _apply_roi_verts(d, roi_verts):
    """Rebuild the ROI selection non-interactively from stored polygon vertices
    and store the masks (set_rois). No figure is opened."""
    from matplotlib.path import Path
    verts = [np.asarray(v, float) for v in roi_verts]

    class _StoredSelector:            # minimal stand-in for selectROI
        pass
    sel = _StoredSelector()
    sel.selVerts = list(verts)
    sel.selPaths = [Path(v) for v in verts]
    d.selector = sel
    d.set_rois()                      # inserts the full-area ROI at index 0
    return d


def get_roi_verts(d):
    """Return the user-drawn polygon vertices as a JSON-serialisable list, or
    None. Any auto-inserted full-area rectangle is skipped."""
    sel = getattr(d, 'selector', None)
    verts = getattr(sel, 'selVerts', None)
    if not verts:
        return None
    xmax, ymax = float(d.X.max()), float(d.Y.max())
    out = []
    for v in verts:
        v = np.asarray(v, float)
        is_full = (v.shape[0] in (4, 5)
                   and np.isclose(v[:, 0].min(), 0) and np.isclose(v[:, 1].min(), 0)
                   and np.isclose(v[:, 0].max(), xmax) and np.isclose(v[:, 1].max(), ymax))
        if not is_full:
            out.append(v.tolist())
    return out or None


def finalize_rois(d, recipe=None):
    """Call this AFTER you finish drawing polygons in the select_rois() figure.
    It stores the ROI masks (set_rois) and, if a recipe is given, records the
    polygon vertices in recipe['roi_verts'] so future runs/restores skip the
    interactive step."""
    verts = get_roi_verts(d)          # capture BEFORE set_rois inserts full-area
    d.set_rois()
    if recipe is not None and verts is not None:
        recipe['roi_verts'] = verts
        print(f"stored {len(verts)} ROI polygon(s) in recipe['roi_verts']")
    return d


# -----------------------------------------------------------------------------
#  SETUP: rebuild dataEBSD from the recipe (raw pixels + crystallography + ROIs)
# -----------------------------------------------------------------------------
def build_dataEBSD(recipe, interactive=True):
    """Read the .oh5 and redo the crystallography, then set up the ROIs.

    ROI handling:
      * recipe['roi_verts'] present  -> non-interactive: rebuild masks from the
        stored polygons (no figure), fully ready to use.
      * else, interactive=True       -> open select_rois() for you to draw, then
        DOES NOT call set_rois (it must wait until you finish). Draw the
        polygon(s) and then call  finalize_rois(dataEBSD, RECIPE).
      * else, interactive=False      -> raise (nothing to select from).
    """
    d = EBSDData()
    d.readEBSDdata(recipe['filename'])
    d.fromCif(recipe['cif_A'], 'A')
    d.fromCif(recipe['cif_M'], 'M')
    d.fromLatticeParams('A', a=recipe['aA'])
    d.fromLatticeParams('M', a=recipe['aM'], b=recipe['bM'],
                        c=recipe['cM'], beta=recipe['betaM'])
    d.setPhaseID(recipe['phaseID'])
    d.getOR()
    d.getDefGrad()          # also runs getTwinSys / setPlotAttributes / getEl
    d.getTwinSys()
    d.setPlotAttributes()
    d.getEl()

    roi_verts = recipe.get('roi_verts')
    if roi_verts:
        _apply_roi_verts(d, roi_verts)                 # non-interactive
    elif interactive:
        d.select_rois(d=recipe['or_d'])                # opens figure; DRAW now
        print("\n>>> Draw ROI polygon(s) in the figure, then run:")
        print(">>>     finalize_rois(dataEBSD, RECIPE)")
        print(">>> (this stores the masks and saves the polygons into the recipe)\n")
    else:
        raise RuntimeError(
            "recipe['roi_verts'] is empty and interactive=False — no ROI to "
            "restore. Run build_dataEBSD(recipe, interactive=True), draw the "
            "polygons, then finalize_rois(dataEBSD, recipe).")
    return d


def default_checkpoint_name(recipe):
    """'<oh5 stem>_roi<ID>_checkpoint.h5' derived from the recipe."""
    import os
    stem = os.path.splitext(os.path.basename(recipe['filename']))[0]
    return os.path.join(os.path.dirname(recipe['filename']),f"{stem}_roi{recipe['roiID']}_checkpoint.h5")


# -----------------------------------------------------------------------------
#  FULL PIPELINE (fresh analysis), all parameters read from `recipe`
# -----------------------------------------------------------------------------
def run_pipeline(recipe=RECIPE, verbose=True, checkpoint_path=None, dataEBSD=None):
    """Run the full analysis from a recipe.

    dataEBSD:
        None  -> build it non-interactively from the recipe (requires
                 recipe['roi_verts']; select ROIs once with build_dataEBSD +
                 finalize_rois to populate them).
        obj   -> use this already-set-up dataEBSD (ROIs already finalized).

    checkpoint_path:
        None       -> do not save a checkpoint (default),
        'auto'     -> save to "<oh5 stem>_roi<ID>_checkpoint.h5" (name from recipe),
        <path str> -> save to that path.
    """
    roiID = recipe['roiID']
    pA, pM = recipe['phaseID']['A'], recipe['phaseID']['M']

    # --- data + crystallography ---------------------------------------------
    if dataEBSD is None:
        dataEBSD = build_dataEBSD(recipe, interactive=False)

    # --- clustering ----------------------------------------------------------
    c = recipe['clustering']
    alg = MisorientationClustering(
        ang_thr=c['ang_thr'], dmax=c['dmax'], minidxs=c['minidxs'], roi=roiID,
        distance=c['distance'], perimeteronly=c['perimeteronly'],
        distance_convention=c['distance_convention'])
    clustering_result = alg.fit(dataEBSD)
    dataEBSD.set_rois2d(0)
    clustering_result.update_grid_2d()

    # --- KAM / GND -----------------------------------------------------------
    k = recipe['kamgnd']
    kg = KamGND(distance=k['distance'], perimeteronly=k['perimeteronly'],
                maxmis=k['maxmis'], roi=roiID, cluster_id=None)
    clustering_result = kg.computeKAM(clustering_result)
    clustering_result = kg.computeGND(clustering_result)

    # --- lamellar clusters ---------------------------------------------------
    lam = recipe['lamellar']
    lamellar_child_clusters = clustering_result.identify_lamellar_clusters(
        phase_id=lam['phase_id'], min_aspect_ratio=lam['min_aspect_ratio'],
        use_hexagonal=lam['use_hexagonal'], use_sklearn=lam['use_sklearn'],
        show_progress=lam.get('show_progress', verbose),
        min_width_threshold=lam['min_width_threshold'],
        min_pixesl_threshold=lam['min_pixesl_threshold'])

    # --- boundaries ----------------------------------------------------------
    dataEBSD.set_rois2d(roiID)
    clustering_result.update_grid_2d()
    boundary_result = BoundaryAnalyzer().analyze(clustering_result)

    # --- physical reconstruction (grains, then lamellae) ---------------------
    rg, rl = recipe['reconstruct_grains'], recipe['reconstruct_lamellas']
    parent_groups, parent_disor_matrix = clustering_result.reconstruct_physical_grains(
        phase='A', boundary_result=boundary_result, child_phase='M',
        threshold_deg=rg['threshold_deg'])
    lamella_groups, lamella_disor_matrix = clustering_result.reconstruct_physical_lamellas(
        phase='M', boundary_result=boundary_result, parent_phase='A',
        parent_groups=parent_groups, threshold_deg=rl['threshold_deg'],
        angle_tol_deg=rl['angle_tol_deg'], collinearity_tol=rl['collinearity_tol'],
        gap_tol_factor=rl['gap_tol_factor'])
    clustering_result.merge_physical_grains(lamella_groups, label='lamella')
    clustering_result.merge_physical_grains(parent_groups,  label='grain')

    # --- re-run boundaries + grains so IDs are post-merge --------------------
    dataEBSD.set_rois2d(roiID)
    clustering_result.update_grid_2d()
    boundary_result = BoundaryAnalyzer().analyze(clustering_result)
    parent_groups, parent_disor_matrix = clustering_result.reconstruct_physical_grains(
        phase='A', boundary_result=boundary_result, child_phase='M',
        threshold_deg=rg['threshold_deg'])

    # --- hierarchy -----------------------------------------------------------
    h = recipe['hierarchy']
    parent_child_hierarchy_with_boundaries = boundary_result.get_parent_child_hierarchy_with_boundaries(
        parent_phase_id=pA, child_phase_id=pM,
        filter_lamellar_to_semi_rois=h['filter_lamellar_to_semi_rois'],
        clustering_result=clustering_result, semi_roi_params=h['semi_roi_params'],
        elongation_cache=clustering_result.lamellar_child_clusters,
        min_aspect_ratio=h['min_aspect_ratio'])

    # --- scan-edge + meeting pairs ------------------------------------------
    parent_phase_id = clustering_result.data.phase_ids['A']
    child_phase_id  = clustering_result.data.phase_ids['M']
    parent_ids = set(clustering_result.labels_by_phase['A'])
    child_ids  = set(clustering_result.labels_by_phase['M'])
    boundary_parents  = boundary_result.detect_boundary_clusters(parent_ids)
    boundary_children = boundary_result.detect_boundary_clusters(child_ids)
    if verbose:
        print(f"Boundary parents:  {len(boundary_parents)}")
        print(f"Boundary children: {len(boundary_children)}")
    meeting_pairs = boundary_result.detect_boundary_meeting_pairs(
        parent_child_hierarchy  = parent_child_hierarchy_with_boundaries,
        lamellar_child_clusters = lamellar_child_clusters,
        parent_phase_id         = parent_phase_id,
        child_phase_id          = child_phase_id,
        tip_distance_threshold  = recipe['meeting']['tip_distance_threshold'],
        verbose                 = verbose)

    # --- HP analyzer ---------------------------------------------------------
    hpa = EbsdHPAnalyzer(parent_child_hierarchy_with_boundaries,
                         clustering_result, boundary_result)
    hpa.listLamellarParents()
    for parent_id in hpa.HP.keys():
        for child_id in hpa.HP[parent_id].keys():
            for roi in ('semi_roi_1', 'semi_roi_2'):
                try:
                    hpa.singleHP(roi, parent_id, child_id,
                                 printout=False, nodirs=False, bestscore=False)
                except Exception:
                    print(f"Failed ROI {roi} for parent {parent_id}, child {child_id}")

    # --- OR preservation + variant compatibility -----------------------------
    params = recipe['params']
    or_results = hpa.analyse_or_preservation(
        meeting_pairs=boundary_result.meeting_pairs,
        threshold_deg=params['or_preserved_deg'], verbose=verbose)
    if verbose:
        print(f"OR preserved: {sum(r['or_preserved'] for r in or_results)}/{len(or_results)}")
    hpa.analyse_variant_compatibility(
        grain_groups=clustering_result.parent_grain_groups,
        disor_matrix=clustering_result.parent_disor_matrix,
        meeting_pairs=boundary_result.meeting_pairs,
        ma_threshold_deg=params['ma_threshold_deg'],
        or_threshold_deg=params['or_preserved_deg'], verbose=verbose)

    aux = dict(parent_groups=parent_groups, parent_disor_matrix=parent_disor_matrix,
               lamella_groups=lamella_groups, lamella_disor_matrix=lamella_disor_matrix,
               meeting_pairs=boundary_result.meeting_pairs,
               boundary_parents=boundary_parents, boundary_children=boundary_children,
               lamellar_child_clusters=lamellar_child_clusters,
               params=params, roiID=roiID)

    result = dict(dataEBSD=dataEBSD, clustering_result=clustering_result,
                  boundary_result=boundary_result,
                  hierarchy=parent_child_hierarchy_with_boundaries,
                  hpa=hpa, or_results=or_results, aux=aux, recipe=recipe)

    if checkpoint_path is not None:
        from hp_checkpoint import save_checkpoint
        if checkpoint_path == 'auto':
            checkpoint_path = default_checkpoint_name(recipe)
        save_checkpoint(checkpoint_path,
                        clustering_result = result['clustering_result'],
                        boundary_result   = result['boundary_result'],
                        hierarchy         = result['hierarchy'],
                        hpa               = result['hpa'],
                        recipe            = result['recipe'],
                        or_results        = result['or_results'],
                        aux               = result['aux'])
    return result


# -----------------------------------------------------------------------------
#  Example workflows
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # ---- First time: select ROIs interactively, then run + checkpoint -------
    # dataEBSD = build_dataEBSD(RECIPE, interactive=True)   # draw polygon(s) ...
    # finalize_rois(dataEBSD, RECIPE)                       # ... then store them
    # R = run_pipeline(RECIPE, dataEBSD=dataEBSD, checkpoint_path='auto')

    # ---- Afterwards (RECIPE now has roi_verts): fully non-interactive -------
    R = run_pipeline(RECIPE, checkpoint_path='auto')
    hpa = R['hpa']
    # ... print / plot / export as usual ...
