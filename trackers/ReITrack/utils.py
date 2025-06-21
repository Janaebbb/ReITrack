import numpy as np


def calibrate_idx(matched, unmatched_dets, unmatched_trks, det_indices, trk_indices):
    new_unmatched_dets = unmatched_dets if len(unmatched_dets) == 0 else det_indices[unmatched_dets]
    new_unmatched_trks = unmatched_trks if len(unmatched_trks) == 0 else trk_indices[unmatched_trks]
    new_matched = []
    for d_i, t_j in matched:
        det_idx = det_indices[d_i]
        trk_idx = trk_indices[t_j]
        new_matched.append((det_idx, trk_idx))
    new_matched = np.asarray(new_matched)
    return new_matched.astype('int64'), new_unmatched_dets.astype('int64'), new_unmatched_trks.astype('int64')
