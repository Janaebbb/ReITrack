import torch
import torch.nn.functional as F
import numpy as np

from trackers.ReITrack.osformer.osformer_motion import get_motion_model, linear_motion_pred, avg_vel
from trackers.ReITrack.association import iou_batch, associate, linear_assignment, oai
from trackers.ReITrack.utils import calibrate_idx
from trackers.ReITrack.embedding import EmbeddingComputer
from trackers.ReITrack.cmc import CMCComputer


class STrack(object):

    count = 0

    def __init__(self, bbox, delta_t=3, emb=None):
        self.time_since_update = 0
        self.id = STrack.count
        STrack.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self._tlwh = bbox.float()
        self.sequence = self._tlwh.unsqueeze(0)
        self.d_sequence = torch.zeros((1, 1, 4))
        self.pred = None

        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t

        self.emb = emb

        self.frozen = False

        self.pred_seq = []

    def update(self, bbox):
        if bbox is not None:
            self.frozen = False
            self.pred_seq = []
            if self.last_observation.sum() >= 0:
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation

                self.velocity = speed_direction(previous_box, bbox)

            self.last_observation = bbox
            self.observations[self.age] = bbox

            # comment for DANCE
            # if self.time_since_update > 1:
            #     self.oos()

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            # motion update
            new_tlwh = self.tlbr_to_tlwh(bbox[:4])
            self.sequence = torch.cat((self.sequence, new_tlwh.unsqueeze(1)), dim=1)
            self.d_sequence = torch.cat((self.d_sequence, self.sequence[:, -1:, :] - self.sequence[:, -2:-1, :]), dim=1)
        else:
            self.sequence = torch.cat((self.sequence, self.pred.unsqueeze(1)), dim=1)
            self.d_sequence = torch.cat((self.d_sequence, self.sequence[:, -1:, :] - self.sequence[:, -2:-1, :]), dim=1)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    @staticmethod
    def multi_predict(stracks, use_osformer=False):
        osformer_pred_stracks = []
        linear_pred_stracks = []
        for st in stracks:
            if len(st.pred_seq) == 0:
                if st.sequence.shape[1] >= 700 and use_osformer:
                    osformer_pred_stracks.append(st)
                else:
                    linear_pred_stracks.append(st)
            else:
                st.motion_from_pred_seq()

        max_seq_len = ReITracker.motion_model.seq_len
        pred_len = ReITracker.motion_model.pred_len
        # pred_len = 5

        # for long-lost tracks
        if len(osformer_pred_stracks) > 0:
            multi_lastest_pos = []
            multi_avg_vel = []
            multi_avg_vel_mask = []
            for st in osformer_pred_stracks:
                multi_lastest_pos.append(st.sequence[:, -1:])
                if st.sequence.shape[1] - 1 < max_seq_len:
                    pad_dim = max_seq_len - st.sequence.shape[1] + 1
                    seq_avg_vel = avg_vel(st.sequence)
                    multi_avg_vel.append(
                        F.pad(seq_avg_vel.permute(0, 2, 1), pad=(pad_dim, 0, 0, 0), mode='constant', value=0).permute(
                            0, 2, 1))
                    multi_avg_vel_mask.append(
                        torch.tensor([0] * pad_dim + [1] * (max_seq_len - pad_dim)).reshape(1, -1, 1))
                else:
                    seq_avg_vel = avg_vel(st.sequence[:, -(max_seq_len + 1):])
                    multi_avg_vel.append(seq_avg_vel)
                    multi_avg_vel_mask.append(torch.ones([1, max_seq_len, 1]))
            multi_avg_vel = torch.cat(multi_avg_vel, dim=0).permute(1, 0, 2).to(
                torch.float32)  # (seq_len, track_num, 4)
            multi_avg_vel_mask = torch.cat(multi_avg_vel_mask, dim=0).permute(1, 0, 2).to(torch.float32)
            multi_lastest_posi = torch.cat(multi_lastest_pos, dim=0).permute(1, 0, 2).to(torch.float32)
            multi_pred = ReITracker.motion_model(multi_avg_vel.cuda(),
                                                 multi_avg_vel_mask.cuda(),
                                                 multi_lastest_posi.cuda())  # (track_num, pred_len, 4)

            multi_pred = multi_pred.detach().cpu()
            for i in range(multi_pred.shape[0]):
                osformer_pred_stracks[i].pred_seq = multi_pred[i][:pred_len]
                osformer_pred_stracks[i].motion_from_pred_seq()

        # for short-lost tracks
        if len(linear_pred_stracks) > 0:
            multi_seq = []
            for st in linear_pred_stracks:
                if st.sequence.shape[1] < max_seq_len:
                    pad_dim = max_seq_len - st.sequence.shape[1]
                    multi_seq.append(F.pad(st.sequence.permute(0, 2, 1), pad=(pad_dim, 0, 0, 0), mode='constant', value=torch.nan).permute(
                            0, 2, 1))
                else:
                    multi_seq.append(st.sequence[:, -max_seq_len:])
            multi_seq = torch.cat(multi_seq, dim=0)
            multi_pred = linear_motion_pred(multi_seq, pred_len=10)
            for i in range(multi_pred.shape[0]):
                linear_pred_stracks[i].pred_seq = multi_pred[i]
                linear_pred_stracks[i].motion_from_pred_seq()

    def motion_from_pred_seq(self):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.pred = self.pred_seq[:1]
        self.pred_seq = self.pred_seq[1:]
        pred_tlbr = self.pred[0].numpy().copy()
        pred_tlbr[2:] = pred_tlbr[:2] + pred_tlbr[2:]

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

    def oos(self):
        last_bbox = self.sequence[0][-1].numpy()
        start_bbox = self.sequence[0][-(self.time_since_update + 1)].numpy()
        time_gap = self.time_since_update
        d_tlbr = (last_bbox - start_bbox) / time_gap
        for i in range(self.time_since_update - 1):
            revised_bbox = start_bbox + (i + 1) * d_tlbr
            revised_bbox[2:] -= revised_bbox[:2]
            self.sequence[0][-self.time_since_update + i] = torch.from_numpy(revised_bbox)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = torch.from_numpy(tlbr.copy()).unsqueeze(0)
        ret[:, 2:] -= ret[:, :2]
        return ret


class ReITracker(object):

    # DANCE
    # motion_model_cfg = "./weights/OneStepFormer/DANCE/DANCE_v1_mse_lr0.001_step/DANCE_v1_mse_lr0.001_step.yml"
    # motion_epoch = 'dance'

    # MOT20
    # motion_model_cfg = "./weights/OneStepFormer/MOT20/MOT20_v1_mse_lr0.001_step/MOT20_v1_mse_lr0.001_step.yml"
    # motion_epoch = 'mot20'

    def __init__(
            self,
            det_thresh,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            delta_t=3,
            inertia=0.2,
            w_association_emb=0.75,
            alpha_fixed_emb=0.95,
            aw_param=0.5,
            aw_off=False,
            **kwargs,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = iou_batch
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        STrack.count = 0

        self.embedder = EmbeddingComputer(kwargs["args"].dataset, kwargs["args"].test_dataset)
        self.cmc = CMCComputer()
        self.aw_off = aw_off

        self.cmc_affine_list = []

        self.osformer = kwargs["args"].OSFormer
        self.oaw = kwargs["args"].OAW

    @classmethod
    def load_motion_model(cls):
        cls.motion_model = get_motion_model(
            motion_model_cfg=cls.motion_model_cfg,
            motion_epoch=cls.motion_epoch
        )

    def update(self, output_results, img_tensor, img_numpy, tag):
        if output_results is None:
            return np.empty((0, 5))
        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()
        self.frame_count += 1
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        bboxes[:, :4] /= scale

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        dets = dets[dets[:, -1] > 0.1]
        scores = scores[scores > 0.1]
        det_idx = np.asarray(list(range(len(dets))))
        det_idx_high = det_idx[scores > self.det_thresh]
        det_idx_low = det_idx[np.logical_and(scores > 0.1, scores < self.det_thresh)]

        dets_embs = np.ones((dets.shape[0], 1))
        if dets.shape[0] != 0:
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)

        # CMC
        transform = self.cmc.compute_affine(img_numpy, dets[:, :4], tag)
        self.cmc_affine_list.append(transform.reshape(6, ))
        for trk in self.trackers:
            trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)

        matched, unmatched_dets, unmatched_trks = self._match(dets, dets_embs, det_idx_high, det_idx_low)

        rm_det_idx_idx = oai(dets, unmatched_dets, self.trackers)
        unmatched_dets = np.delete(unmatched_dets, rm_det_idx_idx)

        ret = []
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        for m in unmatched_trks:
            self.trackers[m].update(None)

        for i in unmatched_dets:
            trk = STrack(
                STrack.tlbr_to_tlwh(dets[i, :4]), delta_t=self.delta_t, emb=dets_embs[i]
            )
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.sequence[0, -1].numpy().copy()
                d[2:] += d[:2]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _match(self, dets, dets_embs, det_idx_high=None, det_idx_low=None):
        stracks_pool = self.trackers
        STrack.multi_predict(stracks_pool, self.osformer)
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        trk_lost_time = []
        for t, trk in enumerate(trks):
            if self.trackers[t].last_observation.sum() < 0:
                pos = self.trackers[t].sequence[0, -1].numpy().copy()
            else:
                pos = self.trackers[t].pred[0].numpy().copy()
            pos[2:] += pos[:2]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
            trk_lost_time.append(self.trackers[t].time_since_update)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trk_embs = np.array(trk_embs)
        trk_lost_time = np.array(trk_lost_time)
        for t in reversed(to_del):
            self.trackers.pop(t)
        last_boxes = np.array([trk.last_observation for trk in self.trackers])

        """
            First round of association
        """
        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])
        matched, unmatched_dets, unmatched_trks = associate(
            dets[det_idx_high],
            trks,
            dets_embs[det_idx_high],
            trk_embs,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
            self.oaw,
            trk_lost_time
        )
        matched, unmatched_dets, unmatched_trks = calibrate_idx(matched, unmatched_dets, unmatched_trks,
                                                                det_idx_high, np.asarray(list(range(len(trks)))))
        """
            Second round of associaton
        """
        if (len(unmatched_dets) + len(det_idx_low)) > 0 and len(unmatched_trks) > 0:
            det_idx_second = np.concatenate((unmatched_dets, det_idx_low), axis=0)
            dets_second = dets[det_idx_second]
            if self.osformer:
                left_trks = last_boxes[unmatched_trks]
            else:
                left_trks = trks[unmatched_trks]

            iou_left = self.asso_func(dets_second, left_trks)
            iou_left = np.array(iou_left)
            final_cost = -iou_left

            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(final_cost)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = det_idx_second[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    matched = np.concatenate((matched.reshape(-1, 2), np.asarray([[det_ind, trk_ind]])), axis=0)
                    to_remove_trk_indices.append(trk_ind)
                    if det_ind not in det_idx_low:
                        to_remove_det_indices.append(det_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        return matched, unmatched_dets, unmatched_trks

    def dump_cache(self):
        self.cmc.dump_cache()
        self.embedder.dump_cache()

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res
