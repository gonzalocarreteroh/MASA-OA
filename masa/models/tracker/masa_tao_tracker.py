"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications:
- Portions of this file have been modified by Gonzalo Carretero with respect to Siyuan Li's original version. Modifications include:
- Integration of Kalman Filter-based prediction and update steps for object tracking.
- Implementation of a matching strategy that combines appearance features with spatial proximity using the Hungarian algorithm.
- Addition of methods to convert bounding box formats for Kalman Filter processing.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.trackers.base_tracker import BaseTracker
from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from torch import Tensor

# GCH
from scipy.optimize import linear_sum_assignment
import numpy as np
from .MASA_OA_KF.kalman_filter import KalmanFilter
from .MASA_OA_KF.track import Track as DSTrack


@MODELS.register_module()
class MasaTaoTracker(BaseTracker):
    """Tracker for MASA on TAO benchmark.

    Args:
        init_score_thr (float): The cls_score threshold to
            initialize a new tracklet. Defaults to 0.8.
        obj_score_thr (float): The cls_score threshold to
            update a tracked tracklet. Defaults to 0.5.
        match_score_thr (float): The match threshold. Defaults to 0.5.
        memo_tracklet_frames (int): The most frames in a tracklet memory.
            Defaults to 10.
        memo_momentum (float): The momentum value for embeds updating.
            Defaults to 0.8.
        distractor_score_thr (float): The score threshold to consider an object as a distractor.
            Defaults to 0.5.
        distractor_nms_thr (float): The NMS threshold for filtering out distractors.
            Defaults to 0.3.
        with_cats (bool): Whether to track with the same category.
            Defaults to True.
        match_metric (str): The match metric. Can be 'bisoftmax', 'softmax', or 'cosine'. Defaults to 'bisoftmax'.
        max_distance (float): Maximum distance for considering matches. Defaults to -1.
        fps (int): Frames per second of the input video. Used for calculating growth factor. Defaults to 1.
    """

    def __init__(
        self,
        init_score_thr: float = 0.8,
        obj_score_thr: float = 0.5,
        match_score_thr: float = 0.5,
        memo_tracklet_frames: int = 10,
        memo_momentum: float = 0.8,
        distractor_score_thr: float = 0.5,
        distractor_nms_thr=0.3,
        with_cats: bool = True,
        max_distance: float = -1,
        fps=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_momentum = memo_momentum
        self.distractor_score_thr = distractor_score_thr
        self.distractor_nms_thr = distractor_nms_thr
        self.with_cats = with_cats

        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []
        self.max_distance = max_distance  # Maximum distance for considering matches
        self.fps = fps
        self.growth_factor = self.fps / 6  # Growth factor for the distance mask
        self.distance_smoothing_factor = 100 / self.fps

        # GCH
        self.assignment_type = 'hungarian' # 'hungarian' or 'greedy'
        self.use_kf = True
        if self.use_kf:
            self.kf = KalmanFilter()

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []

    # GCH
    # Do the KF predict step on all tracks
    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track_id, track_info in self.tracks.items():
            dstrack = track_info.get("dstrack", None)
            if dstrack is not None:
                dstrack.predict(self.kf)

    # GCH
    def bbox_to_xyah(self, bbox):
        """Convert from detection bbox x1, y1, x2, y2 format to (center x, center y, aspect ratio, height) used in KF"""
        x1, y1, x2, y2 = bbox.detach().cpu().tolist()

        w = x2 - x1
        h = y2 - y1
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        ar = w / h

        return np.array([cx, cy, ar, h], dtype=float)
    
    # GCH
    def xyah_to_bbox(self, mean):
        """Convert from (center x, center y, aspect ratio, height) used by KF to detection bbox x1, y1, x2, y2 format"""
        ret = mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        x1, y1, w, h = ret
        ret = [x1, y1, x1 + w, y1 + h]
        return ret
    
    # GCH
    def separate_overlaps(self, bboxes: Tensor) -> Tuple[List[int], List[List[int]]]:
        """Separate overlapping and non-overlapping bboxes"""
        # compute pairwise IoU matrix
        iou_mat = bbox_overlaps(bboxes, bboxes)
        N = bboxes.size(0)
        overlap_flag = (iou_mat > 0).float().sum(dim=1) > 1
        non_ov = [i for i in range(N) if not overlap_flag[i]]
        # union-find for grouping overlapping
        parent = list(range(N))
        def find(x):
            if parent[x]!=x: parent[x]=find(parent[x])
            return parent[x]
        def union(a,b): parent[find(a)] = find(b)
        for i in range(N):
            for j in range(i+1,N):
                if iou_mat[i,j]>0:
                    union(i,j)
        groups = {}
        for i in range(N):
            if overlap_flag[i]:
                root=find(i)
                groups.setdefault(root, []).append(i)
        return non_ov, list(groups.values())

    def update(
        self,
        ids: Tensor,
        bboxes: Tensor,
        embeds: Tensor,
        labels: Tensor,
        scores: Tensor,
        frame_id: int,
        momentum: float = None,
    ) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        m = self.memo_momentum if momentum is None else momentum
        for id, bb, emb, lab, sc in zip(ids, bboxes, embeds, labels, scores):
            id = int(id)
            if id < 0: continue
            if id in self.tracks:
                info = self.tracks[id]
                info['bbox'] = bb
                info['embed'] = (1-m)*info['embed'] + m*emb
                info['last_frame'] = frame_id
                info['label'] = lab
                info['score'] = sc
                if self.use_kf and info.get('dstrack'):
                    info['dstrack'].update(self.kf, self.bbox_to_xyah(bb))
            else:
                # GCH: Initialize new track with Kalman Filter
                if self.use_kf:
                    mean, cov = self.kf.initiate(self.bbox_to_xyah(bb))
                    self.tracks[id] = dict(
                        bbox=bb, embed=emb, label=lab,
                        score=sc, last_frame=frame_id,
                        dstrack=DSTrack(mean=mean, covariance=cov)
                    )
                else:
                    self.tracks[id] = dict(
                        bbox=bb, embed=emb, label=lab,
                        score=sc, last_frame=frame_id
                    )

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory."""
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_frame_ids = []
        # GCH
        memo_ds_tracks = []

        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v["bbox"][None, :])
            memo_embeds.append(v["embed"][None, :])
            memo_ids.append(k)
            memo_labels.append(v["label"].view(1, 1))
            memo_frame_ids.append(v["last_frame"])
            # GCH
            if self.use_kf and v.get("dstrack") is not None:
                memo_ds_tracks.append(v["dstrack"])

        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_frame_ids = torch.tensor(memo_frame_ids, dtype=torch.long).view(1, -1)

        # GCH: Return memo_ds_tracks as well
        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_ids.squeeze(0),
            memo_frame_ids.squeeze(0),
            memo_ds_tracks,
        )

    def compute_distance_mask(self, bboxes1, bboxes2, frame_ids1, frame_ids2):
        """Compute a mask based on the pairwise center distances and frame IDs with piecewise soft-weighting."""
        centers1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2.0
        centers2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2.0
        distances = torch.cdist(centers1, centers2)

        frame_id_diff = torch.abs(frame_ids1[:, None] - frame_ids2[None, :]).to(
            distances.device
        )

        # Define a scaling factor for the distance based on frame difference (exponential growth)
        scaling_factor = torch.exp(frame_id_diff.float() / self.growth_factor)

        # Apply the scaling factor to max_distance
        adaptive_max_distance = self.max_distance * scaling_factor

        # Create a piecewise function for soft gating
        soft_distance_mask = torch.where(
            distances <= adaptive_max_distance,
            torch.ones_like(distances),
            torch.exp(
                -(distances - adaptive_max_distance) / self.distance_smoothing_factor
            ),
        )

        return soft_distance_mask
    
    # GCH: Solve assignment problem using Hungarian or Greedy algorithm
    def match(self, match_scores, memo_ids, scores):
        if self.assignment_type=='hungarian':
            ids = torch.full((match_scores.size(0),), -1, dtype=torch.long)

            if match_scores.numel() == 0:
                return ids

            cost_matrix = -match_scores.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                conf = match_scores[r, c]
                if conf > self.match_score_thr and scores[r] > self.obj_score_thr:
                    ids[r] = memo_ids[c]
            return ids
        else:
            ids = torch.full((match_scores.size(0),), -1, dtype=torch.long)
            used = torch.zeros(match_scores.size(1), dtype=torch.bool)

            for i in range(match_scores.size(0)):
                conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr and not used[memo_ind]:
                    if scores[i] > self.obj_score_thr:
                        ids[i] = id
                        used[memo_ind] = True
            return ids

    def track(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        feats: List[torch.Tensor],
        data_sample: TrackDataSample,
        rescale=True,
        with_segm=False,
        **kwargs
    ) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_instances`.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                True.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_instances.bboxes
        labels = data_sample.pred_instances.labels
        scores = data_sample.pred_instances.scores

        frame_id = metainfo.get("frame_id", -1)
        # create pred_track_instances
        pred_track_instances = InstanceData()

        # GCH: Do predict step to all tracks to change their bboxes, no matter if assigned later or not
        if self.use_kf:
            self.predict()

        # return zero bboxes if there is no track targets
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            pred_track_instances = data_sample.pred_instances.clone()
            pred_track_instances.instances_id = ids
            pred_track_instances.mask_inds = torch.zeros_like(labels)
            return pred_track_instances

        # get track feats
        rescaled_bboxes = bboxes.clone()
        if rescale:
            scale_factor = rescaled_bboxes.new_tensor(metainfo["scale_factor"]).repeat(
                (1, 2)
            )
            rescaled_bboxes = rescaled_bboxes * scale_factor
        track_feats = model.track_head.predict(feats, [rescaled_bboxes])
        # sort according to the object_score
        _, inds = scores.sort(descending=True)
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        embeds = track_feats[inds, :]
        if with_segm:
            mask_inds = torch.arange(bboxes.size(0)).to(embeds.device)
            mask_inds = mask_inds[inds]
        else:
            mask_inds = []

        # GCH: Use only for original MASA model
        """
        bboxes, labels, scores, embeds, mask_inds = self.remove_distractor(
            bboxes,
            labels,
            scores,
            track_feats=embeds,
            mask_inds=mask_inds,
            nms="inter",
            distractor_score_thr=self.distractor_score_thr,
            distractor_nms_thr=self.distractor_nms_thr,
        )
        """

        # init ids container
        ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

        # GCH: Unpack memo_ds_tracks as well
        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (
                memo_bboxes,
                memo_labels,
                memo_embeds,
                memo_ids,
                memo_frame_ids,
                memo_ds_tracks,
            ) = self.memo

            feats = torch.mm(embeds, memo_embeds.t())
            d2t_scores = feats.softmax(dim=1)
            t2d_scores = feats.softmax(dim=0)
            match_scores_bisoftmax = (d2t_scores + t2d_scores) / 2

            match_scores_cosine = torch.mm(
                F.normalize(embeds, p=2, dim=1),
                F.normalize(memo_embeds, p=2, dim=1).t(),
            )

            match_scores = (match_scores_bisoftmax + match_scores_cosine) / 2

            # GCH: Compute KF bboxes for distance mask
            if self.use_kf:
                if len(memo_ds_tracks):
                    kf_bboxes = torch.stack([
                            torch.tensor(self.xyah_to_bbox(track.mean), dtype=torch.float32, device=bboxes.device)
                            for track in memo_ds_tracks
                        ])
                else:
                    kf_bboxes = memo_bboxes

            if self.max_distance != -1:

                # Compute the mask based on spatial proximity
                current_frame_ids = torch.full(
                    (bboxes.size(0),), frame_id, dtype=torch.long
                )

                # GCH: We will use the KF prediction at any given frame for a track to compute the distance against the current detections
                if self.use_kf:
                    distance_mask = self.compute_distance_mask(
                        bboxes, kf_bboxes, current_frame_ids, memo_frame_ids
                    )

                else:
                    distance_mask = self.compute_distance_mask(
                        bboxes, memo_bboxes, current_frame_ids, memo_frame_ids
                    )

                # Apply the mask to the match scores
                match_scores = match_scores * distance_mask

            # GCH: Separate overlapping and non-overlapping detections
            non_ov, groups = self.separate_overlaps(bboxes)
            # phase1: non-overlap
            if non_ov:
                idx = torch.tensor(non_ov,device=ids.device)
                ids[idx] = self.match(match_scores[idx], memo_ids, scores[idx])
                # update those tracks with default momentum
                self.update(ids[idx], bboxes[idx], embeds[idx], labels[idx], scores[idx], frame_id)
            # track used tracks
            used_tracks = set(ids[ids>=0].tolist())
            # phase2: one per overlap-group with momentum=0.2
            # GCH: Use this code for MASA-Clust2 and MASA-Clust8
            """
            for grp in groups:
                cand = [i for i in grp if ids[i] < 0]
                if not cand:
                    continue
                sub = torch.tensor(cand, device=bboxes.device, dtype=torch.long)
                # build list of unused memo indices
                unused_memo = [t for t in memo_ids.tolist() if t not in used_tracks]
                if not unused_memo:
                    continue
                memo_idx_map = {tid: idx for idx, tid in enumerate(memo_ids.tolist())}
                col_indices = [memo_idx_map[tid] for tid in unused_memo]
                sub_scores = match_scores[sub][:, col_indices]
                sub_memo_ids = torch.tensor(unused_memo, device=bboxes.device, dtype=torch.long)
                # find best pair
                flat = sub_scores.flatten()
                max_idx = int(torch.argmax(flat).item())
                num_cols = sub_scores.size(1)
                row = max_idx // num_cols
                col = max_idx % num_cols
                det_idx = int(sub[row].item())
                trk_id = int(sub_memo_ids[col].item())
                if flat[max_idx] > self.match_score_thr and scores[det_idx] > self.obj_score_thr:
                    ids[det_idx] = trk_id
                    used_tracks.add(trk_id)
                    # update with momentum=0.2 for MASA-Clust2 and momentum=0.8 for MASA-Clust8
                    self.update(
                        ids[det_idx:det_idx+1],
                        bboxes[det_idx:det_idx+1],
                        embeds[det_idx:det_idx+1],
                        labels[det_idx:det_idx+1],
                        scores[det_idx:det_idx+1],
                        frame_id,
                        momentum=0, 
                    )
            """
            # phase3: remaining overlap IoU with momentum=0
            rem = [i for grp in groups for i in grp if ids[i]<0]
            rem = torch.tensor(rem,device=bboxes.device) if rem else None
            # build unmatched track indices
            rem_trks = [tid for tid in memo_ids.tolist() if tid not in used_tracks]
            if rem is not None and rem_trks:
                kf_bbs = memo_bboxes
                if self.use_kf and memo_ds_tracks:
                    kf_bbs = torch.stack([torch.tensor(self.xyah_to_bbox(t.mean),device=bboxes.device) for t in memo_ds_tracks])
                # map trk ids to indices
                idx_map = {tid:i for i,tid in enumerate(memo_ids.tolist())}
                trk_idx = torch.tensor([idx_map[tid] for tid in rem_trks],device=bboxes.device)
                iou_mat = bbox_overlaps(bboxes[rem], kf_bbs[trk_idx])
                cost = -iou_mat.cpu().numpy()
                r,c = linear_sum_assignment(cost)
                for rr,cc in zip(r,c):
                    if iou_mat[rr,cc]>0.3:
                        di = rem[rr]
                        ti = rem_trks[cc]
                        ids[di]=ti
                        # update with momentum=0
                        self.update(ids[di:di+1], bboxes[di:di+1], embeds[di:di+1], labels[di:di+1], scores[di:di+1], frame_id, momentum=0)
            # phase4: global IoU for all remaining with momentum=0.35
            det_rem = (ids<0).nonzero(as_tuple=False).view(-1)
            trk_rem = [tid for tid in memo_ids.tolist() if tid not in set(ids[ids>=0].tolist())]
            if len(det_rem) and trk_rem:
                kf_bbs = memo_bboxes
                if self.use_kf and memo_ds_tracks:
                    kf_bbs = torch.stack([torch.tensor(self.xyah_to_bbox(t.mean),device=bboxes.device) for t in memo_ds_tracks])
                idx_map = {tid:i for i,tid in enumerate(memo_ids.tolist())}
                tr_idx = torch.tensor([idx_map[tid] for tid in trk_rem],device=bboxes.device)
                iou_mat = bbox_overlaps(bboxes[det_rem], kf_bbs[tr_idx])
                cost = -iou_mat.cpu().numpy()
                r,c = linear_sum_assignment(cost)
                for rr,cc in zip(r,c):
                    if iou_mat[rr,cc]>0.3:
                        di = det_rem[rr]
                        ti = trk_rem[cc]
                        ids[di]=ti
                        # update with momentum=0.35
                        self.update(ids[di:di+1], bboxes[di:di+1], embeds[di:di+1], labels[di:di+1], scores[di:di+1], frame_id, momentum=0.35)

        # initialize new tracks
        new_inds = (ids == -1) & (scores > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long
        )
        self.num_tracks += num_news

        self.update(ids, bboxes, embeds, labels, scores, frame_id)
        tracklet_inds = ids > -1
        # update pred_track_instances
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]
        if with_segm:
            pred_track_instances.mask_inds = mask_inds[tracklet_inds]

        return pred_track_instances

    def remove_distractor(
        self,
        bboxes,
        labels,
        scores,
        track_feats,
        mask_inds=[],
        distractor_score_thr=0.5,
        distractor_nms_thr=0.3,
        nms="inter",
    ):
        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(scores < distractor_score_thr, as_tuple=False).squeeze(
            1
        )
        if nms == "inter":
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes[:, :])
        elif nms == "intra":
            cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes)
            ious *= cat_same.to(ious.device)
        else:
            raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]
        if track_feats is not None:
            track_feats = track_feats[valid_inds]

        if len(mask_inds) > 0:
            mask_inds = mask_inds[valid_inds]

        return bboxes, labels, scores, track_feats, mask_inds
