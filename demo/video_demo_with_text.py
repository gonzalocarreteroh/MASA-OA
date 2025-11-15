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
- Windows compatibility changes for multiprocessing and resource limits
- Loading detections from .txt files
- Visualization of ground-truth bounding boxes
- Integration with MASA inference for MOT evaluation
"""
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import gc
# GCH: Commenting out resource import for Windows compatibility
# import resource
import argparse
import cv2
import tqdm

import torch
from torch.multiprocessing import Pool, set_start_method

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms

import masa
from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from masa.models.sam import SamPredictor, sam_model_registry
from utils import filter_and_update_tracks

# GCH
from collections import defaultdict
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# GCH: Commenting out resource limit setting for Windows compatibility
"""
def set_file_descriptor_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

# Set the file descriptor limit to 65536
set_file_descriptor_limit(65536)
"""

def visualize_frame(args, visualizer, frame, track_result, frame_idx, fps=None):
    visualizer.add_datasample(
        name='video_' + str(frame_idx),
        image=frame[:, :, ::-1],
        data_sample=track_result[0],
        draw_gt=False,
        show=False,
        out_file=None,
        pred_score_thr=args.score_thr,
        fps=fps,)
    frame = visualizer.get_image()
    gc.collect()
    return frame

def parse_args():

    parser = argparse.ArgumentParser(description='MASA video demo')
    parser.add_argument('video', nargs='?', help='Video file')
    parser.add_argument('--det_config', help='Detector Config file')
    parser.add_argument('--masa_config', help='Masa Config file')
    parser.add_argument('--det_checkpoint', help='Detector Checkpoint file')
    parser.add_argument('--masa_checkpoint', help='Masa Checkpoint file')
    parser.add_argument( '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--save_dir', type=str, help='Output for video frames')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument('--line_width', type=int, default=5, help='Line width')
    parser.add_argument('--unified', action='store_true', help='Use unified model, which means the masa adapter is built upon the detector model.')
    parser.add_argument('--detector_type', type=str, default='mmdet', help='Choose detector type')
    parser.add_argument('--fp16', action='store_true', help='Activation fp16 mode')
    parser.add_argument('--no-post', action='store_true', help='Do not post-process the results ')
    parser.add_argument('--show_fps', action='store_true', help='Visualize the fps')
    parser.add_argument('--sam_mask', action='store_true', help='Use SAM to generate mask for segmentation tracking')
    parser.add_argument('--sam_path',  type=str, default='saved_models/pretrain_weights/sam_vit_h_4b8939.pth', help='Default path for SAM models')
    parser.add_argument('--sam_type', type=str, default='vit_h', help='Default type for SAM models')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    # GCH: New arguments for benchmark evaluation
    parser.add_argument('--det_file', type=str, help='Path to detection/ground truth txt file (MOTChallenge format)')
    parser.add_argument('--input_video', type=str, help='Path to input video file (overrides video argument)')
    parser.add_argument('--output_mot_txt', type=str, help='Path to save tracking results in MOT format')
    parser.add_argument('--output_gt_video', type=str, help='Path to save video with ground truth bboxes visualization')
    parser.add_argument('--tracker_variant', type=str, default='masa-hung-kf-OcclusionAware',
                        help='Tracker variant name (for output filename)')
    args = parser.parse_args()
    return args

# GCH
# Function to extract the detections from a .txt file and translate them into the format expected by MASA
def load_detections(txt_path, device='cpu'):
    """
    Reads a MOTChallenge-style gt.txt and returns a dict:
      frame_idx (0-based) -> (det_bboxes, det_labels)
    where det_bboxes is an Nx5 tensor [x1,y1,x2,y2,score]
    and det_labels is an N tensor of class IDs (here all zeros).
    """
    dets = defaultdict(list)
    with open(txt_path, 'r') as f:
        for line in f:
            vals = line.strip().split(',')
            if len(vals) < 7:
                continue
            # MOTChallenge file: frame, id, bb_left, bb_top, bb_w, bb_h, conf, x,y,z
            fr, _, x, y, w, h, conf = vals[:7]
            fr = int(fr) - 1  # to 0-based
            x, y, w, h, conf = map(float, (x, y, w, h, conf))
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            dets[fr].append([x1, y1, x2, y2, conf])

    mot_dict = {}
    for fr, boxes in dets.items():
        arr = np.array(boxes, dtype=np.float32)
        det_bboxes = torch.from_numpy(arr).to(device)      # (N,5)
        det_labels = torch.zeros((arr.shape[0],),           # all class‑0
                                 dtype=torch.long,
                                 device=device)
        mot_dict[fr] = (det_bboxes, det_labels)
    return mot_dict
  

def main():
    args = parse_args()

    # GCH: Use input_video if provided, otherwise use video argument
    video_path_to_use = args.input_video if args.input_video else args.video

    # For benchmark mode, we don't require --out argument
    if not args.det_file:
        assert args.out, \
            ('Please specify at least one operation (save the '
             'video) with the argument "--out" ')

    # build the model from a config file and a checkpoint file
    if args.unified:
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    else:
        det_model = init_detector(args.det_config, args.det_checkpoint, palette='random', device=args.device)
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
        # build test pipeline
        det_model.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)

    if args.sam_mask:
        print('Loading SAM model...')
        device = args.device
        sam_model = sam_model_registry[args.sam_type](args.sam_path)
        sam_predictor = SamPredictor(sam_model.to(device))

    video_reader = mmcv.VideoReader(video_path_to_use)
    video_writer = None

    #### parsing the text input
    texts = args.texts
    if texts is not None:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
    else:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg)

    if texts is not None:
        masa_model.cfg.visualizer['texts'] = texts
    else:
        masa_model.cfg.visualizer['texts'] = det_model.dataset_meta['classes']

    # init visualizer
    masa_model.cfg.visualizer['save_dir'] = args.save_dir
    masa_model.cfg.visualizer['line_width'] = args.line_width
    if args.sam_mask:
        masa_model.cfg.visualizer['alpha'] = 0.5
    visualizer = VISUALIZERS.build(masa_model.cfg.visualizer)

    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    frame_idx = 0
    instances_list = []
    frames = []
    fps_list = []

    # GCH
    # Load detections from a MOTChallenge-style gt.txt file
    if args.det_file:
        detections_dict = load_detections(args.det_file, device=args.device)
        print(f"Loaded detections from: {args.det_file}")
    else:
        # Fallback to hardcoded path for backward compatibility
        detections_dict = load_detections('./benchmarks/MOT15/Venice-2/gt/gt.txt', device=args.device)

    # Visualize a detection file bboxes coordinates over the corresponding video
    if args.output_gt_video:
        output_path = args.output_gt_video
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Open the video
        cap = cv2.VideoCapture(video_path_to_use)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_idx_det = 0
        print(f"Generating ground truth visualization video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx_det in detections_dict:
                det_bboxes, det_labels = detections_dict[frame_idx_det]
                for bbox in det_bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"Class 0: {score:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
            out.write(frame)
            frame_idx_det += 1
        cap.release()
        out.release()
        print(f"Saved GT visualization video to: {output_path}")
    
    for frame in track_iter_progress((video_reader, len(video_reader))):

        # unified models mean that masa build upon and reuse the foundation model's backbone features for tracking
        if args.unified:
            track_result = inference_masa(masa_model, frame,
                                          frame_id=frame_idx,
                                          video_len=len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          text_prompt=texts,
                                          fp16=args.fp16,
                                          detector_type=args.detector_type,
                                          show_fps=args.show_fps)
            if args.show_fps:
                track_result, fps = track_result
        else:
            # GCH: Load detections for the current frame from the pre-loaded dictionary
            det_bboxes, det_labels = detections_dict.get(frame_idx, (None, None))
            if det_bboxes is None or det_bboxes.numel() == 0:
                # no detections this frame, create empty tensors
                det_bboxes = torch.zeros((0,5), dtype=torch.float32, device=args.device)
                det_labels = torch.zeros((0,), dtype=torch.long, device=args.device)

            track_result = inference_masa(masa_model, frame, frame_id=frame_idx,
                                          video_len=len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          det_bboxes=det_bboxes,
                                          det_labels=det_labels,
                                          fp16=args.fp16,
                                          show_fps=args.show_fps)

            if args.show_fps:
                track_result, fps = track_result

        frame_idx += 1
        if 'masks' in track_result[0].pred_track_instances:
            if len(track_result[0].pred_track_instances.masks) >0:
                track_result[0].pred_track_instances.masks = torch.stack(track_result[0].pred_track_instances.masks, dim=0)
                track_result[0].pred_track_instances.masks = track_result[0].pred_track_instances.masks.cpu().numpy()

        track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
        instances_list.append(track_result.to('cpu'))
        frames.append(frame)
        if args.show_fps:
            fps_list.append(fps)
    
    # GCH: Get the tracker results and puts them in the correct format for evaluation with TrackEval
    if args.output_mot_txt:
        output_mot_txt = args.output_mot_txt
        os.makedirs(os.path.dirname(output_mot_txt), exist_ok=True)
        output_rows = []
        for frame_idx, track_result in enumerate(instances_list, start=1):
            det_sample = track_result.video_data_samples[0]
            pred_tracks = det_sample.pred_track_instances
            ids = pred_tracks.instances_id.cpu().numpy()
            bboxes = pred_tracks.bboxes.cpu().numpy()
            scores = pred_tracks.scores.cpu().numpy()
            for obj_id, bbox, score in zip(ids, bboxes, scores):
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                output_rows.append([
                    frame_idx,             # frame number (1-based)
                    int(obj_id),           # object ID
                    float(x1), float(y1),  # top-left corner
                    float(w), float(h),    # width, height
                    float(score),          # confidence
                    -1, -1, -1             # MOT placeholders
                ])
        # Save as MOT format
        np.savetxt(
            output_mot_txt,
            output_rows,
            fmt=['%d', '%d', '%.2f', '%.2f', '%.2f', '%.2f', '%.6f', '%d', '%d', '%d'],
            delimiter=','
        )
        print("Saved MOT results to:")
        print(output_mot_txt)
    

    if not args.no_post:
        instances_list = filter_and_update_tracks(instances_list, (frame.shape[1], frame.shape[0]))

    if args.sam_mask:
        print('Start to generate mask using SAM!')
        for idx, (frame, track_result) in tqdm.tqdm(enumerate(zip(frames, instances_list))):
            track_result = track_result.to(device)
            track_result[0].pred_track_instances.instances_id = track_result[0].pred_track_instances.instances_id.to(device)
            track_result[0].pred_track_instances = track_result[0].pred_track_instances[(track_result[0].pred_track_instances.scores.float() > args.score_thr).to(device)]
            input_boxes = track_result[0].pred_track_instances.bboxes
            if len(input_boxes) == 0:
                continue
            sam_predictor.set_image(frame)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            track_result[0].pred_track_instances.masks = masks.squeeze(1).cpu().numpy()
            instances_list[idx] = track_result



    if args.out:
        print('Start to visualize the results...')
        num_cores = max(1, min(os.cpu_count() - 1, 16))
        print('Using {} cores for visualization'.format(num_cores))

        if args.show_fps:
            with Pool(processes=num_cores) as pool:

                frames = pool.starmap(
                    visualize_frame, [(args, visualizer, frame, track_result.to('cpu'), idx, fps) for idx, (frame, fps, track_result) in enumerate(zip(frames, fps_list, instances_list))]
                )
        else:
            with Pool(processes=num_cores) as pool:
                frames = pool.starmap(
                    visualize_frame, [(args, visualizer, frame, track_result.to('cpu'), idx) for idx, (frame, track_result) in
                                      enumerate(zip(frames, instances_list))]
                )
        for frame in frames:
            if args.out:
                video_writer.write(frame[:, :, ::-1])

    if video_writer:
        video_writer.release()
    print('Done')


if __name__ == '__main__':
    main()
