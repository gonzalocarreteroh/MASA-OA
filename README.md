# Multiple Object Tracker with Integrated Embedding-Based Optimization and Occlusion-Aware Variants


[Applied Artificial Intelligence Group](https://www.uc3m.es/ss/Satellite/GruposInvestigacion/en/Detalle/Organismo_C/1422545966182/1371325143833/Applied_Artificial_Intelligence_Group_(GIAA)) at UC3M | [Cognitive Science Research Group](https://www.uah.es/es/investigacion/unidades-de-investigacion/grupos-de-investigacion/Grupo-de-Investigacion-en-Ciencia-Cognitiva-Cognitive-Science-Reserch-Group/) at UAH


## Overview

This is the official repository for the _Multiple Object Tracker with Integrated Embedding-Based Optimization and Occlusion-Aware Variants_ work, which builds upon the [Matching Anything by Segmenting Anything](https://matchinganything.github.io/) tracker to include an optimized assignment strategy and occlusion-aware capabilities integrating Kalman Filter motion estimation.


* For our evaluation, we used the MASA-R50: MASA with ResNet-50 backbone to extract object embeddings. However, in practice, other variants could also be used to perform the tracking together with our MASA-OA tracker implementation.

## Structure
- `demo/video_demo_with_text.py`: This is the main script in which you can specify the detection inputs, run the MASA-OA tracking, and generate the output video with tracked objects as well as the text files for performance evaluation with TrackEval.
- `masa/models/tracker/masa_tao_tracker.py`: This file contains the implementation of the MASA-OA tracker with the integrated assignment optimization and occlusion-aware capabilities.
- `masa/models/tracker/MASA_OA_KF/kalman_filter.py`: This file contains the Kalman Filter implementation used for motion estimation in the MASA-OA tracker.
- `masa/models/tracker/MASA_OA_KF/track.py`: This file contains the tracking logic that integrates the Kalman Filter into the tracker.
- `benchmarks`: You can place the input videos and detection text files here from which the model will generate the tracking outputs for evaluation. An example is provided with the MOT15 dataset.
- `benchmarks/imgs_to_video.py`: This script converts image sequences provided by MOTChallenge to video format for easier visualization and evaluation.


## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA compatible GPU

### Environment Setup
1. Clone the repository and install dependencies following the original MASA installation instructions

2. Download the [MASA-R50](https://huggingface.co/dereksiyuanli/masa/resolve/main/masa_r50.pth) weights and place it in `saved_models/masa_models/` directory:
```bash
mkdir -p saved_models/masa_models/
# Download masa_r50.pth to saved_models/masa_models/
```

## Benchmark Evaluation with Ground Truth Detections

This implementation is designed to evaluate the MASA-OA tracker on MOT benchmarks using **ground truth detections**. Currently, the repository includes setup for the MOT15 dataset.

### Benchmark Data Structure

The `benchmarks/` directory should be organized as follows:
```
benchmarks/
└── MOT15/
    ├── SEQUENCE_NAME/
    │   ├── SEQUENCE_NAME.mp4        # Video file
    │   ├── seqinfo.ini              # Sequence information
    │   └── gt/
    │       └── gt.txt               # Ground truth detections
    └── imgs_to_video.py             # Utility to convert image sequences to video
```

### Available MOT15 Sequences
The following sequences are currently included:
- ADL-Rundle-6
- ADL-Rundle-8
- ETH-Bahnhof
- ETH-Pedcross2
- ETH-Sunnyday
- KITTI-13
- KITTI-17
- PETS09-S2L1
- TUD-Campus
- TUD-Stadtmitte
- Venice-2

### Running Evaluation on a Single Sequence (Manual Method)

**Important:** Currently, you need to manually modify `demo/video_demo_with_text.py` for each video sequence you want to evaluate.

#### Step 1: Modify video_demo_with_text.py

Open `demo/video_demo_with_text.py` and locate the following lines (around lines 208-211 and 289):

1. **Line 208** - Detection file path:
```python
detections_dict = load_detections('./benchmarks/MOT15/Venice-2/gt/gt.txt')
```

2. **Line 210** - Input video path:
```python
video_path = "./benchmarks/MOT15/Venice-2/Venice-2.mp4"
```

3. **Line 211** - Output visualization path (optional, for GT bounding boxes visualization):
```python
output_path = "./GCH_eval/MOT15/Venice-2/Venice-2_GT_bboxes.mp4"
```

4. **Line 289** - Output tracking results path:
```python
output_mot_txt = './GCH_eval/MOT15/Venice-2/Venice-2_masa-hung-kf-OcclusionAware.txt'
```

Replace `Venice-2` with the name of the sequence you want to evaluate (e.g., `KITTI-13`, `ETH-Sunnyday`, etc.).

#### Step 2: Run the Tracker

After modifying the paths, run the script:

```bash
python demo/video_demo_with_text.py \
    --masa_config configs/masa-one/masa_r50_plug_and_play.py \
    --masa_checkpoint saved_models/masa_models/masa_r50.pth \
    --score-thr 0.2 \
    --show_fps
```

**Note:** The `--video` and `--out` arguments are not used in the current implementation as the paths are hardcoded in the script.

#### Step 3: Output Files

The script generates two main outputs:

1. **Tracking results** (`.txt` file):
   - Location: `GCH_eval/MOT15/SEQUENCE_NAME/SEQUENCE_NAME_masa-hung-kf-OcclusionAware.txt`
   - Format: MOTChallenge format compatible with TrackEval
   - Contains: frame, track_id, bbox coordinates, confidence score

2. **Visualization video** (`.mp4` file):
   - Location: `GCH_eval/MOT15/SEQUENCE_NAME/SEQUENCE_NAME_GT_bboxes.mp4`
   - Shows ground truth bounding boxes overlaid on the video

### Evaluation with TrackEval

Once you have generated the tracking results (`.txt` files), you can evaluate them using [TrackEval](https://github.com/JonathonLuiten/TrackEval):

1. Install TrackEval:
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
cd TrackEval
pip install -r requirements.txt
```

2. Organize your tracking results and ground truth in the TrackEval format

3. Run evaluation to compute metrics (MOTA, MOTP, IDF1, etc.)

### Tracker Variants

The repository includes several tracker variants. You can modify which variant to use by checking the configuration in `masa/models/tracker/masa_tao_tracker.py`:

- **masa**: Original MASA tracker
- **masa-hung**: MASA with Hungarian algorithm optimization
- **masa-hung-kf**: MASA with Hungarian algorithm and Kalman Filter motion estimation
- **masa-hung-kf-OcclusionAware**: MASA with Hungarian algorithm, Kalman Filter, and occlusion-aware capabilities (default)

The Kalman Filter implementation and tracking logic are located in:
- `masa/models/tracker/MASA_OA_KF/kalman_filter.py`
- `masa/models/tracker/MASA_OA_KF/track.py`

### Converting Image Sequences to Video

If you have MOTChallenge image sequences instead of videos, use the provided utility:

```bash
cd benchmarks
python imgs_to_video.py
```

## Automated Benchmark Evaluation

For easier processing, use the provided batch processing script `run_benchmark.py` to automatically run the tracker on all sequences in a benchmark without manual code modifications.

### Basic Usage

Process all sequences in MOT15:
```bash
python run_benchmark.py --benchmark benchmarks/MOT15
```

### Advanced Options

Process specific sequences only:
```bash
python run_benchmark.py --benchmark benchmarks/MOT15 --sequences Venice-2 KITTI-13 ETH-Sunnyday
```

Generate ground truth visualization videos:
```bash
python run_benchmark.py --benchmark benchmarks/MOT15 --generate-gt-video
```

Use a different tracker variant:
```bash
python run_benchmark.py --benchmark benchmarks/MOT15 --tracker-variant masa-hung-kf
```

Continue processing even if a sequence fails:
```bash
python run_benchmark.py --benchmark benchmarks/MOT15 --continue-on-error
```

### Available Arguments

**Required:**
- `--benchmark`: Path to benchmark directory (e.g., `benchmarks/MOT15`)

**Model Configuration:**
- `--masa-config`: Path to MASA config file (default: `configs/masa-one/masa_r50_plug_and_play.py`)
- `--masa-checkpoint`: Path to MASA checkpoint (default: `saved_models/masa_models/masa_r50.pth`)

**Output Configuration:**
- `--output-dir`: Output directory for results (default: `GCH_eval`)
- `--tracker-variant`: Tracker variant name (default: `masa-hung-kf-OcclusionAware`)

**Sequence Selection:**
- `--sequences`: Specific sequences to process (default: all)

**Processing Options:**
- `--generate-gt-video`: Generate visualization video with GT bboxes
- `--score-thr`: Bbox score threshold (default: 0.2)
- `--device`: Device for inference (default: `cuda:0`)
- `--show-fps`: Show FPS during processing
- `--fp16`: Use FP16 mode
- `--no-post`: Disable post-processing

**Execution Control:**
- `--continue-on-error`: Continue processing remaining sequences if one fails

### Output Structure

The script generates outputs organized by benchmark and sequence:

```
GCH_eval/
└── MOT15/
    ├── Venice-2/
    │   ├── Venice-2_masa-hung-kf-OcclusionAware.txt
    │   └── Venice-2_GT_bboxes.mp4 (if --generate-gt-video is used)
    ├── KITTI-13/
    │   └── KITTI-13_masa-hung-kf-OcclusionAware.txt
    └── ...
```

### Example Workflow

1. **Process all MOT15 sequences with default settings:**
   ```bash
   python run_benchmark.py --benchmark benchmarks/MOT15
   ```

2. **The script will automatically:**
   - Detect all valid sequences in the benchmark
   - Process each sequence sequentially
   - Generate tracking results in MOTChallenge format
   - Display progress and summary

3. **Evaluate with TrackEval:**
   - After processing, use the generated `.txt` files with TrackEval to compute metrics

## Contact
For questions, please contact [Gonzalo Carretero](https://github.com/gonzalocarreteroh).

### Official Citation 
Our work is still under review. We will update this section when appropriate.
```bibtex
@misc{masa,
  author    = {Gonzalo Carretero, Juan P. Llerena, Luis Usero, Miguel A. Patricio},
  title     = {Multiple Object Tracking with Integrated Embedding-Based Optimization and Occlusion-Aware Variants},
  journal   = {},
  year      = {2025},
}
```

If citing our paper, please consider also citing the MASA paper:
```bibtex
@article{masa,
  author    = {Li, Siyuan and Ke, Lei and Danelljan, Martin and Piccinelli, Luigi and Segu, Mattia and Van Gool, Luc and Yu, Fisher},
  title     = {Matching Anything By Segmenting Anything},
  journal   = {CVPR},
  year      = {2024},
}
```

### Acknowledgments

The authors would like to thank: [Siyuan Li](https://siyuanliii.github.io/) and all the authors of the [Matching Anything by Segmenting Anything](https://matchinganything.github.io/) paper.
