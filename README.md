# YOSO-SLAM
Developed by RISET SLAM ELINS UGM
Departement Computer Science and Electronics
Universitas Gadjah Mada
Electronics and Instrumentation


## License
YOSO-SLAM is released under a GPLv3 License.

## Build YOSO-SLAM
### 1. Prerequisistes
#### C++14 or C++0x Compiler
We use the new thread and chrono functionalities of C++14.

#### Pangolin
We use Pangolin for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

#### OpenCV
We use OpenCV to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. Required at leat 3.0. Tested with OpenCV 3.2.0 and 4.4.0.

#### Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. Required at least 3.1.0.

#### DBoW2 and g2o (Included in Thirdparty folder)

We use modified versions of the DBoW2 library to perform place recognition and g2o library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the Thirdparty folder.

#### Python
Required for panoptic-Segmentation inference and alignment of the trajectory with the ground truth. The tested python version is 3.8.10.

#### detectron2
we use detectron2 library to perform panoptic-segmentation inference in python and use the C-python-API to communicate the inference module with the SLAM system. The detectron2 framework can be installed from: https://github.com/facebookresearch/detectron2. We are using the version 0.6 that can be found here: https://github.com/facebookresearch/detectron2@v0.6


### 2. Building YOSO-SLAM library and examples
Clone the repository
```
git clone https://github.com/risetslamelins/YOSO-SLAM
```

We provide a script build.sh to build the Thirdparty libraries and YOSO-SLAM. The program is build and tested with CUDA 11.4 and python3.8 on Jetson Xavier AGX 32Gb (Jetpack 5.1.3).
```
cd YOSO-SLAM
chmod +x build.sh
git clone https://github.com/risetslamelins/YOSO-SLAM
./build.sh
```

### 3. Prepare Panoptic Pretrain Model
- Download YOSO pretrain COCO-Panoptic weight [here](https://github.com/hujiecpp/YOSO/releases/tag/v0.1) and put it in the pretrain model. If you want to try the SLAM with Mask2Former-R50 model, download it [here](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md).
```
cd YOSO-SLAM
mkdir pretrain
mv ~/Downloads/yoso_res50_coco.pth pretrain/
```
- To change the model used, Change **Panoptic_Model** in _config/model_config.yaml_
- _Note:_ Mask2Former CUDA kernel for MSDeformAttn needs to be compiled first, read [here](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md#cuda-kernel-for-msdeformattn).


### 4. TUM/Bonn RGB-D Dataset Example
- Download a sequence from [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) or [Bonn](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html) and uncompress it.
- Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools):
```
python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```
- Execute the following command. Change <Path To Settings> to the path for the TUM1.yaml, TUM2.yaml, TUM3.yaml or bonn.yaml files. Change <Path to Dataset> to the uncompressed sequence folder. Make sure you run the dataset sequence with the correct configuration file to get an accurate result. Change <Path To Association file> to the path to the corresponding associations file.
```
./rgbd_panoptic Vocabulary/ORBvoc.txt <Path To Settings> <Path to Dataset> <Path To Association file>
```

### 5. Launch RealSense RGB-D Example
- Update the parameters in _Examples/RGB-D/RealSense_D455i.yaml_ with your calibration parameters.
- Execute the following command. Change <Path To Settings> to your camera configuration. Change <Path to Output_Directory> to your desired output folder for storing the trajectory files and captured RGB and depth imaged.
```
./rgbd_realsense_D455i Vocabulary/ORBvoc.txt <Path To Settings> <Path to Output_Directory>
```

## Acknowledgements
Our code builds on [Panoptic-SLAM](https://github.com/iit-DLSLab/Panoptic-SLAM) and [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)

## Supplementary data of the JEAS article (Additional files 1–5)

Per-sequence data behind the benchmark, ablation and repeated-run tables of the article (ATE RMSE at full
precision, six decimals, recomputed with `evo`; tracking times as recorded by the timing logs, three decimals),
and the script that reproduces the statistics of Tables 2, 3, 6–8 and 10 and of Sections 5.6, 5.9 and 5.11.
The field-session statistics of Section 5.12 and the ORB-SLAM3/ReFusion tracking times and the initialization
and total times of Table 9 are not covered by these files.

| File | Content |
|---|---|
| `Additional_file_1_ate32_per_sequence.csv` | Per-sequence ATE RMSE (m), 32 sequences × 6 methods, at full precision (6 decimals, the precision of the evo output), recomputed from the stored single-run trajectories with `evo_ape tum <gt> <est> -a` (Tables 2–6; Tables 4–5 show these values rounded to 4 decimals) |
| `Additional_file_2_yoso_slam_statistical_analysis.py` | Python/SciPy script that reproduces Tables 2, 3, 6–8 and 10 of the article: paired Wilcoxon tests with Holm adjustment, effect sizes and bootstrap CIs, global mean, SD, median, minimum and maximum ATE, TOST equivalence analysis with its sensitivity checks (t-based CI, TOST on the log ATE ratio, single-sequence differences beyond the margin), tracking-time tests (with the exact sign-flip p-value), ablation percentages, repeated-run Mann–Whitney tests and the replicated YOSO–Mask2Former equivalence analysis (absolute margin and ATE ratio) |
| `Additional_file_3_tracking_time32_per_sequence.csv` | Mean per-frame tracking-thread latency (s) of YOSO-SLAM, Panoptic-SLAM, Mask2Former-SLAM and YDM-SLAM on the 32 sequences (Section 5.9) |
| `Additional_file_4_ablation10_per_sequence.csv` | Ablation study: ATE RMSE (m, full precision, six decimals, recomputed with `evo_ape tum <gt> <est> -a` from the stored trajectories of the ablation runs; V0, Panoptic-SLAM and the full pipeline are the single runs of Additional file 1) and mean tracking time of the ten sequences under the six configurations of Table 10 |
| `Additional_file_5_repeated_runs_per_run.csv` | Repeated-run experiment: ATE RMSE of each of the four runs of YOSO-SLAM, Mask2Former-SLAM, YDM-SLAM and Panoptic-SLAM on the six repeated sequences (Tables 7–8); Run 1 is the single run of Tables 4–5 |

Reproduce these statistics with:
```
pip install numpy pandas scipy
python Additional_file_2_yoso_slam_statistical_analysis.py
```

### Settings used for the results of the JEAS article
- Test scale 512/800 (640 × 480 frames resized to 683 × 512) for all three panoptic front-ends: `config/YOSO/Base-YOSO.yaml`, `config/Mask2Former/Base-Mask2Former.yaml` and `config/PanopticFPN/panoptic_fpn_R_50_3x.yaml`.
- Panoptic post-processing of YOSO and Mask2Former: `OBJECT_MASK_THRESHOLD: 0.5` and `OVERLAP_THRESHOLD: 0.5` (`config/YOSO/yoso_r50.yaml`, `config/Mask2Former/mask2former_r50.yaml`); PanopticFPN confidence thresholds of 0.5, set in `src/python/inference.py` as in Panoptic-SLAM.
- Epipolar filter (Layer 2, unchanged from Panoptic-SLAM, `src/Tracking.cc`): fundamental matrix estimated by RANSAC from Stuff keypoints (inlier threshold 0.1 px, confidence 0.99, only when more than 40 Stuff matches are found); epipolar-distance threshold τ = 0.1 px.
- Timing: the reported tracking times of the panoptic methods include the per-frame Detectron2 visualization drawn by `VisualizationDemo.run_on_image` (`src/python/predictor.py`) inside the timed call.
