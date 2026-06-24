# YOSO-SLAM
Develope by RISET SLAM ELINS UGM
Electronics and Instrumentation
evaluation and analysis submitted to JOIV : International Journal on Informatics Visualization

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
