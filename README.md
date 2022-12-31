

# LVI-SAM-CC_Comments

## Introduction

本仓库是 [LVI-SAM](https://github.com/TixiaoShan/LVI-SAM) 的粗略中文注释。

- LVI-SAM是由 LIO-SAM 和 VINS-Mono组合而成，本仓库只是对LVI-SAM的简答注释，关于其中LIO-SAM和VINS-Mono两部分的详细注释，在另外两个repo中： [LIO-SAM-CC_Comments](https://github.com/Cc19245/LIO-SAM-CC_Comments) 和 [VINS-Mono-CC_Comments](https://github.com/Cc19245/VINS-Mono-CC_Comments)
- 原版 LVI-SAM 坐标系定义较为复杂和混乱，我仔细分析了LVI-SAM中的坐标系定义，并重新修改了外参部分代码，让外参配置更简单。修改后的代码在另一个repo中：[LVI-SAM-Easyused](https://github.com/Cc19245/LVI-SAM-Easyused)

## Done

- 从 [LVI_SAM_gld](https://gitee.com/gao-lidong/LVI_SAM_gld?_from=gitee_search) clone下来，粗略过一遍代码框架

## TODO

- [ ] 整理注释格式，遵循自己使用的Todotree的注释颜色配置
- [ ] 整理 [LIO-SAM-CC_Comments](https://github.com/Cc19245/LIO-SAM-CC_Comments) 和 [VINS-Mono-CC_Comments](https://github.com/Cc19245/VINS-Mono-CC_Comments) 的详细中文注释到本工程中

## Acknowledgements


- [LVI_SAM_gld](https://gitee.com/gao-lidong/LVI_SAM_gld?_from=gitee_search)
- [LVI-SAM](https://github.com/TixiaoShan/LVI-SAM)



---

# LVI-SAM

This repository contains code for a lidar-visual-inertial odometry and mapping system, which combines the advantages of [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM/tree/a246c960e3fca52b989abf888c8cf1fae25b7c25) and [Vins-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) at a system level.

<p align='center'>
    <img src="./doc/demo.gif" alt="drawing" width="800"/>
</p>

---

## Dependency

- [ROS](http://wiki.ros.org/ROS/Installation) (Tested with kinetic and melodic)
- [gtsam](https://github.com/borglab/gtsam/releases) (Georgia Tech Smoothing and Mapping library)
  ```
  wget -O ~/Downloads/gtsam.zip https://github.com/borglab/gtsam/archive/4.0.2.zip
  cd ~/Downloads/ && unzip gtsam.zip -d ~/Downloads/
  cd ~/Downloads/gtsam-4.0.2/
  mkdir build && cd build
  cmake -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF ..
  sudo make install -j4
  ```
- [Ceres](https://github.com/ceres-solver/ceres-solver/releases) (C++ library for modeling and solving large, complicated optimization problems)
  ```
  sudo apt-get install -y libgoogle-glog-dev
  sudo apt-get install -y libatlas-base-dev
  wget -O ~/Downloads/ceres.zip https://github.com/ceres-solver/ceres-solver/archive/1.14.0.zip
  cd ~/Downloads/ && unzip ceres.zip -d ~/Downloads/
  cd ~/Downloads/ceres-solver-1.14.0
  mkdir ceres-bin && cd ceres-bin
  cmake ..
  sudo make install -j4
  ```

---

## Compile

You can use the following commands to download and compile the package.

```
cd ~/catkin_ws/src
git clone https://github.com/TixiaoShan/LVI-SAM.git
cd ..
catkin_make
```

---

## Datasets

<p align='center'>
    <img src="./doc/sensor.jpeg" alt="drawing" width="600"/>
</p>

The datasets used in the paper can be downloaded from Google Drive. The data-gathering sensor suite includes: Velodyne VLP-16 lidar, FLIR BFS-U3-04S2M-CS camera, MicroStrain 3DM-GX5-25 IMU, and Reach RS+ GPS.

```
https://drive.google.com/drive/folders/1q2NZnsgNmezFemoxhHnrDnp1JV_bqrgV?usp=sharing
```

**Note** that the images in the provided bag files are in compressed format. So a decompression command is added at the last line of ```launch/module_sam.launch```. If your own bag records the raw image data, please comment this line out.

<p align='center'>
    <img src="./doc/jackal-earth.png" alt="drawing" width="286.5"/>
    <img src="./doc/handheld-earth.png" alt="drawing" width="328"/>
</p>

---

## Run the package

1. Configure parameters:

```
Configure sensor parameters in the .yaml files in the ```config``` folder.
```

2. Run the launch file:
```
roslaunch lvi_sam run.launch
```

3. Play existing bag files:
```
rosbag play handheld.bag 
```

---

## Paper 

Thank you for citing our [paper](./doc/paper.pdf) if you use any of this code or datasets.

```
@inproceedings{lvisam2021shan,
  title={LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping},
  author={Shan, Tixiao and Englot, Brendan and Ratti, Carlo and Rus Daniela},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={to-be-added},
  year={2021},
  organization={IEEE}
}
```

---

## Acknowledgement

  - The visual-inertial odometry module is adapted from [Vins-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono).
  - The lidar-inertial odometry module is adapted from [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM/tree/a246c960e3fca52b989abf888c8cf1fae25b7c25).