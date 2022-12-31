# Cloud Info
Header header   ## 标头

int32[] startRingIndex  ## 点云的第i条扫描线（ring）上的第一个可计算曲率的点
int32[] endRingIndex    ## 点云的第i条扫描线（ring）上的最后一个可计算曲率的点

int32[]  pointColInd # point column index in range image ## 点云中每个点在投影图片中的列序号
float32[] pointRange # point range ## 点云中每个点与LiDAR的距离，即投影图片的像素值

int64 imuAvailable  ## imu的旋转测量是否对齐到LiDAR，若对齐说明imu旋转测量可用
int64 odomAvailable ## 是否有与当前帧最近的两个相邻帧之间的位姿变换可用

# Attitude for lidar odometry initialization
## 可用的imu旋转测量，作为LIS帧间位姿变换的预测值
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Odometry
## 可用的相邻帧位姿变换，同样作为LIS帧间位姿变换的预测值
float32 odomX
float32 odomY
float32 odomZ
float32 odomRoll
float32 odomPitch
float32 odomYaw

# Odometry reset ID
## 从里程计获得的位姿变换协方差的取整（四舍五入），可以用于计算该位姿变换值的可信度
int64 odomResetId

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed 原始去畸变点云
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature 角点组成的点云
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature 面点组成的点云