#include "utility.h"
#include "lvi_sam/cloud_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
 */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D

    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll,
                                                                                                       roll)(float,
                                                                                                             pitch,
                                                                                                             pitch)(
                                          float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

class mapOptimization : public ParamServer
{
public:
    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2* isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubOdomAftMappedROS;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subGPS;
    ros::Subscriber subLoopInfo;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lvi_sam::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;    // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;      // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;  // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;    // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec;  // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec;  // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;  // for surrounding key poses of scan-to-map optimization

    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    int imuPreintegrationResetId = 0;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;

    map<int, int> loopIndexContainer;  // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

    mapOptimization()
    {
        //// 设置求解器参数
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;    //// 差值大于0.1，则重新线性化
        parameters.relinearizeSkip = 1;           //// 有一个值需要重新线性化，就更新贝叶斯树
        isam = new ISAM2(parameters);

        //// 发布关键帧位姿
        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/trajectory", 1);
        //// 发布所有关键帧点云和其优化后位姿组成的地图
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_global", 1);
        //// 发布LIS系统得到的位姿（ros格式）
        pubOdomAftMappedROS = nh.advertise<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 1);
        //// 发布LIS系统得到的位移（ros格式）
        pubPath = nh.advertise<nav_msgs::Path>(PROJECT_NAME + "/lidar/mapping/path", 1);

        //// 订阅含有特征信息的cloud_info
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(
            PROJECT_NAME + "/lidar/feature/cloud_info", 5,
            &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        
        //// 订阅GPS
        subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 50, &mapOptimization::gpsHandler, this,
                                                  ros::TransportHints().tcpNoDelay());
        //// 订阅VIS子系统检测到的回环帧
        subLoopInfo = nh.subscribe<std_msgs::Float64MultiArray>(
            PROJECT_NAME + "/vins/loop/match_frame", 5,
            &mapOptimization::loopHandler, this, ros::TransportHints().tcpNoDelay());

        //// 发布回环帧过去帧的局部地图
        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(
                PROJECT_NAME + "/lidar/mapping/loop_closure_history_cloud", 1);
        //// 发布回环帧当前帧的局部地图
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(
                PROJECT_NAME + "/lidar/mapping/loop_closure_corrected_cloud", 1);
        //// 发布rviz可用的回环约束
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(
                PROJECT_NAME + "/lidar/mapping/loop_closure_constraints", 1);

        //// 发布关键帧点云
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_local", 1);
        //// 发布新的优化后的关键帧特征点云
        pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered", 1);
        //// 发布新的优化后的关键帧整体点云
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>(
                PROJECT_NAME + "/lidar/mapping/cloud_registered_raw", 1);

        //// 设置几个降采样滤波器：角点点云组成的地图 面点点云组成的地图
        ////                    回环帧当前帧局部地图 50m内的附近帧位姿数组
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity,
                                                      surroundingKeyframeDensity);  // for surrounding key poses of scan-to-map optimization
        //// 重置内存
        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());    // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());      // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(
                new pcl::PointCloud<PointType>());  // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(
                new pcl::PointCloud<PointType>());    // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());

        for (int i = 0; i < 6; ++i)
        { transformTobeMapped[i] = 0; }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info ana feature cloud
        //// 提取cloudInfo中的当前帧特征点云
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);  //// 作用域锁

        static double timeLastProcessing = -1;
        //// 当前帧点云的时间戳与上一次优化的点云时间戳相差足够时间时，才进行优化处理。控制优化频率
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            //// 获取优化初值（从VIS或IMU中）
            updateInitialGuess();

            //// 提取当前帧50m内的关键帧组成局部地图
            extractSurroundingKeyFrames();

            //// 当前帧特征点云降采样
            downsampleCurrentScan();

            //// 当前帧与局部地图scan2map点云配准
            scan2MapOptimization();

            //// 判定是否为关键帧，如果是，加入所有新的约束节点到因子图中，并更新因子图，添加关键帧点云
            saveKeyFramesAndFactor();

            //// 有回环时，利用回环优化的位姿纠正位姿数组
            correctPoses();

            //// 发布ROS的odom话题和tf话题
            publishOdometry();

            //// 发布关键帧位姿，关键帧位移，关键帧特征点云，关键帧整体点云，关键帧附近局部地图
            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        //// 放入GPS队列
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const* const pi, PointType* const po)
    {
        po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y +
                transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
        po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y +
                transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
        po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y +
                transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr
    transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType* pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z,
                                                          transformIn->roll, transformIn->pitch, transformIn->yaw);

        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x =
                    transCur(0, 0) * pointFrom->x + transCur(0, 1) * pointFrom->y + transCur(0, 2) * pointFrom->z +
                    transCur(0, 3);
            cloudOut->points[i].y =
                    transCur(1, 0) * pointFrom->x + transCur(1, 1) * pointFrom->y + transCur(1, 2) * pointFrom->z +
                    transCur(1, 3);
            cloudOut->points[i].z =
                    transCur(2, 0) * pointFrom->x + transCur(2, 1) * pointFrom->y + transCur(2, 2) * pointFrom->z +
                    transCur(2, 3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 affine3fTogtsamPose3(const Eigen::Affine3f& thisPose)
    {
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(thisPose, x, y, z, roll, pitch, yaw);
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(roll), double(pitch), double(yaw)),
                            gtsam::Point3(double(x), double(y), double(z)));
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch,
                                      thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1],
                                      transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw = transformIn[2];
        return thisPose6D;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok())
        {
            rate.sleep();
            //// 发布1000m以内所有特征点云组成的地图
            publishGlobalMap();
        }

        //// 如果设置savePCD为真，则用pcl格式保存平移轨迹，位姿变换轨迹，所有角点点云、面点点云组成的地图，以及二者共同组成的地图
        if (savePCD == false)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // create directory and remove old files;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
        ++unused;
        // save key frame transformations
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        // extract global point cloud map
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int) cloudKeyPoses3D->size(); i++)
        {
            // clip cloud
            // pcl::PointCloud<PointType>::Ptr cornerTemp(new pcl::PointCloud<PointType>());
            // pcl::PointCloud<PointType>::Ptr cornerTemp2(new pcl::PointCloud<PointType>());
            // *cornerTemp = *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            // for (int j = 0; j < (int)cornerTemp->size(); ++j)
            // {
            //     if (cornerTemp->points[j].z > cloudKeyPoses6D->points[i].z && cornerTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
            //         cornerTemp2->push_back(cornerTemp->points[j]);
            // }
            // pcl::PointCloud<PointType>::Ptr surfTemp(new pcl::PointCloud<PointType>());
            // pcl::PointCloud<PointType>::Ptr surfTemp2(new pcl::PointCloud<PointType>());
            // *surfTemp = *transformPointCloud(surfCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            // for (int j = 0; j < (int)surfTemp->size(); ++j)
            // {
            //     if (surfTemp->points[j].z > cloudKeyPoses6D->points[i].z && surfTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
            //         surfTemp2->push_back(surfTemp->points[j]);
            // }
            // *globalCornerCloud += *cornerTemp2;
            // *globalSurfCloud   += *surfTemp2;

            // origin cloud
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size()
                 << " ...";
        }
        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloud);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloud);
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        downSizeFilterSurf.setInputCloud(globalMapCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius,
                                      pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int) pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                             // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity,
                                                    globalMapVisualizationPoseDensity,
                                                    globalMapVisualizationPoseDensity);  // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int) globalMapKeyPosesDS->size(); ++i)
        {
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) >
                globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int) globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                                                        &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],
                                                        &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                    // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize,
                                                     globalMapVisualizationLeafSize);  // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, "odom");
    }

    //// 通过vis进行闭环检测
    void loopHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        // control loop closure frequency
        static double last_loop_closure_time = -1;
        {
            // std::lock_guard<std::mutex> lock(mtx);
            //// 时间上的简单验证
            if (timeLaserInfoCur - last_loop_closure_time < 5.0)
                return;
            else
                last_loop_closure_time = timeLaserInfoCur;
        }
        //// 进行空间上和相似度上的回环验证
        performLoopClosure(*loopMsg);
    }

    //// 根据vins视觉判断闭环 or 根据距离位置判断闭环
    //// 闭环检测 (通过 距离内搜索 或者 vins 得到的闭环候选帧), loopMsg保存的是时间戳(当前帧, 闭环帧);
    //// 在空间和相似度上验证回环，如果正确则添加闭环约束到容器，发布可视化闭环约束
    void performLoopClosure(const std_msgs::Float64MultiArray& loopMsg)
    {
        //// 获取所有关键帧的位姿
        pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());
        {
            std::lock_guard<std::mutex> lock(mtx);
            *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        }

        //// 通过loopMsg的时间戳来寻找 闭环候选帧对应的关键帧点云
        // get lidar keyframe id
        int key_cur = -1;  // latest lidar keyframe id
        int key_pre = -1;  // previous lidar keyframe id
        {
            loopFindKey(loopMsg, copy_cloudKeyPoses6D, key_cur, key_pre);
            if (key_cur == -1 || key_pre == -1 || key_cur == key_pre)  // || abs(key_cur - key_pre) < 25)
                return;
        }

        // check if loop added before
        //// 检查是否已经添加过这个回环
        {
            // if image loop closure comes at high frequency, many image loop may point to the same key_cur
            auto it = loopIndexContainer.find(key_cur);
            if (it != loopIndexContainer.end())
                return;
        }

        //// 分别为当前帧和闭环帧构造局部地图, 进行map to map的闭环匹配，用来验证闭环
        // get lidar keyframe cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            //// 这里参数为0，所以只保留当前帧，也可以设置其他值，取当前帧附近帧组成局部地图
            loopFindNearKeyframes(copy_cloudKeyPoses6D, cureKeyframeCloud, key_cur, 0);
            //// historyKeyframeSearchNum为25，用闭环帧附近的前后25个帧组成局部地图
            loopFindNearKeyframes(copy_cloudKeyPoses6D, prevKeyframeCloud, key_pre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            //// 发布 闭环帧的localmap点云
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, "odom");
        }

        // get keyframe pose
        Eigen::Affine3f pose_cur;     // 当前帧pose
        Eigen::Affine3f pose_pre;     // 闭环帧pose
        Eigen::Affine3f pose_diff_t;  // serves as initial guess 将两者的相对位姿作为初始位姿
        {
            pose_cur = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_cur]);
            pose_pre = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_pre]);

            Eigen::Vector3f t_diff;
            t_diff.x() = -(pose_cur.translation().x() - pose_pre.translation().x());
            t_diff.y() = -(pose_cur.translation().y() - pose_pre.translation().y());
            t_diff.z() = -(pose_cur.translation().z() - pose_pre.translation().z());
            //// 如果它们之间的距离相差很远，说明漂移很大，不适合用它作为初值
            if (t_diff.norm() < historyKeyframeSearchRadius)
                t_diff.setZero();
            pose_diff_t = pcl::getTransformation(t_diff.x(), t_diff.y(), t_diff.z(), 0, 0, 0);
        }

        //// 使用icp进行闭环匹配(map to map)

        // transform and rotate cloud for matching
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        // pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(100);
        icp.setRANSACIterations(0);
        icp.setTransformationEpsilon(1e-3);
        icp.setEuclideanFitnessEpsilon(1e-3);

        // initial guess cloud
        //// 根据初始相对位姿, 对当前帧点云进行坐标变换
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud_new(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud_new, pose_diff_t);

        // match using icp
        icp.setInputSource(cureKeyframeCloud_new);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud_new, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, "odom");
        }

        // add graph factor
        // 将闭环保存至loopIndexQueue loopPoseQueue loopNoiseQueue中供addLoopFactor()使用
        //// 距离小并且收敛
        if (icp.getFitnessScore() < historyKeyframeFitnessScore && icp.hasConverged() == true)
        {
            // get gtsam pose
            gtsam::Pose3 poseFrom = affine3fTogtsamPose3(
                    Eigen::Affine3f(icp.getFinalTransformation()) * pose_diff_t * pose_cur);
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[key_pre]);
            // get noise
            //// 把相似性分数作为方差噪声
            float noise = icp.getFitnessScore();
            gtsam::Vector Vector6(6);
            Vector6 << noise, noise, noise, noise, noise, noise;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
            // save pose constraint
            mtx.lock();
            //// 添加闭环约束
            loopIndexQueue.push_back(make_pair(key_cur, key_pre));
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();
            // add loop pair to container
            loopIndexContainer[key_cur] = key_pre;
        }

        // visualize loop constraints 发布 所有闭环约束
        if (!loopIndexContainer.empty())
        {
            visualization_msgs::MarkerArray markerArray;
            // loop nodes
            visualization_msgs::Marker markerNode;
            markerNode.header.frame_id = "odom";
            markerNode.header.stamp = timeLaserInfoStamp;
            markerNode.action = visualization_msgs::Marker::ADD;
            markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
            markerNode.ns = "loop_nodes";
            markerNode.id = 0;
            markerNode.pose.orientation.w = 1;
            markerNode.scale.x = 0.3;
            markerNode.scale.y = 0.3;
            markerNode.scale.z = 0.3;
            markerNode.color.r = 0;
            markerNode.color.g = 0.8;
            markerNode.color.b = 1;
            markerNode.color.a = 1;
            // loop edges
            visualization_msgs::Marker markerEdge;
            markerEdge.header.frame_id = "odom";
            markerEdge.header.stamp = timeLaserInfoStamp;
            markerEdge.action = visualization_msgs::Marker::ADD;
            markerEdge.type = visualization_msgs::Marker::LINE_LIST;
            markerEdge.ns = "loop_edges";
            markerEdge.id = 1;
            markerEdge.pose.orientation.w = 1;
            markerEdge.scale.x = 0.1;
            markerEdge.color.r = 0.9;
            markerEdge.color.g = 0.9;
            markerEdge.color.b = 0;
            markerEdge.color.a = 1;

            for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
            {
                int key_cur = it->first;
                int key_pre = it->second;
                geometry_msgs::Point p;
                p.x = copy_cloudKeyPoses6D->points[key_cur].x;
                p.y = copy_cloudKeyPoses6D->points[key_cur].y;
                p.z = copy_cloudKeyPoses6D->points[key_cur].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
                p.x = copy_cloudKeyPoses6D->points[key_pre].x;
                p.y = copy_cloudKeyPoses6D->points[key_pre].y;
                p.z = copy_cloudKeyPoses6D->points[key_pre].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
            }

            markerArray.markers.push_back(markerNode);
            markerArray.markers.push_back(markerEdge);
            pubLoopConstraintEdge.publish(markerArray);
        }
    }

    // 为关键帧key构造localmap点云(nearKeyframes)
    void loopFindNearKeyframes(const pcl::PointCloud<PointTypePose>::Ptr& copy_cloudKeyPoses6D,
                               pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int key_near = key + i;
            if (key_near < 0 || key_near >= cloudSize)
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[key_near],
                                                   &copy_cloudKeyPoses6D->points[key_near]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[key_near],
                                                   &copy_cloudKeyPoses6D->points[key_near]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    // 通过loopMsg的时间戳来寻找 闭环候选帧
    void loopFindKey(const std_msgs::Float64MultiArray& loopMsg,
                     const pcl::PointCloud<PointTypePose>::Ptr& copy_cloudKeyPoses6D, int& key_cur, int& key_pre)
    {
        if (loopMsg.data.size() != 2)
            return;

        double loop_time_cur = loopMsg.data[0];
        double loop_time_pre = loopMsg.data[1];

        // 时间戳在25s之内, 不是闭环
        if (abs(loop_time_cur - loop_time_pre) < historyKeyframeSearchTimeDiff)
            return;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return;

        // latest key
        key_cur = cloudSize - 1;  // 当前帧
        //// 在当前帧以后，且最近的帧
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time > loop_time_cur)
                key_cur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        key_pre = 0;  // 闭环帧
        //// 在闭环帧以前，且最近的帧
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time < loop_time_pre)
                key_pre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }
    }

    // 线程: 通过距离进行闭环检测
    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(0.5);  // 每2s进行一次回环检测
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosureDetection();
        }
    }

    // 通过距离进行闭环检测
    void performLoopClosureDetection()
    {
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;

        // 通过距离找到的闭环候选帧
        int key_cur = -1;
        int key_pre = -1;

        double loop_time_cur = -1;
        double loop_time_pre = -1;

        // find latest key and time
        //// 1.使用kdtree寻找最近的keyframes, 作为闭环检测的候选关键帧 (半径20m以内)
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (cloudKeyPoses3D->empty())
                return;

            //// 当前的关键帧位移数组，构建kdtree
            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            //// 按照半径20m搜索
            kdtreeHistoryKeyPoses->radiusSearch(cloudKeyPoses3D->back(), historyKeyframeSearchRadius,
                                                pointSearchIndLoop, pointSearchSqDisLoop, 0);

            key_cur = cloudKeyPoses3D->size() - 1;
            loop_time_cur = cloudKeyPoses6D->points[key_cur].time;
        }

        // find previous key and time
        //// 2.在候选关键帧集合中，找到与当前帧时间相隔较远的最近帧，设为候选匹配帧 (30s之前)
        {
            for (int i = 0; i < (int) pointSearchIndLoop.size(); ++i)
            {
                int id = pointSearchIndLoop[i];
                if (abs(cloudKeyPoses6D->points[id].time - loop_time_cur) > historyKeyframeSearchTimeDiff)
                {
                    key_pre = id;
                    loop_time_pre = cloudKeyPoses6D->points[key_pre].time;
                    break;
                }
            }
        }

        // 未检测到闭环
        if (key_cur == -1 || key_pre == -1 || key_pre == key_cur || loop_time_cur < 0 || loop_time_pre < 0)
            return;

        //// 将潜在闭环给到回环验证函数
        std_msgs::Float64MultiArray match_msg;
        match_msg.data.push_back(loop_time_cur);  // 当前帧时间戳
        match_msg.data.push_back(loop_time_pre);  // 闭环帧时间戳
        performLoopClosure(match_msg);
    }

    // 计算点云的先验位姿 (通过imu或者vins odom)
    void updateInitialGuess()
    {
        static Eigen::Affine3f lastImuTransformation;

        //// 第一帧点云, 直接使用imu初始化
        // system initialization
        if (cloudKeyPoses3D->points.empty())
        {
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            // 保存下来, 给下一帧使用
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit,
                                                           cloudInfo.imuYawInit);  // save imu before return;
            return;
        }

        // use VINS odometry estimation for pose guess
        //// 非第一帧，优先采用VIS的预测
        static int odomResetId = 0;     //// 一个与cloud_info中odomResetID同名的静态变量，用来检测VIS是否被启动/重启过
        static bool lastVinsTransAvailable = false;
        static Eigen::Affine3f lastVinsTransformation;  //// 上次预测时的vis位姿
        if (cloudInfo.odomAvailable == true && cloudInfo.odomResetId == odomResetId)
        {
            //// VIS预测可用，并且VIS已经被启动/重启过
            // ROS_INFO("Using VINS initial guess");
            if (lastVinsTransAvailable == false)
            {
                //// 上次预测时VIS重新启动了，保存本次的预测值，尝试采用IMU的预测
                // ROS_INFO("Initializing VINS initial guess");
                lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX, cloudInfo.odomY, cloudInfo.odomZ,
                                                                cloudInfo.odomRoll, cloudInfo.odomPitch,
                                                                cloudInfo.odomYaw);
                lastVinsTransAvailable = true;
            } 
            else
            {
                //// 上次预测成功使用了VIS的预测
                //// 用上次预测的位姿与本次预测的位姿之间的位姿变换作为点云配准初值
                // ROS_INFO("Obtaining VINS incremental guess");
                Eigen::Affine3f transBack = pcl::getTransformation(
                    cloudInfo.odomX, cloudInfo.odomY, cloudInfo.odomZ,
                    cloudInfo.odomRoll, cloudInfo.odomPitch,
                    cloudInfo.odomYaw);
                Eigen::Affine3f transIncre = lastVinsTransformation.inverse() * transBack;

                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4],
                                                  transformTobeMapped[5], transformTobeMapped[0],
                                                  transformTobeMapped[1], transformTobeMapped[2]);

                //// 保存本次预测的位姿（VIS和IMU）
                lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX, cloudInfo.odomY, cloudInfo.odomZ,
                                                                cloudInfo.odomRoll, cloudInfo.odomPitch,
                                                                cloudInfo.odomYaw);

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit,
                                                               cloudInfo.imuYawInit);  // save imu before return;
                return;
            }
        } 
        else
        {
            // vins跟丢了, 准备重启
            // ROS_WARN("VINS failure detected.");
            lastVinsTransAvailable = false;
            odomResetId = cloudInfo.odomResetId;
        }

        // use imu incremental estimation for pose guess (only rotation)
        //// VIS预测无法使用或者上一次预测时VIS刚刚启动
        if (cloudInfo.imuAvailable == true)
        {
            //// imu预测可用，则利用imu预测的旋转矩阵与上次预测时的旋转矩阵之间的位姿变换，作为点云配准的初值
            // ROS_INFO("Using IMU initial guess");
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit,
                                                               cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4],
                                              transformTobeMapped[5], transformTobeMapped[0], transformTobeMapped[1],
                                              transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit,
                                                           cloudInfo.imuYawInit);  // save imu before return;
            return;
        }
    }

    // 聚合构建局部地图
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        //// 搜索最后一个关键帧附近的50m以内的关键帧
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);  // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double) surroundingKeyframeSearchRadius,
                                                pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int) pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        //// 把这些关键帧位置进行降采样，构建局部地图时效率更高
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // also extract some latest key frames in case the robot rotates in one position
        //// 同时也将时间较近的关键帧位置加入进去，防止传感器仅仅在一个位置做纯旋转
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses - 1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    // 通过提取到的keyframes, 来提取点云, 从而构造localmap
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

        laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

// extract surrounding map
//// 利用处理器并行计算，将所有关键帧特征点云组合起来
#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < (int) cloudToExtract->size(); ++i)
        {
            int thisKeyInd = (int) cloudToExtract->points[i].intensity;
            //// 剔除距离过远的关键帧
            if (pointDistance(cloudKeyPoses3D->points[thisKeyInd], cloudKeyPoses3D->back()) >
                surroundingKeyframeSearchRadius)
                continue;
            //// 全都变换到map坐标系下
            laserCloudCornerSurroundingVec[i] = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
                                                                     &cloudKeyPoses6D->points[thisKeyInd]);
            laserCloudSurfSurroundingVec[i] = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],
                                                                   &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // fuse the map
        //// 将上面的点云全都保存到一个点云中，也就是局部地图中
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        for (int i = 0; i < (int) cloudToExtract->size(); ++i)
        {
            *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            *laserCloudSurfFromMap += laserCloudSurfSurroundingVec[i];
        }

        //// 降采样
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    }

    // 提取附近的keyframes及其点云, 来构造localmap
    void extractSurroundingKeyFrames()
    {
        //// 检查是否是第一帧
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        //// 降采样
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    // 更新当前帧的位姿 (lidar到map的坐标变换transform)
    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);  //
    }

    // 构造 点到直线 的残差约束（并行计算）
    void cornerOptimization()
    {
        //// 更新点云点投影到地图坐标系的变换矩阵，实际上也就是当前帧位姿
        updatePointAssociateToMap();

        //// 并行计算每个角点的残差
#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            //// 当前角点变换到地图坐标系下
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            //// 如果前五个点与当前点距离都小于1m
            if (pointSearchSqDis[4] < 1.0)
            {
                //// 求五个点的中心点
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++)
                {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5;
                cy /= 5;
                cz /= 5;
                //// 求五个点的协方差矩阵
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++)
                {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

                // 协方差矩阵与点云中角点面点之间的关系:
                // 1.假设点云序列为S，计算 S 的协方差矩阵，记为 cov_mat ，cov_mat 的特征值记为 V ，特征向量记为 E 。
                // 2.如果 S 分布在一条线段上，那么 V 中一个特征值就会明显比其他两个大，E 中与较大特征值相对应的特征向量代表边缘线的方向。(一大两小，大的代表直线方向)
                // 3.如果 S 分布在一块平面上，那么 V 中一个特征值就会明显比其他两个小，E 中与较小特征值相对应的特征向量代表平面片的方向。(一小两大，小方向)边缘线或平面块的位置通过穿过 S 的几何中心来确定。

                // 计算协方差矩阵的特征值和特征向量，用于判断这5个点是不是呈线状分布，此为PCA的原理；
                cv::eigen(matA1, matD1, matV1);

                // 如果5个点呈线状分布，最大的特征值对应的特征向量就是该线的方向向量；
                //// 如果五个点的最大特征值 远大于 第二大的特征值
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
                {
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 从中心点沿着方向向量向两端移动0.1m，构造线上的两个点；
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 向量OA = (x0 - x1, y0 - y1, z0 - z1), 向量OB = (x0 - x2, y0 - y2, z0 - z2)，向量AB = （x1 - x2, y1 - y2, z1 - z2）;
                    // 点到线的距离，d = |向量OA 叉乘 向量OB|/|AB|;
                    // 向量OA 叉乘 向量OB 得到的向量模长 ： 是垂直a、b所在平面，且以|b|·sinθ为高、|a|为底的平行四边形的面积，
                    // 因此|向量OA 叉乘 向量OB|再除以|AB|的模长，则得到高度，即点到线的距离；

                    //// OA×OB叉乘向量的模长，也就是平行四边形面积
                    float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                                      ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                                      ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                                      ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                                      ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                                      ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));
                    //// AB的模长，也就是对角线长度
                    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
                    //// 对当前点x0，y0，z0的偏导
                    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                                (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                                 (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                                 (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;
                    //// 点线之间的距离
                    float ld2 = a012 / l12;
                    //// 点线距离权重，距离越近，权重越大
                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    //// 如果点线距离小于1m
                    if (s > 0.1)
                    {
                        //// 保存优化点，残差（带权重），标记这个被优化点索引
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    // 构建 点到平面 的残差约束（并行计算）
    void surfOptimization()
    {
        //// 大体思路与构建角点的残差差不多，只不过这里构建的是点到平面的距离
        updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0)
            {
                // 求面的法向量不是用的PCA，使用的是最小二乘拟合；
                // 假设平面不通过原点，则平面的一般方程为Ax + By + Cz + 1 = 0，用这个假设可以少算一个参数；
                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // 构建超定方程组： matA0 * norm（A, B, C） = matB0；

                // 求解这个最小二乘问题，可得平面的法向量norm（A, B, C）；
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // Ax + By + Cz + 1 = 0，全部除以法向量的模长，方程依旧成立，而且使得法向量归一化了；
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                // 点(x0, y0, z0)到平面Ax + By + Cz + D = 0 的距离 = fabs(A*x0 + B*y0 + C*z0 + D) / sqrt(A^2 + B^2 + C^2)；
                // 因为法向量（A, B, C）已经归一化了，所以距离公式可以简写为：距离 = fabs(A*x0 + B*y0 + C*z0 + D) ；

                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    // 如果拟合的５个面点中，任何一个点到平面的距离大于阈值，则认为平面拟合不好；
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid)
                {
                    //// 点到平面的距离，带代入平面方程即可
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                    //// 权重
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(
                            pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1)
                    {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    // 联合两类残差 (点到直线, 点到平面)
    void combineOptimizationCoeffs()
    {
        //// 把所有将要被优化的角点及其残差放入容器
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
        {
            if (laserCloudOriCornerFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // 把所有将要被优化的面点及其残差放入容器
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
        {
            if (laserCloudOriSurfFlag[i] == true)
            {
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        //// 重置标记数组，保证下一轮优化
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    //// 手写LM优化方法，返回值为是否收敛
    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        //// 转换到相机坐标系下，与论文公式对应，在建立雅克比矩阵时转换回来
        //// 用于表示雅克比矩阵
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50)
        {
            return false;
        }

        //// matA雅克比矩阵 matB代价函数矩阵
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++)
        {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            //// 旋转矩阵对x方向旋转的偏导
            float arx =
                    (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x +
                    (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y +
                    (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;
            //// 旋转矩阵对y方向旋转的偏导
            float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y +
                         crx * cry * pointOri.z) * coeff.x +
                        ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y -
                         crx * sry * pointOri.z) * coeff.z;
            //// 旋转矩阵对z方向旋转的偏导
            float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) *
                        coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                        ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) *
                        coeff.z;
            // lidar -> camera
            //// 从相机坐标系转换回LiDAR坐标系，z，x，y对应x，y，z
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0)
        {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            //// 为了防止场景几何特征退化，用如下方式进行验证（视觉特征退化只需要统计特征点数量和特征点被跟踪次数）
            // 对AtA进行特征分解
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            //// 理想情况是6个自由度的特征值都很大（都大于某个常数）
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--)
            {
                if (matE.at<float>(0, i) < eignThre[i])
                {
                    for (int j = 0; j < 6; j++)
                    {
                        matV2.at<float>(i, j) = 0;  //// 对特征值小的特征向量置为0，从而使得该自由度上的匹配结果保持为原来的值
                    }
                    // 点云退化了
                    isDegenerate = true;
                } else
                {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        // 点云退化了
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;      //// 更新matX，让退化部分增量为0，从而让退化自由度保持原来的值
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        //// 统计增量的值，保证迭代是收敛的
        float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) + pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05)
        {
            return true;  // converged
        }
        return false;  // keep optimizing
    }

    void scan2MapOptimization()
    {
        //// 验证是否是第一帧
        if (cloudKeyPoses3D->points.empty())
            return;
        //// 特征点数量足够
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            //// 设置局部地图的kdtree，方便构建残差
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();
                //// 构建角点的点线残差，面点的点面残差
                cornerOptimization();
                surfOptimization();
                //// 联合两种残差
                combineOptimizationCoeffs();
                //// LM方法进行非线性优化，如果不收敛，则退出
                if (LMOptimization(iterCount) == true)
                    break;
            }
            //// 更新优化后的位姿
            transformUpdate();
        } else
        {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum,
                     laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = 0.01;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    // 是否将当前帧设为关键帧
    bool saveFrame()
    {
        //// 第一帧不必判定，直接认定为关键帧
        if (cloudKeyPoses3D->points.empty())
            return true;
        //// 计算上一个关键帧与当前帧之间的变换矩阵
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4],
                                                            transformTobeMapped[5], transformTobeMapped[0],
                                                            transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        //// 如果旋转角和位移有全都小于阈值，则判定非关键帧，否则判定为关键帧
        if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    //// 加入里程计因子
    void addOdomFactor()
    {
        //// 第一帧时的特殊对待
        if (cloudKeyPoses3D->points.empty())
        {
            //// 从头捋一遍发现第一帧采用的就是IMU的旋转测量，
            //// 重力在yaw角平面的投影为0，因此yaw角是不可观的，误差较大，因此方差设置较大
            //// 初始启动时运动速度较慢，因此初始位移设为0，0，0，并且方差设置较小
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());  // rad*rad, meter*meter
            //// PriorFactor第一个因子
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        } else
        {
            //// 普通关键帧就按照一般情况设置方差即可
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances(
                    (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
            //// BetweenFactor中间因子，指定其前面因子和自己的序号
            gtSAMgraph.add(
                    BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo),
                                         odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            // if (isDegenerate)
            // {
            // adding VINS constraints is deleted as benefits are not obvious, disable for now
            // gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), vinsPoseFrom.between(vinsPoseTo), odometryNoise));
            // }
        }
    }

    //// 加入GPS因子，GPS的绝对位置测量的误差大约在3~5m
    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        //// 等待系统运行一段距离，短距离内漂移不会太大
        if (cloudKeyPoses3D->points.empty())
            return;
        else if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
            return;

        // pose covariance small, no need to correct
        //// 查看最新关键帧的协方差，如果协方差小于阈值，则不需要GPS因子进行修正
        if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            //// GPS时间戳与点云时间戳对齐
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            } else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            } else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                //// GPS噪声太大时放弃这一次测量
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                //// 可以设置不使用GPS的z坐标，因为通常路面起伏不会太大
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                //// GPS测量接近于0时，说明没有正确初始化，放弃添加
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                //// 本身GPS误差就在3~5m，因此当两次测量距离在5m以内时，放弃本次添加
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                //// 设置最低噪声为1.0
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;

                break;
            }
        }
    }

    //// 加入回环因子
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (size_t i = 0; i < loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            //// 把闭环约束添加到 图 中
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        //// 设置回环标志位，允许根据回环纠正位姿
        aLoopIsClosed = true;
    }

    //// 保存关键帧、因子图优化
    void saveKeyFramesAndFactor()
    {
        //// 判定当前帧是否有资格作为关键帧
        if (saveFrame() == false)
            return;

        //// 加入里程计约束
        addOdomFactor();

        //// 加入GPS约束
        addGPSFactor();

        //// 加入回环约束
        addLoopFactor();    //// 当然，只能优化环内位姿

        //// 保存因子图到isam中，进行因子图优化
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        //// 这一段因子图已经被isam保存了。
        //// 所以重置gtSAMgraph因子图和initialEstimate初始估计，方便建立下一段因子图
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //// 保存优化结果
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size();  //// 借用PCL点云点反射强度的数据结构保存关键帧索引
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
        thisPose6D.roll = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        //// 取出最新关键帧的位姿协方差
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        //// 保存关键帧特征点云数组
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        //// 保存位姿到可视化的topic
        updatePath(thisPose6D);
    }

    //// 根据回环纠正位姿
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        //// 如果存在回环，则将关键帧位姿数组更新为因子图优化后的结果
        if (aLoopIsClosed == true)
        {
            // clear path
            //// 重置ros位移msg，利用新的位姿重写
            globalPath.poses.clear();

            // update key poses
            //// 重写关键帧位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            //// 回环纠正位姿结束，关闭回环纠正的标志位
            aLoopIsClosed = false;
            // ID for reseting IMU pre-integration 增加预积分的索引序号
            ++imuPreintegrationResetId;
        }
    }

    void publishOdometry()
    {
        // Publish odometry for ROS
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = "odom";
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0],
                                                                                         transformTobeMapped[1],
                                                                                         transformTobeMapped[2]);
        laserOdometryROS.pose.covariance[0] = double(imuPreintegrationResetId);
        pubOdomAftMappedROS.publish(laserOdometryROS);

        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(
                tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(
            t_odom_to_lidar, timeLaserInfoStamp, "odom", "lidar_link");
        br.sendTransform(trans_odom_to_lidar);
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = "odom";
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    //// 发布帧位姿，地图，点云，位移
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses 发布关键帧位姿
        publishCloud(&pubKeyPoses, cloudKeyPoses6D, timeLaserInfoStamp, "odom");
        // Publish surrounding key frames  发布当前帧的附近帧聚合的局部地图
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, "odom");
        // publish registered key frame
        //// 如果有订阅者，发布当前帧特征点云
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish registered high-res raw cloud
        //// 如果有订阅者，发布当前帧原始点云
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish path 发布位移
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = "odom";
            pubPath.publish(globalPath);
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    //// 主线程通过mapOptimization的构造函数完成地图优化
    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Lidar Map Optimization Started.\033[0m");

    //// 回环检测线程
    std::thread loopDetectionthread(&mapOptimization::loopClosureThread, &MO);
    //// 点云和地图保存线程
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    //// 主线程等待回环检测线程、地图保存线程结束
    loopDetectionthread.join();
    visualizeMapThread.join();

    return 0;
}