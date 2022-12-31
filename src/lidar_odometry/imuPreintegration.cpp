#include "utility.h"

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
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

class IMUPreintegration : public ParamServer
{
public:
    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    // map -> odom
    tf::Transform map_to_odom;
    tf::TransformBroadcaster tfMap2Odom;
    // odom -> base_link
    tf::TransformBroadcaster tfOdom2BaseLink;

    bool systemInitialized = false;

    //// 先验的位姿噪声，速度噪声，零偏噪声，位姿纠正噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::Vector noiseModelBetweenBias; //// 含零偏噪声模型

    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;
    int imuPreintegrationResetId = 0;

    //; 这里补偿imu2Lidar，旋转就是单位阵
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
    

    IMUPreintegration()
    {
        //// 订阅IMU原始测量和mapping后的Odom话题
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdometry = nh.subscribe<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        //// 发布IMU预积分odom话题和IMU位移可视化话题
        pubImuOdometry = nh.advertise<nav_msgs::Odometry>("odometry/imu", 2000);
        pubImuPath = nh.advertise<nav_msgs::Path>(PROJECT_NAME + "/lidar/imu/path", 1);

        //// 地图坐标系与里程计坐标系的坐标变换
        map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));

        //// 设置IMU的参数：提前标定的加速度协方差，陀螺仪协方差，积分协方差，零偏
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);     // gyro white noise in continuous
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);          // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
        ; // assume zero initial bias

        //// 设置包含零偏的噪声模型：位姿噪声，速度噪声，零偏噪声，
        priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e2);                                                               // m/s
        priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);                                                             // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2);                                                            // meter
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

        //// 初始化预积分器
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
    {
        //// 取出当前odom帧时间戳
        double currentCorrectionTime = ROS_TIME(odomMsg); //// ROS_TIME(msg) == msg->header.stamp.toSec()

        // make sure we have imu data to integrate
        //// 保证存在IMU帧信息用来预积分估计位姿
        if (imuQueOpt.empty())
            return;

        //// 取出odom位姿数据
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        //// 取出odom的id，用来验证预积分系统与odom是否对齐
        int currentResetId = round(odomMsg->pose.covariance[0]);
        //// odom数据转换为Pose3形式
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        // correction pose jumped, reset imu pre-integration
        //// id不同步时，重置预积分系统
        if (currentResetId != imuPreintegrationResetId)
        {
            resetParams();
            imuPreintegrationResetId = currentResetId;
            return;
        }

        // 0. initialize system
        //// 初始化系统
        if (systemInitialized == false)
        {
            resetOptimization();

            // pop old IMU message
            //// 删除旧的IMU帧（odom时间戳以前的IMU帧）
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            //// 用odom得到的位姿初始化imu位姿，作为因子图第一个因子
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // initial velocity
            //// 初始化速度因子
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            //// 初始化零偏因子
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values
            //// 导入优化变量初值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            //// 第一次优化
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            //// 预积分器导入零偏初值
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

            key = 1;
            systemInitialized = true;
            return; //// 初始化结束
        }

        // reset graph for speed
        //// 当索引值达到100时，需要控制因子图规模
        if (key == 100)
        {
            // get updated noise before reset
            //// 获取最新的协方差/噪声
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
            // reset graph
            //// 重置新的因子图和图优化器
            resetOptimization();
            // add pose
            //// 把最新优化的结果作为第一个因子加入因子图，并作为优化初值
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            //// 重置因子索引
            key = 1;
        }

        // 1. integrate imu data and optimize
        //// 预积分旧的imu测量（这里的预积分用来优化零偏），并在队列中删除
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                //// 第一帧dt为1/500，其他为正常的时间间隔
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                imuIntegratorOpt_->integrateMeasurement(
                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                    gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);

                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        //// 加入IMU因子
        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                                                            gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        //// 加入位姿因子
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values
        //// 预测状态和零偏，作为优化变量的初值
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        //// 清空因子图
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        //// 提取优化结果
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_ = result.at<gtsam::Pose3>(X(key));
        prevVel_ = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        //// 更新预积分器的零偏
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization 失效检测
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }

        // 2. after optiization, re-propagate imu odometry preintegration
        //// 把优化结果传播给imuOdom模块
        prevStateOdom = prevState_;
        prevBiasOdom = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        //// 更新的bias预积分器采用新的IMU帧预测位姿
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;               //// 增加索引值
        doneFirstOpt = true; //// 完成了初始化以后的第一次优化
    }

    bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        //// 速度的范数超过30则认为该优化结果无效
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        //// 加速度计零偏和陀螺仪零偏有一个范数超过0.1则认为该优化结果无效
        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 0.1 || bg.norm() > 0.1)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr &imuMsg)
    {
        //// imu测量坐标变换到lidar（旋转）
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        // publish static tf
        //; 这里map_to_odom是I_4，也就是两个坐标系是对齐的
        tfMap2Odom.sendTransform(tf::StampedTransform(
            map_to_odom, thisImu.header.stamp, "map", "odom"));

        //// imu测量存放到队列
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        //// 只有第一次优化过程完成后，才继续后面的过程
        if (doneFirstOpt == false)
            return;

        //// 计算时间间隔，第一帧按照0.002处理
        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        //// 预积分当前imu帧
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt);

        // predict odometry
        //// 预测odom
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        //// 发布odom
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = "odom";
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        //// 把odom坐标变换到lidar坐标系下（平移）
        //; 把使用IMU预测的位姿，弥补LiDAR和IMU之间的平移外参，变成LiDAR预测的位姿
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        //// 发布odom和path和可视化位姿
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        
        
        // information for VINS initialization
        odometry.pose.covariance[0] = double(imuPreintegrationResetId); //// 复位id
        odometry.pose.covariance[1] = prevBiasOdom.accelerometer().x(); //// 加速度计bias
        odometry.pose.covariance[2] = prevBiasOdom.accelerometer().y();
        odometry.pose.covariance[3] = prevBiasOdom.accelerometer().z();
        odometry.pose.covariance[4] = prevBiasOdom.gyroscope().x(); //// 陀螺仪bias
        odometry.pose.covariance[5] = prevBiasOdom.gyroscope().y();
        odometry.pose.covariance[6] = prevBiasOdom.gyroscope().z();
        odometry.pose.covariance[7] = imuGravity; //// 重力，这个就是配置参数中读进来的重力模长
        pubImuOdometry.publish(odometry);

        // publish imu path
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        //; 用10hz的频率发布IMU预测的轨迹
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = thisImu.header.stamp;
            pose_stamped.header.frame_id = "odom";
            pose_stamped.pose = odometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            //// 删去时间间隔太久的旧IMU帧
            while (!imuPath.poses.empty() && abs(imuPath.poses.front().header.stamp.toSec() - imuPath.poses.back().header.stamp.toSec()) > 3.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = thisImu.header.stamp;
                imuPath.header.frame_id = "odom";
                pubImuPath.publish(imuPath);
            }
        }

        // publish transformation
        //; 这里可以发现，base_link其实就是lidar坐标系，odom就是全局不变的世界坐标系
        tf::Transform tCur;
        tf::poseMsgToTF(odometry.pose.pose, tCur);
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(
            tCur, thisImu.header.stamp, "odom", "base_link");
        tfOdom2BaseLink.sendTransform(odom_2_baselink);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar");

    //// 利用IMUPreintegration类构造函数完成订阅、处理、发布
    IMUPreintegration ImuP;

    ROS_INFO("\033[1;32m----> Lidar IMU Preintegration Started.\033[0m");

    ros::spin();

    return 0;
}