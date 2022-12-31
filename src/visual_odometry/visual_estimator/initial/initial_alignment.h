#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>>& _points, 
                   const vector<float> &_lidar_initialization_info,
                   double _t):
        t{_t}, is_key_frame{false}, reset_id{-1}, gravity{9.805}
        {
            points = _points;
            
            // reset id in case lidar odometry relocate
            //; 因为reset_id是int类型，但是为了方便存储后面的数据都统一使用了float的vector来存储，
            //; 所以这里要转成int
            reset_id = (int)round(_lidar_initialization_info[0]);  
            // Pose
            T.x() = _lidar_initialization_info[1];
            T.y() = _lidar_initialization_info[2];
            T.z() = _lidar_initialization_info[3];
            // Rotation
            Eigen::Quaterniond Q = Eigen::Quaterniond(_lidar_initialization_info[7],
                                                      _lidar_initialization_info[4],
                                                      _lidar_initialization_info[5],
                                                      _lidar_initialization_info[6]);
            R = Q.normalized().toRotationMatrix();
            // Velocity
            V.x() = _lidar_initialization_info[8];
            V.y() = _lidar_initialization_info[9];
            V.z() = _lidar_initialization_info[10];
            // Acceleration bias
            Ba.x() = _lidar_initialization_info[11];
            Ba.y() = _lidar_initialization_info[12];
            Ba.z() = _lidar_initialization_info[13];
            // Gyroscope bias
            Bg.x() = _lidar_initialization_info[14];
            Bg.y() = _lidar_initialization_info[15];
            Bg.z() = _lidar_initialization_info[16];
            // Gravity
            gravity = _lidar_initialization_info[17];   //; 这里也只是传递了一个重力的模长而已
        };

        map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>> > > points;
        double t;
        
        IntegrationBase *pre_integration;
        bool is_key_frame;

        // Lidar odometry info
        int reset_id;
        Vector3d T;
        Matrix3d R;
        Vector3d V;
        Vector3d Ba;
        Vector3d Bg;
        double gravity;
};


bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);

//? added by lidar
class odometryRegister
{
public:

    ros::NodeHandle n;
    tf::Quaternion q_lidar_to_cam;   //; R_LI, 即IMU坐标系到LiDAR坐标系的旋转
    Eigen::Quaterniond q_lidar_to_cam_eigen;

    ros::Publisher pub_latest_odometry; 

    odometryRegister(ros::NodeHandle n_in):
    n(n_in)
    {
        //? mod：修改这里
        // Step 1：原作者代码
        //; tf::Quaternion(x, y, z, w), 这里(0, 1, 0, 0)转成旋转矩阵是：[-1, 0, 0;
        //;                                                           0,  1, 0;
        //;                                                           0,  0, -1];
        q_lidar_to_cam = tf::Quaternion(0, 1, 0, 0); // rotate orientation // mark: camera - lidar
        
        //; Eigen::Quaterniond(w, x, y, z)，这里(0, 0, 0, 1)转成旋转矩阵是：[-1, 0, 0;
        //;                                                              0,  -1, 0;
        //;                                                              0,  0, 1];
        //; 其实就是绕着z轴转了180度，和作者的注释是一样的
        q_lidar_to_cam_eigen = Eigen::Quaterniond(0, 0, 0, 1); // rotate position by pi, (w, x, y, z) // mark: camera - lidar
        // pub_latest_odometry = n.advertise<nav_msgs::Odometry>("odometry/test", 1000);


        // Step 2: M2DGR修改
        //; 这里q_lidar_to_cam就是单位阵I_3
        // q_lidar_to_cam = tf::Quaternion(0, 0, 0, 1);
        //; 这里q_lidar_to_cam_eigen也是单位阵I_3
        // q_lidar_to_cam_eigen = Eigen::Quaterniond(1,0,0,0); 


        // Step 3: 学习小组修改
        // q_lidar_to_cam = tf::createQuaternionFromRPY(L_I_RX, L_I_RY, L_I_RZ); // rotate orientation // mark: camera - lidar
        // Eigen::AngleAxisd roll_vector(L_I_RX,Eigen::Vector3d::UnitX());
        // Eigen::AngleAxisd pitch_vector(L_I_RY,Eigen::Vector3d::UnitY());
        // Eigen::AngleAxisd yaw_vector(L_I_RZ,Eigen::Vector3d::UnitZ());
        // Eigen::Quaterniond rotation_vector = yaw_vector*pitch_vector*roll_vector;
        // q_lidar_to_cam_eigen = Eigen::Quaterniond ( rotation_vector );// rotate position by pi, (w, x, y, z) // mark: camera - lidar
    }

    // convert odometry from ROS Lidar frame to VINS camera frame
    //; 根据当前图像的时间戳，从LiDAR里程计中寻找对应时间戳的位姿，作为当前相机位姿的先验
    vector<float> getOdometry(deque<nav_msgs::Odometry>& odomQueue, double img_time)
    {
        vector<float> odometry_channel;

        //; reset id(1), P(3), Q(4), V(3), Ba(3), Bg(3), gravity(1)
        odometry_channel.resize(18, -1); 

        nav_msgs::Odometry odomCur;
        
        // pop old odometry msg
        while (!odomQueue.empty()) 
        {
            //; 先给一个比较大的容忍度，50ms
            if (odomQueue.front().header.stamp.toSec() < img_time - 0.05)
                odomQueue.pop_front();
            else
                break;
        }
        if (odomQueue.empty())
        {
            return odometry_channel;
        }

        // find the odometry time that is the closest to image time
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            odomCur = odomQueue[i];
            //; 500Hz的IMU，所以最近的时间戳应该是0.002s之内
            if (odomCur.header.stamp.toSec() < img_time - 0.002) // 500Hz imu
                continue;
            else
            //; 这里break掉之后，得到的odomCur就是当前距离图像时间戳最近的位姿
                break;
        }

        // time stamp difference still too large
        //; 这个一般不太会发生
        if (abs(odomCur.header.stamp.toSec() - img_time) > 0.05)
        {
            return odometry_channel;
        }

        // convert odometry rotation from lidar ROS frame to VINS camera frame 
        // (only rotation, assume lidar, camera, and IMU are close enough)
        tf::Quaternion q_odom_lidar;
        tf::quaternionMsgToTF(odomCur.pose.pose.orientation, q_odom_lidar);

        //? mod: 修改这里
        // Step 1: 原作者代码
        //! 疑问：为什么要绕着z轴转180度？
        //; 解答：我猜作者是故意为之，就是想让轨迹尽量分开，这样可以让VIO和LIO轨迹结果更方便对比？
        tf::Quaternion q_odom_cam = tf::createQuaternionFromRPY(0, 0, M_PI) 
            * (q_odom_lidar * q_lidar_to_cam); // global rotate by pi // mark: camera - lidar
        
        // Step 2: M2DGR修改
        // tf::Quaternion q_odom_cam = tf::createQuaternionFromRPY(0, 0, M_PI / 2.0) 
            // * (q_odom_lidar * q_lidar_to_cam); 
        
        // Step 3: 学习小组的修改
        //; extQRPY是在params_camera.yaml中新加的参数
        // tf::Quaternion q_odom_cam = extQRPY *(q_odom_lidar * q_lidar_to_cam); // global rotate by pi // mark: camera - lidar

        tf::quaternionTFToMsg(q_odom_cam, odomCur.pose.pose.orientation);

        // convert odometry position from lidar ROS frame to VINS camera frame
        Eigen::Vector3d p_eigen(odomCur.pose.pose.position.x, odomCur.pose.pose.position.y, odomCur.pose.pose.position.z);
        Eigen::Vector3d v_eigen(odomCur.twist.twist.linear.x, odomCur.twist.twist.linear.y, odomCur.twist.twist.linear.z);
        
        //! 疑问：同理，这里p_v也是绕着z轴转了180度
        Eigen::Vector3d p_eigen_new = q_lidar_to_cam_eigen * p_eigen;
        Eigen::Vector3d v_eigen_new = q_lidar_to_cam_eigen * v_eigen;

        odomCur.pose.pose.position.x = p_eigen_new.x();
        odomCur.pose.pose.position.y = p_eigen_new.y();
        odomCur.pose.pose.position.z = p_eigen_new.z();

        odomCur.twist.twist.linear.x = v_eigen_new.x();
        odomCur.twist.twist.linear.y = v_eigen_new.y();
        odomCur.twist.twist.linear.z = v_eigen_new.z();

        odometry_channel[0] = odomCur.pose.covariance[0];
        odometry_channel[1] = odomCur.pose.pose.position.x;
        odometry_channel[2] = odomCur.pose.pose.position.y;
        odometry_channel[3] = odomCur.pose.pose.position.z;
        odometry_channel[4] = odomCur.pose.pose.orientation.x;
        odometry_channel[5] = odomCur.pose.pose.orientation.y;
        odometry_channel[6] = odomCur.pose.pose.orientation.z;
        odometry_channel[7] = odomCur.pose.pose.orientation.w;
        odometry_channel[8]  = odomCur.twist.twist.linear.x;
        odometry_channel[9]  = odomCur.twist.twist.linear.y;
        odometry_channel[10] = odomCur.twist.twist.linear.z;
        odometry_channel[11] = odomCur.pose.covariance[1];
        odometry_channel[12] = odomCur.pose.covariance[2];
        odometry_channel[13] = odomCur.pose.covariance[3];
        odometry_channel[14] = odomCur.pose.covariance[4];
        odometry_channel[15] = odomCur.pose.covariance[5];
        odometry_channel[16] = odomCur.pose.covariance[6];
        odometry_channel[17] = odomCur.pose.covariance[7];

        return odometry_channel;
    }
};
