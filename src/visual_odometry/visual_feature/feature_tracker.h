#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;
};

//? added by lidar ： 利用LiDAR点云获取图像特征点的深度信息
class DepthRegister
{
public:
    ros::NodeHandle n;
    // publisher for visualization
    ros::Publisher pub_depth_feature;
    ros::Publisher pub_depth_image;
    ros::Publisher pub_depth_cloud;

    tf::TransformListener listener;
    tf::StampedTransform transform;
    tf::StampedTransform latest_transform;

    const int num_bins = 360;
    vector<vector<PointType>> pointsArray;

    DepthRegister(ros::NodeHandle n_in) : n(n_in)
    {
        // messages for RVIZ visualization
        //; 这个在rviz中显示的是白色的点云（也不一定是白色的，距离比较近的点是粉红色，远点是白色），
        //; 就是最后筛选出来的这一帧的视觉特征点对应的点云，注意这个点云并不是LiDAR
        //; 扫描得到的点云，而是通过计算视觉特征点的光线和它最近的三个LiDAR特征点构成的平面之间的交线得到的
        pub_depth_feature = n.advertise<sensor_msgs::PointCloud2>(
            PROJECT_NAME + "/vins/depth/depth_feature", 5);

        //; 显示当前帧图像的聚合点云，用于下一步的LiDAR点云辅助图像特征点深度的提取
        pub_depth_image = n.advertise<sensor_msgs::Image>(
            PROJECT_NAME + "/vins/depth/depth_image", 5);

        //; 这个应该是聚合点云，就是前5秒的所有点云累计起来，然后筛选出在相机视野范围内的点云
        pub_depth_cloud = n.advertise<sensor_msgs::PointCloud2>(
            PROJECT_NAME + "/vins/depth/depth_cloud", 5);

        pointsArray.resize(num_bins);
        for (int i = 0; i < num_bins; ++i)
            pointsArray[i].resize(num_bins);
    }

    //? 重要：获取特征点深度的函数
    sensor_msgs::ChannelFloat32 get_depth(const ros::Time &stamp_cur, const cv::Mat &imageCur,
                                          const pcl::PointCloud<PointType>::Ptr &depthCloud,
                                          const camodocal::CameraPtr &camera_model,
                                          const vector<geometry_msgs::Point32> &features_2d)
    {
        // 0.1 initialize depth for return
        //// 初始化深度值通道
        sensor_msgs::ChannelFloat32 depth_of_point;
        depth_of_point.name = "depth";
        depth_of_point.values.resize(features_2d.size(), -1); //; 先把特征点的深度值都赋值成-1

        // 0.2  check if depthCloud available
        //; 没有点云，深度直接都是默认值-1，直接返回
        if (depthCloud->size() == 0)
            return depth_of_point;

        //; 1.因为点云投影需要转换到相机的世界坐标系下，所以这里还是要得到相机当前最新的位姿
        //; 2.注意这里的tf是在哪里广播的，是在vins的imu回调函数中，发送imu频率的里程计位姿的时候，在
        //;   那里面发布了vio位姿的广播
        // 0.3 look up transform at current image time
        //// 收听tf广播
        try{
            listener.waitForTransform("vins_world", "vins_body_ros", stamp_cur, ros::Duration(0.01));
            listener.lookupTransform("vins_world", "vins_body_ros", stamp_cur, transform);
        } 
        catch (tf::TransformException ex){
            // ROS_ERROR("image no tf");
            return depth_of_point;
        }

        //// tf位姿格式转换到仿射变换
        //; Q:这里为什么用仿射变换接受的？不应该是欧式变换吗？
        double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
        xCur = transform.getOrigin().x();
        yCur = transform.getOrigin().y();
        zCur = transform.getOrigin().z();
        tf::Matrix3x3 m(transform.getRotation());
        m.getRPY(rollCur, pitchCur, yawCur);
        Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

        // 0.4 transform cloud from global frame to camera frame
        //// 世界坐标系点云转换到相机坐标系点云，注意相机系是flu坐标表示
        pcl::PointCloud<PointType>::Ptr depth_cloud_local(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*depthCloud, *depth_cloud_local, transNow.inverse());

        // 0.5 project undistorted normalized (z) 2d features onto a unit sphere
        //// 把2d关键点投影到球坐标系，并做坐标变换
        //; 注意这里是把相机观测到的特征点投影到球面上，然后里面的坐标变换是因为相机和LiDAR的xyz轴不对应
        pcl::PointCloud<PointType>::Ptr features_3d_sphere(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)features_2d.size(); ++i)
        {
            // normalize 2d feature to a unit sphere
            Eigen::Vector3f feature_cur(features_2d[i].x, features_2d[i].y, features_2d[i].z); // z always equal to 1
            feature_cur.normalize();    //; 这里又normalize，就是把归一化平面上的点都强制投影到单位球面上

            // convert to ROS standard
            //// LiDAR是Front-Left-Up坐标系，Camera是Right-Down-Front坐标系
            //; LiDAR的xyz，分别对应camera的z/-x/-y
            PointType p;
            p.x = feature_cur(2);
            p.y = -feature_cur(0);
            p.z = -feature_cur(1);
            p.intensity = -1; // intensity will be used to save depth
            features_3d_sphere->push_back(p);
        }

        //; 注意下面这个点云投影到图像上的操作和LeGO-LOAM是一样的
        // 3. project depth cloud on a range image, filter points satcked in the same region
        //// 筛选在相机FoV以内的，距离相机近的点云点，用来构建用于搜索关键点深度的点云
        //// 点云投影图分辨率, 这里是0.5度一个点云
        //;  bin_res = 0.5,  num_bins = 360
        float bin_res = 180.0 / (float)num_bins;  // currently only cover the space in front of lidar (-90 ~ 90)
        cv::Mat rangeImage = cv::Mat(num_bins, num_bins, CV_32F, cv::Scalar::all(FLT_MAX)); //// 点云投影图

        //; 遍历在相机坐标系下表示的点云，投影到一个 360x360 的矩形上，为什么投影的大小是360*360呢？
        // std::cout << "before : " << (int)depth_cloud_local->size() << "\n";
        for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
        {
            PointType p = depth_cloud_local->points[i];
            // filter points not in camera view
            //// 排除相机FoV以外的点
            //; 这里相机FOV是atan2(10) = 84.3度（准确的说这个是半锥角）
            if (p.x < 0 || abs(p.y / p.x) > 10 || abs(p.z / p.x) > 10)
                continue;
            // find row id in range image
            //// 计算俯仰角，横排id
            //; atan2 (-pi, pi], 但是这里结果只能在-90 ~ 90之间，所以+90变成 0 ~ 180之间
            float row_angle = atan2(p.z, sqrt(p.x * p.x + p.y * p.y)) * 180.0 / M_PI + 90.0; // degrees, bottom -> up, 0 -> 360
            int row_id = round(row_angle / bin_res);   //; /0.5的分辨率，在mat中的行序号就在0 ~ 360之间
            // find column id in range image
            //// 计算偏航角，竖排id  
            //; 1.关于atan2: http://c.biancheng.net/ref/atan2.html  
            //; 2.下面计算的是atan2(x/y)，在LiDAR前面，从左到右是0~180度，
            float col_angle = atan2(p.x, p.y) * 180.0 / M_PI; // degrees, left -> right, 0 -> 360
            int col_id = round(col_angle / bin_res);
            // id may be out of boundary
            //// 排除过界id
            if (row_id < 0 || row_id >= num_bins || col_id < 0 || col_id >= num_bins)
                continue;
            // only keep points that's closer
            //// 计算点云点与相机的距离，如果同时有两个点在同一个像素坐标上，则保留较小的那个
            //; 保留距离小的那个是因为距离越远点云越稀疏，扫描得到的距离可能就越不准确
            float dist = pointDistance(p);
            if (dist < rangeImage.at<float>(row_id, col_id))
            {
                rangeImage.at<float>(row_id, col_id) = dist;
                pointsArray[row_id][col_id] = p;
            }
        }

        // 4. extract downsampled depth cloud from range image
        //// 构建用于搜索关键点深度的点云并发布
        //; 这里就是把上面点云投影到一个图像上之后，在各个位置有值的点云拿出来
        pcl::PointCloud<PointType>::Ptr depth_cloud_local_filter2(new pcl::PointCloud<PointType>());
        for (int i = 0; i < num_bins; ++i)
        {
            for (int j = 0; j < num_bins; ++j)
            {
                //; 不是无穷大就说明有深度点，那么就把它放到点云中
                if (rangeImage.at<float>(i, j) != FLT_MAX)
                    depth_cloud_local_filter2->push_back(pointsArray[i][j]);
            }
        }
        *depth_cloud_local = *depth_cloud_local_filter2;
        //; 把筛选后的聚合点云发布出去。此时聚合点云的个数最大应该是360*360 = 129600
        //; 但是经过测试作者的数据集，在投影到矩阵上之前有45000个点左右，筛选后有15000个点左右，远远没有达到360*360的个数
        publishCloud(&pub_depth_cloud, depth_cloud_local, stamp_cur, "vins_body_ros");

        // 5. project depth cloud onto a unit sphere
        //// 把上面的点云投影到单位球球面，intensity保存距离信息
        pcl::PointCloud<PointType>::Ptr depth_cloud_unit_sphere(new pcl::PointCloud<PointType>());
        // std::cout << "after : " << (int)depth_cloud_local->size() << "\n";
        for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
        {
            PointType p = depth_cloud_local->points[i];
            float range = pointDistance(p);
            p.x /= range;
            p.y /= range;
            p.z /= range;
            p.intensity = range;
            depth_cloud_unit_sphere->push_back(p);
        }
        //// 如果球面上的点云点太少则直接返回
        if (depth_cloud_unit_sphere->size() < 10)
            return depth_of_point;

        // 6. create a kd-tree using the spherical depth cloud
        //// 建立kdtree
        pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());
        kdtree->setInputCloud(depth_cloud_unit_sphere);

        // 7. find the feature depth using kd-tree
        //// kdtree搜索与关键点球面最近的三个点
        vector<int> pointSearchInd;
        vector<float> pointSearchSqDis;
        //! 这个阈值 * 5 再平方怎么来的？没看懂
        //; 这个值是sin(0.5度)，也就是一个网格的值
        float dist_sq_threshold = pow(sin(bin_res / 180.0 * M_PI) * 5.0, 2); //// 搜索距离阈值

        //; 遍历相机特征点在单位球面上的投影
        for (int i = 0; i < (int)features_3d_sphere->size(); ++i)
        {
            kdtree->nearestKSearch(features_3d_sphere->points[i], 3, pointSearchInd, pointSearchSqDis);

            //// 保证有三个点并且最远距离仍不超过阈值
            if (pointSearchInd.size() == 3 && pointSearchSqDis[2] < dist_sq_threshold)
            {
                //; 把这3个点都乘以他们的深度，也就是再恢复成原来的真实的坐标
                float r1 = depth_cloud_unit_sphere->points[pointSearchInd[0]].intensity;
                Eigen::Vector3f A(depth_cloud_unit_sphere->points[pointSearchInd[0]].x * r1,
                                  depth_cloud_unit_sphere->points[pointSearchInd[0]].y * r1,
                                  depth_cloud_unit_sphere->points[pointSearchInd[0]].z * r1);

                float r2 = depth_cloud_unit_sphere->points[pointSearchInd[1]].intensity;
                Eigen::Vector3f B(depth_cloud_unit_sphere->points[pointSearchInd[1]].x * r2,
                                  depth_cloud_unit_sphere->points[pointSearchInd[1]].y * r2,
                                  depth_cloud_unit_sphere->points[pointSearchInd[1]].z * r2);

                float r3 = depth_cloud_unit_sphere->points[pointSearchInd[2]].intensity;
                Eigen::Vector3f C(depth_cloud_unit_sphere->points[pointSearchInd[2]].x * r3,
                                  depth_cloud_unit_sphere->points[pointSearchInd[2]].y * r3,
                                  depth_cloud_unit_sphere->points[pointSearchInd[2]].z * r3);

                // https://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
                Eigen::Vector3f V(features_3d_sphere->points[i].x,
                                  features_3d_sphere->points[i].y,
                                  features_3d_sphere->points[i].z);

                //// 计算关键点与ABC平面之间的距离
                //! 这里暂时没有弄明白，怎么计算的？
                //! 解答：这个计算方式就在上面的那个链接里，高赞的那个回答就是。
                //!    注意仔细思考的话这里的s其实是点到相机中心的直线距离，和LiDAR的距离比较像（一个点打出去测量的距离）
                //!    而不是点到相机的直线距离（z轴距离，逆深度就是用这个距离的倒数），而最后我们要的是逆深度这个距离，
                //!    所以可以看到最后对深度进行赋值的时候，使用的是Z轴距离，也就是逆深度的距离
                Eigen::Vector3f N = (A - B).cross(B - C); //; 得到ABC三点共面的法向量
                float s = (N(0) * A(0) + N(1) * A(1) + N(2) * A(2)) / (N(0) * V(0) + N(1) * V(1) + N(2) * V(2));

                float min_depth = min(r1, min(r2, r3));
                float max_depth = max(r1, max(r2, r3));

                //// 三个点之间的距离也不能太远，距离s应该在ABC距离之间
                //; 1.首先这个判断>2是在论文中防止深度歧义说了的，因为不同帧得到的点云组成一个大的聚合点云，会导致后面的点
                //;   实际被前面的点挡住了，但是仍然在聚合点云中存在，这样再投影到球面上之后，寻找的3个最近的点可能就是在
                //;   不同物体上的点，比如2个是后面的物体，1个是前面的物体。这个时候存在两个问题：
                //;   (1)恢复的深度很奇怪，在前后两个物体的深度之间，但是实际我们知道像素点的深度要么是后面物体的深度，
                //;      要么是前面物体的深度
                //;   (2)如(1)所述，到底是后面物体的深度还是前面物体的深度？其实是很难知道的。所以这里干脆直接排除这种情况
                //;      不给这种情况下的像素点赋值深度。
                //; 2.判断s<=0.5这个很好理解，就是点的深度不可能距离太近。
                if (max_depth - min_depth > 2 || s <= 0.5)
                {
                    continue;
                }
                //; 下面这个判断应该也比较好理解，因为一般情况下三个点如果是在一个倾斜的平面上，
                //; 那么求得的像素点的光线和平面的交点这个深度，应该是在这三个点的深度之间的。
                else if (s - max_depth > 0)
                {
                    s = max_depth;
                }
                else if (s - min_depth < 0)
                {
                    s = min_depth;
                }
                // convert feature into cartesian space if depth is available
                //// 关键点乘以距离，其中x坐标应当就等于深度值
                //; 注意这里是x坐标，因为前面进行了相机和LIDAR坐标系的转换，这里的x代表的就是相机的
                //; z轴，也就是深度
                features_3d_sphere->points[i].x *= s;
                features_3d_sphere->points[i].y *= s;
                features_3d_sphere->points[i].z *= s;

                //; 注意这里深度就赋值给点云数据结构的强度了，所以后面就可以利用这个来判断是否有有效的深度
                // the obtained depth here is for unit sphere, VINS estimator needs depth
                // for normalized feature (by value z), (lidar x = camera z)
                features_3d_sphere->points[i].intensity = features_3d_sphere->points[i].x;
            }
        }

        // visualize features in cartesian 3d space (including the feature without depth (default 1))
        //// 发布包含深度的关键点点云
        publishCloud(&pub_depth_feature, features_3d_sphere, stamp_cur, "vins_body_ros");

        // update depth value for return
        //// 保存大于3的深度值，返回到通道
        for (int i = 0; i < (int)features_3d_sphere->size(); ++i)
        {
            if (features_3d_sphere->points[i].intensity > 3.0)
                depth_of_point.values[i] = features_3d_sphere->points[i].intensity;
        }

        // visualization project points on image for visualization (it's slow!)
        //// 可视化，对应Rviz中的Image Depth
        if (pub_depth_image.getNumSubscribers() != 0)
        {
            vector<cv::Point2f> points_2d;
            vector<float> points_distance;

            //; 遍历投影到矩形上，并且经过筛选之后的LiDAR点云
            for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
            {
                // convert points from 3D to 2D
                //; 注意这里换轴是因为depth_cloud_local点的坐标轴仍然保持和LiDAR的方向一样，
                //; 也就是xyz仍然是前左上的坐标系，而现在我们要把点云利用相机模型进行投影，所以
                //; 这里应该用相机坐标系下的点云，也就是xyz是右下前坐标系的值，所以这里要换轴
                Eigen::Vector3d p_3d(-depth_cloud_local->points[i].y,
                                     -depth_cloud_local->points[i].z,
                                     depth_cloud_local->points[i].x);
                Eigen::Vector2d p_2d;
                camera_model->spaceToPlane(p_3d, p_2d);

                points_2d.push_back(cv::Point2f(p_2d(0), p_2d(1)));
                points_distance.push_back(pointDistance(depth_cloud_local->points[i]));
            }

            cv::Mat showImage, circleImage;
            cv::cvtColor(imageCur, showImage, cv::COLOR_GRAY2RGB);

            circleImage = showImage.clone();
            for (int i = 0; i < (int)points_2d.size(); ++i)
            {
                float r, g, b;
                //; 根据点的深度，给不同深度的点赋不同的颜色
                getColor(points_distance[i], 50.0, r, g, b);
                cv::circle(circleImage, points_2d[i], 0, cv::Scalar(r, g, b), 5);
            }

            //; 这里就混合当前相机的图像和LiDAR点云的图像
            //; 参数1：src1，第一个原数组.  参数2：alpha，第一个数组元素权重
            //; 参数3：src2第二个原数组     参数4：beta，第二个数组元素权重
            //; 参数5：gamma，图1与图2作和后添加的数值。不要太大，不然图片一片白。总和等于255以上就是纯白色了。
            //; 参数6：dst，输出图片
            cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage); // blend camera image and circle image

            cv_bridge::CvImage bridge;
            bridge.image = showImage;
            bridge.encoding = "rgb8";
            sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
            imageShowPointer->header.stamp = stamp_cur;
            pub_depth_image.publish(imageShowPointer);
        }

        return depth_of_point;
    }

    //; 高级操作，实在没有看懂是怎么根据深度来赋值颜色的
    void getColor(float p, float np, float &r, float &g, float &b)
    {
        float inc = 6.0 / np;  //; np传入的时候是50，这里就是6/50 = 0.12
        float x = p * inc;
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
        if ((0 <= x && x <= 1) || (5 <= x && x <= 6))
            r = 1.0f;
        else if (4 <= x && x <= 5)
            r = x - 4;
        else if (1 <= x && x <= 2)
            r = 1.0f - (x - 1);

        if (1 <= x && x <= 3)
            g = 1.0f;
        else if (0 <= x && x <= 1)
            g = x - 0;
        else if (3 <= x && x <= 4)
            g = 1.0f - (x - 3);

        if (3 <= x && x <= 5)
            b = 1.0f;
        else if (2 <= x && x <= 3)
            b = x - 2;
        else if (5 <= x && x <= 6)
            b = 1.0f - (x - 5);
        r *= 255.0;
        g *= 255.0;
        b *= 255.0;
    }
};