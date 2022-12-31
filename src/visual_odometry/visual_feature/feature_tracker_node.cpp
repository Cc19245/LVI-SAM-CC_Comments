#include "feature_tracker.h"
#include <sensor_msgs/PointCloud2.h>


#define SHOW_UNDISTORTION 0


// mtx lock for two threads
std::mutex mtx_lidar;

// global variable for saving the depthCloud shared between two threads
pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());

// global variables saving the lidar point cloud
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double> timeQueue;

// global depth register for obtaining depth of a feature
DepthRegister *depthRegister;

// feature publisher for VINS estimator
ros::Publisher pub_feature;
ros::Publisher pub_match;
ros::Publisher pub_restart;
ros::Publisher depth_cloud_publisher;

// feature tracker variables
FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;



void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    double cur_img_time = img_msg->header.stamp.toSec();

    //// 第一帧图像则直接返回
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = cur_img_time;
        last_image_time = cur_img_time;
        return;
    }
    //// 如果当前帧与上一帧间隔有问题，则跟踪不稳定，清空、重新初始化
    //// detect unstable camera stream
    if (cur_img_time - last_image_time > 1.0 || cur_img_time < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = cur_img_time;
    // frequency control 频率控制 默认pub频率20
    //// 每一帧图片都要处理，但不一定要全都pub，因为所有pub出去的内容都会增加后端优化的规模
    if (round(1.0 * pub_count / (cur_img_time - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (cur_img_time - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = cur_img_time;
            pub_count = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    //// 提取照片个，格式转换
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        //// 每个相机都有一个跟踪实例
        //// 对于单目相机：提取特征+光流跟踪  对于双目相机：左目按照单目处理，右目灰度均衡化处理或仅保存
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cur_img_time);
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

        #if SHOW_UNDISTORTION
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
        #endif
    }

    //// 更新特征点ID
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        //// 用pointcloud消息类型，这种消息类型包含下面各种通道：归一化平面坐标， 关键点索引，关键点像素坐标，关键点速度，关键点的深度
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header.stamp = img_msg->header.stamp;
        feature_points->header.frame_id = "vins_body";

        //// 记录每个相机的关键点id
        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                //// 对于跟踪次数大于1的关键点
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;            //// 归一化平面上的点

                    //; 注意这里feature_points特征点和下面的各种sensor_msgs::ChannelFloat32信息是一一对应的
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }

        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);

        //? added by lidar : 从LiDAR中提取点云，获得特征点的深度信息
        // get feature depth from lidar point cloud
        //// 获取连续几帧聚合的点云
        pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
        mtx_lidar.lock();
        *depth_cloud_temp = *depthCloud;  //; 在lidar_callback中得到的世界坐标系下的5s内的激光点云
        mtx_lidar.unlock();

        //// 通过聚合点云获取图像的特征点深度信息
        sensor_msgs::ChannelFloat32 depth_of_points = depthRegister->get_depth(
            img_msg->header.stamp, show_img, depth_cloud_temp, trackerData[0].m_camera, feature_points->points);
        feature_points->channels.push_back(depth_of_points);
        
        // skip the first image; since no optical speed on frist image
        //// 第一次跟踪的图片只有关键点坐标信息，没有速度信息，跳过
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_feature.publish(feature_points);

        //; 注意这下面的显示也跟原来的vins-mono不一样了，可以用颜色来显示跟踪的情况了
        // publish features in image
        if (pub_match.getNumSubscribers() != 0)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::RGB8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    if (SHOW_TRACK)
                    {
                        // track count
                        //// 如果设定SHOW_TRACK，在关键点上画圈，关键点跟踪次数越多，圆圈越绿，反之则越蓝
                        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(255 * (1 - len), 255 * len, 0), 4);
                    } else {
                        // depth
                        //// 如果不设定SHOW_TRACK，则对有深度的关键点画绿圈，对没有深度的关键点画红圈
                        if(j < depth_of_points.values.size())
                        {
                            if (depth_of_points.values[j] > 0)
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 10, cv::Scalar(0, 255, 0), 4);
                            else
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 0, 255), 4);
                        }
                    }
                }
            }

            pub_match.publish(ptr->toImageMsg());
        }
    }
}


/**
 * @brief 收到的LiDAR坐标系下的点云的回调函数
 * 
 * @param[in] laser_msg 
 */
void lidar_callback(const sensor_msgs::PointCloud2ConstPtr& laser_msg)
{
    static int lidar_count = -1;
    //// 默认隔三帧取一帧
    //; 这里的原因是对于低速运动的物体来说，360度的激光雷达扫描到的点云
    //; 大部分都是重复的，所以没必要把每一帧LiDAR点云都投影到图像上进行匹配
    if (++lidar_count % (LIDAR_SKIP+1) != 0)
        return;

    // 0. listen to transform 获取camera在世界坐标系下的位姿(camera to world)
    //// 收听tf广播
    static tf::TransformListener listener;
    static tf::StampedTransform transform;   //; camera to world
    try{
        listener.waitForTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, ros::Duration(0.01));
        listener.lookupTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, transform);
    } 
    catch (tf::TransformException ex){
        // ROS_ERROR("lidar no tf");
        return;
    }

    //// 获取位姿
    double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
    //; xyz平移
    xCur = transform.getOrigin().x();
    yCur = transform.getOrigin().y();
    zCur = transform.getOrigin().z();
    //; 旋转
    tf::Matrix3x3 m(transform.getRotation());
    m.getRPY(rollCur, pitchCur, yawCur);
    Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

    // 1. convert laser cloud message to pcl 激光点云数据格式转换
    //// 获取点云
    pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*laser_msg, *laser_cloud_in);

    // 2. downsample new cloud (save memory) 
    //// 降采样
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
    static pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(laser_cloud_in);
    downSizeFilter.filter(*laser_cloud_in_ds);
    *laser_cloud_in = *laser_cloud_in_ds;

    // 3. filter lidar points (only keep points in camera view)
    //// 仅保留相机FoV中的点， 删除不在相机视野范围内的激光点
    //; 相机FoV一般也就是100度左右，100度已经比较大了
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
    {
        PointType p = laser_cloud_in->points[i];
        // 位于相机前方，水平FoV小于阈值，垂直FoV小于阈值
        if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
            laser_cloud_in_filter->push_back(p);
    }
    *laser_cloud_in = *laser_cloud_in_filter;

    // TODO: transform to IMU body frame
    // 4. offset T_lidar -> T_camera
    //// 把点云 从lidar坐标系转换到camera坐标系
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
    //; transOffset得到的就是LiDAR坐标系到相机坐标系的坐标变换关系，其实就是一个固定的外参
    Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
    *laser_cloud_in = *laser_cloud_offset;

    // 5. transform new cloud into global odom frame
    //// 把点云 从camera坐标系转换到global坐标系 
    //; 利用相机当前的位姿，把这阵点云转换到世界坐标系
    //; 注意：这里用的transNow位姿，是VINS的IMU的前左上坐标系 到 vins_world的坐标变换，所以前面对
    //;      点云转换的时候，也要把LiDAR点云转换到相机的前左上坐标系下
    pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_global, transNow);

    // 6. save new cloud
    //// 保存到队列
    //; 保存这一帧点云的时间戳和点云数据
    double timeScanCur = laser_msg->header.stamp.toSec();
    cloudQueue.push_back(*laser_cloud_global);
    timeQueue.push_back(timeScanCur);

    // 7. pop old cloud
    //// 删除与最新的点云时间戳相差5s以上的旧点云
    //; 因为一帧LiDAR的点云是比较少的，为了让点云能够尽量的覆盖图像的扫描范围，
    //; 所以这里保留最新的一段时间内的点云数据
    while (!timeQueue.empty())
    {
        if (timeScanCur - timeQueue.front() > 5.0)
        {
            cloudQueue.pop_front();
            timeQueue.pop_front();
        } else {
            break;
        }
    }

    std::lock_guard<std::mutex> lock(mtx_lidar);
    // 8. fuse global cloud
    //// 聚合队列中的点云，这里的操作就是把点云 + 起来
    depthCloud->clear();
    for (int i = 0; i < (int)cloudQueue.size(); ++i)
        *depthCloud += cloudQueue[i];

    // 9. downsample global cloud
    //// 聚合点云降采样
    //; 因为5S内的点云肯定还是有很大的重合的，所以这里再次进行一个降采样
    pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(depthCloud);
    downSizeFilter.filter(*depthCloudDS);
    *depthCloud = *depthCloudDS;

    // Step 结束：最终得到depthCloud， 就是位于世界坐标系下的、可能在相机FoV内的点云地图，
    //      后面将会把这个点云地图投影到相机平面上去

    //? add 发布这个点云显示
    sensor_msgs::PointCloud2 depthCloud_ros;
    pcl::toROSMsg(*depthCloud, depthCloud_ros);
    //; LiDAR最后一个点的时间
    depthCloud_ros.header = laser_msg->header;        // world; camera_init
    depth_cloud_publisher.publish(depthCloud_ros);
}

int main(int argc, char **argv)
{
    //// 初始化ROS
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Feature Tracker Started.\033[0m");
    //// 设置控制台日志显示信息
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Error);
    readParameters(n);

    //// 读取相机参数
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    // load fisheye mask to remove features on the boundary
    //// 对鱼眼相机，贴上掩膜防止提取边界特征点
    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_ERROR("load fisheye mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // initialize depthRegister (after readParameters())
    //// 初始化深度注册器
    depthRegister = new DepthRegister(n);
    
    // subscriber to image and lidar
    //// 订阅img话题和去畸变点云话题
    ros::Subscriber sub_img   = n.subscribe(IMAGE_TOPIC,       5,    img_callback);
    ros::Subscriber sub_lidar = n.subscribe(POINT_CLOUD_TOPIC, 5,    lidar_callback);
    if (!USE_LIDAR)
        sub_lidar.shutdown();

    // messages to vins estimator
    //// 发布自定义特征话题，含特征点图片话题，重启标志位话题
    pub_feature = n.advertise<sensor_msgs::PointCloud>(PROJECT_NAME + "/vins/feature/feature",     5);
    pub_match   = n.advertise<sensor_msgs::Image>     (PROJECT_NAME + "/vins/feature/feature_img", 5);
    pub_restart = n.advertise<std_msgs::Bool>         (PROJECT_NAME + "/vins/feature/restart",     5);

    //; add
    depth_cloud_publisher = n.advertise<sensor_msgs::PointCloud2>("/vins/depth/full_gathered_cloud", 100);

    // two ROS spinners for parallel processing (image and lidar)
    //// 两个线程共同处理
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}