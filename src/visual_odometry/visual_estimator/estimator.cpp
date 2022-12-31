#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    failureCount = -1;
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    ++failureCount;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    //// 第一次收到的imu测量直接设为初值
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    //// 当前帧如果没有创建预积分类实例时，自动创建
    if (!pre_integrations[frame_count]) 
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    //// 窗口内的第一帧会直接跳过预积分过程，因为滑动窗口第一帧是起点帧，没有预积分值
    if (frame_count != 0)
    {
        //// 导入imu测量和时间间隔dt到各个容器
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        // 由于重力向量的约束，使得横滚俯仰角是可观的;
        //// 中值积分计算状态
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    //// 更新上一帧测量
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, 
                             const vector<float> &lidar_initialization_info,
                             const std_msgs::Header &header)
{
    // 每帧图像处理后的特征信息，由一个map的数据结构组成；
    // 键为特征ID，值为该特征在每个相机中的x,y,z,u,v,velocity_x,velocity_y,depth 8个变量;
    // map< 特征ID, vector< pair<相机ID, x,y,z,u,v,vel_x,vel_y,depth >>>

    // Add new image features
    // 将新跟踪到的特征加入容器，检查新跟踪到的特征的视差，判断是否添加新的关键帧
    // 添加特征点, 检测视差 (根据视差决定marge最老帧还是次新帧)
    // (当前帧是关键帧，扔掉最老的关键帧看，控制规模)(当前帧不是关键帧，则扔掉当前帧)
    // 迭代当前帧检测到的每个路标点 id,看看feature队列中是否包含,若不含,则添加到feature队列中;若包含,则添加到对应id的FeaturePerFrame队列。
    // 计算第2最新帧与第3最新帧之间的平均视差（当前帧是第1最新帧），然后判断是否把第2最新帧添加为关键帧
    // 在未完成初始化时，如果窗口没有塞满，那么是否添加关键帧的判定结果不起作用，滑动窗口要塞满；
    // 只有在滑动窗口塞满后，或者初始化完成之后，才需要滑动窗口，此时才需要做关键帧判别，根据第2最新关键帧是否为关键帧选择相应的边缘化策略
    //// 根据平均视差（优秀旧特征总视差/优秀旧特征数量）确定是边缘化旧帧还是边缘化次新帧
    //// 如果平均视差小，说明当前帧的移动距离较小，不够作为新关键帧，应当边缘化上一帧，
    //// 反之说明平均视差大说明当前帧有资格作为新关键帧，应当边缘化窗口最旧的帧。
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    // Marginalize old imgs if lidar odometry available for initialization
    //// 在初始化阶段时，如果有lidar的信息可用来帮助初始化，则边缘化旧帧
    if (solver_flag == INITIAL && lidar_initialization_info[0] >= 0)
        marginalization_flag = MARGIN_OLD;

    //// 记录标头
    Headers[frame_count] = header;

    // 构造测量帧 (imageFrame包含图像数据, 激光里程计数据, 预积分)
    //; 注意这一个普通帧的信息在这里构造的时候就赋值了它的P V Q ba bg等信息，这个和VINS-Mono有很大的不同
    ImageFrame imageframe(image, lidar_initialization_info, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));

    //// 用最新一帧的bias和测量重构预积分类
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // Calibrate rotational extrinsics
    // 在线外参标定 (camera to imu 的旋转)
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 得到前后两帧的关联特征的坐标(归一化坐标)
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // 标定camera to imu的旋转
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // 滑动窗口满了才能进行初始化
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            // 外参标定成功, 并且距离上次初始化超过0.1s
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                //// 相机与IMU初始化
                //// 相机可以通过lidar或sfm初始化
                //// imu用雅克比矩阵标定初值并用相机IMU松耦合初始化 视觉和imu对齐，求解bias, 重力，速度；///
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            if(result)
            {
                //// 初始化
                solver_flag = NON_LINEAR;
                // 初始化成功, 先进行一次优化再滑动窗口
                solveOdometry();      // 主要的VIO优化过程，初始化成功以后每个滑动窗口都要优化一遍
                slideWindow();
                f_manager.removeFailures();
                // ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else
                // 初始化失败, 直接滑动窗口
                slideWindow(); 
        }
        else
            frame_count++;
    }
    else
    {
        solveOdometry();

        if (failureDetection())     // 对跟踪和优化效果进行评估
        {
            ROS_ERROR("VINS failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_ERROR("VINS system reboot!");
            return;
        }

        slideWindow();      // 根据之前的视差滑动窗口

        //根据计算的位姿对匹配点进行重投影，将重投影误差较大的点去除； 剔除求解失败的，即质量差的特征
        f_manager.removeFailures();

        // prepare output of VINS   更新位姿信息
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
    // 至此，VIO优化过程已经完成，其实VINS-fusion的流程还是比较简单的；
    // 1. 图像回调函数将图像数据push到image_buf中；
    // 2. 然后rosnode节点调用inputImage函数对图像进行光流跟踪，将跟踪到的特征点按照ID构建成一个map容器；然后push到feature_buf中；
    // 3. IMU回调函数调用inputIMU函数将IMU数据push到accBuf，gyrBuf中，同时通过中值积分对滑动窗口中最新的位姿进行递推预测；
    // 4. 选择多线程时，在另一个线程执行processMeasurements函数，读取上面得到的feature_buf和accBuf，gyrBuf，并进行优化处理；
    // 5. 读取到一帧feature_buf的数据后，接着调用getIMUInterval函数读取两帧图像数据之间的所有IMU数据；
    // 6. 调用processIMU函数，计算图像帧间的imu预积分值，即帧间平移，旋转，速度，以及bias；并利用imu对系统最新状态进行传播，为视觉三角化及重投影提供位姿初值；
    // 7. 调用processImage函数，对上述处理好的数据进行滑动窗口优化，是VIO优化的主要过程；
    // 7.1 首先将当前帧特征点添加到f_manager.feature容器中，然后进行视差的计算，用来决定滑动窗口边缘化新帧还是老帧；
    // 7.2 然后进行初始化；初始化采用视觉和IMU的松耦合,首先用SFM求解滑窗内所有帧的位姿,和所有路标点的3D位置,然后跟IMU预积分的值对齐,求解重力方向、尺度因子、陀螺仪bias及每一帧对应的速度;
    // 7.3 接着进行紧耦合优化；(将视觉约束、IMU约束放在一个大的目标函数中进行非线性优化,求解滑窗内所有帧的PVQ、bias等)
    // 7.3.1 如果不使用imu，也即只使用双目，则调用initFramePoseByPnP函数，，根据3d点利用pnp计算当前帧位姿的初值；如果使用imu的话，则前面处理IMU数据的时候已经使用预积分传播的值作为位姿初值；
    // 7.3.2 调用triangulate函数，对每帧图像还未具有深度信息的特征点进行三角化；后续准备利用上面估计的初始位姿构建重投影误差，对位姿进行优化；
    // 7.3.3 调用optimization函数,根据新的视觉观测及IMU信息构建ceres的problem，在滑动窗口内进行非线性优化；
    // 7.3.3.1 添加待优化的状态量: p,q,speed,ba,bg,相机和IMU外参p_cb,q_cb,时间戳偏移td;
    // 7.3.3.2 添加残差:添加边缘化残差;添加IMU残差;添加视觉残差;
    // 7.3.3.3 然后设置求解器属性,进行求解问题;
    // 7.3.3.4 边缘化流程：
            // 1.创建marginalization_info，从头到位都是marginalization_info这个变量来进行统筹安排进行边缘化；
            //   然后通过marginalization_info->addResidualBlockInfo()来添加约束，有三个方面的来源：（1）旧的（2）imu预积分项（3）特征点
            // 2.首先通过last_marginalization_info 构建出marginalization_factor，这个factor就是先验的残差项；
            // 3.然后将滑窗内第0帧和第1帧间的 IMU 预积分值(pre_integrations[1])构建预积分因子，然后构建imu残差项；
            // 4.接着挑选出第一次观测帧为第 0 帧的路标点，和其他共视帧构建重投影因子，然后构建视觉残差项；
            // 5.以上3项通过marginalization_info->addResidualBlockInfo()来添加到边缘化约束中；
            // 6.调用preMarginalize->Evaluate计算每次IMU和视觉观测(cost_function)对应的参数块(parameter_blocks)，雅可比矩阵(jacobians)，残差值(residuals);
            // 7.多线程计整个先验项的参数块，雅可比矩阵和 残差值;
            // 8.最后得到了优化项需要的两个变量：last_marginalization_info和last_marginalization_parameter_blocks；

}
bool Estimator::initialStructure()
{
    //? added by lidar ： 使用LiDAR的信息辅助初始化
    // Lidar initialization
    {
        bool lidar_info_available = true;

        // clear key frame in the container    
        //; 先把所有的普通帧都标志为 非关键帧    
        for (map<double, ImageFrame>::iterator frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
            frame_it->second.is_key_frame = false;

        // check if lidar info in the window is valid
        //// 必须窗口内的lidar信息全都可用
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            if (all_image_frame[Headers[i].stamp.toSec()].reset_id < 0 || 
                all_image_frame[Headers[i].stamp.toSec()].reset_id != all_image_frame[Headers[0].stamp.toSec()].reset_id)
            {
                // lidar odometry not available (id=-1) or lidar odometry relocated due to pose correction
                lidar_info_available = false;
                ROS_INFO("Lidar initialization info not enough.");
                break;
            }
        }

        //// 如果lidar信息完全可用，则可以直接通过lidar成功初始化
        if (lidar_info_available == true)
        {
            // Update state
            //// 采用lidar信息更新参数，重新传播预积分值，并设为关键帧
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                Ps[i] = all_image_frame[Headers[i].stamp.toSec()].T;
                Rs[i] = all_image_frame[Headers[i].stamp.toSec()].R;
                Vs[i] = all_image_frame[Headers[i].stamp.toSec()].V;
                Bas[i] = all_image_frame[Headers[i].stamp.toSec()].Ba;
                Bgs[i] = all_image_frame[Headers[i].stamp.toSec()].Bg;

                pre_integrations[i]->repropagate(Bas[i], Bgs[i]);

                all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
            }

            // update gravity
            //! 问题：这里重力加速度g在lio也进行了估计吗？
            //! 1.这个g是类成员变量，在原始的vins-mono中，是初始化的时候求解线性方程组得到的，然后根据重力
            //!   模长进行优化，最后第1帧和重力对齐，得到的g应该就是(0, 0, 9.8).
            //! 2.这里也可以看到，传入的gravity只是lio估计的一个模长，并且lio初始化的时候会根据9轴IMU
            //!   给出初始位姿，也就是已经和重力对齐了，所以这里给g赋值也直接和重力对齐即可。
            g = Eigen::Vector3d(0, 0, all_image_frame[Headers[0].stamp.toSec()].gravity);

            // reset all features
            //// 清空所有特征的深度值
            VectorXd dep = f_manager.getDepthVector();
            for (int i = 0; i < dep.size(); i++)
                dep[i] = -1;
            f_manager.clearDepth(dep);

            // triangulate all features
            //// 根据pose对所有特征三角化（相信根据lidar的pose三角化的深度值）
            Vector3d TIC_TMP[NUM_OF_CAM];
            for(int i = 0; i < NUM_OF_CAM; i++)
                TIC_TMP[i].setZero();
            ric[0] = RIC[0];
            f_manager.setRic(ric);
            //; 注意这里是对还没有深度的那些特征点重新进行三角化，实际就是原始的vins-mono中在初始话的时候全局sfm的最后，
            //; 会对所有帧的特征点中那些没有深度的特征点重新进行三角化。
            //! 警告：上面的解释是错误的！在初始化的时候，只会使用LiDAR的位姿对VIO进行初始化，而前端特征提取发来
            //!     的并没有深度值，因为这个时候VIO还没有初始化完成，还没有发送VIO的tf变换，所以前端特征提取无法
            //!     得到特征点对应的深度。因此这里只是利用LiDAR初始化VIO的位姿，然后认为LiDAR的位姿是准确的，用
            //!     LiDAR的位姿直接三角化视觉特征点（也就是初始化时候的深度仍然是由三角化得到的）
            f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

            return true;
        }
    }


    //check imu observibility
    //// 检查imu能观性
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        //// 计算平均加速度
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        //// 计算加速度均方差
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        //// 均方差小，则代表imu激励不足，imu能观性差
        if(var < 0.25)
        {
            ROS_INFO("Trying to initialize VINS, IMU excitation not enough!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    //// 将所有特征点的所有观测导入到sfm问题中
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    //// 找到窗口内的某一帧，与最新一帧的平均视差足够大并且通过二者计算的相对位姿足够精准
    //// 称这一帧为参考帧l，现在知道了参考帧与最新帧之间的位姿变换relative_R和T
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    //// 利用sfm方法求解位姿从窗口内所有关键帧位姿
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        //// 跳过关键帧
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        //// 这个帧位姿的初值应当为与它最近的并且晚于它的关键帧位姿（Twc-->Tcw）
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        //// 这里又将Tcw转换为Twc，旋转矩阵是正交矩阵，其逆矩阵就是转置
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    //// 相机IMU松耦合初始化
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    VectorXd x;
    //solve scale
    //// 相机IMU松耦合初始化计算尺度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_INFO("solve gravity failed, try again!");
        return false;
    }

    // change state
    //// 获取最新的位姿结果
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // reset all depth to -1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    //// 利用新的位姿结果重新三角化
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    //// 用最新的bias更新预积分值
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    //// 用最新的尺度更新各个参数
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    //// 用最新的重力加速度更新各个参数
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            //// 计算当前帧平均视差
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    //// 保证窗口填满
    if (frame_count < WINDOW_SIZE)
        return;

    //// 保证VIO处于非线性优化阶段
    if (solver_flag == NON_LINEAR)
    {
        // 利用初始位姿（来源于IMU预积分），对每帧图像还未具有深度信息的特征点进行三角化；
        // 后续准备利用imu传播的最新位姿构建重投影误差，对位姿进行优化；
        f_manager.triangulate(Ps, tic, ric);

        // 根据新的视觉观测及IMU信息在滑动窗口内对位姿进行优化；
        optimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_ERROR("VINS little feature %d!", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_ERROR("VINS big IMU acc bias estimation %f, restart estimator!", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_ERROR("VINS big IMU gyr bias estimation %f, restart estimator!", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    if (Vs[WINDOW_SIZE].norm() > 30.0)
    {
        ROS_ERROR("VINS big speed %f, restart estimator!", Vs[WINDOW_SIZE].norm());
        return true;
    }
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5.0)
    {
        ROS_ERROR("VINS big translation, restart estimator!");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_ERROR("VINS big z translation, restart estimator!");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / M_PI * 180.0;
    if (delta_angle > 50)
    {
        ROS_ERROR("VINS big delta_angle, moving too fast!");
        //return true;
    }
    return false;
}

// 整体思路:
// Step1:添加待优化的状态量
// 1.1 添加p,q,speed,ba,bg
// 1.2 添加相机和IMU外参p_cb,q_cb
// 1.3 将优化变量存入数组.因为ceres用的是double类型的数组,所以要做vector到double类型的变换 < WINDOW_SIZE - 2))

// Step2:添加残差  (预积分残差：IMU测量值与IMU增量的差) （视觉残差：重投影误差）  （边缘化残差：被扔掉的旧帧包含的位姿与滑动窗口位姿的残差）
// 2.1 添加边缘化残差
// 2.2 添加IMU残差.滑动窗口中的相邻两帧之间都有一个IMU残差.滑动窗口的大小是10.共有10个IMU残差项.
//       (注意:这里的IMU项和camera项之间是有一个系数的,这个系数就是他们各自的协方差矩阵;
//       IMU的协方差就是预计分的协方差,视觉的协方差就是一个固定的系数,f/1.5.(1.5是特征点追踪的方差))
// 2.3 添加视觉残差.针对滑动窗口中的所有特征点.只要该特征点被观测的次数大于2次并且观测到该特征点的首帧在滑动窗口的前7才行.然后通过观测该特征点的两帧建立残差.这里忽略闭环校正的情况
// 2.4 然后设置求解器属性,进行求解问题.这里设置的最大迭代次数是8,最大求解时间是0.04s,为了保证实时.

// Step3:marg部分
// 3.1 对于边缘化首帧
// 3.1.1 把之前存的残差部分加进来
// 3.1.2 把与首帧相关的残差项加进来,包含IMU,vision.
// 3.1.3 计算所有残差项的残差和雅克比
// 3.1.4 多线程构造Hx=b的结构,(需要细看)
// 3.1.5 marg结束,调整参数块在下一次window的位置
// 3.2 对于边缘化倒数第二帧
// 3.2.1 如果倒数第二帧不是关键帧,保留该帧的IMU测量,去掉该帧的visual,代码中都没写.
// 3.2.2 计算所有残差项的残差和雅克比
// 3.2.3 多线程构造Hx=b的结构,(需要细看)
// 3.2.4 marg结束,调整参数块在下一次window的位置
void Estimator:: optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);     // 柯西核函数
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 位姿中包含四元数，四元数是一种过参数化的表示，若按照四元数本身的维度，优化方向会有4个，而实际优化的方向(维度)只有3个;
        // 为了移除多余的空的优化方向，因此需要特殊定义优化过程中其运算的过程；采用LocalParameterization
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); // 参数三是优化变量增量更新方式，默认为相加
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    //添加相机与imu外参优化变量；
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]); // （优化变量块）参数块设为常值，不优化
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }

    //// 添加相机与imu时间戳偏移优化变量  TD = time delay
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    vector2double();    //// 优化前的参数保存到double数组参数，作为初值

    // marginalization residual
    // 添加边缘化残差因子
    // Step2.1:添加边缘化的残差
    // last_marginalization_parameter_blocks指的就是和被边缘化的变量有约束关系的变量,也就是heyijia博客中的Xb.
    // 这个marginalization的结构是始终存在的,随着marg结构的不断更新,last_marginalization_parameter_blocks对应的还是滑动窗口中的变量
    // last_marginalization_info 就是Xb对应的测量Zb,将这个约束作为Xb的先验,
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // IMU pre-integration residual
    // 添加imu预积分残差因子
    // Step2.2:添加IMU的residual
    // 这里IMU项和camera项之间是有一个系数的,这个系数就是他们各自的协方差矩阵；
    // IMU的协方差就是预积分的协方差(IMUFacotor::Evaluate,中添加IMU协方差,求解jacibian矩阵),
    // 而camera的测量残差则是一个固定的系数,f/1.5.(1.5是特征点追踪的方差)
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;

        // 因子就是代价函数，里面定义了测量值与优化变量之间误差的求解方式，误差对优化变量的导数（雅克比），以及测量值的协方差矩阵；
        // 由IMU预积分得到的相邻帧位姿变化量为测量约束构建代价函数；创建代价函数的时候传入值为待优化变量运算过程中的目标值；
        // 并且结构体内部定义了残差的计算方式，计算残差的函数输入参数即为待优化变量；
        // 此定义的代价函数会被ceres调用，用来计算优化变量和目标值之间的残差；
        // 注意计算残差的时候需要考虑测量值的协方差；方差越小，该项测量值产生的残差比重越大；
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);

        // AddResidualBlock输入包括代价函数和优化变量；
        // 注意此处优化变量要和代价函数中定义的误差求解需要的变量要一致，也就是误差代表着要优化变量之间的误差；误差最小时，也即变量已最优；
        // 注意此处传入的待优化变量数量以及顺序要和代价函数中定义的计算残差所用到的变量形式必须一致；
        // AddResidualBlock就是告诉ceres，以当前输入的变量为参数，按照代价函数定义的方式计算残差；
        // 相邻优化变量的相对位姿和IMU预积分值-之间的误差最小，优化变量(即全局位姿)则最优；////
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    // Image feature re-projection residual
    // 添加视觉重投影残差因子；
    // camera的测量协方差是一个固定的系数,f/1.5.(1.5个像素是特征点测量误差，和环境尺度也相关)
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();

        // 必须满足出现2次以上且在上上帧之前出现；
        // 观测的较好的特征才能提供好的视觉约束；
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        // 观测到该特征点的首帧
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        // 被首帧观测到的归一化坐标 (第一个归一化坐标)
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历观测到该feature的所有frames
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            // 当前帧观测到的特征坐标；
            // 有了首帧观测坐标，当前帧观测坐标，准备根据位姿构建重投影误差；
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                 it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                
                // depth is obtained from lidar, skip optimizing it
                //// 如果深度值来自于lidar，则不再优化逆深度
                if (it_per_id.lidar_depth_flag == true)
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);

                // depth is obtained from lidar, skip optimizing it
                if (it_per_id.lidar_depth_flag == true)
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    //// 信赖区域策略采用狗腿法
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;

    //// 如果边缘化旧帧，则最大求解时间设置少一些
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    double2vector();    // 数组转vec  优化完毕后double数组保存到vec中

    // Step3：marg部分
    //1.把之前存的残差部分加进来
    //2.把与当前要marg掉的帧的所有相关残差项加进来,IMU,odom,vision.
    //3.preMarginalize->调用Evaluate计算所有ResidualBlock的残差和雅克比,parameter_block_data是margniliazation中存参数块的容器
    //4.多线程构造Hx=b的结构,H是边缘化后的结果,First Estimate Jacobian,在X0处进行线性化,需要去看!!
    //5.marg结束,调整参数块在下一次window中对应的位置

    // 边缘化流程：
    // 1.创建marginalization_info，从头到位都是marginalization_info这个变量来进行统筹安排进行边缘化；
    //   然后通过marginalization_info->addResidualBlockInfo()来添加约束，有3个方面的来源：（1）旧的（2）imu预积分项（3）特征点
    // 2.首先通过last_marginalization_info 构建出marginalization_factor，这个factor就是先验的残差项；
    // 3.然后将滑窗内第0帧和第1帧间的 IMU 预积分值(pre_integrations[1])构建预积分因子，然后构建imu残差项；
    // 4.接着挑选出第一次观测帧为第 0 帧的路标点，和其他共视帧构建重投影因子，然后构建视觉残差项；
    // 5.以上4项通过marginalization_info->addResidualBlockInfo()来添加到边缘化约束中；
    // 6.调用preMarginalize->Evaluate计算每次IMU和视觉观测(cost_function)对应的参数块(parameter_blocks)，雅可比矩阵(jacobians)，残差值(residuals);
    // 7.多线程计算整个先验项的参数块，雅可比矩阵和 残差值;
    // 8.最后得到了优化项需要的两个变量：last_marginalization_info和last_marginalization_parameter_blocks；

    // 当次新帧为关键帧时,MARGIN_OLD,将 marg 掉最老帧,及其看到的路标点和相关联的 IMU 数据,将其转化为先验信息加到整体的目标函数中:
    // 1) 把上一次先验项中的残差项(尺寸为 n)传递给当前先验项,并从中去除需要丢弃的状态量;
    // 2) 将滑窗内第0帧和第1帧间的 IMU 预积分因子(pre_integrations[1])放到marginalization_info 中；
    // 3) 挑选出第一次观测帧为第0帧的路标点, 将对应的多组视觉观测放到marginalization_info 中；
    // 4) marginalization_info->preMarginalize():得到每次 IMU 和视觉观测(cost_function)对应的参数块(parameter_blocks),雅可比矩阵,残差值;
    // 5) marginalization_info->marginalize():多线程计整个先验项的参数块,雅可比矩阵和残差值；

    if (marginalization_flag == MARGIN_OLD)
    {
        //// 边缘化问题类marginalization_info
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        // 先验误差会一直保存，而不是只使用一次,如果上一次边缘化的信息存在
        // 要边缘化的参数块是 para_Pose[0] para_SpeedBias[0] 以及 para_Feature[feature_index](滑窗内的第feature_index个点的逆深度)
        //// 采用逆深度更加近似于高斯分布并且数值稳定性更好
        //// 如果有上一次的边缘化问题类
        if (last_marginalization_info)
        {
            //// 找到待边缘化参数块的索引并保存
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            //// 导入边缘化残差块信息
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            //// 导入imu残差块信息
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 因子定义的就是残差的计算方式，残差相对与优化变量的雅克比的计算方式；
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                // 残差块信息包含了要边缘化的帧与后一帧的残差信息，相关的参数块，以及要边缘化掉的变量块；
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                //// 同样只保留高质量特征
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    //// 导入重投影残差快信息（含时间偏移or不含）到marginalization_info
                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        //// 边缘化
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //// 构建下一轮非线性优化使用的保留变量参数块，地址前进一位，因为后续要滑动窗口
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        //// 更新last_marginalization_info和last_marginalization_parameter_blocks，准备下一轮非线性优化
        if (last_marginalization_info)
            delete last_marginalization_info;
        //// 更新last_marginalization_info
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    // 如果倒数第二帧不是关键帧
    // 当次新帧不是关键帧时,MARGIN_SECOND_NEW,我们将直接扔掉次新帧及它的视觉观测边,而不对次新帧进行 marg,
    // 因为我们认为当前帧和次新帧很相似,也就是说当前帧跟路标点之间的约束和次新帧与路标点的约束很接近,直接丢弃并不会造成整个约束关系丢失过多信息。
    // 但是值得注意的是,我们要保留次新帧的 IMU 数据,从而保证 IMU 预积分的连贯性。
    else
    {
        //// last_marginalization_info非空并且其中包含了次新帧位姿变量
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                //// 本轮只边缘化掉次新帧的pose而不边缘化次新帧V和bias
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    //// 正常的边缘化次新帧策略中，不可能出现次新帧[V,bias]优化变量
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                //// 导入边缘化残差块信息
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            //// 不导入imu残差块信息和视觉残差块信息，直接边缘化
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            //// 滑动窗口以后，在第WINDOW_SIZE - 1帧之前的帧，地址不变
            //// 在第WINDOW_SIZE - 1帧之后的帧，即第WINDOW_SIZE帧，会变成第WINDOW_SIZE - 1帧
            //// 按照这种方式构建addr_shift，从而挑选出保留的变量参数块地址last_marginalization_parameter_blocks
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            //// 更新last_marginalization_info和last_marginalization_parameter_blocks
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

            // 通过以上过程先验项就构造完成了,在对滑动窗口内的状态量进行优化时,
            // 把它与IMU残差项和视觉残差项放在一起优化,从而得到不丢失历史信息的最新状态估计的结果。
            
        }
    }
}

// 实际滑动窗口的地方，如果第二最新帧是关键帧的话，那么这个关键帧就会留在滑动窗口中，时间最长的一帧和其测量值就会被边缘化掉；
// 如果第二最新帧不是关键帧的话，则把这帧的视觉测量舍弃掉而保留IMU测量值在滑动窗口中这样的策略会保证系统的稀疏性
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                //// 把所有数组前移一位
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            //// 前移后，最新的一帧数据初值与前一帧相同
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            //// 更新最新的预积分值
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            //// 清空最新一帧的测量数据
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            //// 清空最旧的测量帧
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            //// 把窗口左边缘旧帧滑走
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            //// 次新帧合并最新帧
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            //// 更新最新的预积分，清空测量数据
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            //// 滑走次新帧
            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    //// 直接丢弃次新帧观测
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    //// 如果系统处于非线性优化状态，更新初始观测帧在最旧帧的特征点的深度估计
    //// 否则直接丢弃最旧帧观测
    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}