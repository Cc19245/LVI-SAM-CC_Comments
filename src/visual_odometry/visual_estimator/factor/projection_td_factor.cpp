#include "projection_td_factor.h"

Eigen::Matrix2d ProjectionTdFactor::sqrt_info;
double ProjectionTdFactor::sum_t;

ProjectionTdFactor::ProjectionTdFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j, 
                                       const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                       const double _td_i, const double _td_j, const double _row_i, const double _row_j) : 
                                       pts_i(_pts_i), pts_j(_pts_j), 
                                       td_i(_td_i), td_j(_td_j)
{
    velocity_i.x() = _velocity_i.x();
    velocity_i.y() = _velocity_i.y();
    velocity_i.z() = 0;
    velocity_j.x() = _velocity_j.x();
    velocity_j.y() = _velocity_j.y();
    velocity_j.z() = 0;
    row_i = _row_i - ROW / 2;
    row_j = _row_j - ROW / 2;

#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

bool ProjectionTdFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];

    double td = parameters[4][0];

    Eigen::Vector3d pts_i_td, pts_j_td;
    //// 由于所有的pose都是以IMU为原点的，所以在计算重投影误差时，应该相机-IMU-IMU-相机
    // 根据时间戳采imu数据时，已经给图像时间戳进行了偏移补偿；
    // 所以此处加入时间偏移的增量即可，用来估计更准确的时间偏移；
    // 如果估计的时间偏移已经很准确，即imu与图像时间戳已经对齐，则此处增量应该为0；
    // 用传入的变量计算残差，同时更新雅克比矩阵，然后根据残差减小的梯度方向调整传入的待优化变量；(直至收敛)
    pts_i_td = pts_i - (td - td_i + TR / ROW * row_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j + TR / ROW * row_j) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;    //将i时刻特征点从归一化图像平面转换成3D坐标(观测的特征点存储在归一化平面)；逆深度
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;   //将i时刻特征点从相机坐标系转到imu坐标系；
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;            //将i时刻特征点从imu坐标系转到世界坐标系；
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);//将i时刻特征点从世界坐标系转到j时刻imu坐标系；
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);//将i时刻特征点从j时刻imu坐标系转到j时刻相机坐标系；
    Eigen::Map<Eigen::Vector2d> residual(residuals);        //i,j点都在j时刻相机坐标系下，准备计算重投影误差；

#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    // 准备除以深度转化到归一化平面，求重投影误差；
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif

    // 做优化时给残差除以协方差(乘以信息矩阵)，是为了将误差归一化；
    // 比如视觉，认为以当前预测的位姿投影的特征点和观测到的特征点应该离得很近，只不过带有误差，这个误差就是当前估计的协方差；
    // 因为当前估计值理论上应该符合以当前测量值为均值，当前估计的Sigma为协方差的正态分布；
    // 减去当前测量值后，则符合零均值，Sigma协方差的正态分布；
    // 然后除以协方差后，则符合零均值，协方差为1的正态分布；
    //// IMU有协方差的递推（传播）模型，但是视觉没有，因此视觉的协方差是认为它有1.5个像素误差
    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce;

        // 雅克比矩阵：此处计算的重投影误差对传入的各个优化变量的导数，雅克比矩阵地址块与传入参数是一一对应的；

        if (jacobians[0])  // 重投影误差对i时刻位姿的导数；
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])  // 重投影误差对j时刻位姿的导数；
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])  // 重投影误差对相机与IMU外参的导数；
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])  // 重投影误差对特征点逆深度的导数；
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
        }
        if (jacobians[4])  // 时间偏移td对重投影误差的导数，时间标定论文的公式6对td求导；
        {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);
            jacobian_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0  +
                          sqrt_info * velocity_j.head(2);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}