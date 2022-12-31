#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    //// 形参(代价函数，损失函数，参数块，边缘化的变量id)
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}
    //// 残差计算函数
    void Evaluate();
    //// 代价函数 损失函数
    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    //// 这个约束包含的所有优化变量
    std::vector<double *> parameter_blocks;
    //// 待边缘化变量的索引
    std::vector<int> drop_set;
    //// 雅可比矩阵
    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    //// 残差
    Eigen::VectorXd residuals;

    //// 位姿变量，7维6个自由度（四元数3个自由度）
    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    //// 位姿变量，7维6个自由度（四元数3个自由度）
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    //// 建立容器parameter_block_data
    void preMarginalize();
    //// 多线程计算H和b，并分解得到J和e
    void marginalize();
    //// 建立保留变量的相关容器keep_block_size;keep_block_idx;keep_block_data;keep_block_addr;
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    //// 所有残差块信息
    std::vector<ResidualBlockInfo *> factors;
    //// 与边缘化变量相关的优化变量的维度n，将要边缘化的变量维度m
    int m, n;
    //// 所有优化变量的内存地址和size的键值对（global size维度size）
    std::unordered_map<long, int> parameter_block_size; //global size
    int sum_block_size;
    //// 边缘化与和边缘化相关的优化变量的内存地址和索引的键值对（local size自由度size）
    std::unordered_map<long, int> parameter_block_idx; //local size
    //// 所有优化变量的内存地址和值的键值对
    std::unordered_map<long, double *> parameter_block_data;

    //// 要保留的变量的size，idx，data
    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;
    //// 边缘化之后构造出来的雅可比J和残差e
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    //// 按照MarginalizationInfo的J和e，更新x和e的值，拷贝jacobian的值
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    //// 复合一个MarginalizationInfo
    MarginalizationInfo* marginalization_info;
};
