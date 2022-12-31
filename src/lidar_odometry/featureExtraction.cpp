#include "utility.h"
#include "lvi_sam/cloud_info.h"

struct smoothness_t
{
    float value; //// 曲率值
    size_t ind;  //// 点序号
};

struct by_value
{
    //// 仿函数用来做比较器
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    {
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:
    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lvi_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    FeatureExtraction()
    {
        //// 订阅去畸变、有序化的自定义格式点云cloudInfo，回调函数负责去除干扰点，根据曲率提取角点和平面点
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/deskew/cloud_info",
                                                              5, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        //// 发布注册了角点点云和面点点云，并且“瘦身”之后的cloudInfo
        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/feature/cloud_info", 5);
        //// 发布角点点云和面点点云
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_corner", 5);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_surface", 5);

        //// 重置所有参数，设置体素化滤波器分辨率
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN * Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
        cloudLabel = new int[N_SCAN * Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr &msgIn)
    {
        cloudInfo = *msgIn;                                      // new cloud info
        cloudHeader = msgIn->header;                             // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            //// 依然采用距离来计算曲率，对之前投影成图片时计算过的距离再次利用，省去了对坐标的计算
            float diffRange = cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4] + cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2] + cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10 + cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2] + cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4] + cloudInfo.pointRange[i + 5];

            cloudCurvature[i] = diffRange * diffRange; //diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i + 1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));

            //// 相近点：在有序点云中顺序相邻，并且在距离图像上的列序号之差小于10
            if (columnDiff < 10)
            {
                // 10 pixel diff in range image
                //// 两个相近的点与LiDAR的距离之差大于0.3m，则认为它们是互相遮挡的点
                //// 标记后景点，防止把遮挡点当做角点
                if (depth1 - depth2 > 0.3)
                {
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }
                else if (depth2 - depth1 > 0.3)
                {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            //// 理论上物体距离越远时，点云才应该越稀疏，相邻点距离较远
            //// 但是对于平行于激光光束的表面来说，即使距离激光雷达很近，依然会导致两个相邻点很远
            //// 所以当相邻点距离之差大于0.02倍距离时，认为它是平行于激光光束的平面，排除掉该点
            float diff1 = std::abs(float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {

                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20)
                        {
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++)
                        {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();         //// clear仅改变size
        cloudInfo.startRingIndex.shrink_to_fit(); //// 释放数组的capacity
        cloudInfo.endRingIndex.clear();
        cloudInfo.endRingIndex.shrink_to_fit();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointColInd.shrink_to_fit();
        cloudInfo.pointRange.clear();
        cloudInfo.pointRange.shrink_to_fit();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner = publishCloud(
            &pubCornerPoints, cornerCloud, cloudHeader.stamp, "base_link");
        cloudInfo.cloud_surface = publishCloud(
            &pubSurfacePoints, surfaceCloud, cloudHeader.stamp, "base_link");
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar");

    //// 与imageProjection相似，核心工作都由FeatureExtraction的构造函数完成
    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Lidar Feature Extraction Started.\033[0m");

    ros::spin();

    return 0;
}