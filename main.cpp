
#include "FastICP.h"
#include <opencv2/opencv.hpp>
#include <opencv2/flann/flann.hpp>
#include "clock.h"
#include "nanoflann.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <omp.h>

#include <iostream>              //标准C++库中的输入输出的头文件
#include <pcl/io/pcd_io.h>       //PCD读写类相关的头文件
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

using namespace std;
//using namespace FastICP;



int main()
{

	int numIterations = 100;
	float threshold = 0.01;
	float maxDistance = 40;
	bool picky = true;
	bool point2PlaneMetric = true;
	bool incremental = true;
	
	glm::mat4x4 matrix;
	

	// 声明需要用到的点云（读入的，转换的，ICP调整的三个点云）
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc1(new pcl::PointCloud<pcl::PointXYZ>);  // Original point cloud  
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst1(new pcl::PointCloud<pcl::PointXYZ>); // Transformed point cloud  
	//pcl::PointCloud<pcl::PointXYZ>::Ptr normalDst1(new pcl::PointCloud<pcl::PointXYZ>); // ICP output point cloud  

	pcl::io::loadPCDFile<pcl::PointXYZ>("球面.pcd", *cloudSrc1);
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setLeafSize(1, 1, 1);
	voxel_grid.setInputCloud(cloudSrc1);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid.filter(*cloud_src);
	std::cout << "down size *cloudSrc1 from " << cloudSrc1->size() << "to" << cloud_src->size() << endl;
	int numSrc = cloud_src->size();
	glm::vec3 *cloudSrc = new glm::vec3[numSrc];
    for (int i = 0; i < cloud_src->size(); i++)
	{
		cloudSrc[i] = glm::vec3(cloud_src->points[i].x, cloud_src->points[i].y, cloud_src->points[i].z);
	}
	const glm::vec3 *cloudSrc0 = cloudSrc;

	pcl::io::loadPCDFile<pcl::PointXYZ>("球面1.pcd", *cloudDst1);
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_2;
	voxel_grid_2.setLeafSize(1, 1, 1);
	voxel_grid_2.setInputCloud(cloudDst1);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_dst(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid_2.filter(*cloud_dst);
	std::cout << "down size *cloudDst1 from " << cloudDst1->size() << "to" << cloud_dst->size() << endl;
	int numDst = cloud_dst->size();
	glm::vec3 *cloudDst = new glm::vec3[numDst];
    for (int i = 0; i < cloud_dst->points.size(); i++)
	{
		cloudDst[i] = glm::vec3(cloud_dst->points[i].x, cloud_dst->points[i].y, cloud_dst->points[i].z);
	}
	const glm::vec3 *cloudDst0 = cloudDst;

	int numNormal = cloud_dst->size();
	glm::vec3 *normalDst = new glm::vec3[numNormal];
	const float radius = 2;
	FastICP::computeNormals(numNormal, cloudDst0, normalDst, radius);
	
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	vector<float> Weights(cloud_dst->size());
	int number = 1;
	const double PI = 3.1415;
	double ab, a1, b1, cosr;

	for (int i = 0; i < cloud_dst->size(); i++)
	{
		a1 = sqrt(normalDst[i].x * normalDst[i].x + normalDst[i].y * normalDst[i].y + normalDst[i].z* normalDst[i].z);
		for (int j = 0; j < cloud_dst->size(); j++)
		{
			ab = normalDst[i].x * normalDst[j].x + normalDst[i].y *normalDst[j].y + normalDst[i].z * normalDst[j].z;
			b1 = sqrt(normalDst[j].x *normalDst[j].x + normalDst[j].y * normalDst[j].y + normalDst[j].z*normalDst[j].z);
			cosr = ab / a1 / b1;
			if (acos(cosr) == 0)//<*PI)
				number = number + 1;
		}
		Weights[i] = number;

		number = 1;
	}

	for (int i = 0; i < cloud_dst->size(); i++)
	{
		//Weights[i] = cloud_dst->size() / Weights[i];
		if (Weights[i] == 0)
			Weights[i] = 0;
		else
		{
			Weights[i] = 1 / Weights[i];
		}

	}
	double begin = clock();
	FastICP::ICP(Weights, matrix, numSrc, cloudSrc0, numDst, cloudDst0, normalDst, numIterations, threshold, maxDistance, picky, point2PlaneMetric, incremental);
	double end = clock();
	cout << "total time: " << (double)(end - begin) / (double)CLOCKS_PER_SEC << " s" << endl;

	return (0);
}