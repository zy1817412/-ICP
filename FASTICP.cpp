#include "FastICP.h"
#include <opencv2/opencv.hpp>
#include <opencv2/flann/flann.hpp>
#include "clock.h"
#include "nanoflann.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <omp.h>


using  namespace std;
using  namespace cv;

namespace FastICP
{
	struct PointCloud
	{
		size_t numPoints = 0;//size_t 类型表示C中任何对象所能达到的最大长度，它是无符号整数。
		glm::vec3 *points = nullptr;

		// Must return the number of data points
		inline size_t kdtree_get_point_count() const//建议编译器将指定的函数体插入并取代每一处调用该函数的地方（上下文），从而节省了每次调用函数带来的额外时间开支。
		{
			return numPoints;
		}

		// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
		inline float kdtree_distance(const float *point, const size_t index, size_t) const
		{
			
			//return add1.m128_f32[0];
			//const glm::vec3 temp = glm::vec3(point[0], point[1], point[2]) - points[index];
			//return glm::dot(temp, temp);
			const float d0 = point[0] - points[index].x;
			const float d1 = point[1] - points[index].y;
			const float d2 = point[2] - points[index].z;
			return d0 * d0 + d1 * d1 + d2 * d2;
		}

		// Returns the dim'th component of the idx'th point in the class:
		// Since this is inlined and the "dim" argument is typically an immediate value, the
		//  "if/else's" are actually solved at compile time.
		inline float kdtree_get_pt(const size_t index, int dim) const
		{
			if (dim == 0)
				return points[index].x;
			else if (dim == 1)
				return points[index].y;
			else
				return points[index].z;
		}

		template <class BBOX>
		bool kdtree_get_bbox(BBOX &) const
		{
			return false;
		}
	};

	void ICP(std::vector<float> &Weights, glm::mat4 &matrix, const int &numSrc, const glm::vec3 *cloudSrc, const int &numDst, const glm::vec3 *cloudDst, const glm::vec3 *normalDst, const int &numIterations, const float &threshold, const float &maxDistance, const bool &picky, const bool &point2PlaneMetric, const bool &incremental)
	{
		sapClock clock;//日期时间函数
		matrix = glm::mat4();
		//Eigen::Matrix4f  matrix = Eigen::Matrix4f::Identity();
		if (numSrc < 6 || numDst < 1)
			return;

		bool scaling = true; //顶点
	//	bool picky = false;挑剔的点
		glm::vec3 *cloudSrcMoved = new glm::vec3[numSrc];
		glm::vec3 *cloudSrcScaled = new glm::vec3[numSrc];
		PointCloud pointCloudDstScaled;
		pointCloudDstScaled.numPoints = numDst;
		pointCloudDstScaled.points = new glm::vec3[numDst];
		glm::vec3 *cloudDstScaled = pointCloudDstScaled.points;

		size_t *indices = new size_t[numSrc];
		float *distances = new float[numSrc];
		glm::vec3 meanAverage;
		float scale = 1.0f;
		clock.begin();
		if (scaling)//顶点
		{
			//Hartley-Zissermann Scaling:
			glm::vec3 meanSrc;
			for (int i = 0; i < numSrc; ++i)
				meanSrc += cloudSrc[i];
			meanSrc /= numSrc;
			glm::vec3 meanDst;
			for (int i = 0; i < numDst; ++i)
				meanDst += cloudDst[i];
			meanDst /= numDst;
			meanAverage = (meanSrc + meanDst) * 0.5f;

			for (int i = 0; i < numSrc; ++i)
				cloudSrcScaled[i] = cloudSrc[i] - meanAverage;
			for (int i = 0; i < numDst; ++i)
				cloudDstScaled[i] = cloudDst[i] - meanAverage;

			//compute average dist from origin
			float averageDistance = 0.0f;
			for (int i = 0; i < numSrc; ++i)
				averageDistance += glm::length(cloudSrcScaled[i]);
			for (int i = 0; i < numDst; ++i)
				averageDistance += glm::length(cloudDstScaled[i]);
			averageDistance *= 0.5f;

			//scale to unit sphere
			scale = numSrc / averageDistance;
			for (int i = 0; i < numSrc; ++i)
				cloudSrcScaled[i] *= scale;
			for (int i = 0; i < numDst; ++i)
				cloudDstScaled[i] *= scale ;// *Weights[i];
		}
		else
		{
			for (int i = 0; i < numSrc; ++i)
				cloudSrcScaled[i] = cloudSrc[i];
			for (int i = 0; i < numDst; ++i)
				cloudDstScaled[i] = cloudDst[i];
		}
		for (int i = 0; i < numSrc; ++i)
			cloudSrcMoved[i] = cloudSrcScaled[i];
		clock.end();
		//printf("scale：%.2fms\n", clock.getInterval());

		clock.begin();
		
		nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> index(3, pointCloudDstScaled, nanoflann::KDTreeSingleIndexAdaptorParams(15));
		index.buildIndex();

		std::vector<int> matchIndicesSrc(numSrc);
		std::vector<int> matchIndicesDst(numSrc);
		std::vector<float> minDistances(numDst);
		std::vector<size_t> minIndices(numDst);//指标
		for (int i = 0; i < numSrc; ++i)
			matchIndicesSrc[i] = i;

		//Distance error in last iteration最后一次迭代的距离错误
		float previousError = FLT_MAX;

		//Change in distance error between two iterations两次迭代之间距离误差的变化
		float errorRatio = 0.0f;

		//we will keep track of the pose with minimum residual我们将跟踪的姿态与最小剩余
		float minError = FLT_MAX;

		int iterations = 0;
		clock.end();
		printf("build kdtree：%.2fms\n", clock.getInterval()); //时间间隔

		//Start main loop of ICP 开始ICP的主循环 配准
		while (!(errorRatio < 1.0f + threshold && errorRatio > 1.0f - threshold) && iterations < numIterations)
		{
			clock.begin();
			
#pragma omp parallel for schedule (dynamic, 1)//（动态）结果不对了，indexMatrix和distanceMatrix不是局部的，还是不对(indexMatrix用的现有内存)
			for (int i = 0; i < numSrc; ++i)
			{
				
				index.knnSearch((float *)(cloudSrcMoved + i), 1, indices + i, distances + i, -1);
			}
			clock.end();
			printf("search：%.2fms\n", clock.getInterval());
			

			float scaledMaxDistance = maxDistance;
			if (scaling)
				scaledMaxDistance *= scale;
			int count = 0;
			for (int i = 0; i < numSrc; ++i)
			{
				int index = indices[i];
			
				if (glm::distance(cloudSrcMoved[i], cloudDstScaled[index]) < scaledMaxDistance && glm::dot(normalDst[index], normalDst[index]) > 0.0f)
				{
					matchIndicesSrc[count] = i;
					matchIndicesDst[count] = indices[i];
				
					++count;
				}
			}
		

			if (count < 6)
				break;

			glm::mat4 transformMatrix;
			if (picky)
			{
				clock.begin();
				for (int i = 0; i < minDistances.size(); ++i)
					minDistances[i] = FLT_MAX;
				for (int i = 0; i < count; ++i)
				{
					int indexDst = matchIndicesDst[i];
					float distance = glm::distance(cloudSrcMoved[matchIndicesSrc[i]], cloudDstScaled[indexDst]);
					if (distance < minDistances[indexDst])
					{
						minDistances[indexDst] = distance;
						minIndices[indexDst] = matchIndicesSrc[i];
					}
				}
				count = 0;
				for (int i = 0; i < minDistances.size(); ++i)
				{
					if (minDistances[i] < FLT_MAX)
					{
						matchIndicesSrc[count] = minIndices[i];
						matchIndicesDst[count] = i;
						++count;
					}
				}
				clock.end();
				//printf("picky：%.2fms\n", clock.getInterval());
			}

			if (count < 6)
				break;

			clock.end();
			if (point2PlaneMetric)
			{
				if (incremental)//增量
					transformMatrix = minimizePoint2Plane(Weights, count, cloudSrcMoved, cloudDstScaled, normalDst, matchIndicesSrc, matchIndicesDst);
				else
					transformMatrix = minimizePoint2Plane(Weights, count, cloudSrcScaled, cloudDstScaled, normalDst, matchIndicesSrc, matchIndicesDst);
			}
			else
			{
				if (incremental)
					transformMatrix = minimizePoint2Point( count, cloudSrcMoved, cloudDstScaled, matchIndicesSrc, matchIndicesDst);//
				else
					transformMatrix = minimizePoint2Point( count, cloudSrcScaled, cloudDstScaled, matchIndicesSrc, matchIndicesDst);
			}
			clock.end();
			printf("minimize：%.2fms\n", clock.getInterval());

			if (incremental)
				matrix = transformMatrix * matrix;
			else
				matrix = transformMatrix;
			for (int i = 0; i < numSrc; ++i)
				cloudSrcMoved[i] = glm::vec3(matrix * glm::vec4(cloudSrcScaled[i], 1.0f));

			float error = 0.0f;
			for (int i = 0; i < numSrc; ++i)
			{
				float temp = glm::distance(cloudSrcMoved[i], cloudSrcScaled[i]);
				error += temp * temp;
			}
			error = sqrtf(error / numSrc);

			float error2 = 0.0f;
			for (int i = 0; i < count; ++i)
			{
				float temp = glm::distance(cloudSrcMoved[matchIndicesSrc[i]], cloudDstScaled[matchIndicesDst[i]]);
				error2 += temp * temp;
			}
			error2 = sqrtf(error2 / numSrc);

			errorRatio = error / previousError;
			previousError = error;
			if (error < minError)
				minError = error;

			printf("iter %d error %f errorRatio %f\n", iterations, error2 / scale, errorRatio);

			++iterations;
		}

		if (scaling)
		{
			glm::vec3 temp = glm::mat3(matrix) * meanAverage;
			matrix[3][0] /= scale;
			matrix[3][0] += (meanAverage.x - temp.x);
			matrix[3][1] /= scale;
			matrix[3][1] += (meanAverage.y - temp.y);
			matrix[3][2] /= scale;
			matrix[3][2] += (meanAverage.z - temp.z);
		}

		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
				printf("%f ", matrix[j][i]);
			printf("\n");
		}

		delete[]cloudSrcMoved;
		delete[]cloudSrcScaled;
		delete[]cloudDstScaled;
		delete[]indices;
		delete[]distances;
	}

	glm::mat4 projectedICP(const std::vector<glm::vec3> &cloudSrc, const glm::mat4 &projectionMatrix, const int &widthDst, const int &heightDst, const glm::vec3 *cloudDst, const glm::vec3 *normalsDst, const int &numIterations, const float &threshold, const float &maxDistance)
	{
		sapClock clock;
		glm::mat4 matrix;

		if (cloudSrc.size() < 6)
			return matrix;

		//Distance error in last iteration
		float previousError = FLT_MAX;

		//Change in distance error between two iterations
		float errorRatio = 0.0f;

		//we will keep track of the pose with minimum residual
		float minError = FLT_MAX;

		int iterations = 0;

		float distance;
		std::vector<glm::vec3> cloudSrcMoved = cloudSrc;
		glm::vec4 projectedPoint;
		glm::vec3 pointSrc;
		glm::vec3 pointDst;
		glm::vec3 normalDst;
		std::vector<glm::vec3> matchedPointsSrc(cloudSrc.size());
		std::vector<glm::vec3> matchedPointsDst(cloudSrc.size());
		std::vector<glm::vec3> matchedNormalsDst(cloudSrc.size());
		int numMatched = 0;
		while (!(errorRatio < 1.0f + threshold && errorRatio > 1.0f - threshold) && iterations < numIterations)
		{
			numMatched = 0;
			clock.begin();
			for (int i = 0; i < cloudSrcMoved.size(); ++i)
			{
				pointSrc = cloudSrcMoved[i];
				projectedPoint = projectionMatrix * glm::vec4(pointSrc, 1.0f);
				projectedPoint /= projectedPoint.z;
				int x = glm::round(projectedPoint.x);
				int y = glm::round(projectedPoint.y);
				int index = y * widthDst + x;
				if (x >= 0 && y >= 0 && x < widthDst && y < heightDst)
				{
					pointDst = cloudDst[index];
					if (pointDst.z > 0.0f)
					{
						normalDst = normalsDst[index];
						distance = glm::distance(pointSrc, pointDst);
						if (distance < maxDistance && glm::dot(normalDst, normalDst) > 0.0f)
						{
							matchedPointsSrc[numMatched] = pointSrc;
							matchedPointsDst[numMatched] = pointDst;
							matchedNormalsDst[numMatched] = normalDst;
							++numMatched;
						}
					}
				}
			}
			clock.end();
			printf("project：%.2fms\n", clock.getInterval());

			if (numMatched < 6)
				break;

			clock.begin();
			glm::mat4 transformMatrix = minimizePoint2Plane(numMatched, &matchedPointsSrc[0],
				&matchedPointsDst[0], &matchedNormalsDst[0]);
			clock.end();
			printf("minimize：%.2fms\n", clock.getInterval());

			matrix = transformMatrix * matrix;
			for (int i = 0; i < cloudSrc.size(); ++i)
				cloudSrcMoved[i] = glm::vec3(matrix * glm::vec4(cloudSrc[i], 1.0f));

			float error = 0.0f;
			for (int i = 0; i < cloudSrc.size(); ++i)
			{
				float temp = glm::distance(cloudSrcMoved[i], cloudSrc[i]);
				error += temp * temp;
			}
			error = sqrtf(error / cloudSrc.size());

			errorRatio = error / previousError;
			previousError = error;
			if (error < minError)
				minError = error;

			printf("iter %d errorRatio %f\n", iterations, errorRatio);

			++iterations;
		}

		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
				printf("%f ", matrix[j][i]);
			printf("\n");
		}

		return matrix;
	}

	glm::mat4 minimizePoint2Plane(const int &numMatched, const glm::vec3 *cloudSrc, const glm::vec3 *cloudDst, const glm::vec3 *normalDst)
	{
		glm::mat4 transformMatrix;
		if (numMatched > 5)
		{
		
			cv::Mat matA(numMatched, 6, CV_32FC1);
			cv::Mat vectorX(6, 1, CV_32FC1);
			cv::Mat vectorB(numMatched, 1, CV_32FC1);
			for (int i = 0; i < numMatched; ++i)
			{
				//vectorB(i, 0) = glm::dot(cloudDst[i] - cloudSrc[i], normalDst[i]);
				vectorB.at<float>(i, 0) = glm::dot(cloudDst[i] - cloudSrc[i], normalDst[i]);
				glm::vec3 crossed = glm::cross(cloudSrc[i], normalDst[i]);
				matA.at<float>(i, 0) = crossed.x;
				matA.at<float>(i, 1) = crossed.y;
				matA.at<float>(i, 2) = crossed.z;
				matA.at<float>(i, 3) = normalDst[i].x;
				matA.at<float>(i, 4) = normalDst[i].y;
				matA.at<float>(i, 5) = normalDst[i].z;
			}
			//vectorX = matA.householderQr().solve(vectorB);
			cv::solve(matA, vectorB, vectorX, cv::DECOMP_NORMAL | cv::DECOMP_CHOLESKY);
			glm::mat3 Rx;
			Rx[1][1] = Rx[2][2] = cos(vectorX.at<float>(0, 0));
			Rx[1][2] = sin(vectorX.at<float>(0, 0));
			Rx[2][1] = -sin(vectorX.at<float>(0, 0));
			glm::mat3 Ry;
			Ry[0][0] = Ry[2][2] = cos(vectorX.at<float>(1, 0));
			Ry[2][0] = sin(vectorX.at<float>(1, 0));
			Ry[0][2] = -sin(vectorX.at<float>(1, 0));
			glm::mat3 Rz;
			Rz[0][0] = Rz[1][1] = cos(vectorX.at<float>(2, 0));
			Rz[1][0] = -sin(vectorX.at<float>(2, 0));
			Rz[0][1] = sin(vectorX.at<float>(2, 0));
			transformMatrix = glm::mat4(Rx * Ry * Rz);
			transformMatrix[3][0] = vectorX.at<float>(3, 0);
			transformMatrix[3][1] = vectorX.at<float>(4, 0);
			transformMatrix[3][2] = vectorX.at<float>(5, 0);
		}
		return transformMatrix;
	}

	glm::mat4 minimizePoint2Plane(std::vector<float> &Weights, const int &numMatched, const glm::vec3 *cloudSrc, const glm::vec3 *cloudDst, const glm::vec3 *normalDst, const std::vector<int> &indicesSrc, const std::vector<int> &indicesDst)
	{
		glm::mat4 transformMatrix;
		if (numMatched > 5)
		{
		
			cv::Mat matA(numMatched, 6, CV_32FC1);
			cv::Mat vectorX(6, 1, CV_32FC1);
			cv::Mat vectorB(numMatched, 1, CV_32FC1);
			for (int i = 0; i < numMatched; ++i)
			{
				int indexSrc = indicesSrc[i];
				int indexDst = indicesDst[i];
				//vectorB(i, 0) = glm::dot(cloudDst[indexDst] - cloudSrc[indexSrc], normalDst[indexDst]);
				vectorB.at<float>(i, 0) = glm::dot(cloudDst[indexDst] - cloudSrc[indexSrc], normalDst[indexDst] * sqrt(Weights[i]));
				glm::vec3 crossed = glm::cross(cloudSrc[indexSrc], normalDst[indexDst]);
				matA.at<float>(i, 0) = crossed.x * sqrt(Weights[i]);
				matA.at<float>(i, 1) = crossed.y * sqrt(Weights[i]);
				matA.at<float>(i, 2) = crossed.z * sqrt(Weights[i]);
				matA.at<float>(i, 3) = normalDst[indexDst].x * sqrt(Weights[i]);
				matA.at<float>(i, 4) = normalDst[indexDst].y * sqrt(Weights[i]);
				matA.at<float>(i, 5) = normalDst[indexDst].z * sqrt(Weights[i]);
			}
			//vectorX = matA.householderQr().solve(vectorB);
			cv::solve(matA, vectorB, vectorX, cv::DECOMP_NORMAL | cv::DECOMP_CHOLESKY);
			glm::mat3 Rx;
			Rx[1][1] = Rx[2][2] = cos(vectorX.at<float>(0, 0));
			Rx[1][2] = sin(vectorX.at<float>(0, 0));
			Rx[2][1] = -sin(vectorX.at<float>(0, 0));
			glm::mat3 Ry;
			Ry[0][0] = Ry[2][2] = cos(vectorX.at<float>(1, 0));
			Ry[2][0] = sin(vectorX.at<float>(1, 0));
			Ry[0][2] = -sin(vectorX.at<float>(1, 0));
			glm::mat3 Rz;
			Rz[0][0] = Rz[1][1] = cos(vectorX.at<float>(2, 0));
			Rz[1][0] = -sin(vectorX.at<float>(2, 0));
			Rz[0][1] = sin(vectorX.at<float>(2, 0));
			transformMatrix = glm::mat4(Rx * Ry * Rz);
			//transformMatrix[3][0] = vectorX(3, 0);
			//transformMatrix[3][1] = vectorX(4, 0);
			//transformMatrix[3][2] = vectorX(5, 0);
			transformMatrix[3][0] = vectorX.at<float>(3, 0);
			transformMatrix[3][1] = vectorX.at<float>(4, 0);
			transformMatrix[3][2] = vectorX.at<float>(5, 0);
		}
		return transformMatrix;
	}

	//计算单点的法线
	//参数：点集points，邻域索引indices
	//返回值：法线
	glm::vec3 computeNormal(const glm::vec3 *&points, const std::vector<std::pair<size_t, float>> &indices)
	{
		int numParameters = 3;
		glm::vec3 normal  (0.0f,0.0f,0.0f) ;
		if (indices.size() >= numParameters)
		{
			glm::vec3 center ;
			for (int i = 0; i < indices.size(); ++i)
				center += points[indices[i].first];
			center /= indices.size();
			cv::Mat matCovariance(indices.size(), 3, CV_32FC1);
			for (int i = 0; i < indices.size(); ++i)
			{
				matCovariance.at<float>(i, 0) = points[indices[i].first].x - center.x;
				matCovariance.at<float>(i, 1) = points[indices[i].first].y - center.y;
				matCovariance.at<float>(i, 2) = points[indices[i].first].z - center.z;
			}
			matCovariance = matCovariance.t() * matCovariance;
			cv::Mat matEigenValue;
			cv::Mat matEigenVector;
			cv::eigen(matCovariance, matEigenValue, matEigenVector);
			normal.x = matEigenVector.at<float>(2, 0);
			normal.y = matEigenVector.at<float>(2, 1);
			normal.z = matEigenVector.at<float>(2, 2);
		}

		return glm::normalize(normal);//正常化
	}

	//计算单点的曲率
	//参数：点集points，邻域索引indices
	//返回值：曲率
	float computeCurvature(const glm::vec3 *&points, const std::vector<std::pair<size_t, float>> &indices)
	{
		//返回float &会所有都是0
		int numParameters = 6;
		if (indices.size() >= numParameters)
		{
			glm::vec3 normal = computeNormal(points, indices);
			glm::mat3 rotateMatrix = glm::mat3(glm::rotate(glm::mat4(), glm::degrees(acos(glm::dot(normal, glm::vec3(0.0f, 0.0f, -1.0f)))), glm::cross(normal, glm::vec3(0.0f, 0.0f, -1.0f))));
			//float效果不行，速度还没提高，SVD也不行
			cv::Mat matA(indices.size(), numParameters, CV_64FC1);
			cv::Mat vectorX(numParameters, 1, CV_64FC1);
			cv::Mat vectorB(indices.size(), 1, CV_64FC1);
			for (int i = 0; i < indices.size(); ++i)
			{
				const glm::vec3 &point = rotateMatrix * points[indices[i].first];
				matA.at<double>(i, 0) = point.x * point.x;
				matA.at<double>(i, 1) = point.x * point.y;
				matA.at<double>(i, 2) = point.y * point.y;
				matA.at<double>(i, 3) = point.x;
				matA.at<double>(i, 4) = point.y;
				matA.at<double>(i, 5) = 1.0f;
				vectorB.at<double>(i, 0) = -point.z;
			}
			//NORMAL|SVD结果也不太对，但是边缘更完整，而且对不同方向适应性较强
			cv::solve(matA, vectorB, vectorX, cv::DECOMP_NORMAL | cv::DECOMP_LU);
			return -vectorX.at<double>(0, 0) - vectorX.at<double>(2, 0);
		}
		else
			return 0.0f;
	}

	int computeCurvatures(const int &numPoints, const glm::vec3 *points, float *&curvatures, const float &radius)
	{
		if (numPoints <= 6)
			return -1;

		PointCloud cloud;
		cloud.numPoints = numPoints;
		cloud.points = new glm::vec3[numPoints];
		for (int i = 0; i < numPoints; ++i)
			cloud.points[i] = points[i];
		int numThreads = omp_get_max_threads();
		std::vector<std::vector<std::pair<size_t, float>>> indices(numThreads);
		for (int i = 0; i < indices.size(); ++i)
			indices[i].resize(numPoints);

		nanoflann::SearchParams param;
		param.sorted = false;
		nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(100));
		tree.buildIndex();
		curvatures = new float[numPoints];
		int numParameters = 6;

		sapClock clock;
		clock.begin();
#pragma omp parallel for schedule (dynamic, 1)
		for (int i = 0; i < numPoints; ++i)
		{
			const int &threadIndex = omp_get_thread_num();
			tree.radiusSearch((float *)(cloud.points + i), radius * radius, indices[threadIndex], param);
			curvatures[i] = computeCurvature(points, indices[threadIndex]);
		}
		clock.end();
		printf("curvature：%.2fms\n", clock.getInterval());

		delete[]cloud.points;

		return 0;
	}

	int computeNormals(const int &numPoints, const glm::vec3 *points, glm::vec3 *normals, const float &radius)
	{
		if (numPoints <= 3)
			return -1;

		PointCloud cloud;
		cloud.numPoints = numPoints;
		cloud.points = new glm::vec3[numPoints];
		for (int i = 0; i < numPoints; ++i)
			cloud.points[i] = points[i];
		int numThreads = omp_get_max_threads();
		std::vector<std::vector<std::pair<size_t, float>>> indices(numThreads);
		for (int i = 0; i < indices.size(); ++i)
			indices[i].resize(numPoints);

		nanoflann::SearchParams param;
		param.sorted = false;
		nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(100));
		tree.buildIndex();

		sapClock clock;
		clock.begin();
#pragma omp parallel for schedule (dynamic, 1)
		for (int i = 0; i < numPoints; ++i)
		{
			const int &threadIndex = omp_get_thread_num();
			tree.radiusSearch((float *)(cloud.points + i), radius * radius, indices[threadIndex], param);
			normals[i] = computeNormal(points, indices[threadIndex]);
		}
		clock.end();
		printf("normal：%.2fms\n", clock.getInterval());

		delete[]cloud.points;

		return 0;
	}

	bool compareFunction(const glm::vec4 &v1, const glm::vec4 &v2)
	{
		return v1.w < v2.w;
	}

	int uniformDownsample(const int &numPoints, glm::vec3 *points, glm::vec3 *normals, const float &distance, const float &preserveCurvature, const bool &preserveEdge)
	{
		if (numPoints <= 1)
			return numPoints;

		float *curvatures = 0;
		if (preserveCurvature > 0.0f)
		{
			if (numPoints <= 6)
				return numPoints;
			computeCurvatures(numPoints, points, curvatures, distance);
			std::vector<glm::vec4> pointsCurvatures(numPoints);
			for (int i = 0; i < numPoints; ++i)
				pointsCurvatures[i] = glm::vec4(points[i], curvatures[i]);
			std::sort(pointsCurvatures.begin(), pointsCurvatures.end(), compareFunction);
			for (int i = 0; i < numPoints; ++i)
			{
				points[i] = glm::vec3(pointsCurvatures[i]);
				curvatures[i] = pointsCurvatures[i].w;
			}
		}

		PointCloud cloud;
		cloud.numPoints = numPoints;
		cloud.points = new glm::vec3[numPoints];
		for (int i = 0; i < numPoints; ++i)
			cloud.points[i] = points[i];
		std::vector<unsigned char> markers(numPoints, 255);
		//不用reserve而用resize也可以
		//速度从100ms->80ms
		std::vector<std::pair<size_t, float>> indices(numPoints);
		std::vector<bool> isEdges(numPoints);

		sapClock clock;
		clock.begin();
		nanoflann::SearchParams param;
		//和true对比速度
		//100ms vs 180ms
		param.sorted = false;
		//参数越大越好？建树快，搜索速度一样。应该和点云密度有关
		nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(100));
		index.buildIndex();
		clock.end();
		printf("build kdtree：%.2fms\n", clock.getInterval());

		if (preserveEdge)
		{
			glm::vec3 minPoint = points[0];
			glm::vec3 maxPoint = points[0];
			for (int i = 0; i < numPoints; ++i)
			{
				minPoint = glm::min(minPoint, points[i]);
				maxPoint = glm::max(maxPoint, points[i]);
			}
			float interval = 0.002f * glm::distance(minPoint, maxPoint);
			minPoint -= glm::vec3(interval);
			maxPoint += glm::vec3(interval);
			glm::ivec3 numGrid = glm::ivec3((maxPoint - minPoint) / interval);
			std::vector<std::vector<std::vector<unsigned char>>> labelsGrid(numGrid.x);
			for (int i = 0; i < labelsGrid.size(); ++i)
			{
				labelsGrid[i].resize(numGrid.y);
				for (int j = 0; j < labelsGrid[i].size(); ++j)
					labelsGrid[i][j].resize(numGrid.z, 0);
			}
			for (int i = 0; i < numPoints; ++i)
			{
				glm::ivec3 idx = static_cast<glm::ivec3>((points[i] - minPoint) / interval);
				if (idx.x > 0 && idx.y > 0 && idx.z > 0 && idx.x < numGrid.x && idx.y < numGrid.y && idx.z < numGrid.z)
					labelsGrid[idx.x][idx.y][idx.z] = 255;
			}
			std::vector<std::vector<std::vector<unsigned char>>> edgesGrid(labelsGrid);
			for (int i = 0; i < numGrid.x; ++i)
				for (int j = 0; j < numGrid.y; ++j)
					for (int k = 0; k < numGrid.z; ++k)
						edgesGrid[i][j][k] = 255;
			glm::ivec3 center;
			glm::ivec3 zero;
			for (int i = 1; i < numGrid.x - 1; ++i)
				for (int j = 1; j < numGrid.y - 1; ++j)
					for (int k = 1; k < numGrid.z - 1; ++k)
					{
						edgesGrid[i][j][k] = 0;
						center = zero;
						for (int x = -1; x < 2; ++x)
							for (int y = -1; y < 2; ++y)
								for (int z = -1; z < 2; ++z)
									if (labelsGrid[i + x][j + y][k + z] == 255)
										center += glm::ivec3(x, y, z);
						if (center.x != 0 || center.y != 0 || center.z != 0)
							edgesGrid[i][j][k] = 255;
					}

			int numThreads = omp_get_max_threads();
			std::vector<std::vector<std::pair<size_t, float>>> indices2(numThreads);
			for (int i = 0; i < indices2.size(); ++i)
				indices2[i].resize(numPoints);

#pragma omp parallel for schedule (dynamic, 1) //加了反而变慢(因为indices冲突了)
			for (int i = 0; i < numPoints; ++i)
			{
				const int &threadIndex = omp_get_thread_num();
				glm::ivec3 idx = glm::ivec3((points[i] - minPoint) / interval);
				if (idx.x > 0 && idx.y > 0 && idx.z > 0 && idx.x < numGrid.x && idx.y < numGrid.y && idx.z < numGrid.z)
				{
					if (edgesGrid[idx.x][idx.y][idx.z] == 255)
					{
						glm::vec3 meanPoint;
						index.radiusSearch((float *)(cloud.points + i), distance * distance, indices2[threadIndex], param);
						for (int j = 0; j < indices2[threadIndex].size(); ++j)
							meanPoint += points[indices2[threadIndex][j].first];
						meanPoint /= indices2[threadIndex].size();
						isEdges[i] = (glm::distance(points[i], meanPoint) > 0.25f * distance);
					}
				}
			}
		}

		clock.begin();
		int count = 0;
		glm::vec3 zero;
		for (int i = 0; i < numPoints; ++i)
		{
			if (markers[i] == 255)
			{
				float radius = distance;
				if (preserveCurvature > 0.0f)
					radius = glm::min(distance, 0.2f / fabs(curvatures[i]) * distance / preserveCurvature);
				//kdtree_distance返回的是平方
				index.radiusSearch((float *)(cloud.points + i), radius * radius, indices, param);
				for (int j = 0; j < indices.size(); ++j)
					if (!preserveEdge || !isEdges[indices[j].first])
						markers[indices[j].first] = 0;
				points[count] = points[i];
				normals[count] = normals[i];
				++count;
			}
		}
		clock.end();
		printf("downsample：%.2fms\n", clock.getInterval());

		delete[]cloud.points;
		if (curvatures)
			delete[]curvatures;

		return count;
	}

	void compare(const int &numPointsTest, const glm::vec3 *pointsTest, const int &numPointsRef, glm::vec3 *pointsRef, glm::vec3 *&deviations, char *&signs)
	{
		PointCloud pointCloudRef;
		pointCloudRef.numPoints = numPointsRef;
		pointCloudRef.points = pointsRef;
		size_t *indices = new size_t[numPointsTest];
		float *distances = new float[numPointsTest];

		nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3> index(3, pointCloudRef, nanoflann::KDTreeSingleIndexAdaptorParams(15));
		index.buildIndex();

		if (!deviations)
			deviations = new glm::vec3[numPointsTest];
		if (!signs)
			signs = new char[numPointsTest];

#pragma omp parallel for schedule (dynamic, 1)
		for (int i = 0; i < numPointsTest; ++i)
		{
			index.knnSearch((float *)(pointsTest + i), 1, indices + i, distances + i, -1);
			signs[i] = -1;
			deviations[i] = pointsTest[i] - pointsRef[indices[i]];
		}

		delete[]indices;
		delete[]distances;
	}


}






