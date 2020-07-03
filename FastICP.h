#ifndef FASTICP_H
#define FASTICP_H

#include "glm/glm.hpp"
#include <vector>
#include <Eigen/Eigen>


namespace FastICP
{
	int uniformDownsample(const int &numPoints, glm::vec3 *points, glm::vec3 *normals, const float &distance = 1.0f, const float &preserveCurvature = 1.0f, const bool &preserveEdge = false);

	void compare(const int &numPointsTest, const glm::vec3 *pointsTest, const int &numPointsRef, glm::vec3 *pointsRef, glm::vec3 *&deviations, char *&signs);

	void ICP(std::vector<float> &Weights, glm::mat4 &matrix, const int &numSrc, const glm::vec3 *cloudSrc, const int &numDst, const glm::vec3 *cloudDst, const glm::vec3 *normalDst, const int &numIterations = 100, const float &threshold = 0.02f, const float &maxDistance = 1.0f, const bool &picky = true, const bool &point2PlaneMetric = true, const bool &incremental = true);

	glm::mat4 projectedICP(const std::vector<glm::vec3> &cloudSrc, const glm::mat4 &projectionMatrix, const int &widthDst, const int &heightDst, const glm::vec3 *cloudDst, const glm::vec3 *normalsDst, const int &numIterations = 100, const float &threshold = 0.02f, const float &maxDistance = 1.0f);

	template <typename Scalar>
	glm::mat4 minimizePoint2Point(const int &cloudSize, const glm::detail::tvec3<Scalar> *cloudSrc, const glm::detail::tvec3<Scalar> *cloudDst)
	{
		glm::mat4 transformMatrix;

		// dual quaternion optimization
		Eigen::Matrix<Scalar, 4, 4> C1 = Eigen::Matrix<Scalar, 4, 4>::Zero();
		Eigen::Matrix<Scalar, 4, 4> C2 = Eigen::Matrix<Scalar, 4, 4>::Zero();
		Scalar *c1 = C1.data();
		Scalar *c2 = C2.data();

		glm::detail::tvec3<Scalar> a;
		glm::detail::tvec3<Scalar> b;
		for (int i = 0; i < cloudSize; ++i)
		{
			a = cloudSrc[i];
			b = cloudDst[i];
			/*const Scalar axbx = a.x * b.x * Weights[i];
			const Scalar ayby = a.y * b.y * Weights[i];
			const Scalar azbz = a.z * b.z * Weights[i];
			const Scalar axby = a.x * b.y * Weights[i];
			const Scalar aybx = a.y * b.x * Weights[i];
			const Scalar axbz = a.x * b.z * Weights[i];
			const Scalar azbx = a.z * b.x * Weights[i];
			const Scalar aybz = a.y * b.z * Weights[i];
			const Scalar azby = a.z * b.y * Weights[i];*/
			const Scalar axbx = a.x * b.x;
			const Scalar ayby = a.y * b.y;
			const Scalar azbz = a.z * b.z;
			const Scalar axby = a.x * b.y;
			const Scalar aybx = a.y * b.x;
			const Scalar axbz = a.x * b.z;
			const Scalar azbx = a.z * b.x;
			const Scalar aybz = a.y * b.z;
			const Scalar azby = a.z * b.y;
			c1[0] += axbx - azbz - ayby;
			c1[5] += ayby - azbz - axbx;
			c1[10] += azbz - axbx - ayby;
			c1[15] += axbx + ayby + azbz;
			c1[1] += axby + aybx;
			c1[2] += axbz + azbx;
			c1[3] += aybz - azby;
			c1[6] += azby + aybz;
			c1[7] += azbx - axbz;
			c1[11] += axby - aybx;

			c2[1] += a.z + b.z;
			c2[2] -= a.y + b.y;
			c2[3] += a.x - b.x;
			c2[6] += a.x + b.x;
			c2[7] += a.y - b.y;
			c2[11] += a.z - b.z;
		}

		c1[4] = c1[1];
		c1[8] = c1[2];
		c1[9] = c1[6];
		c1[12] = c1[3];
		c1[13] = c1[7];
		c1[14] = c1[11];
		c2[4] = -c2[1];
		c2[8] = -c2[2];
		c2[12] = -c2[3];
		c2[9] = -c2[6];
		c2[13] = -c2[7];
		c2[14] = -c2[11];

		C1 *= -2.0f;
		C2 *= 2.0f;

		const Eigen::Matrix<Scalar, 4, 4> A = (0.25f / float (cloudSize)) * C2.transpose() * C2 - C1;

		const Eigen::EigenSolver< Eigen::Matrix<Scalar, 4, 4>> es(A);

		ptrdiff_t i;
		es.eigenvalues().real().maxCoeff(&i);
		const Eigen::Matrix<Scalar, 4, 1> qmat = es.eigenvectors().col(i).real();
		const Eigen::Matrix<Scalar, 4, 1> smat = -(0.5f / float(cloudSize)) * C2 * qmat;

		const Eigen::Quaternion<Scalar> q(qmat(3), qmat(0), qmat(1), qmat(2));
		const Eigen::Quaternion<Scalar> s(smat(3), smat(0), smat(1), smat(2));

		const Eigen::Quaternion<Scalar> t = s * q.conjugate();

		const Eigen::Matrix<Scalar, 3, 3> R(q.toRotationMatrix());

		for (int i =0 ; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				transformMatrix[j][i] = R(i,j);

		transformMatrix[3][0] = -t.x();
		transformMatrix[3][1] = -t.y();
		transformMatrix[3][2] = -t.z();

		return transformMatrix;
	}

	template <typename Scalar>
	glm::mat4 minimizePoint2Point( const int &cloudSize, const glm::detail::tvec3<Scalar> *cloudSrc, const glm::detail::tvec3<Scalar> *cloudDst, const std::vector<int> &indicesSrc, const std::vector<int> &indicesDst)
	{
		glm::mat4 transformMatrix;

		// dual quaternion optimization
		Eigen::Matrix<Scalar, 4, 4> C1 = Eigen::Matrix<Scalar, 4, 4>::Zero();
		Eigen::Matrix<Scalar, 4, 4> C2 = Eigen::Matrix<Scalar, 4, 4>::Zero();
		Scalar *c1 = C1.data();
		Scalar *c2 = C2.data();

		glm::detail::tvec3<Scalar> a;
		glm::detail::tvec3<Scalar> b;
		for (int i = 0; i < cloudSize; ++i)
		{
			a = cloudSrc[indicesSrc[i]];
			b = cloudDst[indicesDst[i]];
			const Scalar axbx = a.x * b.x;
			const Scalar ayby = a.y * b.y;
			const Scalar azbz = a.z * b.z;
			const Scalar axby = a.x * b.y;
			const Scalar aybx = a.y * b.x;
			const Scalar axbz = a.x * b.z;
			const Scalar azbx = a.z * b.x;
			const Scalar aybz = a.y * b.z;
			const Scalar azby = a.z * b.y;
			c1[0] += axbx - azbz - ayby;
			c1[5] += ayby - azbz - axbx;
			c1[10] += azbz - axbx - ayby;
			c1[15] += axbx + ayby + azbz;
			c1[1] += axby + aybx;
			c1[2] += axbz + azbx;
			c1[3] += aybz - azby;
			c1[6] += azby + aybz;
			c1[7] += azbx - axbz;
			c1[11] += axby - aybx;

			c2[1] += a.z + b.z;
			c2[2] -= a.y + b.y;
			c2[3] += a.x - b.x;
			c2[6] += a.x + b.x;
			c2[7] += a.y - b.y;
			c2[11] += a.z - b.z;
		}

		c1[4] = c1[1];
		c1[8] = c1[2];
		c1[9] = c1[6];
		c1[12] = c1[3];
		c1[13] = c1[7];
		c1[14] = c1[11];
		c2[4] = -c2[1];
		c2[8] = -c2[2];
		c2[12] = -c2[3];
		c2[9] = -c2[6];
		c2[13] = -c2[7];
		c2[14] = -c2[11];

		C1 *= -2.0f;
		C2 *= 2.0f;

		const Eigen::Matrix<Scalar, 4, 4> A = (0.25f / float (cloudSize)) * C2.transpose() * C2 - C1;

		const Eigen::EigenSolver< Eigen::Matrix<Scalar, 4, 4>> es(A);

		ptrdiff_t i;
		es.eigenvalues().real().maxCoeff(&i);
		const Eigen::Matrix<Scalar, 4, 1> qmat = es.eigenvectors().col(i).real();
		const Eigen::Matrix<Scalar, 4, 1> smat = -(0.5f / float(cloudSize)) * C2 * qmat;

		const Eigen::Quaternion<Scalar> q(qmat(3), qmat(0), qmat(1), qmat(2));
		const Eigen::Quaternion<Scalar> s(smat(3), smat(0), smat(1), smat(2));

		const Eigen::Quaternion<Scalar> t = s * q.conjugate();

		const Eigen::Matrix<Scalar, 3, 3> R(q.toRotationMatrix());

		for (int i =0 ; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				transformMatrix[j][i] = R(i,j);

		transformMatrix[3][0] = -t.x();
		transformMatrix[3][1] = -t.y();
		transformMatrix[3][2] = -t.z();

		return transformMatrix;
	}

	glm::mat4 minimizePoint2Plane(const int &numMatched, const glm::vec3 *cloudSrc, const glm::vec3 *cloudDst, const glm::vec3 *normalDst);

	glm::mat4 minimizePoint2Plane(std::vector<float> &Weights, const int &numMatched, const glm::vec3 *cloudSrc, const glm::vec3 *cloudDst, const glm::vec3 *normalDst, const std::vector<int> &indicesSrc, const std::vector<int> &indicesDst);

	int computeNormals(const int &numPoints, const glm::vec3 *points, glm::vec3 *normals, const float &radius);
}

#endif