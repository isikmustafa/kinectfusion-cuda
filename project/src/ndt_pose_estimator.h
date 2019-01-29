#pragma once

#include "ndt_map_3d.h"
#include "data_helper.h"
#include "device_helper.cuh"

#include <cmath>

template<size_t N>
class NdtPoseEstimator
{
public:
	NdtPoseEstimator(float total_width_in_meters, unsigned int frame_width, unsigned int frame_height)
		: m_ndt_map(total_width_in_meters)
		, m_frame_width(frame_width)
		, m_frame_height(frame_height)
	{
		m_c1 = 0.95f;
		m_c2 = (1.0f - m_c1) / pow(m_ndt_map.getVoxelWidth(), 3.0f);
		m_d3 = -log(m_c2);
		m_d1 = -log(m_c1 + m_c2) - m_d3;
		constexpr float exp_0p5 = 1.648721271f;
		m_d2 = -2.0f * log((-log(m_c1 * exp_0p5 + m_c2) - m_d3) / m_d1);
	}

	// Vertices always expected in world coordinates
	void initialize(glm::vec4* vertices)
	{
		Coords2D coords;
		for (coords.x = 0; coords.x < m_frame_height; coords.x++)
		{
			for (coords.y = 0; coords.y < m_frame_width; coords.y++)
			{
				glm::vec4 vertex = vertices[calculateIndexFromPixelCoords(coords)];

				m_ndt_map.updateMap(glm::vec3(vertex));
			}
		}
	}

	// Vertices always expected in world coordinates
	void computePose(glm::vec4* vertices, const glm::mat4x4& previous_pose)
	{
		/*
			For each vertex:
			1. Calculate grid coordinates
			2. Get normal distribution of corresponding voxel
			3. Optimize pose with Newton, following chapter V of
			http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.7059&rep=rep1&type=pdf
			4. return if converged, else back to 1
		*/

		const int max_iter = 5;
		const float convergence_thresh = 1e-6;

		Coords2D coords;
		for (int i = 0; i < max_iter; i++)
		{
			//Initialize gradient and hessian with zeros at each iteration.
			m_gradient.fill(0.0f);
			m_hessian.fill(m_gradient);

			for (coords.x = 0; coords.x < m_frame_height; coords.x++)
			{
				for (coords.y = 0; coords.y < m_frame_width; coords.y++)
				{
					glm::vec4 vertex = vertices[calculateIndexFromPixelCoords(coords)];

					if (vertex.w == device_helper::cInvalid)
					{
						continue;
					}
					vertex.w = 1.0f;

					glm::vec3 vertex_global = previous_pose * vertex;
					auto grid_coordinate = m_ndt_map.getGridCoordinate(vertex_global);
					auto index = m_ndt_map.calculateGridIndexFromGridCoordinate(grid_coordinate);

					if (m_ndt_map.isIndexValid(index))
					{
						auto& voxel = m_ndt_map.getVoxel(index);

						// Suggested by the thesis to avoid singular covariance matrix.
						if (voxel.count > 5.0f)
						{
							constructGradientAndHessian(m_ndt_map.toVoxelCoordinates(grid_coordinate, index), voxel);
						}
					}
				}
			}

			bool converged = solveNewtonStep();

			if (converged)
			{
				break;
			}
		}
	}

private:
	NdtMap3D<N> m_ndt_map;
	unsigned int m_frame_width;
	unsigned int m_frame_height;

	// Constant parameters for Newton's algorithm
	float m_c1;
	float m_c2;
	float m_d1;
	float m_d2;
	float m_d3;

	std::array<float, 6> m_gradient;
	std::array<std::array<float, 6>, 6> m_hessian;

private:
	Coords2D calculatePixelCoords(unsigned int idx)
	{
		return { idx / m_frame_width, idx % m_frame_width };
	}

	int calculateIndexFromPixelCoords(const Coords2D& coords)
	{
		return coords.x * m_frame_width + coords.y;
	}

	void constructGradientAndHessian(const glm::vec3& local_point, const NdtVoxel& voxel)
	{
		// Calculate some reusable intermediate results
		glm::vec3 x = local_point - voxel.mean;
		glm::mat3 sigma_inv = computeInverseCovMat(voxel);

		std::array<glm::vec3, 6> jacobian{ { { 1.0f, 0.0f, 0.0f },
											   { 0.0f, 1.0f, 0.0f },
											   { 0.0f, 0.0f, 0.0f },
											   { 0.0f, -x.z,  x.y },
											   {  x.z, 0.0f, -x.x },
											   { -x.y,  x.x, 0.0f } } };

		glm::vec3 x_T_sigma_inv = sigma_inv * x;

		std::array<float, 6> x_T_sigma_inv_dx_dp;
		for (int i = 0; i < 6; i++)
		{
			x_T_sigma_inv_dx_dp[i] = glm::dot(x_T_sigma_inv, jacobian[i]);
		}

		float exp_x_T_sigma_inv_x = m_d1 * m_d2 * exp(-m_d2 / 2.0f * glm::dot(x_T_sigma_inv, x));

		// Update gradient
		for (int i = 0; i < 6; i++)
		{
			m_gradient[i] += x_T_sigma_inv_dx_dp[i] * exp_x_T_sigma_inv_x;
		}

		// Update Hessian
		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				float dx_dpj_T_sigma_inv_dx_dpi = glm::dot(sigma_inv * jacobian[i], jacobian[j]);
				m_hessian[i][j] = exp_x_T_sigma_inv_x * (-m_d2 * x_T_sigma_inv_dx_dp[i] * x_T_sigma_inv_dx_dp[j]
					+ dx_dpj_T_sigma_inv_dx_dpi);
			}
		}
	}

	// Beware that this function assumes that the voxel has more than 5 points inside (voxel.count > 5)
	glm::mat3 computeInverseCovMat(const NdtVoxel& voxel)
	{
		auto one_over_count = 1.0f / voxel.count;

		glm::mat3 cov_mat;
		cov_mat[0][0] = voxel.co_moments_diag.x * one_over_count;
		cov_mat[1][1] = voxel.co_moments_diag.y * one_over_count;
		cov_mat[2][2] = voxel.co_moments_diag.z * one_over_count;
		cov_mat[0][1] = cov_mat[1][0] = voxel.co_moments_triangle.x * one_over_count;
		cov_mat[0][2] = cov_mat[2][0] = voxel.co_moments_triangle.y * one_over_count;
		cov_mat[1][2] = cov_mat[2][1] = voxel.co_moments_triangle.z * one_over_count;

		return glm::inverse(cov_mat);
	}

	bool solveNewtonStep()
	{
		// TODO: implement, signature is not finite
		return false;
	}
};