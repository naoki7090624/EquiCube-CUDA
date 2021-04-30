#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <cuda_runtime.h>


#define PI 3.1415

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int H, int W, int rows, int cols, double sqr)
{

	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;


	if (dst_x < cols && dst_y < rows)
	{
		double tx = 0.0, ty = 0.0;
		double x = 0.0, y = 0.0, z = 0.0;
		
		if (dst_y + 1 < sqr + 1) {
			if (dst_x + 1 < sqr + 1) {
				tx = dst_x + 1;
				ty = dst_y + 1;
				x = tx - 0.5 * sqr;
				y = 0.5 * sqr;
				z = ty - 0.5 * sqr;
			}
			else if (dst_x + 1 < 2 * sqr + 1) {
				tx = dst_x + 1 - sqr;
				ty = dst_y + 1;
				x = 0.5 * sqr;
				y = (tx - 0.5 * sqr) * -1;
				z = ty - 0.5 * sqr;
			}
			else {
				tx = dst_x + 1 - sqr * 2;
				ty = dst_y + 1;
				x = (tx - 0.5 * sqr) * -1;
				y = -0.5 * sqr;
				z = ty - 0.5 * sqr;
			}
		}
		else {
			if (dst_x + 1 < sqr + 1) {
				tx = dst_x + 1;
				ty = dst_y + 1 - sqr;
				x = int(-0.5 * sqr);
				y = int(tx - 0.5 * sqr);
				z = int(ty - 0.5 * sqr);
			}
			else if (dst_x + 1 < 2 * sqr + 1) {
				tx = dst_x + 1 - sqr;
				ty = dst_y + 1 - sqr;
				x = (ty - 0.5 * sqr) * -1;
				y = (tx - 0.5 * sqr) * -1;
				z = 0.5 * sqr;
			}
			else {
				tx = dst_x + 1 - sqr * 2;
				ty = dst_y + 1 - sqr;
				x = ty - 0.5 * sqr;
				y = (tx - 0.5 * sqr) * -1;
				z = -0.5 * sqr;
			}
		}

		double rho = sqrt(x * x + y * y + z * z);
		double normTheta = y < 0 ? atan2(y, x) * -1 : PI + (PI - atan2(y, x));
		normTheta /= (2.0 * PI);
		double normPhi = (PI - acos(z / rho)) / PI;

		int iX = normTheta * W;
		int iY = normPhi * H;

		if (iX >= W)
			iX -= W;
		if (iY >= H)
			iY -= H;


		dst(dst_y, dst_x).x = src(iY, iX).x;
		dst(dst_y, dst_x).y = src(iY, iX).y;
		dst(dst_y, dst_x).z = src(iY, iX).z;
	}
}

int divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, double sqr)
{
	const dim3 block(32, 8);
	const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

	process << <grid, block >> > (src, dst, src.rows, src.cols, dst.rows, dst.cols, sqr);

}

