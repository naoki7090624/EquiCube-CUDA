#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <cuda_runtime.h>
#include <algorithm>


#define PI 3.1415
#define M(a, b) ((a) > (b) ? (a) : (b))

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int Y, int X)
{

	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;


	if (dst_x < cols && dst_y < rows)
	{
		double U = double(dst_x) / (cols - 1);
		double V = double(dst_y) / (rows - 1);

		double theta = U * 2 * PI;
		double phi = V * PI;

		double x = cos(theta) * sin(phi);
		double y = sin(theta) * sin(phi);
		double z = cos(phi);

		double maximum = M(abs(x), M(abs(y), abs(z)));
		double xx = x / maximum;
		double yy = y / maximum;
		double zz = z / maximum;

		double x2D;
		double y2D;
		double rho;

		int iX, iY;

		if (xx == 1.0 || xx == -1.0) {
			x = xx * 0.5;
			rho = x / (cos(theta) * sin(phi));
			y = rho * sin(theta) * sin(phi);
			z = rho * cos(phi);
			if (xx == 1) { // X+
				x2D = y + 0.5;
				y2D = 1.0 - (z + 0.5);
				iX = x2D * X + X;
				iY = y2D * Y;
			}
			else { //X-
				x2D = (y * -1.0) + 0.5;
				y2D = 1.0 - (z + 0.5);
				iX = x2D * X;
				iY = y2D * Y + Y;
			}
		}
		else if (yy == 1 || yy == -1) {
			y = yy * 0.5;
			rho = double(y) / (sin(theta) * sin(phi));
			x = rho * cos(theta) * sin(phi);
			z = rho * cos(phi);
			if (yy == 1) { //Y+
				x2D = (x * -1.0) + 0.5;
				y2D = 1.0 - (z + 0.5);
				iX = x2D * X + 2*X;
				iY = y2D * Y;
			}
			else { //Y-
				x2D = x + 0.5;
				y2D = 1.0 - (z + 0.5);
				iX = x2D * X;
				iY = y2D * Y;
			}
		}
		else {
			z = zz * 0.5;
			rho = double(z) / cos(phi);
			x = rho * cos(theta) * sin(phi);
			y = rho * sin(theta) * sin(phi);
			if (zz == 1) { //Z+
				x2D = y + 0.5;
				y2D = 1.0 - ((x * -1.0) + 0.5);
				iX = x2D * X + 2*X;
				iY = y2D * Y + Y;
			}
			else { //Z-
				x2D = y + 0.5;
				y2D = 1.0 - (x + 0.5);
				iX = x2D * X + X;
				iY = y2D * Y + Y;
			}

		}

		dst(dst_y, dst_x).x = src(iY, iX).x;
		dst(dst_y, dst_x).y = src(iY, iX).y;
		dst(dst_y, dst_x).z = src(iY, iX).z;

	}
}

int divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int y, int x)
{
	const dim3 block(32, 8);
	const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

	process << <grid, block >> > (src, dst, dst.rows, dst.cols, y, x);

}

