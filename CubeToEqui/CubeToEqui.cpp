#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock


using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int y, int x);

int main(int argc, char** argv)
{
	//cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
	//cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
	cv::Mat h_img = cv::imread(argv[1]);

	int x = h_img.cols / 3;
	int y = h_img.rows / 2;
	int outW = x * 4;
	int outH = h_img.rows;
	cv::Mat result;

	if (strstr(argv[1], "stereo")) {
		cv::Rect crop_L = cv::Rect(0, 0, h_img.cols, h_img.rows / 2); // top
		cv::Rect crop_R = cv::Rect(0, h_img.rows / 2, h_img.cols, h_img.rows / 2); // bottom
		cv::Mat h_img_L = h_img(crop_L);
		cv::Mat h_img_R = h_img(crop_R);
		cv::cuda::GpuMat l_img, r_img, l_result, r_result;
		cv::Mat dst_img = cv::Mat::zeros(outH/2, outW, CV_8UC3);

		l_img.upload(h_img_L);
		r_img.upload(h_img_R);
		l_result.upload(dst_img);
		r_result.upload(dst_img);

		auto begin = chrono::high_resolution_clock::now();

		startCUDA(l_img, l_result, y/2, x);
		startCUDA(r_img, r_result, y/2, x);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end - begin;

		//cv::imshow("Processed Image", d_result);
		cv::Mat sl_result(l_result);
		cv::Mat sr_result(r_result);
		vconcat(sl_result, sr_result, result);

		cv::imwrite("rEqui_left.png", sl_result);
		cv::imwrite("rEqui_right.png", sr_result);
		cv::imwrite("rEqui_stereo.png", result);

		cout << diff.count() << endl;

		cv::waitKey();
		return 0;

		return 0;

	}
	else {

		cv::cuda::GpuMat img, result;
		cv::Mat h_result;
		cv::Mat dst_img = cv::Mat::zeros(outH, outW, CV_8UC3);

		img.upload(h_img);
		result.upload(dst_img);

		//cv::imshow("Original Image", img);

		cout << result.rows << result.cols << endl;

		auto begin = chrono::high_resolution_clock::now();

		startCUDA(img, result, y, x);

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> diff = end - begin;

		//cv::imshow("Processed Image", d_result);
		cv::Mat s_result(result);
		cv::imwrite("rEqui.png", s_result);

		cout << diff.count() << endl;

		cv::waitKey();
		return 0;

		return 0;
	}
}
