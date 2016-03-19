#pragma once

#include <opencv2\opencv.hpp>

#include "Learning.h"

// learns the relation between coordinates (x,y) and pixel color
void learnImage(std::string fileName) {

	// loading the image
	cv::Mat src = cv::imread(fileName);
	if (src.empty()) { std::cerr << "can't read " << fileName << std::endl; return; }

	// image dimensions
	cv::resize(src, src, cv::Size(8, 8), 0, 0, cv::INTER_AREA);
	int w = src.size().width, h = src.size().height;
	int cols = src.channels();

	// fetching the sample pixels
	std::vector<Sample> samples(w*h);
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			std::vector<double> colors(cols);
			for (int c = 0; c < cols; c++) {
				colors[c] = src.data[cols*(y*w + x) + c] / 255.0;
			}
			samples[y*w + x] = {
				{ (1.0*x) / w, (1.0*y) / h }, // coordinates in [0;1]
				colors // pixel colors
			};
		}
	}

	// learning
	NetLearner learner(Network({ 2, 10, 10, cols }));
	learner.learn(samples);

	// getting the output of the learnt function
	int dstSize = 512;
	cv::Mat dst(cv::Size(dstSize, dstSize), CV_8UC4);
	for (int y = 0; y < dstSize; y++) {
		for (int x = 0; x < dstSize; x++) {
			std::vector<double> result = learner.apply({
				(x*1.0) / dstSize,
				(y*1.0) / dstSize
			});
			for (int c = 0; c < cols; c++) {
				dst.data[4 * (y*dstSize + x) + c] = (uchar)(255 * result[c]);
			}
		}
	}
	cv::imshow("Learnt image", dst); cv::waitKey();
}

#include "Image.h"

// TODO : make dense patches
void learnImageFilter(std::string image, std::string testImage) {

	// reading the images
	cv::Mat src = cv::imread(image);
	if (src.empty()) { std::cerr << "can't read " << image << std::endl; return; }

	cv::Mat testIm = cv::imread(testImage);
	if (testIm.empty()) { std::cerr << "can't read " << testImage << std::endl; return; }

	// filtering the image
	//cv::Mat dst;// = src;//cv::Scalar(255, 255, 255) - src;
	//cv::GaussianBlur(src, dst, cv::Size(21, 21), 2, 2, cv::BORDER_REPLICATE);
	src = cv::imread("../../data/cityLow.png");
	cv::Mat dst = cv::imread("../../data/city.png");
	testIm = cv::imread("../../data/statue.png");

	// displaying the images
	cv::imshow("Input Image (learning)", src);
	cv::imshow("Output Image (learning)", dst);
	cv::waitKey(16);

	int patchSize = 4;
	ImageFilterLearner learner(patchSize, { 10, 10 });
	learner.learn(src, dst);

	cv::imshow("result on the learning image", learner.apply(src).im);
	cv::waitKey(16);
	cv::imshow("test image", testIm);
	cv::waitKey(16);
	cv::imshow("result on a test image", learner.apply(testIm).im);
	cv::waitKey();

	// testing on random patches
	for (int p = 0; p < 32; p++) {

		// random patch
		cv::Mat srcPatch(cv::Size(patchSize, patchSize), CV_8UC3);
		for (int i = 0; i < 3 * patchSize*patchSize; i++) {
			srcPatch.data[i] = (255 * rand()) / RAND_MAX;
		}
		cv::Mat dstPatch = learner.apply(srcPatch).im;

		cv::resize(srcPatch, srcPatch, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
		cv::resize(dstPatch, dstPatch, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
		cv::imshow("input patch", srcPatch);
		cv::imshow("output patch", dstPatch);
		cv::waitKey();
	}
}

void learnImage() {

	int dstSize = 512;

	std::string videoFileName = "../../data/dst.avi";
	cv::VideoWriter video(videoFileName,
		cv::VideoWriter::fourcc('F', 'M', 'P', '4'),
		30, cv::Size(dstSize, dstSize), true
		);
	if (!video.isOpened()) { std::cerr << "can't open " << videoFileName << std::endl; return; }

	NetLearner learner(Network({ 2, 3, 3, 3 }));

	for (std::string fileName : {
		"../../data/blender.png",
			"../../data/sublime.png",
			"../../data/steam.png",
			"../../data/chrome.png",
			"../../data/olivier.png"
	}) {
		// loading the image
		cv::Mat src = cv::imread(fileName);
		if (src.empty()) { std::cerr << "can't read " << fileName << std::endl; return; }

		// image dimensions
		cv::resize(src, src, cv::Size(16, 16), 0, 0, cv::INTER_AREA);
		int w = src.size().width, h = src.size().height;
		int cols = src.channels();

		// fetching the sample pixels
		std::vector<Sample> samples(w*h);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				std::vector<double> colors(cols);
				for (int c = 0; c < cols; c++) {
					colors[c] = src.data[cols*(y*w + x) + c] / 255.0;
				}
				samples[y*w + x] = {
					{ (1.0*x) / w, (1.0*y) / h }, // coordinates in [0;1]
					colors // pixel colors
				};
			}
		}

		// learning
		for (int i = 0; i < 100; i++) {
			learner.learn(samples);

			// getting the output of the learnt function
			cv::Mat dst(cv::Size(dstSize, dstSize), CV_8UC3);
			for (int y = 0; y < dstSize; y++) {
				for (int x = 0; x < dstSize; x++) {
					std::vector<double> result = learner.apply({
						(x*1.0) / dstSize,
						(y*1.0) / dstSize
					});
					for (int c = 0; c < max(cols, 3); c++) {
						dst.data[3 * (y*dstSize + x) + c] = (uchar)(255 * result[c]);
					}
				}
			}
			cv::imshow("Learnt image", dst); cv::waitKey(1);
			video.write(dst);
		}
	}
	video.release();
}