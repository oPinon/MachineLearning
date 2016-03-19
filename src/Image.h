#pragma once


#include "Learning.h"

#include <opencv2\opencv.hpp>

// image container
class Image {
public:
	cv::Mat im;
	const int w, h, chans;
	uchar* pixels;
	Image(cv::Mat im) :
		w(im.size().width),
		h(im.size().height),
		chans(im.channels()),
		im(im),
		pixels(im.data)
		{}
	Image(int w, int h, int chans) :
		w(w), h(h), chans(chans),
		im(cv::Size(w,h), CV_MAKETYPE(CV_8U, chans)),
		pixels(im.data) {}
};

// TODO : output size = 1 or patchSize*patchSize ?
class ImageFilterLearner {
	const int patchSize;
	NetLearner learner;
public:
	ImageFilterLearner(int patchSize = 8, std::vector<int> hiddenLayers = { 10 });
	void learn(const Image& input, const Image& output);
	Image apply(const Image& input);
};