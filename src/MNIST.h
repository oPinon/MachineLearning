#pragma once

#include <string>
#include <fstream>
#include <iostream>

#include "Learning.h"

typedef unsigned char uchar;

int byteSwap(int src) {
	int dst;
	uchar* srcP = (uchar*)&src;
	uchar* dstP = (uchar*)&dst;
	int bytes = sizeof(int) / sizeof(uchar);
	for (int i = 0; i < bytes; i++) {
		dstP[i] = srcP[bytes - 1 - i];
	}
	return dst;
}

int readInt(std::fstream& src) {
	int dst;
	src.read((char*)&dst, sizeof(int));
	return byteSwap(dst);
}

// returns the index of the class with the highest probability
int maxProb(const std::vector<double> probs) {

	double bestProb = 0; int bestClass;
	for (int i = 0; i < probs.size(); i++) {
		if (probs[i] > bestProb) {
			bestProb = probs[i];
			bestClass = i;
		}
	}
	return bestClass;
}

// classifying hand writen digits ffrom the MNIST dataset
// http://yann.lecun.com/exdb/mnist/
void learnMNIST(std::string imagesFileName, std::string labelsFileName) {

	// loading the images
	std::fstream imagesFile(imagesFileName, std::ios::in | std::ios::binary);
	if (!imagesFile.is_open()) { std::cerr << "can't open " << imagesFileName.c_str() << std::endl; return; }

	// loading the labels file
	std::fstream labelsFile(labelsFileName, std::ios::in | std::ios::binary);
	if (!labelsFile.is_open()) { std::cerr << "can't open " << labelsFileName.c_str() << std::endl; return; }

	// header of the images file
	int magicNumber = readInt(imagesFile),
		nbOfImages = readInt(imagesFile),
		nbRows = readInt(imagesFile),
		nbColumns = readInt(imagesFile);

	// header of the labels file
	int magicNumber2 = readInt(labelsFile),
		nbLabels = readInt(labelsFile);

	// all sample digits
	std::vector<Sample> samples(nbOfImages);

	// parsing both files
	for (int i = 0; i < nbOfImages; i++) {

		// image pixels
		std::vector<uchar> pixels(nbRows*nbColumns);
		imagesFile.read((char*)pixels.data(), nbRows*nbColumns);

		// label
		char label;
		labelsFile.read(&label, 1);

		Sample& s = samples[i];

		s = {
			std::vector<double>(nbRows*nbColumns), // image size
			std::vector<double>(10,0) // ten digits
		};

		// converting pixels to double
		for (int j = 0; j < nbRows*nbColumns; j++) {
			s.input[j] = pixels[j] / 255.0;
		}

		s.output[label] = 1.0;

	}
	// learning the dataset
	std::random_shuffle(samples.begin(), samples.end());
	int learnSize = 1000;//(samples.size() * 80) / 100;
	std::vector<Sample> learningSamples(samples.begin(), samples.begin() + learnSize);
	NetLearner classifier(Network({ nbRows*nbColumns, 300, 10 }));

	while (true) {
		classifier.learn(learningSamples);

#include <opencv2\opencv.hpp>

		// testing the classifier
		int errors = 0;
		int testSize = min<int>(1.1 * learnSize, samples.size()) - learnSize;
		for (int i = learnSize; i < learnSize + testSize; i++) {

			cv::Mat im(cv::Size(nbColumns, nbRows), CV_64FC1);
			const Sample& s = samples[i];
			std::copy(s.input.begin(), s.input.end(), (double*)im.data);

			// results of the classification
			std::vector<double> result = classifier.apply(s.input);
			int bestClass = maxProb(result);
			if (bestClass != maxProb(s.output)) { errors++; }

#if 0 // displaying the result
			std::cout << "classified as " << bestClass << "; real class is " << maxProb(s.output) << std::endl;
			cv::resize(im, im, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
			cv::imshow("digit", im); cv::waitKey();
#endif
		}

		std::cout << (errors * 100) / (testSize) << "% errors on the test set" << std::endl;

		// display the net coeffs for each class (in colum)
		cv::Mat coeffsViz(cv::Size(nbColumns, 10 * nbRows), CV_64FC1);
		double* coeffsP = (double*)classifier.net.synapses[0].coefficients.data();
		for (int i = 0; i < 10; i++) {
			std::copy( // dont forget the bias
				coeffsP + i + i*(nbRows*nbColumns),
				coeffsP + i + (i + 1)*(nbRows*nbColumns),
				((double*)coeffsViz.data) + i*nbColumns*nbRows
				);
		}
		cv::normalize(coeffsViz, coeffsViz, -1, 1, cv::NORM_MINMAX);
		coeffsViz = 128 * coeffsViz + 128; coeffsViz.convertTo(coeffsViz, CV_8U);
		cv::applyColorMap(coeffsViz, coeffsViz, cv::COLORMAP_BONE);
		cv::resize(coeffsViz, coeffsViz, cv::Size(3 * nbColumns, 3 * nbRows * 10), 0, 0, cv::INTER_NEAREST);
		cv::imshow("coeffs", coeffsViz); cv::waitKey(16);
	}
}