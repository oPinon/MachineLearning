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
	if (!labelsFile.is_open()) { std::cerr << "can't open " <<labelsFileName.c_str() << std::endl; return; }

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
		uchar label; labelsFile >> label;

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
	int learnSize = samples.size() - 32;//(samples.size() * 80) / 100;
	std::vector<Sample> learningSamples(samples.begin(),samples.begin()+learnSize);
	KNearestNeighbors classifier(100);
	classifier.learn(learningSamples);

#include <opencv2\opencv.hpp>

	// testing the classifier
	int errors = 0;
	for (int i = learnSize; i < samples.size(); i++) {

		cv::Mat im(cv::Size(nbColumns, nbRows), CV_64FC1);
		const Sample& s = samples[i];
		std::copy(s.input.begin(), s.input.end(), (double*)im.data);

		// results of the classification
		std::vector<double> result = classifier.apply(s.input);
		int bestClass = maxProb(result);
		if (bestClass != maxProb(s.output)) { errors++; }

		std::cout << "classified as " << bestClass << std::endl;
		cv::resize(im, im, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
		cv::imshow("digit", im); cv::waitKey();
	}
	std::cout << (errors*100)/(samples.size()-learnSize) << "% errors on the test set" << std::endl;
}