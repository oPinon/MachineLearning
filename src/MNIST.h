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
int maxProb(const std::vector<double>& probs) {

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

#if 0	// adding images randomly offseted
	for (int i = 0; i < nbOfImages; i++) {

		int offX = (1 - 2*double(rand()) / RAND_MAX) * nbColumns;
		int offY = (1 - 2 * double(rand()) / RAND_MAX) * nbRows;
		// the translation must be big enough
		if (offX*offX + offY*offY < nbRows*nbColumns / 16) { continue; }

		int j = rand() % nbOfImages;

		std::vector<double>& original = samples[j].input;
		std::vector<double> input(nbRows*nbColumns);
		for (int y = 0; y < nbRows; y++) {
			for (int x = 0; x < nbColumns; x++) {
				if(
					x+offX < 0 || x+offX >= nbColumns
					|| y+offY < 0 || y+offY >= nbRows
				) { continue; }
				input[y*nbColumns + x] = original[(y + offY)*nbColumns + x + offX];
			}
		}
		samples.push_back({ input, std::vector<double>(10,0) });
	}
#endif

	// learning the dataset
	std::random_shuffle(samples.begin(), samples.end());
	int learnSize = (samples.size() * 80) / 100;
	std::vector<Sample> learningSamples(samples.begin(), samples.begin() + learnSize);
	NetLearner classifier(Network({ nbRows*nbColumns, 10 }, 0));

	while (true) {
		classifier.learn(learningSamples, 1, 0);

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

		// display the net coeffs for each class (in a row)
		cv::Mat coeffsViz;
		double* coeffsP = (double*)classifier.net.synapses[0].coefficients.data();
		std::vector<cv::Mat> coeffs;
		for (int i = 0; i < 10; i++) {
			cv::Mat coeff(cv::Size(nbColumns, nbRows), CV_64FC1);
			std::copy( // dont forget the bias
				coeffsP + i + i*(nbRows*nbColumns),
				coeffsP + i + (i + 1)*(nbRows*nbColumns),
				((double*)coeff.data)
			);
			coeffs.push_back(coeff);
		}
		cv::hconcat(coeffs, coeffsViz);
		cv::normalize(coeffsViz, coeffsViz, -1, 1, cv::NORM_MINMAX);
		coeffsViz = 128 * coeffsViz + 128; coeffsViz.convertTo(coeffsViz, CV_8U);
		cv::applyColorMap(coeffsViz, coeffsViz, cv::COLORMAP_BONE);
		cv::resize(coeffsViz, coeffsViz, cv::Size(10 * 4 * nbColumns, 4 * nbRows), 0, 0, cv::INTER_NEAREST);
		cv::imshow("coeffs", coeffsViz);
		if (cv::waitKey(16) == 27) { break; };
	}

	// find digits in an image (TODO : multiscale)
	{
		// reading the source image
		std::string imPath = "../../data/digits.png";
		cv::Mat src = cv::imread(imPath);
		if (src.empty()) { std::cerr << "cant read " << imPath << std::endl; return; }
		cv::cvtColor(src, src, cv::COLOR_RGB2GRAY);
		cv::imshow("image", src); cv::waitKey(16);
		int w = src.size().width, h = src.size().height;

		// where to stock the results of the classifier
		struct Result {
			int foundClass;
			double confidence; // the higher, the better
		};
		std::vector<Result> results((w - nbColumns)*(h - nbRows)); // per pixel

		// Convolving the classifier with the image
		cv::Mat segmentation(cv::Size(w - nbColumns, h - nbRows), CV_64FC1);
		double* segP = (double*)segmentation.data;
		for (int y = 0; y < h - nbRows; y++) {
			for (int x = 0; x < w - nbColumns; x++) {

				// input patch
				std::vector<double> input(nbRows*nbColumns);
				for (int y2 = 0; y2 < nbRows; y2++) {
					for (int x2 = 0; x2 < nbColumns; x2++) {
						input[y2*nbColumns + x2] =
							src.data[((y+y2)*w+x+x2)*src.channels()] / 255.0;
					}
				}

				// output classes
				std::vector<double> classes = classifier.apply(input);
				segP[y*(w - nbColumns) + x] = classes[3];
				int bestClass = maxProb(classes);
				double bestProb = classes[bestClass];
				classes[bestClass] = 0;
				double secondBestProb = classes[maxProb(classes)];
				results[y*(w - nbColumns) + x] = { bestClass, bestProb - secondBestProb };
			}
		}
		segmentation.convertTo(segmentation, CV_8U, 255);
		cv::applyColorMap(segmentation, segmentation, cv::COLORMAP_BONE);
		cv::imshow("probabilities of 3", segmentation); cv::waitKey(16);

		// finding the maximums in the result image
		double bestConfidence;
		int i = 0;
		std::vector<cv::Mat> channels = { cv::Mat(src.size(),CV_8UC1),
			cv::Mat(src.size(),CV_8UC1), src };
		channels[0] = 0; channels[1] = 0;
		cv::merge( channels, src); // display the gray image in the red channel
		do {
			int bestX, bestY, bestClass;
			bestConfidence = 0;

			// searching for the best result among all pixels
			for (int y = 0; y < h - nbRows; y++) {
				for (int x = 0; x < w - nbColumns; x++) {
					Result& r = results[y*(w - nbColumns) + x];
					double confidence = r.confidence;
					if (confidence > bestConfidence) {
						bestConfidence = confidence;
						bestX = x;
						bestY = y;
						bestClass = r.foundClass;
					}
				}
			}

			// Displaying the result
			cv::rectangle(src, cv::Rect(bestX, bestY, nbColumns, nbRows), { 0,255,0 });
			std::stringstream ss; ss << bestClass
				<< " : " << int(100 * bestConfidence) << "%";
			cv::putText(src, ss.str(), cv::Point(bestX - 12, bestY + nbRows - 1),
				cv::FONT_HERSHEY_PLAIN, 1, { 0,255,0 });

			// Removing neighboring results (they might be the same digit)
			for (int y = max(0, bestY - nbRows); y < min(h - nbRows, bestY + nbRows); y++) {
				for (int x = max(0, bestX - nbColumns); x < min(w - nbColumns, bestX + nbColumns); x++) {
					results[y*(w - nbColumns) + x].confidence = 0;
				}
			}
			i++;
		} while (i < 8); // only display the best results

		cv::imshow("Digits found", src); cv::waitKey(16);
		cv::waitKey();
	}
}