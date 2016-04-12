#pragma once

#include "Learning.h"

#include <iostream>
#include <fstream>
#include <ctime>
#include <opencv2\opencv.hpp>

// learns the XOR with different parameters, and outputs a CSV
void testXOR(std::string outputCsvPath) {

	// testing different parameters, to find the best ones
	srand(time(NULL));
	struct Parameters {
		double initCoeffs,
			miniBatch,
			learningRate;
	};
	std::vector<Parameters> params = {
		{ 0, -1, 1 },
		{ 0, 0, 1 }
	};
	for (double i = 1E-3; i <= 10; i*=10) {
		for (double l = 1E-3; l <= 10; l *= 10) {
			for (double m = -1; m <= 5; m++) {
				params.push_back({i,m,l});
			}
		}
	}

	std::vector<Sample> samples = {
		{ { 0,0 },{ 0 } },
		{ { 0,1 },{ 1 } },
		{ { 1,0 },{ 1 } },
		{ { 1,1 },{ 0 } }
	};
	std::fstream results(outputCsvPath, std::ios::out);
	if (!results.is_open()) { std::cerr << "can't write "
		<< outputCsvPath << std::endl; return; }

	results << "Initial Coefficients, Batch Size, Learning Rate"
		", Error after 10, Error after 100, Error after 1000" << std::endl;
	for (auto param : params) {

		results << param.initCoeffs << ','
			<< param.miniBatch << ',' << param.learningRate << ',';
		NetLearner learner(Network({ 2, 3, 1 }, param.initCoeffs));
		results << learner.learn(samples, 10, param.miniBatch, param.learningRate)
			<< ',' << learner.learn(samples, 90, param.miniBatch, param.learningRate)
			<< ',' << learner.learn(samples, 900, param.miniBatch, param.learningRate)
			<< std::endl;
	}
	results.close();

	// how robust are parameters ? Testing several random starts
	std::vector<double> errors;
	int nbStarts = 100;
	clock_t start = clock();
	std::cout << "testing " << nbStarts << " random starts :" << std::endl;
	for (int i = 0; i < nbStarts; i++) {
		NetLearner learner(Network({ 2, 10, 1 }, 1));
		errors.push_back(learner.learn(samples, 1000, -1, 10));

		/*cv::Mat im(cv::Size(512, 512), CV_8UC1);
		for (int y = 0; y < 512; y++) {
			for (int x = 0; x < 512; x++) {
				im.data[y * 512 + x] = 255 * learner.apply({ x / 512.0,y / 512.0 })[0];
			}
		}
		cv::imshow("im", im); cv::waitKey(1);*/
	}
	double mean = 0;
	for (double e : errors) { mean += e; }
	mean /= errors.size();
	double std = 0;
	for (double e : errors) { std += (e - mean)*(e - mean); }
	std = sqrt(std / errors.size());
	std::cout << "done in " << 
		((clock() - start) * 1000.0 / CLOCKS_PER_SEC / nbStarts) << " ms" << std::endl;
	std::cout << "mean error is " << mean
		<< " and std is " << std << std::endl;
	double minE = INFINITY, maxE = 0;
	for (double e : errors) {
		if (e < minE) { minE = e; }
		if (e > maxE) { maxE = e; }
	}
	std::cout << "min error is " << minE
		<< " and max is " << maxE << std::endl;
}

// simulates and learns a [0,1]->[0,1] function
void test1DFunction(double(*func)(double)) {

	// generating samples
	std::vector<Sample> samples(512);
	for (int i = 0; i < samples.size(); i++) {
		double x = double(rand()) / RAND_MAX;
		double y = func(x);
		samples[i] = { {x},{y} };
	}

	// learning from samples
	NetLearner learner(Network({ 1,10,10,1 }));
	for (int i = 0; true; i++) {
		learner.learn(samples, 100, 5);

		// plotting the results
		int w = 1024, h = 256;
		cv::Mat plot(cv::Size(w, h), CV_8UC3); plot = cv::Scalar{ 255, 255, 255 };
		for (int x = 0; x < w - 1; x++) { // plotting the real function
			cv::line(plot,
			{ x, int(h*func(double(x) / w)) },
			{ x + 1, int(h*func(double(x + 1) / w)) },
			{ 255, 200, 200 } // in light blue
			);
		}
		for (int x = 0; x < w - 1; x++) { // plotting the learned function
			cv::line(plot,
			{ x, int(h * learner.apply({ double(x) / w })[0]) },
			{ x + 1, int(h * learner.apply({ double(x + 1) / w })[0]) }, { 0, 0, 255 }, // in red
				2
				);
		}
		for (const auto& s : samples) { // plotting the samples
			cv::circle(plot, { int(s.input[0] * w), int(s.output[0] * h) }, 2, { 0, 0,0 }, cv::FILLED);
		}
		cv::imshow("plot", plot); cv::waitKey(16);
	}
}

void testAll() {

	//test1DFunction([](double x) { return x*x; });
	//test1DFunction([](double x) { return 0.5+0.3*sin(42*x); });
	//test1DFunction([](double x) { return x < 0.5 ? 0.2 : 0.7; });
	test1DFunction([](double x) { return 0.5 + 0.3*sin(1 / pow(1-x, 2)); });
}