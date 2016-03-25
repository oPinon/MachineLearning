#pragma once

#include "Learning.h"

#include <iostream>

void testXOR() {

	NetLearner learner(Network({ 2,3,1 }));
	learner.learn({
		{ { 0,0 },{ 0 } },
		{ { 0,1 },{ 1 } },
		{ { 1,0 },{ 1 } },
		{ { 1,1 },{ 0 } }
	});
}

#include <opencv2\opencv.hpp>

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
	NetLearner learner(Network({ 1,10,1 }));
	for (int i = 0; true; i++) {
		learner.learn(samples, 100);

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

	testXOR();

	//test1DFunction([](double x) { return x*x; });
	//test1DFunction([](double x) { return 0.5+0.3*sin(42*x); });
	//test1DFunction([](double x) { return x < 0.5 ? 0.2 : 0.7; });
	test1DFunction([](double x) { return 0.5 + 0.3*sin(1 / pow(x, 2)); });
}