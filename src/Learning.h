#pragma once

#include <vector>

// learning sample
struct Sample {
	std::vector<double> input;
	std::vector<double> output;
};

// TODO : learn on Sample iterator, instead of vector
class Learner {
public:
	virtual void learn(const std::vector<Sample>& samples) = 0;
	virtual std::vector<double> apply(const std::vector<double>& input) = 0;
};

#include "NeuralNetwork.h"

class NetLearner : Learner {

public:
	Network net; // TODO : private
	NetLearner(Network net) : net(net) {};
	void learn(
		const std::vector<Sample>& samples,
		int iterations,
		int miniBatch = -1 // set to -1 to disable minibatches
	);
	void learn(const std::vector<Sample>& samples) { learn(samples, 1); }; // HACK ?
	std::vector<double> apply(const std::vector<double>& input); // TODO : make it const
};

class NearestNeighbor : Learner {

	std::vector<Sample> samples;
public:
	void learn(const std::vector<Sample>& samples);
	std::vector<double> apply(const std::vector<double>& input);
};

class KNearestNeighbors : Learner {

	std::vector<Sample> samples;
	int nbNeighbors;
public:
	KNearestNeighbors(int nbNeighbors = 10) : nbNeighbors(nbNeighbors) {};
	void learn(const std::vector<Sample>& samples);
	std::vector<double> apply(const std::vector<double>& input);
};