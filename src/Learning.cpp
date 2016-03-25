#include "Learning.h"

void NetLearner::learn(const std::vector<Sample>& samples, int iterations, int miniBatch) {

	for (int i = 0; i < iterations; i++) { // TODO : stop criterion
		int count = 0;
		double error = 0;
		for (const Sample& s : samples) {

			net.setInput(s.input.data());
			net.activate();
			auto output = net.getOuput();
			for (int i = 0; i < output.size(); i++) {
				double diff = output[i] - s.output[i];
				error += diff*diff;
			}
			net.setDesiredOutput(s.output.data());
			net.backtrack();
			if (count > miniBatch) { // minibatch
				count = 0;
				net.update();
			}
			count++;
		}
		net.update();
	}
}

std::vector<double> NetLearner::apply(const std::vector<double>& input) {

	net.setInput(input.data());
	net.activate();
	return net.getOuput();
}

/*void NearestNeighbor::learn(const std::vector<Sample>& samples) {

	this->samples.insert(this->samples.end(),samples.begin(), samples.end());
}

std::vector<double> NearestNeighbor::apply(const std::vector<double>& input) {

	double bestDist = INFINITY;
	const Sample* bestSample = NULL;

	for (const Sample& s : samples) {
		double dist = 0;
		for (int i = 0; i < input.size(); i++) {
			double diff = input[i] - s.input[i];
			dist += diff*diff;
		}
		if (dist < bestDist) {
			bestDist = dist;
			bestSample = &s;
		}
	}
	return bestSample->output;
}

#include <queue>

void KNearestNeighbors::learn(const std::vector<Sample>& samples) {

	this->samples.insert(this->samples.end(), samples.begin(), samples.end());
}

std::vector<double> KNearestNeighbors::apply(const std::vector<double>& input) {

	struct Result {
		double dist; const Sample* s;
	};
	struct ResultComparator {
		bool operator()(const Result& a, const Result& b) { return a.dist < b.dist; };
	};

	std::priority_queue<Result,std::vector<Result>, ResultComparator> bestSamples;
	for (const Sample& s : samples) {
		double dist = 0;
		for (int i = 0; i < input.size(); i++) {
			double diff = input[i] - s.input[i];
			dist += diff*diff;
		}
		if (bestSamples.size() < this->nbNeighbors) {
			bestSamples.push({ dist, &s });
		}
		else if (dist < bestSamples.top().dist) {
			bestSamples.pop();
			bestSamples.push({ dist, &s });
		}
	}
	
	// return the average of the best samples
	int nbFound = bestSamples.size();
	std::vector<double> dst(bestSamples.top().s->output.size());
	while (bestSamples.size() > 0) {
		Result r = bestSamples.top();
		for (int i = 0; i < dst.size(); i++) {
			dst[i] += r.s->output[i];
		}
		bestSamples.pop();
	}
	for (double& d : dst) { d /= nbFound; }
	return dst;
}*/