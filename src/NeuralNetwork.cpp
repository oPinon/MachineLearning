#include "NeuralNetwork.h"

Network::Synapses::Synapses(int input, int output) : inputLayer(input), outputLayer(output) {

	coefficients = vector<double>(input*output);
	gradient = vector<double>(input*output, 0);
	for (int i = 0; i < coefficients.size(); i++) { coefficients[i] = 1 - 2 * ((double)rand()) / RAND_MAX; }
}

Network::Network(vector<int> layerSizes) : layers(), synapses() {

	for (int i = 0; i < layerSizes.size(); i++) {
		vector<Neuron> layer = vector<Neuron>(layerSizes[i] + 1);
		layer[layer.size() - 1].value = -1; // the bias neuron has a constant value of -1
		if (i < layerSizes.size() - 1) {  // for every layer but the last :

			Synapses s = Synapses(layerSizes[i] + 1, layerSizes[i + 1]);  // don't forget the bias in the input layer !
			synapses.push_back(s);
		}
		layers.push_back(layer);
	}
}

void Network::setInput(const double* values) {

	vector<Neuron>& inputLayer = layers[0];
	for (int i = 0; i < inputLayer.size()-1; i++) { inputLayer[i].value = values[i]; }
}

void Network::activate() {

	for (int i = 1; i < layers.size(); i++) {  // for every layer but the input

		vector<Neuron>& layer = layers[i];
		Synapses& synapse = synapses[i - 1]; // synapses between layer i-1 and i
		vector<Neuron>& prevLayer = layers[i - 1];
		for (int nextNeuron = 0; nextNeuron < layer.size() - 1; nextNeuron++) {
			layer[nextNeuron].input = 0;
			for (int prevNeuron = 0; prevNeuron < synapse.inputLayer; prevNeuron++) {
				layer[nextNeuron].input += synapse.get(prevNeuron, nextNeuron) * prevLayer[prevNeuron].value;
			}
			layer[nextNeuron].value = sigmoid(layer[nextNeuron].input);
		}
	}
}

void Network::setDesiredOutput(const double* values) {

	vector<Neuron>& outputLayer = layers[layers.size() - 1];
	for (int i = 0; i < outputLayer.size() - 1; i++) {
		outputLayer[i].diff = sigmoidDeriv(outputLayer[i].input) * (values[i] - outputLayer[i].value); // delta = g'(in) * (y - a)
	}
}

vector<double> Network::getOuput() {

	vector<Neuron>& outputLayer = layers[layers.size() - 1];
	vector<double> dst(outputLayer.size() - 1);
	for (int i = 0; i < outputLayer.size() - 1; i++) {
		dst[i] = outputLayer[i].value;
	}
	return dst;
}

void Network::update() {

	for (auto& s : synapses) { s.updateCoeffs(); }
}

void Network::backtrack() {

	for (size_t l = layers.size() - 2; l >= 0; l--) {
		vector<Neuron>& layer = layers[l];; // local input layer
		Synapses& synapse = synapses[l];
		vector<Neuron>& nextLayer = layers[l + 1]; // local output layer
		for (int j = 0; j < layer.size(); j++) {
			double diffSum = 0;
			for (int i = 0; i < nextLayer.size() - 1; i++) {
				diffSum += synapse.get(j, i) * nextLayer[i].diff;
				double diff = layer[j].value * nextLayer[i].diff;
				synapse.addDiff(j, i, diff);
			}
			layer[j].diff = sigmoidDeriv(layer[j].input) * diffSum;
		}
	}
}