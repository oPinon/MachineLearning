#pragma once

#include <vector>
#include <string>

using namespace std;

class Network
{
	inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
	inline double sigmoidDeriv(double x) { return sigmoid(x) * (1 - sigmoid(x)); } // TODO : optimize

	struct Neuron
	{
		double input; // sum of all incoming synapses
		double value; // = activation( input )
		double diff; // difference between the desired value and the value
	};

	// connections between several neural layers
	struct Synapses
	{
		const double learningRate = 1; // TODO : change it

		int inputLayer; // nb of neurons in the input layer
		int outputLayer; // nb of neurons in the output layer
	//private: TODO
		vector<double> coefficients; // coefficients of each connection
		vector<double> gradient; // delta to add to the coefficient for the next step

	public:
		inline double get(int input, int output) const { return coefficients[output * inputLayer + input]; } // get coefficient
		inline void set(int input, int output, double value) { coefficients[output * inputLayer + input] = value; } // set coefficient
		void addDiff(int input, int output, double value) { gradient[output * inputLayer + input] += value; }

		Synapses(int input, int output);

		void updateCoeffs() { for (int i = 0; i < coefficients.size(); i++) { coefficients[i] += learningRate * gradient[i]; gradient[i] = 0; } }
	};

public:

	vector<vector<Neuron>> layers;
	vector<Synapses> synapses;

	Network(vector<int> layers);
	void setInput(const double* values);
	void activate();
	void setDesiredOutput(const double* values);
	vector<double> getOuput();
	void update();
	void backtrack();
	void exportToFile(const std::string& fileName) const; // TODO
	static Network importFromFile(const std::string& fileName); // TODO
};