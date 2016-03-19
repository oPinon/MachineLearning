#include "Image.h"

ImageFilterLearner::ImageFilterLearner(int patchSize, std::vector<int> hiddenLayers) :
	patchSize(patchSize),
	learner(Network({ patchSize*patchSize,1 })) // HACK
{
	std::vector<int> layers;
	layers.push_back(patchSize*patchSize);
	layers.insert(layers.end(),hiddenLayers.begin(), hiddenLayers.end());
	layers.push_back(1);
	learner = {Network(layers)};
}

void ImageFilterLearner::learn(const Image& src, const Image& dst) {

	// checking image dimensions
	int w = src.w, h = src.h, cols = src.chans;
	if (w != dst.w || h != dst.h || src.chans != dst.chans) {
		std::cerr << "input and output images have different dimensionrs" << std::endl;
		throw 1;
	}

	// getting the image patches (assuming independent channels)
	int patchSize = 8;
	std::vector<Sample> samples;;
	for (int y = 0; y < h - patchSize; y++) {
		for (int x = 0; x < w - patchSize; x++) {
			for (int k = 0; k < cols; k++) {
				Sample s = {
					std::vector<double>(patchSize*patchSize),
					std::vector<double>(1)
				};
				for (int y2 = 0; y2 < patchSize; y2++) {
					for (int x2 = 0; x2 < patchSize; x2++) {
						s.input[y2*patchSize + x2] = src.pixels[cols*((y + y2)*w + (x + x2)) + k] / 255.0;
						//s.output[y2*patchSize + x2] = dst.pixels[cols*((y + y2)*w + (x + x2)) + k] / 255.0;
					}
				}
				s.output[0] = dst.pixels[cols*((y + patchSize/2)*w + (x + patchSize/2)) + k] / 255.0;
				samples.push_back(s);
			}
		}
	}

	// HACK : reducing the number of patches
	std::random_shuffle(samples.begin(), samples.end());
	samples = std::vector<Sample>(samples.begin(), samples.begin() + 512);

	// learning the filter from patches
	learner.learn(samples);
}

Image ImageFilterLearner::apply(const Image& src) {

	int w = src.w, h = src.h, cols = src.chans;
	Image dst(w, h, cols);
	
	// dense patches overlap : we sum them, then average
	std::vector<double> sums(w*h*cols, 0); // sum of values per pixel
	std::vector<int> counts(w*h*cols, 0); // number of values added per pixel

										  // applying the filter to each patch
	for (int y = 0; y < h - patchSize; y ++) {
		for (int x = 0; x < w - patchSize; x ++) {
			for (int k = 0; k < cols; k++) {

				// compiling the patch into a column vector
				std::vector<double> input(patchSize*patchSize);
				for (int y2 = 0; y2 < patchSize; y2++) {
					for (int x2 = 0; x2 < patchSize; x2++) {
						input[y2*patchSize + x2] = src.pixels[cols*((y + y2)*w + (x + x2)) + k] / 255.0;
					}
				}

				// applying the learnt function
				std::vector<double> output = learner.apply(input);

				// putting the patch to the destination image
				/*for (int y2 = 0; y2 < patchSize; y2++) {
					for (int x2 = 0; x2 < patchSize; x2++) {
						sums[cols*((y + y2)*w + (x + x2)) + k] += output[y2*patchSize + x2];
						counts[cols*((y + y2)*w + (x + x2)) + k]++;
					}
				}*/
				sums[cols*((y + patchSize/2)*w + (x + patchSize/2)) + k] += output[0];
				counts[cols*((y + patchSize / 2)*w + (x + patchSize / 2)) + k]++;
			}
		}
	}

	// averaging the sums on each pixel
	for (int i = 0; i < w*h*cols; i++) {
		if (counts[i] == 0) { continue; } // HACK
		dst.pixels[i] = 255.0 * sums[i] / counts[i];
	}

	return dst;
}