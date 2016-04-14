#include "ImageTest.h"

#include <fstream>
#include <assert.h>

void faceTest(std::string folder) {

	// converting the image to samples
	std::vector<Sample> samples;
	int w = 32, h = 32;

	// compiling all images to binary, for faster reading
	std::fstream binImages(folder + "allImages", std::ios::in | std::ios::binary);
	if (!binImages.is_open()) {

		// reading the list of face images
		std::string fileListPath = folder + "list.txt";
		std::fstream fileList(fileListPath, std::ios::in);
		if (!fileList.is_open()) {
			std::cerr << "can't open the list of faces : "
				<< fileListPath << std::endl;
			return;
		}

		// reading all the images
		std::vector<cv::Mat> images;
		{
			std::string line;
			while (std::getline(fileList, line)) {
				std::cout << line << std::endl;
				cv::Mat im = cv::imread(folder + line);
				if (im.empty()) { continue; }
				images.push_back(im);
			}
			fileList.close();
		}

		// assuming that the first image has the correct size
		int w0 = images[0].size().width, h0 = images[0].size().height;

		// filtering images that have the wrong size
		{
			std::vector<cv::Mat> filteredImages;
			for (cv::Mat& im : images) {
				if (im.size().width == w0 && im.size().height == h0) {
					filteredImages.push_back(im);
				}
			}
			images = filteredImages;
			std::cout << "found " << images.size()
				<< " face images" << std::endl;
		}

		for (cv::Mat& im : images) {
			cv::resize(im, im, cv::Size(w, h), 0, 0, cv::INTER_AREA);
			assert(im.depth() == CV_8U); // im must be uint8
			Sample s;
			for (int i = 0; i < w*h; i++) {
				s.input.push_back(double(im.data[i*im.channels()]) / 255);
			}
			s.output = s.input;
			samples.push_back(s);
		}

		// exporting the samples to a binary file
		binImages = std::fstream(folder + "allImages", std::ios::out | std::ios::binary);
		if (binImages.is_open()) {
			int size = samples.size();
			binImages.write((char*)&size, sizeof(int));
			for (Sample& s : samples) {
				std::vector<unsigned char> inputC(w*h);
				for (int i = 0; i < s.input.size(); i++) { inputC[i] = 255 * s.input[i]; }
				binImages.write((char*)inputC.data(), inputC.size() * sizeof(unsigned char));
			}
		}
		binImages.close();
	}
	else { // reading the binary file
		int nbSamples;
		binImages.read((char*)&nbSamples, sizeof(int));
		std::cout << "importing " << nbSamples
			<< " samples (from bin file)" << std::endl;
		for (int i = 0; i < nbSamples; i++) {
			std::vector<unsigned char> inputC(w*h);
			binImages.read((char*)inputC.data(), w*h*sizeof(unsigned char));
			std::vector<double> input(w*h);
			for (int i = 0; i < inputC.size(); i++) { input[i] = inputC[i] / 255.0; }
			samples.push_back({ input, input });
		}
		binImages.close();
	}

	// learning the images
	int principalComponents = 8;
	NetLearner learner(Network({ w*h, principalComponents, w*h }, 0.01));

	while (true) {
		learner.learn(samples, 1, 32, 0.1);

		// display the net coeffs for each class (in a row)
		cv::Mat coeffsViz;
		double* coeffsP = (double*)learner.net.synapses[0].coefficients.data();
		std::vector<cv::Mat> coeffs;
		for (int i = 0; i < principalComponents; i++) {
			cv::Mat coeff(cv::Size(w, h), CV_64FC1);
			std::copy( // dont forget the bias
				coeffsP + i + i*(w*h),
				coeffsP + i + (i + 1)*(w*h),
				((double*)coeff.data)
				);
			cv::normalize(coeff, coeff, -1, 1, cv::NORM_MINMAX);
			coeffs.push_back(coeff);
		}
		cv::hconcat(coeffs, coeffsViz);
		coeffsViz = 128 * coeffsViz + 128; coeffsViz.convertTo(coeffsViz, CV_8U);
		cv::applyColorMap(coeffsViz, coeffsViz, cv::COLORMAP_BONE);
		cv::resize(coeffsViz, coeffsViz, cv::Size(principalComponents * 4 * w, 4 * h), 0, 0, cv::INTER_NEAREST);
		cv::imshow("coeffs", coeffsViz);
		if (cv::waitKey(16) == 27) { break; }
	}

#if 1 // reconstructing faces from learnt components
	for (Sample& s : samples) {
		auto out = learner.apply(s.input);
		cv::Mat original(cv::Size(w, h), CV_64FC1);
		std::copy(s.input.begin(), s.input.end(), (double*)original.data);
		cv::Mat output(cv::Size(w, h), CV_64FC1);
		std::copy(out.begin(), out.end(), (double*)output.data);
		cv::resize(original, original, cv::Size(256,256));
		cv::resize(output, output, cv::Size(256,256));
		cv::imshow("src", original); cv::imshow("dst", output);
		if (cv::waitKey() == 27) { break; }
	}
#endif

	// detecting faces in an image
	{
		// reading the image
		cv::Mat src = cv::imread("../../data/kid.png");
		cv::cvtColor(src, src, cv::COLOR_RGB2GRAY);
		cv::imshow("face to detect", src); cv::waitKey(16);
		src.convertTo(src, CV_64FC1); src = src / 255;

		// for different scales
		for (float scale = 1; int(src.size().width * scale) > w && int(src.size().height) > h; scale *= 0.8) {
			
			// scaling the image
			cv::Mat srcSmall;
			cv::resize(
				src, srcSmall,
				cv::Size(src.size().width*scale, src.size().height*scale),
				0, 0, cv::INTER_AREA
				);
			int wI = srcSmall.size().width, hI = srcSmall.size().height;
			cv::Mat dst(cv::Size(wI - w, hI - h), CV_64FC1);
			double* dstP = (double*)dst.data;

			// for each region of the image
			for (int y = 0; y < hI - h; y++) {
				for (int x = 0; x < wI - w; x++) {

					cv::Mat patch(cv::Size(w, h), CV_64FC1);
					srcSmall(cv::Rect(x, y, w, h)).copyTo(patch);
					cv::normalize(patch, patch, 0, 1, cv::NORM_MINMAX);
					std::vector<double> input((double*)patch.data, (double*)patch.data + w*h);
					std::vector<double> output = learner.apply(input);
					double error = 0; // reconstruction error from the PCA
					for (int k = 0; k < w*h; k++) {
						double diff = input[k] - output[k];
						error += abs(diff);
					}
					dstP[y*(wI - w) + x] = exp(-error/(w*h));
				}
			}
			cv::normalize(dst, dst, -1, 1, cv::NORM_MINMAX);
			dst = -128 * dst + 128; dst.convertTo(dst, CV_8U);
			cv::resize(dst, dst, cv::Size(src.size().width - w, src.size().height - h));
			cv::applyColorMap(dst, dst, cv::COLORMAP_BONE);
			cv::imshow("face probabilities", dst); cv::waitKey();
		}
	}
}

// http://www.anefian.com/research/GTDB_README.txt
void faceTest2(std::string folder) {

	int wF = 32, hF = 32; // faces dimensions
	std::vector<Sample> samples;

	std::string binPath = folder + "allSamples";
	std::fstream binSamples(binPath, std::ios::in | std::ios::binary);
	if (binSamples.is_open()) {
		int nb; binSamples.read((char*)&nb, sizeof(int));
		std::cout << "importing " << nb << " samples from " << binPath << std::endl;
		for (int i = 0; i < nb; i++) {
			std::vector<unsigned char> pixels(wF*hF);
			binSamples.read((char*)pixels.data(), wF*hF*sizeof(unsigned char));
			unsigned char label; binSamples.read((char*)&label, sizeof(unsigned char));
			Sample s;
			s.input = std::vector<double>(wF*hF);
			for (int i = 0; i < wF*hF; i++) { s.input[i] = pixels[i] / 255.0; }
			s.output = { label / 255.0 };
			samples.push_back(s);
		}
	}
	else {
		// for each person
		for (int i = 1; i <= 50; i++) {
			std::stringstream ss;
			ss << 100 + i;
			std::string personFolder = "s" + ss.str().substr(1) + "/",
				imFolder = folder + "gt_db/" + personFolder;

			// for each image of the person
			for (int j = 1; j <= 15; j++) {
				std::stringstream ss2;
				ss2 << 100 + j;
				std::string imPath = imFolder + ss2.str().substr(1) + ".jpg";

				// reading the image
				cv::Mat im = cv::imread(imPath);
				if (im.empty()) {
					std::cerr << "can't read " << imPath << std::endl;
					return;
				}

				// reading the label
				std::stringstream ss3;
				ss3 << 1000 + 20 * (i - 1) + j;
				std::string labelPath = folder + "labels/" +
					+"lab" + ss3.str().substr(1);
				std::fstream label(labelPath, std::ios::in);
				if (!label.is_open()) {
				std:cerr << "can't read " << labelPath << std::endl;
					return;
				}
				int x, y, x2, y2;
				label >> x >> y >> x2 >> y2;

				// Getting the face
				int w = x2 - x, h = y2 - y;
				cv::Mat face(cv::Size(w, h), CV_8UC3);
				{
					im(cv::Rect(x, y, w, h)).copyTo(face);

					// downscaling it (TODO : check ratio)
					double ratio = min(double(w) / wF, double(h) / hF);
					cv::resize(face, face, cv::Size(w / ratio, h / ratio), 0, 0, cv::INTER_AREA);

					int w = face.size().width, h = face.size().height;
					cv::Mat cropped(cv::Size(wF, hF), CV_8UC1);
					face(cv::Rect(0, h - hF, wF, hF)).copyTo(cropped);
					//cv::imshow("z", cropped); cv::waitKey();
					cv::cvtColor(cropped, cropped, cv::COLOR_RGB2GRAY);

					Sample s;
					s.input = std::vector<double>(wF*hF);
					for (int i = 0; i < wF*hF; i++) { s.input[i] = cropped.data[i] / 255.0; }
					s.output = { 1.0 }; // 1 because it is a face
					samples.push_back(s);

					cv::Mat imS;
					cv::resize(im, imS, cv::Size(im.size().width / ratio, im.size().height / ratio), 0, 0, cv::INTER_AREA);
					cv::cvtColor(imS, imS, cv::COLOR_RGB2GRAY);

					// extracting a non-face image
					int xS = x / ratio, yS = y / ratio;
					bool found = false;
					while (!found) {
						int xRand = (imS.size().width - wF) * double(rand()) / RAND_MAX;
						int yRand = (imS.size().height - hF) * double(rand()) / RAND_MAX;

						// if to close from face, discard it
						if ((xRand - xS)*(xRand - xS) + (yRand - yS)*(yRand - yS) < wF*hF / 16) { continue; }

						found = true;
						cv::Mat nonFace;
						im(cv::Rect(xRand, yRand, wF, hF)).copyTo(nonFace);
						cv::cvtColor(nonFace, nonFace, cv::COLOR_RGB2GRAY);

						Sample s;
						s.input = std::vector<double>(wF*hF);
						for (int i = 0; i < wF*hF; i++) { s.input[i] = nonFace.data[i] / 255.0; }
						s.output = { 0 };
						samples.push_back(s);
					}
				}

				//cv::rectangle(im, { x,y,w,h }, { 0,255,0 });
				//cv::imshow("k", im); cv::waitKey(1);
			}
		}

		// exporting to binary file
		binSamples = std::fstream(binPath, std::ios::out | std::ios::binary);
		if (!binSamples.is_open()) { std::cerr << "can't write to " << binPath << std::endl; }
		else {
			int nb = samples.size();
			binSamples.write((char*)&nb, sizeof(int));
			for (const Sample& s : samples) {
				std::vector<unsigned char> pixels(wF*hF);
				for (int i = 0; i < wF*hF; i++) { pixels[i] = s.input[i] * 255; }
				binSamples.write((char*)pixels.data(), wF*hF*sizeof(unsigned char));
				unsigned char label = s.output[0] * 255;
				binSamples.write((char*)&label, sizeof(unsigned char));
			}
		}
	}

#if 0
	// displaying the samples
	for (const Sample& s : samples) {
		cv::Mat face(cv::Size(wF,hF),CV_8UC1);
		for (int i = 0; i < wF*hF; i++) { face.data[i] = s.input[i] * 255; }
		cv::resize(face, face, cv::Size(8 * wF, 8 * hF), 0, 0, cv::INTER_NEAREST);
		cv::imshow(s.output[0] > 0.5 ? "Face" : "Non-Face", face); cv::waitKey();
	}
#endif

	int nbComponents = 10;
	NetLearner classifier(Network({ wF*hF, nbComponents, 1 }, 0.001));

	int nbToLearn = 80 * samples.size() / 100;
	//std::random_shuffle(samples.begin(), samples.end());
	std::vector<Sample>& learningSamples = std::vector<Sample>(samples.begin(),samples.begin()+nbToLearn);
	std::vector<Sample>& testingSamples = std::vector<Sample>(samples.begin() + nbToLearn,samples.end());
	
	while (true) {
		classifier.learn(learningSamples, 1, 8);
		int error = 0;
		for (Sample& s : testingSamples) {
			double result = classifier.apply(s.input)[0];
			if ((result < 0.5) != (s.output[0] < 0.5)) { error++; }
		}
		std::cout << "error on Test is " << (error*100.0) / testingSamples.size() << "%" << std::endl;

		// TODO : generic function to display the coeffs of a network
		cv::Mat coeffsViz;
		double* coeffsP = (double*)classifier.net.synapses[0].coefficients.data();
		std::vector<cv::Mat> coeffs;
		for (int i = 0; i < nbComponents; i++) {
			cv::Mat coeff(cv::Size(wF, hF), CV_64FC1);
			std::copy( // dont forget the bias
				coeffsP + i + i*(wF*hF),
				coeffsP + i + (i + 1)*(wF*hF),
				((double*)coeff.data)
				);
			coeffs.push_back(coeff);
		}
		cv::hconcat(coeffs, coeffsViz);
		cv::normalize(coeffsViz, coeffsViz, -1, 1, cv::NORM_MINMAX);
		coeffsViz = 128 * coeffsViz + 128; coeffsViz.convertTo(coeffsViz, CV_8U);
		cv::applyColorMap(coeffsViz, coeffsViz, cv::COLORMAP_BONE);
		cv::resize(coeffsViz, coeffsViz, cv::Size(nbComponents * 4 * wF, 4 * hF), 0, 0, cv::INTER_NEAREST);
		cv::imshow("coeffs", coeffsViz);
		if (cv::waitKey(16) == 27) { break; };
	}
}