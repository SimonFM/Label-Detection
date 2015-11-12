/*
 * Simon Markham 
 *
 * Label Detection using OpenCV
 *
 * For my implementation I examined the images through the HLS channels
 * and saw that the "best" way to detect a label was through the Luminance
 * channel or the Saturation Channels and then caculated the Sd of those
 * images in those channels. If they had a high SD, then they probably had
 * a label due to the various different colours in those spectrums. If they
 * had a relatively low SD, then they shouldn't have a label.
 */
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


// Location of the images in the project
char * fileLocation = "Media/";
char * imageFiles[] = { "Glue1.jpg", "Glue2.jpg",
						"Glue3.jpg", "Glue4.jpg",
						"Glue5.jpg", "Glue6.jpg"};

// Actual results, analysed by me.
bool groundTruth[6][5] = {{true, true, true, true, false},
						  {true, false, true, true, true},
                          {true, true, true, false, true},
                          {true, false, false, true, true},
                          {false, true, true, false, true},
                          {true, true, true, true, false} };
string str = "";
Mat hlsChannels[3];
Mat * image, * cropped;
Mat sampleImage, sampleImageSD;
int i, j;
int channelIndex = 1;
double standard_dev = 0;

// Testing
int FP = 0,FN = 0, TP = 0,TN = 0;
double precision, accuracy, recall, f1, specificity;

/**
 * Sets the sample SD to be used, from the sample image.
 */
void setSampleSD(int index){
	Mat  mean, std, testImage, channels[3] ;

	sampleImage =  image[index](Rect (283,177,69,90));
	cvtColor(sampleImage,testImage,CV_BGR2HLS);
	split(testImage,channels);
	meanStdDev(channels[channelIndex],mean,sampleImageSD);
}

/**
 * This function calculates the True Positives, True Negatives,
 * False Positives and the False Negatives. I looked at how Ken did
 * it in his book and came up with my own method that does it.
 */
void test(bool results){
	// True Positive
	// When there was a label and it detected it.
	if(results == true && groundTruth[i][j] == true) TP++;
	// True Negative
	// There wasn't a label and it didn't detect one.
	else if(results == false && groundTruth[i][j] == false ) TN++;
	// False Postive
	// It detected a label, but there wasn't a label
	else if(results == true && groundTruth[i][j] == false) FP++;
	// False Negative
	// There was a label, but it was not classified as a label
	else if(results == false && groundTruth[i][j] == true) FN++;
	else cout << "This shouldn't have happened"<<endl;
}

/**
 * A simple method to get the standard deviation and determine if there's
 * a label or not. If the label has an SD of less than 19.0 then shouldn't
 * be a label otherwise, there should be.
 */
void detectLabel(Mat img){
	Mat  mean, std ;
	meanStdDev(img,mean,std);

	float sd =  std.at<double>(0,0);
	float sampleSD = sampleImageSD.at<double>(0,0);

	bool result = sd >= sampleSD;

	if(result) cout << "YES" <<endl;
	else cout << "NO" <<endl;
	test(result);

	imshow("IMAGE",image[i]);
	imshow(str+" : result_"+i,img);
}

// Function to run program
int main(int argc, const char** argv){
	Mat display;
	Mat * hls =	new Mat[5];
	int number_of_images = sizeof(imageFiles)/sizeof(imageFiles[0]);

	image = new Mat[number_of_images];
	cropped = new Mat[number_of_images];
	int width, height;

	// This code snippet was taken from your OpenCVExample and it loads the images
	for (i = 0; i < number_of_images; i++){
			string filename(fileLocation);
			filename.append(imageFiles[i]);
			image[i] = imread(filename, -1);

			if (image[i].empty()) {
				cout << "Could not open " << filename << endl;
				return -1;
			}
		}
	/*
	 * Loop to iterate through every channel of HLS
	 */
	for(channelIndex = 0; channelIndex < 3; channelIndex++){
		// reset the metrics
		precision = 0.0;
		recall = 0.0;
		accuracy = 0.0;
		f1 = 0.0;
		specificity = 0.0;
		TP = 0;
		TN = 0;
		FP = 0;
		FN = 0;
		// set the sample Standard Deviation
		setSampleSD(1);
		for (i = 0; i < number_of_images; i++){
			// This the area my solution crops for every image loaded in.
			// values for each image.
			switch(i){
				case 0:
					cropped[0] = image[i](Rect (22,174,74,90));
					cropped[1] = image[i](Rect (149,168,74,90));
					cropped[2] = image[i](Rect (274,173,74,90));
					cropped[3] = image[i](Rect (400,170,74,90));
					cropped[4] = image[i](Rect (529,173,74,90));
					break;
				case 1:
					cropped[0] = image[i](Rect (27,182,71,89));
					cropped[1] = image[i](Rect (164,183,57,79));
					cropped[2] = image[i](Rect (283,177,69,90));
					cropped[3] = image[i](Rect (398,170,72,90));
					cropped[4] = image[i](Rect (520,173,68,90));
					break;
				case 2:
					cropped[0] = image[i](Rect (28,176,72,89));
					cropped[1] = image[i](Rect (142,174,57,79));
					cropped[2] = image[i](Rect (272,177,69,90));
					cropped[3] = image[i](Rect (387,181,72,90));
					cropped[4] = image[i](Rect (500,171,68,90));
					break;
				case 3:
					cropped[0] = image[i](Rect (30,179,71,85));
					cropped[1] = image[i](Rect (156,186,57,79));
					cropped[2] = image[i](Rect (277,181,69,90));
					cropped[3] = image[i](Rect (393,180,72,90));
					cropped[4] = image[i](Rect (517,175,68,90));
					break;
				case 4:
					cropped[0] = image[i](Rect (25,177,71,85));
					cropped[1] = image[i](Rect (143,173,57,79));
					cropped[2] = image[i](Rect (263,176,69,90));
					cropped[3] = image[i](Rect (381,171,69,90));
					cropped[4] = image[i](Rect (501,174,68,90));
					break;
				case 5:
					cropped[0] = image[i](Rect (28,182,70,88));
					cropped[1] = image[i](Rect (155,176,70,88));
					cropped[2] = image[i](Rect (282,178,70,88));
					cropped[3] = image[i](Rect (408,179,70,88));
					cropped[4] = image[i](Rect (539,179,70,88));
					break;
			}

			str = imageFiles[i];
			// Iterate through every cropped image to determine if there's a label.
			// By first convert to the HLS spectrum, then getting those separate hlsChannels
			// and passing the luminence channel in to the detectLabel function
			for(j = 0; j < 5; j++){
				cvtColor(cropped[j],hls[j],CV_BGR2HLS);
				split(hls[j],hlsChannels);
				detectLabel(hlsChannels[channelIndex]);
				waitKey(0);
			}

		}
		cout<<"---------------------------------------"<<endl;
		switch(channelIndex){
			case 0:
				cout<<"Hue Channel"<<endl;
				break;
			case 1:
				cout<<"Luminance"<<endl;
				break;
			case 2:
				cout<<"Saturation"<<endl;
				break;
			default:
				cout<<"Nope"<<endl;
				break;

		}

		// Calculate and display the metrics.
		precision = ((double) TP) / ((double) (TP+FP));
		recall = ((double) TP) / ((double) (TP+FN));
		accuracy = ((double) (TP+TN)) / ((double) (TP+FP+TN+FN));
		f1 = 2.0* ((precision*recall) / (precision + recall));
		specificity = ((double) TN) / ((double) (FP+TN));
		cout<<"---------------------------------------"<<endl;
		cout<<"TP"<<TP<<endl;
		cout<<"TN"<<TN<<endl;
		cout<<"FP"<<FP<<endl;
		cout<<"FN"<<FN<<endl;
		cout<<"---------------------------------------"<<endl;
		// Ouputs the tests to the console.
		cout<<"Precision: "<<precision<<endl;
		cout<<"Recall: "<<recall<<endl;
		cout<<"Accuracy: "<<accuracy<<endl;
		cout<<"Specificity: "<<specificity<<endl;
		cout<<"F1 Measure: "<<f1<<endl;
		cout<<"---------------------------------------"<<endl;
		waitKey(0);
	}
    return 0;
}
