#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <cpprest/json.h>
#include <Windows.h>
#include "FlyCapture2.h"


using namespace std;
using namespace cv;
using namespace utility;
using namespace web;
using namespace FlyCapture2;

#define TOP_CAMERA_SERIAL_NUMBER 15322821
#define SIDE_CAMERA_SERIAL_NUMBER 15322827

#define SMOOTHMASK false
#define LOAD_BG_MODELS false
#define BG_MODEL_SIDE_FILENAME "side_bg.model"
#define BG_MODEL_TOP_FILENAME "top_bg.model"

static void help()
{
	 printf("\nBackground segmentation.\n"
		"Learning is togged by the space key. Will read from file or camera\n"
		" Based on OpenCV.\n"
		"Usage: \n"
		" ./FishDetect <path to movie file> <path to output movie file> \n\n");
}

const char* keys =
{
    "{c  camera   |         | use camera or not}"
    "{m  method   |mog2     | method (knn or mog2) }"
    "{s  smooth   |         | smooth the mask }"
    "{fn file_name|../data/tree.avi | movie file        }"
};

void OnTopImageGrabbed(Image* pImage, const void* pCallbackData)
{

}
void OnSideImageGrabbed(Image* pImage, const void* pCallbackData)
{

}

void GetImageFromCamera(Camera* cam, Mat* img)
{
	Image monoImage;
	FlyCapture2::Error error = cam->RetrieveBuffer(&monoImage);
	if (error != PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return;
	}

	// convert to OpenCV Mat
	unsigned int rowBytes = (double)monoImage.GetReceivedDataSize() / (double)monoImage.GetRows();
	cv::Mat(monoImage.GetRows(), monoImage.GetCols(), CV_8UC3, monoImage.GetData(), rowBytes).copyTo(*img);  // Not sure if this is necessary or if I could just return the new Mat, or define the input mat better and just set data...
	
	return;
}

vector<vector<Point>>* ProcessSideImage(Mat img, Ptr<BackgroundSubtractor> bg_model, Mat* fgmask, Mat fgimg)
{
	vector<vector<Point>>* goodFish = new vector<vector<Point>>();
	vector<Point> tankEdges(14);
	tankEdges[0] = Point(54, 234);
	tankEdges[1] = Point(128, 606);
	tankEdges[2] = Point(247, 750);
	tankEdges[3] = Point(409, 889);
	tankEdges[4] = Point(514, 958);
	tankEdges[5] = Point(775, 958);
	tankEdges[6] = Point(945, 826);
	tankEdges[7] = Point(1050, 717);
	tankEdges[8] = Point(1158, 552);
	tankEdges[9] = Point(1243, 267);
	tankEdges[10] = Point(1045, 114);
	tankEdges[11] = Point(711, 21);
	tankEdges[12] = Point(555, 21);
	tankEdges[13] = Point(265, 111);


	vector<vector<Point>> tankContours(1);
	tankContours[0] = tankEdges;

	// Bright areas
	Mat bands[3];
	split(img, bands);
	Mat lights;
	threshold(bands[1], lights, 150, 255, THRESH_BINARY_INV);
	//imshow("Lights", lights);

	if (fgimg.empty())
		fgimg.create(img.size(), img.type());

	//update the model
	bg_model->apply(img, *fgmask, -1);
	if (SMOOTHMASK) {
		GaussianBlur(*fgmask, *fgmask, Size(11, 11), 3.5, 3.5);
		threshold(*fgmask, *fgmask, 10, 255, THRESH_BINARY);
	}

	Mat fgmaskMed;
	medianBlur(*fgmask, fgmaskMed, 3);

	fgimg = Scalar::all(0);
	img.copyTo(fgimg, fgmaskMed);

	Mat bgimg;
	bg_model->getBackgroundImage(bgimg);
	*fgmask = *fgmask & lights;

	// Render contours on the image.
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(*fgmask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	int minContourLen = 30;
	bool oneFishDone = false;
	for (int i = 0; i< contours.size(); i++)
	{
		double area = contourArea(contours[i], false);
		if (area > 30)
		{
			/* TO DO
			For each blob I've detected, draw a line perpendicular to each tank edge (mirror).  Search those lines for blobs
			If you find one additional blob along this line, remove it from the list
			If you find two (or more) blobs along this line, remove the one that is closest in size to your own
			*/
			// Get contour's center of mass
			Moments m = moments(contours[i], false);
			Point2f mc = Point2f(m.m10 / m.m00, m.m01 / m.m00);
			cv::drawMarker(img, mc, Scalar(255, 0, 0), cv::MarkerTypes::MARKER_STAR, 10);
			// if the center of mass is within the bounding box of the tank bottom, I already know it's not a reflection
			vector<vector<Point>> possiblyBadFish;
			if (cv::pointPolygonTest(tankEdges, mc, false) < 0)  // possibly a reflection
			{
				drawContours(img, contours, i, Scalar(255, 0, 255), 1, 8, hierarchy, 0, Point());
				possiblyBadFish.push_back(contours[i]);

			}
			else  // guaranteed good fish
			{
				drawContours(img, contours, i, Scalar(0, 255, 255), 1, 8, hierarchy, 0, Point());
				goodFish->push_back(contours[i]);
			}
		}
	}
	cv::drawContours(img, tankContours, -1, Scalar(0, 255, 0));

	return goodFish;
}

vector<vector<Point>>* ProcessTopImage(Mat img, Ptr<BackgroundSubtractor> bg_model, Mat fgmask, Mat fgimg)
{
	vector<vector<Point>>* goodFish = new vector<vector<Point>>();

	vector<Point> tankBottom(6);
	tankBottom[0] = Point(480, 293);
	tankBottom[1] = Point(746, 258);
	tankBottom[2] = Point(911, 475);
	tankBottom[3] = Point(803, 731);
	tankBottom[4] = Point(539, 758);
	tankBottom[5] = Point(373, 539);
	vector<vector<Point>> bottomContours(1);
	bottomContours[0] = tankBottom;

	vector<Point[2]> tankWalls(6);
	tankWalls[0][0] = tankBottom[0];
	tankWalls[0][1] = tankBottom[1];
	tankWalls[1][0] = tankBottom[1];
	tankWalls[1][1] = tankBottom[2];
	tankWalls[2][0] = tankBottom[2];
	tankWalls[2][1] = tankBottom[3];
	tankWalls[3][0] = tankBottom[3];
	tankWalls[3][1] = tankBottom[4];
	tankWalls[4][0] = tankBottom[4];
	tankWalls[4][1] = tankBottom[5];
	tankWalls[5][0] = tankBottom[5];
	tankWalls[5][1] = tankBottom[0];

	// Bright areas
	Mat bands[3];
	split(img, bands);
	Mat lights;
	threshold(bands[1], lights, 150, 255, THRESH_BINARY_INV);
	//imshow("Lights", lights);

	if (fgimg.empty())
		fgimg.create(img.size(), img.type());

	//update the model
	bg_model->apply(img, fgmask, -1);
	if (SMOOTHMASK)  {
		GaussianBlur(fgmask, fgmask, Size(11, 11), 3.5, 3.5);
		threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
	}

	Mat fgmaskMed;
	medianBlur(fgmask, fgmaskMed, 3);

	fgimg = Scalar::all(0);
	img.copyTo(fgimg, fgmaskMed);

	Mat bgimg;
	bg_model->getBackgroundImage(bgimg);
	fgmask = fgmask & lights;

	// Render contours on the image.
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(fgmask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	int minContourLen = 30;
	bool oneFishDone = false;
	for (int i = 0; i< contours.size(); i++)
	{
		double area = contourArea(contours[i], false);
		if (area > 30)
		{
			/* TO DO
			For each blob I've detected, draw a line perpendicular to each tank edge (mirror).  Search those lines for blobs
			If you find one additional blob along this line, remove it from the list
			If you find two (or more) blobs along this line, remove the one that is closest in size to your own
			*/
			// Get contour's center of mass
			Moments m = moments(contours[i], false);
			Point2f mc = Point2f(m.m10 / m.m00, m.m01 / m.m00);
			cv::drawMarker(img, mc, Scalar(255, 0, 0), cv::MarkerTypes::MARKER_STAR, 10);
			// if the center of mass is within the bounding box of the tank bottom, I already know it's not a reflection
			vector<vector<Point>> possiblyBadFish;
			if (cv::pointPolygonTest(tankBottom, mc, false) < 0)  // possibly a reflection
			{
				drawContours(img, contours, i, Scalar(255, 0, 255), 2, 8, hierarchy, 0, Point());
				possiblyBadFish.push_back(contours[i]);

			}
			else  // guaranteed good fish
			{
				drawContours(img, contours, i, Scalar(0, 255, 255), 2, 8, hierarchy, 0, Point());
				goodFish->push_back(contours[i]);
				if (!oneFishDone)
				{
					// for each tank edge, compute the perpendicular line to each edge of the tank
					for (int i = 0; i < tankWalls.size(); i++)
					{
						// Step 1, compute the angle between the fish point and the closest point I have on the line
						float xd = mc.x - tankWalls[i][0].x;
						float yd = mc.y - tankWalls[i][0].y;
						float dist[3];
						dist[0] = sqrt(xd*xd + yd*yd);
						xd = mc.x - tankWalls[i][1].x;
						yd = mc.y - tankWalls[i][1].y;
						dist[1] = sqrt(xd*xd + yd*yd);
						xd = tankWalls[i][0].x - tankWalls[i][1].x;
						yd = tankWalls[i][0].y - tankWalls[i][1].y;
						dist[2] = sqrt(xd*xd + yd*yd);
						int nearest = dist[0] <= dist[1] ? 0 : 1;
						int farthest = nearest == 0 ? 1 : 0;
						float angle = acos((dist[2] * dist[2] + dist[nearest] * dist[nearest] - dist[farthest] * dist[farthest]) / (2 * dist[2] * dist[nearest]));
						Point2f nearestPoint = Point2f(tankWalls[i][nearest]);
						Point2f farthestPoint = Point2f(tankWalls[i][farthest]);
						// if that angle is > 90, I know the perpendicular line intersects outside my two points
						if (angle > CV_PI / 2 || angle < -1 * CV_PI / 2)
						{
							continue;
						}
						// Step 2, get the new point along the line where my fish point would intersect
						float d = cos(angle)*dist[nearest];
						Point2f v = nearestPoint - farthestPoint;
						float mag = sqrt(v.x * v.x + v.y * v.y);
						Point2f u = v / mag;
						Point2f pointOnLine = nearestPoint - d*u;
						line(img, mc, pointOnLine, CvScalar(0, 128, 255));
					}
					//oneFishDone = true;
				}

			}
		}
	}
	cv::drawContours(img, bottomContours, -1, Scalar(0, 255, 0));

	return goodFish;
}

//this is a sample for foreground detection functions
int main(int argc, const char** argv)
{
	bool useCamera = false;
	
	if (argc != 4)
	{
		help();
		useCamera = true;
		fprintf(stderr, "Usage: FishDetect <path to top movie file> <path to side movie file> <path to output movie file> \nPress [Enter] to exit.\n");
		fprintf(stderr, "Got:\n");
		for (int i = 0; i < argc; i++)
		{
			fprintf(stderr, "%d: %s\n", i, argv[i]);
		}
		getchar();
		return -1;
	}

    string file_top = argv[1];
	string file_side = argv[2];
	string outfile = argv[3];
    string method = "mog";
    VideoCapture cap_top;
	VideoCapture cap_side;
	Camera* camera_top = NULL;
	Camera* camera_side = NULL;

	namedWindow("Side Camera");
	namedWindow("Top Camera");
	moveWindow("Side Camera", 0, 0);
	moveWindow("Top Camera", 1280, 0);

	Ptr<BackgroundSubtractor> bg_model_top_tank = method == "knn" ?
		createBackgroundSubtractorKNN().dynamicCast<BackgroundSubtractor>() :
		createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
	Ptr<BackgroundSubtractor> bg_model_side_tank = method == "knn" ?
		createBackgroundSubtractorKNN().dynamicCast<BackgroundSubtractor>() :
		createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

	Mat topimg, sideimg, fgmask_top_tank, fgmask_side_tank, fgimg_top_tank, fgimg_side_tank;
	int frameNo = 0;

	if (useCamera)
	{
		// Get available pointgrey cameras
		BusManager busMgr;
		unsigned int numCameras;
		FlyCapture2::Error error = busMgr.GetNumOfCameras(&numCameras);
		if (error != PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return -1;
		}

		cout << "Number of cameras detected: " << numCameras << endl;

		if (numCameras < 2)
		{
			cout << "Insufficient number of cameras... exiting" << endl;
			return -1;
		}

		cout << "Getting cameras from serial numbers...";
		FlyCapture2::PGRGuid guid;
		error = busMgr.GetCameraFromSerialNumber(TOP_CAMERA_SERIAL_NUMBER, &guid);
		if (error != PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return -1;
		}
		cout << "[OK]" << endl;
		cout << "Connecting to cameras...";

		camera_top = new FlyCapture2::Camera();
		error = camera_top->Connect(&guid);
		if (error != PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return -1;
		}
		cout << "Top...";

		camera_side = new FlyCapture2::Camera();
		error = camera_side->Connect(&guid);
		if (error != PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return -1;
		}
		cout << "Side...[OK]" << endl;

		cout << "Setting camera modes...";
		// greyscale image mode
		FlyCapture2::Format7ImageSettings fmt7ImageSettings;
		fmt7ImageSettings.width = 1280;
		fmt7ImageSettings.height = 10024;
		fmt7ImageSettings.mode = MODE_0;
		fmt7ImageSettings.offsetX = 0;
		fmt7ImageSettings.offsetY = 0;
		fmt7ImageSettings.pixelFormat = PIXEL_FORMAT_MONO8;
		// Validate Format 7 settings
		bool valid;
		Format7PacketInfo fmt7PacketInfo;
		error = camera_top->ValidateFormat7Settings(&fmt7ImageSettings,
			&valid,
			&fmt7PacketInfo);
		unsigned int num_bytes =
			fmt7PacketInfo.recommendedBytesPerPacket;

		// Set Format 7 (partial image mode) settings
		error = camera_top->SetFormat7Configuration(&fmt7ImageSettings,
			num_bytes);
		if (error != PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return -1;
		}
		cout << "Top...";
		error = camera_side->ValidateFormat7Settings(&fmt7ImageSettings,
			&valid,
			&fmt7PacketInfo);
		num_bytes =
			fmt7PacketInfo.recommendedBytesPerPacket;

		// Set Format 7 (partial image mode) settings
		error = camera_side->SetFormat7Configuration(&fmt7ImageSettings,
			num_bytes);
		if (error != PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return -1;
		}
		cout << "Side...[OK]" << endl;

		cout << "Starting cameras...";

		error = camera_top->StartCapture();// OnTopImageGrabbed);
		if (error != PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return -1;
		}
		cout << "Top...";

		error = camera_side->StartCapture();// OnSideImageGrabbed);
		if (error != PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return -1;
		}
		cout << "Side...[OK]" << endl;
	}
	else
	{
		cap_top.open(file_top.c_str());
		cap_side.open(file_side.c_str());

		if (!cap_top.isOpened() || !cap_side.isOpened())
		{
			printf("can not open camera or video file %s and %s\n", file_top.c_str(), file_side.c_str());
			getchar();
			return -1;
		}
	}

    for(;;)
    {
		if (!useCamera)
		{
			cap_top >> topimg;
			if (topimg.empty()) break;
			cap_side >> sideimg;
			if (sideimg.empty()) break;
		}
		else
		{
			GetImageFromCamera(camera_top, &topimg);
			GetImageFromCamera(camera_side, &sideimg);
		}
		vector<vector<Point>>* goodFish_top = ProcessTopImage(topimg, bg_model_top_tank, fgmask_top_tank, fgimg_top_tank);
		vector<vector<Point>>* goodFish_side = ProcessSideImage(sideimg, bg_model_side_tank, &fgmask_side_tank, fgimg_side_tank);

		// create the json to submit
		// this is expected format:
		/* [{"FishId":"RedFish","FishLocationDateTime":{"DateTime":"\/Date(-62135596800000)\/","OffsetMinutes":0},"XPos":1.234,"YPos":2.342,"ZPos":2.4445},{"FishId":"RedFish","FishLocationDateTime":{"DateTime":"\/Date(-62135596800000)\/","OffsetMinutes":0},"XPos":1.234,"YPos":2.342,"ZPos":2.4445}] */
		/*
		json::value jsonArray = json::value::array(goodFish->size());
		for (int i = 0; i < goodFish->size(); i++)
		{
			Moments m = moments((*goodFish)[i], false);
			Point2f mc = Point2f(m.m10 / m.m00, m.m01 / m.m00);
			json::value fish;
			fish[L"FishId"] = json::value::string(U("RandomFish"));
			fish[L"XPos"] = json::value::number(mc.x);
			fish[L"YPos"] = json::value::number(mc.y);
			fish[L"ZPos"] = json::value::number(0.0);
			jsonArray[i] = fish;
		}
		utility::stringstream_t stream;
		utility::string_t jsonString = jsonArray.serialize();
		std::replace(jsonString.begin(), jsonString.end(), '"', '\'');
		int size_needed = WideCharToMultiByte(CP_UTF8, 0, jsonString.c_str(), jsonString.length(), NULL, 0, NULL, NULL);
		std::string strTo(size_needed, 0);
		WideCharToMultiByte(CP_UTF8, 0, jsonString.c_str(), jsonString.length(), &strTo[0], size_needed, NULL, NULL);

		char* command = (char*)malloc(strTo.size() + 20);
		strcpy(command, "FishData.exe \"");
		command = strcat(command, strTo.c_str());
		strcat(command, "\"");
		
		int key = waitKey(10);
		if (key == 'A')
		{
			printf("Command: [%s]\n", command);
			system(command);
		}
		*/
        imshow("Side Camera", sideimg);
		imshow("Top Camera", topimg);
		//imwrite("out\\out" + std::to_string(frameNo) +".jpg", sideimg);
		frameNo++;
		
		if(goodFish_top != NULL)
			delete goodFish_top;
		if(goodFish_side != NULL)
			delete goodFish_side;

        char k = (char)waitKey(30);
        if( k == 27 ) break;
    }
	bg_model_side_tank->save(BG_MODEL_SIDE_FILENAME);
	bg_model_top_tank->save(BG_MODEL_TOP_FILENAME);

    return 0;
}
