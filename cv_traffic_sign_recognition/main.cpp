#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

#define PI 3.1415926

using namespace cv;
using namespace std;
std::vector<cv::KeyPoint> keypoints1, keypoints2;
cv::Ptr<cv::ORB> orb = cv::ORB::create();
cv::Mat descriptors1, descriptors2;
string standardPath = "../template_6.jpeg";

void RGB2HSV(double red, double green, double blue, double &hue, double &saturation, double &intensity) {
    double r, g, b;
    double h, s, i;

    double sum;
    double minRGB, maxRGB;
    double theta;

    r = red / 255.0;
    g = green / 255.0;
    b = blue / 255.0;

    minRGB = ((r < g) ? (r) : (g));
    minRGB = (minRGB < b) ? (minRGB) : (b);

    maxRGB = ((r > g) ? (r) : (g));
    maxRGB = (maxRGB > b) ? (maxRGB) : (b);

    sum = r + g + b;
    i = sum / 3.0;

    if (i < 0.001 || maxRGB - minRGB < 0.001) {
        h = 0.0;
        s = 0.0;
    } else {
        s = 1.0 - 3.0 * minRGB / sum;
        theta = sqrt((r - g) * (r - g) + (r - b) * (g - b));
        theta = acos((r - g + r - b) * 0.5 / theta);
        if (b <= g)
            h = theta;
        else
            h = 2 * PI - theta;
        if (s <= 0.01)
            h = 0;
    }

    hue = (int) (h * 180 / PI);
    saturation = (int) (s * 100);
    intensity = (int) (i * 100);
}

int ORB_demo(Mat tar, Mat ori, string name) {
    cvtColor(ori, ori, cv::COLOR_BGR2BGRA);

    orb->detect(ori, keypoints1);
    orb->compute(ori, keypoints1, descriptors1);

    orb->detect(tar, keypoints2);
    orb->compute(tar, keypoints2, descriptors2);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher bfmatcher(cv::NORM_HAMMING);
    bfmatcher.match(descriptors1, descriptors2, matches);

    double min_dist = 0, max_dist = 3000;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance <= cv::max(2 * min_dist, 30.0))
            good_matches.push_back(matches[i]);
    }

    cv::Mat img_match;
    drawMatches(ori, keypoints1, tar, keypoints2, good_matches, img_match);

    string txt;
    if (good_matches.size() == 0) {
        txt = "impossible";
    } else if (good_matches.size() > 0 && good_matches.size() <= 3) {
        txt = "less possible";
    } else if (good_matches.size() > 3 && good_matches.size() <= 7) {
        txt = "featured possible";
    } else if (good_matches.size() > 7) {
        txt = "highly possible";
    }
    cv::imshow(name + ": " + txt, img_match);

//    std::cout << "good matches size = " << good_matches.size() << std::endl;
    return good_matches.size();
}

void fillHole(const Mat srcBw, Mat &dstBw) {
    Size m_Size = srcBw.size();
    Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));

    Mat cutImg;
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

bool isInside(Rect rect1, Rect rect2) {
    Rect t = rect1 & rect2;
    if (rect1.area() > rect2.area()) {
        return false;
    } else {
        if (t.area() != 0)
            return true;
    }
}

vector<Mat> getCircle(Mat srcImg) {
    Mat srcImgCopy;
    srcImg.copyTo(srcImgCopy);

    int width = srcImg.cols;
    int height = srcImg.rows;
    double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
    Mat matRgb = Mat::zeros(srcImg.size(), CV_8UC1);
    int x, y;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            B = srcImg.at<Vec3b>(y, x)[0];
            G = srcImg.at<Vec3b>(y, x)[1];
            R = srcImg.at<Vec3b>(y, x)[2];
            RGB2HSV(R, G, B, H, S, V);
            if ((H >= 330 && H <= 360 || H >= 0 && H <= 10) && S >= 21 && S <= 100 && V > 16 &&
                V < 99) {
                matRgb.at<uchar>(y, x) = 255;
            }
        }
    }
    imshow("step1: to hsv model", matRgb);

    medianBlur(matRgb, matRgb, 3);
    medianBlur(matRgb, matRgb, 5);
    imshow("step2: Median Blur", matRgb);

    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
    Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 3 + 1, 2 * 3 + 1), Point(3, 3));
    erode(matRgb, matRgb, element);
    imshow("step3: Erode", matRgb);
    dilate(matRgb, matRgb, element1);
    imshow("step4: Dilate", matRgb);
    fillHole(matRgb, matRgb);//填充
    imshow("step5: FillHole", matRgb);

    Mat matRgbCopy;
    matRgb.copyTo(matRgbCopy);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Mat> r;
    findContours(matRgb, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point>> contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());

    for (int i = 0; i < contours.size(); i++) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
    }

    Mat drawing = Mat::zeros(matRgb.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
        Rect rect = boundRect[i];
        bool inside = false;
        for (int j = 0; j < contours.size(); j++) {
            Rect t = boundRect[j];
            if (rect == t)
                continue;
            else if (isInside(rect, t)) {
                inside = true;
                break;
            }
        }
        if (inside)
            continue;

        float ratio = (float) rect.width / (float) rect.height;
        float Area = (float) rect.width * (float) rect.height;
        float dConArea = (float) contourArea(contours[i]);
        float dConLen = (float) arcLength(contours[i], 1);
        if (dConArea < 700)
            continue;
        if (ratio > 1.3 || ratio < 0.4)
            continue;

        Scalar color = (0, 0, 255);
        drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);

        cv::Mat roi = srcImg(boundRect[i]);
        r.push_back(roi);

        Mat grayImg, dstImg, normImg, scaledImg;
        cvtColor(drawing, grayImg, COLOR_BGR2GRAY);
        cornerHarris(grayImg, dstImg, 2, 3, 0.04);

        normalize(dstImg, normImg, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        convertScaleAbs(normImg, scaledImg);

        int harrisNum = 0;
        for (int j = 0; j < normImg.rows; j++) {
            for (int i = 0; i < normImg.cols; i++) {
                if ((int) normImg.at<float>(j, i) > 160) {
                    circle(scaledImg, Point(i, j), 4, Scalar(0, 10, 255), 2, 8, 0);
                    harrisNum++;
                }
            }
        }
        if (harrisNum > 33)
            continue;
//        imshow("result.jpg", srcImg);
//        imshow("cornerHarris.jpg", scaledImg);
    }

    //waitKey(0);
    return r;
}

int main(int argc, char **argv) {
    int action;
    cout << "--------------------欢迎使用交通标志检测识别系统----------------------" << endl;
    cout << "Enter '1': 使用系统内置交通标志的样本." << endl;
    cout << "Enter '2': 使用系统不带内置交通标志的样本." << endl;
    cout << "Enter '3': 使用第三方图片，输入URL." << endl;
    cin >> action;
    string originPath;
    if (action == 1) {
        originPath = "../image/example_1.jpeg";
    } else if (action == 2) {
        originPath = "../image/example_2.jpeg";
    } else if (action == 3) {
        cout << "Enter the ABSOLUTE path of your image: ";
        cin >> originPath;
    } else {
        cout << "Bad action." << endl;
        exit(502);
    }

    Mat origin = imread(originPath, IMREAD_COLOR);
    Mat standard = imread(standardPath, IMREAD_COLOR);
    if (standard.empty()) {
        cout << "Could not load sample image due to the wrong path! Please fix the path of sample path in source code line 13!" << endl;
        exit(404);
    }
    if (origin.empty()) {
        cout << "Could not load image due to the wrong path! " << endl;
        exit(404);
    }
    vector<Mat> circled = getCircle(origin);
    int i = 0;
    int max = 0;
    for (auto it:circled) {
        i++;
        imshow("Detected Sign No." + to_string(i), it);
        cout << "Detected Sign No." + to_string(i) << endl;
        int num = ORB_demo(standard, it, "Sign No." + to_string(i));
        cout << "Matched feature points number of No." + to_string(i) + " sign = " << num << endl;
        if (num >= max) max = num;
        string result;
        if (num == 0) {
            result = "Detected sign No." + to_string(i) + " is completely impossible to be the target sign.";
        } else if (num > 0 && num <= 3) {
            result = "Detected sign No." + to_string(i) + " is less possible to be the target sign.";
        } else if (num > 3 && num <= 7) {
            result = "Detected sign No." + to_string(i) + " is featured possible to be the target sign.";
        } else if (num > 7) {
            result = "Detected sign No." + to_string(i) + " is highly possible to be the target sign.";
        }
        cout << result << endl << endl;
    }
    if (i == 0) {
        cout << "Cannot detect any sign!" << endl;
    }
    string sumup;
    if (max == 0) {
        sumup = "Target sign is completely impossible in this image.";
    } else if (max > 0 && max <= 7) {
        sumup = "At least one target sign is possible in this image.";
    } else if (max > 7) {
        sumup = "At least one target sign is in this image.";
    }
    cout << "RESULT: " + sumup << endl;
    waitKey(0);
    return 0;
}
