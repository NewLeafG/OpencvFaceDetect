package org.opencv.samples.colorblobdetect;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.opencv.imgproc.Imgproc.boundingRect;

public class ColorBlobDetector {
    // Lower and Upper bounds for range checking in HSV color space
    private Scalar mLowerBound = new Scalar(0);
    private Scalar mUpperBound = new Scalar(0);
    private Scalar mLowerBoundForCompare = new Scalar(0);
    private Scalar mUpperBoundForCompare = new Scalar(0);
    // Minimum contour area in percent for contours filtering
    private static double mMinContourArea = 0.1;
    // Color radius for range checking in HSV color space
    private Scalar mColorRadius = new Scalar(15, 30, 30, 0);
    private Scalar mColorRadiusForCompare = new Scalar(10, 20, 20, 0);
    //    private Scalar mColorRadius = new Scalar(25, 50, 50, 0);
    private Scalar m_mainColor = new Scalar(15, 30, 30, 0);
//    private Scalar m_compareColor = new Scalar(15, 30, 30, 0);
    private Mat mSpectrum = new Mat();
    private List<MatOfPoint> mContours = new ArrayList<MatOfPoint>();
    public int m_maxIndex = 0;

    // Cache
    Mat mPyrDownMat = new Mat();
    Mat mHsvMat = new Mat();
    Mat mMask = new Mat();
    Mat mDilatedMask = new Mat();
    Mat mHierarchy = new Mat();

    // The number of rows and cols for dividing grid.
    int m_rows = 5;
    int m_cols = 5;
    double m_maxArea = 0;

    public void setColorRadius(Scalar radius) {
        mColorRadius = radius;
    }

    public void setHsvColor(Scalar hsvColor) {

                m_mainColor = hsvColor;

                double minH = (hsvColor.val[0] >= mColorRadiusForCompare.val[0]) ? hsvColor.val[0] - mColorRadiusForCompare.val[0] : 0;
                double maxH = (hsvColor.val[0] + mColorRadiusForCompare.val[0] <= 255) ? hsvColor.val[0] + mColorRadiusForCompare.val[0] : 255;

                mLowerBound.val[0] = minH;
                mUpperBound.val[0] = maxH;

                mLowerBound.val[1] = hsvColor.val[1] - mColorRadius.val[1];
                mUpperBound.val[1] = hsvColor.val[1] + mColorRadius.val[1];

                mLowerBound.val[2] = hsvColor.val[2] - mColorRadius.val[2];
                mUpperBound.val[2] = hsvColor.val[2] + mColorRadius.val[2];

                mLowerBound.val[3] = 0;
                mUpperBound.val[3] = 255;

                minH = (hsvColor.val[0] >= mColorRadiusForCompare.val[0]) ? hsvColor.val[0] - mColorRadiusForCompare.val[0] : 0;
                maxH = (hsvColor.val[0] + mColorRadiusForCompare.val[0] <= 255) ? hsvColor.val[0] + mColorRadiusForCompare.val[0] : 255;

                mLowerBoundForCompare.val[0] = minH;
                mUpperBoundForCompare.val[0] = maxH;

                mLowerBoundForCompare.val[1] = hsvColor.val[1] - mColorRadiusForCompare.val[1];
                mUpperBoundForCompare.val[1] = hsvColor.val[1] + mColorRadiusForCompare.val[1];

                mLowerBoundForCompare.val[2] = hsvColor.val[2] - mColorRadiusForCompare.val[2];
                mUpperBoundForCompare.val[2] = hsvColor.val[2] + mColorRadiusForCompare.val[2];

                mLowerBoundForCompare.val[3] = 0;
                mUpperBoundForCompare.val[3] = 255;

//        Mat spectrumHsv = new Mat(1, (int) (maxH - minH), CvType.CV_8UC3);
//
//        for (int j = 0; j < maxH - minH; j++) {
//            byte[] tmp = {(byte) (minH + j), (byte) 255, (byte) 255};
//            spectrumHsv.put(0, j, tmp);
//        }
//
//        Imgproc.cvtColor(spectrumHsv, mSpectrum, Imgproc.COLOR_HSV2RGB_FULL, 4);
    }

    public Mat getSpectrum() {
        return mSpectrum;
    }

    public void setMinContourArea(double area) {
        mMinContourArea = area;
    }

    public double process(Mat rgbaImage) {
        mContours.clear();
        Imgproc.pyrDown(rgbaImage, mPyrDownMat);
        Imgproc.pyrDown(mPyrDownMat, mPyrDownMat);

        Imgproc.cvtColor(mPyrDownMat, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL);

        Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask);
        Imgproc.dilate(mMask, mDilatedMask, new Mat());

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        Imgproc.findContours(mDilatedMask, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        if (0 == contours.size()) return 0;

        // Find max contour area
        double maxArea = 0;
        int index = 0;
        Log.d("TAG", "process: m_maxIndex = -1;");
        m_maxIndex = 0;
        Log.d("findMainColor", "findMainColor: " + contours.size());
        Iterator<MatOfPoint> each = contours.iterator();
        while (each.hasNext()) {
            MatOfPoint wrapper = each.next();
            double area = Imgproc.contourArea(wrapper);
            if (0 == area) { // FIXME: 2016/9/9 注意：area有可能为0！！
                Log.d("findMainColor", "findMainColor: area equal 0!");
//                return 0;
            }
            if (area > maxArea) {
                maxArea = area;
                m_maxIndex = index;
            }
            index++;
        }

        // Filter contours by area and resize to fit the original image size
//        each = contours.iterator();
//        while (each.hasNext()) {
//            MatOfPoint contour = each.next();
//            if (Imgproc.contourArea(contour) > mMinContourArea * maxArea) {
//                Core.multiply(contour, new Scalar(4, 4), contour);
//                mContours.add(contour);
//            }
//        }
        Log.d("TAG", "process: mContours.add(contours.get(m_maxIndex));");
        MatOfPoint contour = contours.get(m_maxIndex);
        Core.multiply(contour, new Scalar(4, 4), contour);
        mContours.add(contour);

        return maxArea;
    }

    public List<MatOfPoint> getContours() {
        return mContours;
    }

    public Scalar findMainColor(Mat mRgba, boolean colorSetted) {

        /**
         * If colorSetted is true, then we can use the exist color to reduce the
         * probability of finding out wrong main color.
         */
        if (colorSetted) {
            process(mRgba);
            if (0 == mContours.size()) {
                return null;
            }
            Rect rlt = boundingRect(mContours.get(0));
            if (rlt.width < 5 || rlt.height < 5) return null;

            mRgba = mRgba.submat(rlt);
        }

        m_maxArea = 0;
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        int colsPerCell = cols / m_cols;
        int rowsPerCell = rows / m_rows;

        // Calculate number of cells discarding head and tail.
//        int verticalNum = m_rows - 2;
//        int horizontalNum = m_cols - 2;

        Scalar mainColor = new Scalar(0);
        Mat RegionRgba = new Mat(), RegionHsv = new Mat();
        Scalar previousMainColor=m_mainColor;
        for (int i = 1; i < m_rows - 1; i++) {
            for (int j = 1; j < m_cols - 1; j++) {
                Rect rect = new Rect();

                rect.x = colsPerCell * j;
                rect.y = rowsPerCell * i;
                rect.width = colsPerCell;
                rect.height = rowsPerCell;

                RegionRgba = mRgba.submat(rect);

                RegionHsv = new Mat();
                Imgproc.cvtColor(RegionRgba, RegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

                // Calculate average color of touched region
                Scalar colorHsv = Core.sumElems(RegionHsv);
                int pointCount = rect.width * rect.height;
                for (int k = 0; k < colorHsv.val.length; k++)
                    colorHsv.val[k] /= pointCount;

                setHsvColor(colorHsv);
                double area = process(mRgba);
                if (m_maxArea < area) {
                    m_maxArea = area;
                    mainColor = colorHsv;
                }
            }
        }
//        setHsvColor(mainColor);
        setHsvColor(previousMainColor);
        RegionRgba.release();
        RegionHsv.release();

        return mainColor;
    }

    public boolean Compare(Scalar color){
        if((color.val[0]<mLowerBoundForCompare.val[0])||(color.val[0]>mUpperBoundForCompare.val[0]))return false;
        if((color.val[1]<mLowerBoundForCompare.val[1])||(color.val[1]>mUpperBoundForCompare.val[1]))return false;
        if((color.val[2]<mLowerBoundForCompare.val[2])||(color.val[2]>mUpperBoundForCompare.val[2]))return false;
        if((color.val[3]<mLowerBoundForCompare.val[3])||(color.val[3]>mUpperBoundForCompare.val[3]))return false;

        return true;
    }
}
