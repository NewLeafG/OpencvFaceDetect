package org.opencv.samples.facedetect;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.samples.colorblobdetect.ColorBlobDetector;

import java.util.List;

import static org.opencv.imgproc.Imgproc.boundingRect;

/**
 * Created by Muse on 2016/9/6.
 */
public class FaceTracker {
    public boolean m_faceLocated = false;
    public boolean m_colorSetted = false;
    private static Rect m_searchArea = new Rect();
    private static Rect m_target = new Rect();
    public static Point m_lastPosition = new Point();
    private static ColorBlobDetector m_colorDetector = new ColorBlobDetector();
    private static int m_imgWidth = 0;
    private static int m_imgHeight = 0;
    private static float m_searchScale = 1.5f;
    private Scalar CONTOUR_COLOR = new Scalar(255, 0, 0, 255);
    public boolean m_bSearchColor = false;

    public FaceTracker(int imgWidth, int imgHeight) {
        m_imgWidth = imgWidth;
        m_imgHeight = imgHeight;
    }

    public Rect HandleFaces(Mat img, Rect[] facesArray) {
        int num = facesArray.length;
        Point imgCentre = new Point(img.cols() / 2, img.rows() / 2);
        int index = 0;
        // If the number of faces is larger than 1, choose one which is closer to the image's centre.
        if (num > 1) {
            index = GetClosestRect(imgCentre, facesArray);
        }

        return facesArray[index];
    }

    public Rect GetSearchArea() {
        return m_searchArea;
    }

    private Point CalcRectCentre(Rect rect) {
        Point centreP = new Point();
        centreP.x = rect.x + rect.width / 2;
        centreP.y = rect.y + rect.height / 2;

        return centreP;
    }

    private double CalcDistanceOfTwoPoints(Point p0, Point p1) {
        return Math.sqrt((p0.x - p1.x) * (p0.x - p1.x) + (p0.y - p1.y) * (p0.y - p1.y));
    }

    public Rect GetBestMatch(Rect searchArea, Rect[] facesArray) {

        Log.d("NAN", "GetBestMatch: rectThree " + searchArea.x + " " + System.currentTimeMillis());
        int len = facesArray.length;
        if (1 == len) {
            Log.d("NAN", "GetBestMatch: rectFour " + facesArray[0].x + " " + System.currentTimeMillis());
            //return new Rect(facesArray[0].x + searchArea.x, facesArray[0].y + searchArea.y, facesArray[0].width, facesArray[0].height);
            return new Rect(facesArray[0].x, facesArray[0].y, facesArray[0].width, facesArray[0].height);
        } else {
            Rect temp = facesArray[GetClosestRect(m_lastPosition, facesArray)];
            Log.d("NAN", "GetBestMatch: rectFive " + temp.x + " " + System.currentTimeMillis());
            return temp;
        }
    }

    public int GetClosestRect(Point origin, Rect[] rectArray) {
        int num = rectArray.length;
        Point centreP;
        double distance = 0;
        centreP = CalcRectCentre(rectArray[0]);
        double minDistance = CalcDistanceOfTwoPoints(origin, centreP);
        int index = 0;
        for (int i = 1; i < num; i++) {
            centreP = CalcRectCentre(rectArray[i]);
            distance = CalcDistanceOfTwoPoints(centreP, origin);
            if (distance < minDistance) {
                minDistance = distance;
                index = i;
            }
        }

        return index;
    }

    public Rect SearchColor(Mat matROI) {
        m_colorDetector.process(matROI);
        List<MatOfPoint> contours = m_colorDetector.getContours();
        Log.d("ColorCount: ", "ColorCount: " + contours.size());
        if (0 == contours.size()) {
            return null;
        }
        MatOfPoint contour = contours.get(0);
        Rect rlt = boundingRect(contour);
        if (rlt.width < 5 || rlt.height < 5) return null;

        return rlt;
    }

    public void UpdateData(Rect target, Mat rgba, boolean updateColor) {
        m_target = target;
        Point centerP = new Point(target.x + target.width / 2, target.y + target.height / 2);
        m_lastPosition = centerP;
        int width = (int) (target.width * m_searchScale);
        int height = (int) (target.height * m_searchScale);
        int left = (int) (centerP.x - width / 2);
        if (left < 0) {
            left = 0;
        }
        int top = (int) (centerP.y - height / 2);
        if (top < 0) {
            top = 0;
        }
        if (left + width > m_imgWidth) {
            width = m_imgWidth - left;
        }
        if (top + height > m_imgHeight) {
            height = m_imgHeight - top;
        }

//        Log.d("UpdateData: ", "UpdateData: "+String.valueOf(left));
        m_searchArea.x = left;
        m_searchArea.y = top;
        m_searchArea.width = width;
        m_searchArea.height = height;

        if (updateColor) {
            try {
                m_colorDetector.findMainColor(rgba.submat(target), m_colorSetted);
                m_colorSetted = true;
            } catch (Exception e) {
                Log.d("findMainColor", "findMainColor: " + e.getMessage());
            }
        }

        m_faceLocated = true;
    }

    public Mat drawContour(Mat rgba) {
        List<MatOfPoint> contours = m_colorDetector.getContours();
        if (0 == contours.size()) {
            return rgba;
        }
        MatOfPoint contour = contours.get(0);

        Imgproc.drawContours(rgba, contours, -1, CONTOUR_COLOR, 6);
        return rgba;
    }

    public boolean Verify(Rect target) {
        if (IsInRect(CalcRectCentre(target), m_target)) {
            if (target.area() > m_target.area() * 0.5 && target.area() < m_target.area() * 2) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    private boolean IsInRect(Point p, Rect rect) {
        if (rect.tl().x < p.x && p.x < rect.br().x && rect.tl().y < p.y && p.y < rect.br().y) {
            return true;
        } else {
            return false;
        }
    }
}
