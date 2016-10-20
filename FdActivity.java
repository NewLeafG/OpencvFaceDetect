package org.opencv.samples.facedetect;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.webkit.MimeTypeMap;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

import com.muse.imgselector.ImgSelectorActivity;
import com.muse.imgselector.utils.MediaScanner;
import com.nostra13.universalimageloader.cache.disc.naming.Md5FileNameGenerator;
import com.nostra13.universalimageloader.core.ImageLoader;
import com.nostra13.universalimageloader.core.ImageLoaderConfiguration;
import com.nostra13.universalimageloader.core.assist.QueueProcessingType;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.samples.colorblobdetect.ColorBlobDetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;
import static org.opencv.imgproc.Imgproc.warpAffine;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";

    static {
        Log.i(TAG, "Load OpenCV library!");
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV load not successfully!");
        } else {
            System.loadLibrary("detection_based_tracker");
        }
    }

    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar FACE_RECT_COLOR_REC = new Scalar(0.5, 0, 255, 255);
    private static final Scalar WHITE_COLOR = new Scalar(255, 255, 255, 255);
    public static final int JAVA_DETECTOR = 0;
    public static final int NATIVE_DETECTOR = 1;
    public static List<Camera.Size> m_previewSizes;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mItemType;

    private Mat mRgba;
    private Mat mRgbaRaw;
    private Mat mGray;
    private Mat mGrayRaw;
    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mJavaDetectorEye;
    private CascadeClassifier mJavaDetectorNose;
    private DetectionBasedTracker mNativeDetector;
    private DetectionBasedTracker mNativeDetectorProfile;

    private int mDetectorType = NATIVE_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CamView mOpenCvCameraView;

    private String m_ImgDir;
    private boolean m_bSaveFaceImg = false;
    private static Mat m_roiImage;
    private boolean m_bModelTrainded = false;
    private double m_area = 0;
    private boolean m_bFRONTCAMERA = false;
    private ColorBlobDetector m_Detector;
    private Scalar CONTOUR_COLOR = new Scalar(255, 0, 0, 255);
    private double m_threshold = 0.02;
    private static FaceTracker m_faceTracker;
    private double m_fontScale = 1;
//    private static final int COLOR_COUNT = 100;
//    private static int m_colorCount = COLOR_COUNT;

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CamView) findViewById(R.id.fd_activity_surface_view);

        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        ImageButton imgBtn = (ImageButton) findViewById(R.id.btnImgSelector);
        imgBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onBtnImgSltClicked(v);
            }
        });
        imgBtn = (ImageButton) findViewById(R.id.btn_takePhoto);
        imgBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onTakePhotoClicked(v);
            }
        });
        imgBtn = (ImageButton) findViewById(R.id.btn_switchCam);
        imgBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onSwitchCamClicked(v);
            }
        });

        Button btn=(Button)findViewById(R.id.btn_faceReset);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onFaceReset(v) ;
            }
        });

        CreateImgDir(); //Fixme: 此处未做防错处理，如果创建目录失败(返回值为false)会导致未知后果!!!

        Init();

        mJavaDetector = loadCascadeFile(R.raw.haarcascade_frontalface_default, "haarcascade_frontalface_default");
        mJavaDetectorEye = loadCascadeFile(R.raw.haarcascade_eye_tree_eyeglasses, "haarcascade_eye_tree_eyeglasses");
        mJavaDetectorNose = loadCascadeFile(R.raw.haarcascade_mcs_nose, "haarcascade_mcs_mouth");

        mNativeDetector = loadNativeCascadeFile(R.raw.haarcascade_frontalface_default, "native_haarcascade_frontalface_default");
        mNativeDetectorProfile = loadNativeCascadeFile(R.raw.haarcascade_profileface, "native_haarcascade_profileface");

        mOpenCvCameraView.enableView();
    }

    private void onFaceReset(View v) {
        Button btn=(Button)v;
        btn.setText("Clicked");
    }


    /**
     * Mainly to set the resolution of the camera.
     */
    private void camInit() {
//        mOpenCvCameraView.disableView();

        // Resolution index of the camera.
        int index = 0;
        if (m_bFRONTCAMERA) {
            index = 4;
        } else {
            index = 5;
        }
        mOpenCvCameraView.setResolution(m_previewSizes.get(index));
        Toast.makeText(FdActivity.this, "Resolution:" +
                        String.valueOf(m_previewSizes.get(index).width) + "*" +
                        String.valueOf(m_previewSizes.get(index).height),
                Toast.LENGTH_LONG).show();

        m_faceTracker = new FaceTracker(m_previewSizes.get(index).width, m_previewSizes.get(index).height);

//        mOpenCvCameraView.enableView();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.enableView();
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mGrayRaw = new Mat();
        mRgba = new Mat();
        mRgbaRaw = new Mat();
        m_previewSizes = mOpenCvCameraView.getResolutionList();
        camInit();
        m_Detector = new ColorBlobDetector();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mGrayRaw.release();
        mRgba.release();
        mRgbaRaw.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        long previous = System.currentTimeMillis();

        mRgba = inputFrame.rgba();
//        mGrayRaw = inputFrame.gray();
//        equalizeHist(mGrayRaw, mGray);
        mGray = inputFrame.gray();
        if (m_bFRONTCAMERA) {
//            mRgbaRaw = inputFrame.rgba();
            Core.flip(mRgba, mRgba, 1);
            Core.flip(mGray, mGray, 1);
        }
        // Minize the size for faster processing.
//        Imgproc.resize(mGray,mGray,new Size(),0.5,0.5,1);
//        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();
        Rect target = new Rect();
        Rect[] facesArray;


        if (m_faceTracker.m_faceLocated) {
            Log.d(TAG, "onCameraFrame: Null returned! m_faceLocated!");
            Rect searchArea = m_faceTracker.GetSearchArea();
//            Mat searchROI = new Mat(mGray, searchArea);
            mNativeDetector.detect(mGray, faces);

            facesArray = faces.toArray();
            if (facesArray.length > 0) {
                Log.d(TAG, "onCameraFrame: Searching face...");
//                Log.d(TAG, "GetBestMatch: rectTwo "+String.valueOf(searchROI.isSubmatrix()));
                target = m_faceTracker.GetBestMatch(searchArea, facesArray);
                if (m_faceTracker.Verify(target)) {
                    m_faceTracker.UpdateData(target, mRgba, false);
//                Log.d(TAG, "onCameraFrame: Searching face ended.");
                    Log.d(TAG, "onCameraFrame: rectSnd " + String.valueOf(target.tl().x));
                } else {
                    m_faceTracker.m_bSearchColor = true;
                }
//                m_colorCount = COLOR_COUNT;
            } else {
                m_faceTracker.m_bSearchColor = true;
            }
            if (m_faceTracker.m_bSearchColor) {
                Log.d(TAG, "onCameraFrame: Searching color...");
                m_faceTracker.m_bSearchColor = false;
                target = m_faceTracker.SearchColor(new Mat(mRgba, searchArea));
                if ((null == target)) {
                    Log.d(TAG, "onCameraFrame: Null returned!");
                    m_faceTracker.m_faceLocated = false;
//                    m_faceTracker.m_colorSet = false;
                    return mRgba;
                }
//                m_colorCount--;
                target.x += searchArea.x;
                target.y += searchArea.y;
                m_faceTracker.UpdateData(target, mRgba, false);
                mRgba = m_faceTracker.drawContour(mRgba);
            }
        } else {
            Log.d(TAG, "onCameraFrame: Null returned! !m_faceLocated");
            mNativeDetector.detect(mGray, faces);
            facesArray = faces.toArray();
            int len = facesArray.length;
            if (len > 0) {
                Rect[] verifiedFaces;
                int verifiedIndex[] = new int[len];
                int verifiedCnt = 0;

                for (int i = 0; i < len; i++) {
                    if ( VerifyFace(facesArray[i])) {
                        verifiedIndex[verifiedCnt] = i;
                        verifiedCnt++;
                    }
                }
                if (0 == verifiedCnt) {
                    return mRgba;
                }

                verifiedFaces = new Rect[verifiedCnt];
                for (int m = 0; m < verifiedCnt; m++) {
                    verifiedFaces[m] = facesArray[verifiedIndex[m]];
                }
                target = m_faceTracker.HandleFaces(mGray, verifiedFaces);
                m_faceTracker.UpdateData(target, mRgba, true);
                Log.d(TAG, "onCameraFrame: rectFst " + String.valueOf(target.tl().x));
            } else {
                // // TODO: 2016/9/10 因为侧脸易误检，故暂时移除。 
//                mNativeDetectorProfile.detect(mGray, faces);
//                facesArray = faces.toArray();
//                int len0 = facesArray.length;
//                if (len0 > 0) {
//                    target = m_faceTracker.HandleFaces(mGray, facesArray);
//                    m_faceTracker.UpdateData(target, mRgba, true);
//                    Log.d(TAG, "onCameraFrame: Null returned! Profile!!");
//                }else{
//                    // TODO: 2016/9/10 应添加图片翻转后的侧脸检测。 
////                    Mat temp = mRgba.clone();
////                    Core.flip(mRgba, temp);
//                }

                return mRgba;
            }
        }

        if (!VerifyTarget(target)) {
            m_faceTracker.m_faceLocated = false;
            m_faceTracker.m_colorSet = false;

            return mRgba;
        }

        Imgproc.rectangle(mRgba, target.tl(), target.br(), FACE_RECT_COLOR_REC, -1);
        Imgproc.putText(mRgba, String.valueOf(FaceTracker.m_lastPosition), new Point((target.tl().x + target.br().x) / 2, (target.tl().y + target.br().y) / 2), 3, m_fontScale, WHITE_COLOR, 3);
        Imgproc.putText(mRgba, String.valueOf(target.area()), new Point(target.tl().x, target.tl().y), 3, m_fontScale, WHITE_COLOR, 1);

        long current = System.currentTimeMillis();
        long elapsed = current - previous;
        Imgproc.putText(mRgba, String.format("FPS = %5.2f",1000.0 / elapsed), new org.opencv.core.Point(20, 30), 3, 1, new Scalar(255, 255, 255, 255), 3);

//        m_Detector.findMainColor(new Mat(mRgba, target));
//        m_Detector.process(mRgba);
//        List<MatOfPoint> contours = m_Detector.getContours();
//        Log.e(TAG, "Contours count: " + contours.size());
//        Imgproc.drawContours(mRgba, contours, -1, CONTOUR_COLOR, 6);




  /*      if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        } else {
            Log.e(TAG, "Detection method is not selected!");
        }

        facesArray = faces.toArray();

        if (true) {
            for (int i = 0; i < facesArray.length; i++) {

                int len = facesArray[i].height > facesArray[i].width ?
                        facesArray[i].width : facesArray[i].height;

                // Find eyes
                MatOfRect eyes = new MatOfRect();

                Mat faceROI = new Mat(mGray, facesArray[i]);
                if (mJavaDetectorEye != null)
                    mJavaDetectorEye.detectMultiScale(faceROI, eyes, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                            new Size(mAbsoluteFaceSize * 0.3, mAbsoluteFaceSize * 0.3), new Size());

                Rect[] eyesArray = eyes.toArray();

                // 图片截取
                final Rect rect = new Rect(facesArray[i].x, facesArray[i].y, len, len);
                m_roiImage = new Mat(mRgba, rect);

                // 缩放为同一尺寸，因为faceRecognizer需要这样的图片
//                Imgproc.resize(m_roiImage, m_roiImage, new Size(200, 200));

                double confidence = 0;
               if (2 == eyesArray.length) {
                    // Judge whether this two rect are overlapped.
                    if (eyesArray[0].tl().x < eyesArray[1].tl().x && eyesArray[0].br().x > eyesArray[1].tl().x) {
                        Log.d(TAG, "onCameraFrameError: Overlap detected!");
                        break;
                    }
                    if (eyesArray[1].tl().x < eyesArray[0].tl().x && eyesArray[1].br().x > eyesArray[0].tl().x) {
                        Log.d(TAG, "onCameraFrameError: Overlap detected!");
                        break;
                    }

                    Point centRight = new Point((eyesArray[0].tl().x + eyesArray[0].br().x) / 2,
                            (eyesArray[0].tl().y + eyesArray[0].br().y) / 2);
                    Point centLeft = new Point((eyesArray[1].tl().x + eyesArray[1].br().x) / 2,
                            (eyesArray[1].tl().y + eyesArray[1].br().y) / 2);
                    Point[] landMarks = new Point[]{centLeft, centRight};
                    if (centLeft.x > centRight.x) {
                        landMarks = new Point[]{centRight, centLeft};
                    }
                    final Mat alignedFace = m_roiImage;//getwarpAffineImg(m_roiImage, landMarks);
                    if (null == alignedFace) break;
                    Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
                    for (int m = 0; m < eyesArray.length; m++) {
                        Imgproc.rectangle(mRgba, new Point(eyesArray[m].tl().x + facesArray[i].tl().x, eyesArray[m].tl().y + facesArray[i].tl().y),
                                new Point(eyesArray[m].br().x + facesArray[i].tl().x, eyesArray[m].br().y + facesArray[i].tl().y), FACE_RECT_COLOR, 3);
                    }
                    if (m_bSaveFaceImg) {
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                TakePhoto(alignedFace);
                            }
                        });
                        m_bSaveFaceImg = false;
                    }
//                    confidence = judge(alignedFace.getNativeObjAddr());
                }

                if (m_bModelTrainded) {
//                Mat bgrMat=new Mat(m_roiImage.rows(), m_roiImage.cols(), CvType.CV_8UC3);
//                Imgproc.cvtColor(m_roiImage, bgrMat, Imgproc.COLOR_RGBA2BGR, 3);
                    Log.d(TAG, "onCameraFrame: 用于Train的图片是经过颜色转换的，所以用于predict的图片也要进行颜色转换！");
//                equalizeHist(m_roiImage, m_roiImage);
                    if (true) {
                        m_Detector.findMainColor(m_roiImage);
//                    Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR_REC, -1);
                        m_area = facesArray[i].area();
                        Imgproc.putText(mRgba, String.valueOf(facesArray[i].width), new Point((facesArray[i].tl().x + facesArray[i].br().x) / 2, (facesArray[i].tl().y + facesArray[i].br().y) / 2), 3, 2, WHITE_COLOR, 3);
                        Imgproc.putText(mRgba, String.valueOf(confidence), new Point(facesArray[i].tl().x, facesArray[i].tl().y), 3, 2, WHITE_COLOR, 1);
                        m_Detector.process(mRgba);
                        List<MatOfPoint> contours = m_Detector.getContours();
                        Log.e(TAG, "Contours count: " + contours.size());
                        Imgproc.drawContours(mRgba, contours, -1, CONTOUR_COLOR, 6);
                    }
                }
            }
        } else {
            for (int i = 0; i < facesArray.length; i++) {
                Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            }
        }*/

        return mRgba;
    }

    private boolean VerifyTarget(Rect target) {
        double proportion = target.width * 1.0 / target.height;
        if ((proportion > 1.5))
            return false;
        if ((proportion < 0.2)) {
            Log.d(TAG, "VerifyTarget: " + target.width + "  " + target.height + "  " + target.width / target.height);
            return false;
        }

        return true;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }

    /**
     * 跳转到face recogniser界面
     *
     * @param view
     */
    public void onBtnImgSltClicked(View view) {
        Intent slt = new Intent(FdActivity.this, ImgSelectorActivity.class);
        startActivityForResult(slt, 0);
    }

    /**
     * 拍取face
     *
     * @param view
     */
    public void onTakePhotoClicked(View view) {

        m_bSaveFaceImg = true;
    }

    /**
     * 切换前后相机，注意：默认手机具有前后两个摄像头
     *
     * @param view
     */
    public void onSwitchCamClicked(View view) {

        m_bFRONTCAMERA = !m_bFRONTCAMERA;
        mOpenCvCameraView.disableView();

        if (m_bFRONTCAMERA) {
            mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        } else {
            mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
        }

        mOpenCvCameraView.enableView();
    }

    /**
     * Create directories for this program.
     *
     * @return
     */
    private boolean CreateImgDir() {
        String exStorSta = Environment.getExternalStorageState();
        if (exStorSta.equals(Environment.MEDIA_MOUNTED)) {
            String sdcardRoot = Environment.getExternalStorageDirectory().getAbsolutePath();

            // 创建存放face图片的目录
            File folder = new File(sdcardRoot + "/FaceDR/Face_Img");
            if (!folder.exists()) {
                if (folder.mkdirs()) {
                    Toast.makeText(getApplicationContext(), "目录已创建", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(getApplicationContext(), "Failed to create directory!", Toast.LENGTH_LONG).show();
                    return false;
                }
            }

            m_ImgDir = folder.getAbsolutePath();
            return true;
        } else {
            Toast.makeText(this, "外部存储加载异常！", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    /**
     * 1.初始化ImageLoader。
     * 2.初始化人脸识别器。
     */
    private void Init() {
        ImageLoaderConfiguration config = new ImageLoaderConfiguration.Builder(this)
                .threadPriority(Thread.NORM_PRIORITY - 2)
                .denyCacheImageMultipleSizesInMemory()
                .discCacheFileNameGenerator(new Md5FileNameGenerator())
                .tasksProcessingOrder(QueueProcessingType.LIFO)
                .memoryCacheExtraOptions(96, 120)
                .build();
        // Initialize ImageLoader with configuration.
        ImageLoader.getInstance().init(config);

        // Extract the root directory of the file of this program;
        int index = m_ImgDir.lastIndexOf('/');
        String root = m_ImgDir.substring(0, index + 1);
        // Check whether the file of face model is extant.
        File modelFile = new File(root + "face-rec-model.txt");//Fixme: The name of model file is hardcoded when it was being saved @trainRecognizer.
        long fileSize = Integer.parseInt(String.valueOf(modelFile.length()));
        if (modelFile.exists() && fileSize > 0) {
            m_bModelTrainded = true;
            loadRecognizer(modelFile.getAbsolutePath(), m_threshold);
        } else {
            Toast.makeText(FdActivity.this, "识别模型未加载！", Toast.LENGTH_LONG).show();
        }
    }

    public void TakePhoto(Mat face) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String fileName = m_ImgDir + "/" + currentDateandTime + ".jpg";
        Mat IntermediateMat = new Mat(face.rows(), face.cols(), CvType.CV_8UC3);
        Imgproc.cvtColor(face, IntermediateMat, Imgproc.COLOR_RGB2GRAY, 1);
//        equalizeHist(IntermediateMat, IntermediateMat);
        Imgcodecs.imwrite(fileName, IntermediateMat);

        MediaScanner mediaScanner = new MediaScanner(this);
        String[] filePaths = new String[]{fileName};
        String[] mimeTypes = new String[]{MimeTypeMap.getSingleton().getMimeTypeFromExtension("jpg")};
        mediaScanner.scanFiles(filePaths, mimeTypes);

        Toast.makeText(this, fileName + "has been saved & added to mediaStore.", Toast.LENGTH_LONG).show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        String msg;
        if (resultCode == RESULT_OK) {
            createCSVFile();
            msg = trainRecognizer(m_ImgDir + "/../csv.ext", m_threshold);
            if (msg.equals("Done")) {
                m_bModelTrainded = true;
                Log.d(TAG, "onActivityResult: test");
            }
        }
    }

    private void createCSVFile() {
        File csvFile = new File(m_ImgDir + "/..", "csv.ext");
        FileOutputStream os;
        try {
            os = new FileOutputStream(csvFile);
            int num = ImgSelectorActivity.m_imgPaths.length;
            String separater = ";";
            String label = "0";
            for (int i = 0; i < num; i++) {
                os.write(ImgSelectorActivity.m_imgPaths[i].getBytes());
                os.write(separater.getBytes());
                os.write(label.getBytes());
                os.write("\n".getBytes());
            }
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(FdActivity.this, e.getMessage(), Toast.LENGTH_LONG).show();
            csvFile.delete();
            return;
        }

    }

    private static native String trainRecognizer(String csvPath, double threshold);

    private static native String loadRecognizer(String modelPath, double threshold);

    private static native double judge(long faceMat);


    /**
     * 根据眼睛坐标对图像进行仿射变换,算法参考：http://blog.csdn.net/yanyan_xixi/article/details/36372901
     *
     * @param src       Face image.
     * @param landmarks The center of two eyes.
     * @return null if calculated angle is bigger than 30.
     */
    Mat getwarpAffineImg(Mat src, Point[] landmarks) {
        Mat oral = new Mat();
        src.copyTo(oral);

        //计算两眼中心点，按照此中心点进行旋转， 第0个为左眼坐标，1为右眼坐标
        Point eyesCenter = new Point((landmarks[0].x + landmarks[1].x) * 0.5f, (landmarks[0].y + landmarks[1].y) * 0.5f);

        // 计算两个眼睛间的角度
        double dy = (landmarks[1].y - landmarks[0].y);
        double dx = (landmarks[1].x - landmarks[0].x);
        double angle = Math.atan2(dy, dx) * 180.0 / Math.PI; // Convert from radians to degrees.

        if (Math.abs(angle) > 30) {
//            Toast.makeText(FdActivity.this,"Angle:"+String.valueOf(angle),Toast.LENGTH_SHORT).show();
            Log.d(TAG, "getwarpAffineImg: " + String.valueOf(angle));
            return null;
        }

        //由eyesCenter, angle, scale按照公式计算仿射变换矩阵，此时1.0表示不进行缩放
        Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, 1.0);
        Mat rot = new Mat();
        // 进行仿射变换，变换后大小为src的大小
        warpAffine(src, rot, rot_mat, src.size());

        /**
         * 寻找nose的位置，注意：这一步必须在图片经过水平校正之后。
         */
        // Find noses
        MatOfRect mouths = new MatOfRect();

        if (mJavaDetectorNose != null)
            mJavaDetectorNose.detectMultiScale(rot, mouths, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize * 0.1, mAbsoluteFaceSize * 0.1), new Size());

        Rect[] mouthsArray = mouths.toArray();
        boolean mouthDetected = false;
        Point noseCentre = new Point();
        if (1 == mouthsArray.length) {
            eyesCenter.x = mouthsArray[0].x + mouthsArray[0].width / 2;
            double xLeft = mouthsArray[0].x;
            double xRight = mouthsArray[0].x + mouthsArray[0].width;
            double yTop = mouthsArray[0].y;
            double yBottom = mouthsArray[0].y + mouthsArray[0].height;
            noseCentre.x = (xLeft + xRight) / 2;
            noseCentre.y = (yBottom + yTop) / 2;
//            Imgproc.rectangle(rot, new Point(xLeft, yTop),
//                    new Point(xRight, yBottom), FACE_RECT_COLOR, 3);
//            Imgproc.line(rot, new Point(xLeft, yTop), new Point(xRight, yBottom), FACE_RECT_COLOR, 3);
//            Imgproc.line(rot, new Point(xLeft, yBottom), new Point(xRight, yTop), FACE_RECT_COLOR, 3);
            mouthDetected = true;
        }

        // 变换后的左眼中心点
        Point p = new Point(0, 0);
        p.x = rot_mat.get(0, 0)[0] * landmarks[0].x + rot_mat.get(0, 1)[0] * landmarks[0].y + rot_mat.get(0, 2)[0];
        // 变换后的右眼中心点
        Point pr = new Point(0, 0);
        pr.x = rot_mat.get(0, 0)[0] * landmarks[1].x + rot_mat.get(0, 1)[0] * landmarks[1].y + rot_mat.get(0, 2)[0];
        // 计算出双眼水平距离的一半，将其作为一个基尺，因为就算眼球转动，该距离也是相对稳定的。
        double unitLen = (pr.x - p.x) / 2;
        // Crop the image
        Rect rect = new Rect();
        if (mouthDetected) {
            int left = (int) (eyesCenter.x - unitLen * 1.5);
            rect.x = left < 0 ? 0 : left;
            int top = (int) (noseCentre.y - unitLen * 1.5);
            rect.y = top < 0 ? 0 : top;
            int width = (int) (eyesCenter.x - rect.x) * 2;
            rect.width = rect.x + width > rot.cols() ? rot.cols() - rect.x : width;
            int height = (int) unitLen * 3;
            rect.height = rect.y + height > rot.rows() ? rot.rows() - rect.y : height;
        } else {
            int left = (int) (eyesCenter.x - unitLen * 1.5);
            rect.x = left < 0 ? 0 : left;
            int top = (int) (eyesCenter.y - unitLen * 0.5);
            rect.y = top < 0 ? 0 : top;
            int width = (int) (eyesCenter.x - rect.x) * 2;
            rect.width = rect.x + width > rot.cols() ? rot.cols() - rect.x : width;
            int height = (int) unitLen * 4;
            rect.height = rect.y + height > rot.rows() ? rot.rows() - rect.y : height;
        }
        Mat roiMat = new Mat(rot, rect);
        Imgproc.resize(roiMat, roiMat, new Size(200, 200));

        return roiMat;
    }

    private CascadeClassifier loadCascadeFile(int resourceId, String casName) {
        try {
            CascadeClassifier javaDetector;
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(resourceId);
            Log.d(TAG, "onCreate: ");
            File cascadeFile;
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            cascadeFile = new File(cascadeDir, casName + ".xml");//haarcascade_frontalface_default
            FileOutputStream os = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            javaDetector = new CascadeClassifier(cascadeFile.getAbsolutePath());
            if (javaDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                javaDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());

            cascadeDir.delete();

            return javaDetector;
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }

        Toast.makeText(FdActivity.this, "Error occurred when loading cascade file!!!", Toast.LENGTH_LONG).show();
        return null;
    }

    private DetectionBasedTracker loadNativeCascadeFile(int resourceId, String casName) {
        try {
            DetectionBasedTracker detector;
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(resourceId);
            Log.d(TAG, "onCreate: ");
            File cascadeFile;
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            cascadeFile = new File(cascadeDir, casName + ".xml");//haarcascade_frontalface_default
            FileOutputStream os = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            detector = new DetectionBasedTracker(cascadeFile.getAbsolutePath(), 0);

            cascadeDir.delete();

            return detector;
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load Native cascade. Exception thrown: " + e);
        }

        Toast.makeText(FdActivity.this, "Error occurred when loading Native cascade file!!!", Toast.LENGTH_LONG).show();
        return null;
    }

    private boolean VerifyFace(Rect faceRect) {

        Mat faceROI;
        if(m_faceTracker.m_colorSet) {
            faceROI= new Mat(mRgba,faceRect);
            return m_faceTracker.VerifyColor(faceROI);
        }

        faceROI= new Mat(mGray,faceRect);
        MatOfRect eyes = new MatOfRect();
        if (mJavaDetectorEye != null)
            mJavaDetectorEye.detectMultiScale(faceROI, eyes, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize * 0.3, mAbsoluteFaceSize * 0.3), new Size());
        Rect[] eyesArray = eyes.toArray();
        if (eyesArray.length > 0) return true;

        // Find noses
        MatOfRect noses = new MatOfRect();
        if (mJavaDetectorNose != null)
            mJavaDetectorNose.detectMultiScale(faceROI, noses, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize * 0.1, mAbsoluteFaceSize * 0.1), new Size());
        Rect[] nosesArray = noses.toArray();
        if (nosesArray.length > 0) return true;


        Log.d(TAG, "VerifyFace: verified!");

        return false;
    }

}
