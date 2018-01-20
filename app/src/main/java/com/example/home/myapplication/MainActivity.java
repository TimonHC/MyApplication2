package com.example.home.myapplication;

import android.content.Context;
import android.hardware.Camera;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.AttributeSet;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;

import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    public static final String TAG = "ShapeDetect";
    //проверяем, загрузилась ли наша либа
    static {
        if (!OpenCVLoader.initDebug()) {
            Log.wtf(TAG, "OpenCV failed to load!");
        }
    }

    private JavaCameraView cameraView;
    private MatOfPoint2f approximatedCurve;
    private Mat contourImage;
    private Mat edges;
    private Mat hierarchyOutputVector;
    private Mat downscaledImage;
    private Mat upscaledImage;
    private Mat grayFrame;
    private Mat circlesFrame;
    private Mat circles;

    //колбэк от загрузчика библиотеки
    private BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    approximatedCurve = new MatOfPoint2f();

                    contourImage = new Mat();
                    edges = new Mat();
                    hierarchyOutputVector = new Mat();
                    downscaledImage = new Mat();
                    upscaledImage = new Mat();
                    grayFrame = new Mat();
                    circlesFrame = new Mat();
                    circles = new Mat();

                    cameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //задаем вьюху для приложения
        cameraView = findViewById(R.id.cameraview);
        cameraView.setCvCameraViewListener(this);
        cameraView.setMaxFrameSize(1024, 768);
    }

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, loaderCallback);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null)
            cameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }


    @Override
    public void onCameraViewStopped() {

    }



    //
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat rgbaFrame = inputFrame.rgba();
        //матрица Input обращенная в grayscale
        inputFrame.gray().copyTo(grayFrame);
        inputFrame.gray().copyTo(circlesFrame);

        //Матрица хранящая 3 значения : x_{c}, y_{c}, r для каждого найденного круга.

        //используем размытие, чтоб уменьшить шум и погрешность при поиске кругов

        // down-scale and upscale the image to filter out the noise
        Imgproc.pyrDown(grayFrame, downscaledImage, new Size(grayFrame.cols()/2, grayFrame.rows()/2));
        Imgproc.pyrUp(downscaledImage, upscaledImage, grayFrame.size());

        Imgproc.Canny(upscaledImage, edges, 0, 255);
        Imgproc.dilate(edges, edges, new Mat(), new Point(-1, 1), 1);

        // find contours and store them all as a list
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(
                edges,
                contours,
                hierarchyOutputVector,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        // loop over all found contours
        for (MatOfPoint contour : contours) {

            MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());

            // approximates a polygonal curve with the specified precision
            Imgproc.approxPolyDP(
                    curve,
                    approximatedCurve,
                    0.02 * Imgproc.arcLength(curve, true),
                    true
            );

            int edgesCount = (int) approximatedCurve.total();

            Log.i(TAG, "Edges count: "+ edgesCount);

            // triangle detection
            if(edgesCount == 3) {
                Log.i(TAG, "triangle found");

                draw(rgbaFrame, approximatedCurve);

                //setLabel(rgbaFrame, contour);
            }

            if (edgesCount == 4) {

                Log.i(TAG, "looking for rectangle");

                List<Double> cos = new ArrayList<>();
                for (int j = 2; j < 4; j++) {
                    cos.add(
                            angle(
                                    approximatedCurve.toArray()[j % 4],
                                    approximatedCurve.toArray()[j - 2],
                                    approximatedCurve.toArray()[j - 1]
                            )
                    );
                }
                Collections.sort(cos);

                double mincos = cos.get(0);
                double maxcos = cos.get(1);

                if (mincos >= -0.1 && maxcos <= 0.3) {
                    Log.i(TAG, "rectangle found");
                    draw(rgbaFrame, approximatedCurve);
                }

            }
            }


        /*
        input: входящее изображение в grayscale
        circles: Матрица хранящая 3 значения : x_{c}, y_{c}, r для каждого найденного круга.
        CV_HOUGH_GRADIENT: Задает метод распознания (градиентный метод Хафа)
        dp = 2: The inverse ratio of resolution
        min_dist = 100: Минимальное расстояние между обнаруженными центрами
        param_1 = 100: Верхний порог для краевого детектора Canny
        param_2 = 90: Порог для обнаружения центра
        min_radius = 0: Минимальный радиус обнаружения
        max_radius = 1000: Максимальный радиус обнаружения*/

        Imgproc.blur(circlesFrame, circlesFrame, new Size(7, 7), new Point(2, 2));

        Imgproc.HoughCircles(circlesFrame, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 100, 125, 125, 0, 1000);

        Log.i(TAG, String.valueOf("size: " + circles.cols()) + ", " + String.valueOf(circles.rows()));

        // Рисует круги, если они распознаны
        if (circles.cols() > 0) {
            for (int x=0; x < Math.min(circles.cols(), 5); x++ ) {
                double circleVec[] = circles.get(0, x);

                if (circleVec == null) {
                    break;
                }

                Point center = new Point((int) circleVec[0], (int) circleVec[1]);
                int radius = (int) circleVec[2];

                // центр круга
                Imgproc.circle(rgbaFrame, center, 3, new Scalar(255, 255, 255), 5);
                // внешний край
                Imgproc.circle(rgbaFrame, center, radius, new Scalar(255, 255, 255), 2);
            }
        }

        //очищает память
//        releaseMemory();
        
        //возвращает кадр в RGB
        return rgbaFrame;
        //return dst;
    }

    private void draw(Mat rgbaFrame, MatOfPoint2f curve) {
        List<MatOfPoint> contours = new ArrayList<>();
        contours.add(new MatOfPoint(curve.toArray()));

        Imgproc.drawContours(rgbaFrame, contours, 0, new Scalar(255, 255, 255), 3);
    }


    private void releaseMemory(){
        circles.release();
        grayFrame.release();
        contourImage.release();
        edges.release();
        hierarchyOutputVector.release();
        downscaledImage.release();
        upscaledImage.release();
        grayFrame.release();
        circles.release();
    }

    private static double angle(Point pt1, Point pt2, Point pt0)
    {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2)
                / Math.sqrt(
                (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
        );
    }
}

