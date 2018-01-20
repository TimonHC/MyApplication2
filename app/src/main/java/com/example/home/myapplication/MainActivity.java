package com.example.home.myapplication;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    public static final String TAG = "src";
    //проверяем, загрузилась ли наша либа
    static {
        if (!OpenCVLoader.initDebug()) {
            Log.wtf(TAG, "OpenCV failed to load!");
        }
    }

    private JavaCameraView cameraView;
    //колбэк от загрузчика библиотеки
    private BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
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
        cameraView = (JavaCameraView) findViewById(R.id.cameraview);
        cameraView.setCvCameraViewListener(this);
        cameraView.setMaxFrameSize(1280, 720);
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
        //матрица Input обращенная в grayscale
        Mat input = inputFrame.gray();
        //Матрица хранящая 3 значения : x_{c}, y_{c}, r для каждого найденного круга.
        Mat circles = new Mat();
        Mat contours = new Mat();
        //используем размытие, чтоб уменьшить шум и погрешность при поиске кругов
        Imgproc.blur(input, input, new Size(7, 7), new Point(2, 2));

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
        Imgproc.HoughCircles(input, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 100, 100, 90, 0, 1000);



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
                Imgproc.circle(input, center, 3, new Scalar(255, 255, 255), 5);
                // внешний край
                Imgproc.circle(input, center, radius, new Scalar(255, 255, 255), 2);
            }
        }
        //очищает память
        circles.release();
        input.release();
        //возвращает кадр в RGB
        return inputFrame.rgba();
    }
}