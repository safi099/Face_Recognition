package com.developer.iamsafi.face_recognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.AsyncTask;
import android.util.Log;
import android.util.SparseArray;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;

public class FaceOverLay {
    private final Context c;
    private Bitmap mBitmap;
    private SparseArray<Face> mFaces;

    public FaceOverLay(Context context) {
        c = context;
    }


    public Bitmap setBitmap(Bitmap bitmap) {
        mBitmap = bitmap;
        FaceDetector detector = new FaceDetector.Builder(c)
                .setTrackingEnabled(true)
                .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                .setMode(FaceDetector.ACCURATE_MODE)
                .build();
        Log.i("check", "Dector is operational " + (!detector.isOperational()));
        if (detector.isOperational()) {

            //Handle contingency        } else {
            Frame frame = new Frame.Builder().setBitmap(bitmap).build();
            mFaces = detector.detect(frame);
            detector.release();
            return CropFace();
        }


        return null;
    }


    protected Bitmap CropFace() {


        if ((mBitmap != null) && (mFaces != null)) {
            Log.i("check", "Entered in Body og");
            return drawBitmap();
        }
        Log.i("check", "No values Entered " + "Bitmap is " + (mBitmap != null) + " Faces is " + (mFaces != null));

        return null;
    }

    private Bitmap drawBitmap() {
//        double viewWidth = canvas.getWidth();
//        double viewHeight = canvas.getHeight();
        double imageWidth = mBitmap.getWidth();
        double imageHeight = mBitmap.getHeight();
        int size = 200;
        float left = 0;
        float top = 0;
        float right = 0;
        float bottom = 0;
        float width = 0;
        float height = 0;
        for (int i = 0; i < mFaces.size(); i++) {
            Face face = mFaces.valueAt(i);
            left = (float) (face.getPosition().x);
            top = (float) (face.getPosition().y);
            right = (float) (face.getPosition().x + face.getWidth());
            width = face.getWidth();
            height = face.getHeight();
            bottom = (float) (face.getPosition().y + face.getHeight());
        }
        Rect src = new Rect((int) left, (int) top, (int) right, (int) bottom);
        Rect dst = new Rect(0, 0, size, size);
        Bitmap faceBitmap = Bitmap.createBitmap(mBitmap, (int) left, (int) top, (int) width, (int) height);
//        canvas.drawBitmap(mBitmap, src, dst, null);
        return faceBitmap;
    }

}
