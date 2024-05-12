package reconnaissanceFacial;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Camera {
    private static CascadeClassifier faceDetector;
    private static CascadeClassifier eyeDetector;
    private static Net genderNet;
    private static Net ageNet;
    private static CascadeClassifier smileDetector;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        faceDetector = new CascadeClassifier("src/haarcascade_frontalface_default.xml");
        eyeDetector = new CascadeClassifier("src/haarcascade_eye.xml");

        String genderModelTxt = "src/deploy_gender.prototxt";
        String genderModelBin = "src/gender_net.caffemodel";
        smileDetector = new CascadeClassifier("src/haarcascade_smile.xml");
        
        genderNet = Dnn.readNetFromCaffe(genderModelTxt, genderModelBin);

        String ageModelTxt = "src/deploy_age2.prototxt";
        String ageModelBin = "src/age_net.caffemodel";
        ageNet = Dnn.readNetFromCaffe(ageModelTxt, ageModelBin);

        List<File> images = loadImagesFromDirectory("src/images");
        List<String> names = loadNamesFromImages(images);

        VideoCapture capture = new VideoCapture(0);
        JFrame frame = new JFrame("Reconnaissance faciale");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel panel = new JPanel();
        frame.setContentPane(panel);
        JLabel label = new JLabel();
        panel.add(label);
        JButton saveImageButton = new JButton("Enregistrer Image");
        panel.add(saveImageButton);
        frame.pack();
        frame.setVisible(true);

        MatOfRect faceDetections = new MatOfRect();
        Mat webcamImage = new Mat();

        saveImageButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String imageName = JOptionPane.showInputDialog(frame, "Entrez le nom de l'image :", "Enregistrer Image", JOptionPane.PLAIN_MESSAGE);
                if (imageName != null && !imageName.isEmpty()) {
                    saveImage(webcamImage, imageName);
                }
            }
        });

        while (true) {
            capture.read(webcamImage);
            Mat grayImage = new Mat();
            Imgproc.cvtColor(webcamImage, grayImage, Imgproc.COLOR_BGR2GRAY);
            faceDetector.detectMultiScale(grayImage, faceDetections);

            for (Rect rect : faceDetections.toArray()) {
                Mat faceROI = grayImage.submat(rect);
                MatOfRect eyeDetections = new MatOfRect();
                MatOfRect smileDetections = new MatOfRect();

                eyeDetector.detectMultiScale(faceROI, eyeDetections, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());

                smileDetector.detectMultiScale(faceROI, smileDetections, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());

                for (Rect eyeRect : eyeDetections.toArray()) {
                    Point eyeRectStart = new Point(rect.x + eyeRect.x, rect.y + eyeRect.y);
                    Point eyeRectEnd = new Point(eyeRectStart.x + eyeRect.width, eyeRectStart.y + eyeRect.height);
                    Imgproc.rectangle(webcamImage, eyeRectStart, eyeRectEnd, new Scalar(255, 0, 0), 2);
                }

              /*  for (Rect smileRect : smileDetections.toArray()) {
                    Point smileRectStart = new Point(rect.x + smileRect.x, rect.y + smileRect.y);
                    Point smileRectEnd = new Point(smileRectStart.x + smileRect.width, smileRectStart.y + smileRect.height);
                    Imgproc.rectangle(webcamImage, smileRectStart, smileRectEnd, new Scalar(0, 255, 0), 2);
                }*/

                String recognizedInfo = recognizeFace(webcamImage, rect, images, names, eyeDetections.toArray(), smileDetections.toArray());
                Imgproc.rectangle(webcamImage, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
                Imgproc.putText(webcamImage, recognizedInfo, new Point(rect.x, rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
            }

            BufferedImage imageToShow = matToBufferedImage(webcamImage);
            label.setIcon(new ImageIcon(imageToShow));
            panel.revalidate();
        }
    }

    public static List<File> loadImagesFromDirectory(String directoryPath) {
        List<File> images = new ArrayList<>();
        File folder = new File(directoryPath);
        File[] listOfFiles = folder.listFiles();
        if (listOfFiles != null) {
            for (File file : listOfFiles) {
                if (file.isFile()) {
                    images.add(file);
                }
            }
        }
        return images;
    }

    public static List<String> loadNamesFromImages(List<File> images) {
        List<String> names = new ArrayList<>();
        for (File file : images) {
            String name = file.getName().replaceFirst("[.][^.]+$", "");
            names.add(name);
        }
        return names;
    }
    public static boolean isSimilarSize(Mat face1, Mat face2) {
        double threshold = 0.2;
        double sizeDiff = Math.abs(face1.size().area() - face2.size().area()) / (double) face1.size().area();
        return sizeDiff < threshold;
    }


    public static String recognizeFace(Mat webcamImage, Rect faceRect, List<File> images, List<String> names, Rect[] eyeRects, Rect[] smileRects) {
        Mat face = new Mat(webcamImage, faceRect);

        Mat genderBlob = Dnn.blobFromImage(face, 1.0, new Size(227, 227), new Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);

        Mat ageBlob = Dnn.blobFromImage(face, 1.0, new Size(227, 227), new Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);

        genderNet.setInput(genderBlob);
        ageNet.setInput(ageBlob);
        Mat genderPreds = genderNet.forward();
        Mat agePreds = ageNet.forward();

        String gender = genderPreds.get(0, 0)[0] > 0.5 ? "Masculin" : "Féminin";

        double predictedAgeDouble = agePreds.get(0, 0)[0] * 100.0;
        String ageCategory = getAgeCategory(predictedAgeDouble);

        int numEyes = eyeRects.length;
        int numSmiles = smileRects.length;
        String eyeInfo = "Yeux : " + numEyes;
        String smileInfo = "Sourires : " + numSmiles;

        for (int i = 0; i < images.size(); i++) {
            Mat img = Imgcodecs.imread(images.get(i).getAbsolutePath());
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(img, faceDetections);
            for (Rect rect : faceDetections.toArray()) {
                Mat detectedFace = new Mat(img, rect);
                if (isSimilarSize(face, detectedFace)) {
                    return names.get(i) + " : " + gender + ", Âge : " + ageCategory;
                }
            }
        }

        return   " Inconnu   "+" : " + gender + " :" + ageCategory;
    }



    public static String getAgeCategory(double age) {
        if (age < 0.005) {
            return "Adulte";
        } else if (age < 0.01) {
            return "Adulte";
        } else if (age < 0.018) {
            return "Adulte";
        } else if (age < 0.4) {
            return "Adulte";
        } else {
            return "Adulte";
        }
    }

    public static void saveImage(Mat image, String imageName) {
        String outputFolder = "src/images/";
        File folder = new File(outputFolder);
        if (!folder.exists()) {
            folder.mkdirs();
        }
        String filename = outputFolder + imageName + ".jpg";
        Imgcodecs.imwrite(filename, image);
        System.out.println("Image enregistrée : " + filename);
    }

    public static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        mat.get(0, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData());
        return image;
    }
}