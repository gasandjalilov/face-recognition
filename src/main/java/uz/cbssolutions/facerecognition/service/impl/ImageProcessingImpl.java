package uz.cbssolutions.facerecognition.service.impl;

import lombok.RequiredArgsConstructor;
import org.datavec.image.loader.NativeImageLoader;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import uz.cbssolutions.facerecognition.dto.FaceRecognitionResult;
import uz.cbssolutions.facerecognition.exception.MultipleFacesException;
import uz.cbssolutions.facerecognition.exception.NoFacesPresentedException;
import uz.cbssolutions.facerecognition.model.FaceData;
import uz.cbssolutions.facerecognition.service.ImageProcessing;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ImageProcessingImpl implements ImageProcessing {

    public final CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_alt.xml");
    public final NativeImageLoader nativeImageLoader = new NativeImageLoader(224, 224, 3);

    @Override
    public Mono<FaceRecognitionResult> recogniseFaces(Flux<FilePart> images) {
        return images.flatMap(filePart -> DataBufferUtils.join(filePart.content())
                        .map(dataBuffer -> {
                            byte[] bytes = new byte[dataBuffer.readableByteCount()];
                            dataBuffer.read(bytes);
                            DataBufferUtils.release(dataBuffer);
                            return bytes;
                        }))
                        .map(bytes -> {
                            Mat image = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_UNCHANGED);
                            MatOfRect faceDetections = new MatOfRect();
                            List<Mat> faces = new ArrayList<>();
                            faceDetector.detectMultiScale(image, faceDetections);
                            for (Rect rect : faceDetections.toArray()) {
                                faces.add(new Mat(image, rect));
                            }
                            return FaceData.builder().mat(faces).matOfRect(faceDetections).build();
                        })
                .flatMap(faceData -> {
                    if(faceData.getMat().isEmpty()) return Mono.error(new NoFacesPresentedException("No Faces Presented"));
                    else if (faceData.getMat().size()>1) return Mono.error(new MultipleFacesException("Multiple Faces Presented"));
                    else return Mono.just(faceData);
                })
                .map(nativeImageLoader.asMatrix());



                filePartMono.map(filePart ->)
        Mat image1 = Imgcodecs. ("image1.jpg");
        Mat image2 = Imgcodecs.imread("image2.jpg");
        MatOfRect faceDetections1 = new MatOfRect();
        MatOfRect faceDetections2 = new MatOfRect();
        faceDetector.detectMultiScale(image1, faceDetections1);
        faceDetector.detectMultiScale(image2, faceDetections2);
        System.out.println(String.format("Detected %s faces in image1", faceDetections1.toArray().length));
        System.out.println(String.format("Detected %s faces in image2", faceDetections2.toArray().length));
        List<Mat> faces1 = new ArrayList<Mat>();
        List<Mat> faces2 = new ArrayList<Mat>();
        for (Rect rect : faceDetections1.toArray()) {
            faces1.add(new Mat(image1, rect));
            Imgproc.rectangle(image1, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }
        for (Rect rect : faceDetections2.toArray()) {
            faces2.add(new Mat(image2, rect));
            Imgproc.rectangle(image2, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }
        Imgcodecs.imwrite("output1.jpg", image1);
        Imgcodecs.imwrite("output2.jpg", image2);
        double threshold = 0.6;
        return null;
    }


    public Mono<Mat> image2Mat(Flux<InputStream> imageStream) throws IOException {
        return imageStream.map(stream -> {
                    try {
                        BufferedImage image = ImageIO.read(stream);
                        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                        ImageIO.write(image, "jpg", byteArrayOutputStream);
                        byteArrayOutputStream.flush();
                        return Imgcodecs.imdecode(new MatOfByte(byteArrayOutputStream.toByteArray()), Imgcodecs.IMREAD_UNCHANGED);
                    }

                })
                .onErrorMap(throwable ->);


    }
}
