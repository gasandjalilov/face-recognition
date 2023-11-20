package uz.cbssolutions.facerecognition.service.impl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import uz.cbssolutions.facerecognition.dto.FaceRecognitionResult;
import uz.cbssolutions.facerecognition.exception.ErrorParsingMat;
import uz.cbssolutions.facerecognition.exception.MultipleFacesException;
import uz.cbssolutions.facerecognition.exception.NoFacesPresentedException;
import uz.cbssolutions.facerecognition.math.EuclideanDistance;
import uz.cbssolutions.facerecognition.model.FaceData;
import uz.cbssolutions.facerecognition.model.NLPFaceData;
import uz.cbssolutions.facerecognition.service.ImageProcessing;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

@Service
@Slf4j
public class ImageProcessingImpl implements ImageProcessing {

    public final CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_alt.xml");
    private DataNormalization scalerNormalizer = new VGG16ImagePreProcessor();;
    private EuclideanDistance euclideanDistance = new EuclideanDistance();
    public final NativeImageLoader nativeImageLoader = new NativeImageLoader(224, 224, 3);
    private TransferLearningHelper transferLearningHelper;

    public ImageProcessingImpl() {
        try {
            ZooModel objZooModel = VGG16.builder().build();
            ComputationGraph objComputationGraph = null;
            objComputationGraph = (ComputationGraph) objZooModel.initPretrained(PretrainedType.VGGFACE);
            transferLearningHelper = new TransferLearningHelper(objComputationGraph,"pool4");
        }
        catch (IOException exception){
            log.error("Error Creating dataset: {}", exception.getLocalizedMessage());
        }
    }

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
                            Mat image = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_GRAYSCALE);
                            MatOfRect faceDetections = new MatOfRect();
                            LinkedList<Mat> faces = new LinkedList<>();
                            faceDetector.detectMultiScale(image, faceDetections);
                            for (Rect rect : faceDetections.toArray()) {
                                faces.add(new Mat(image, rect));
                            }
                            return FaceData.builder().mat(faces).matOfRect(faceDetections).build();
                        })
                .log()
                .flatMap(faceData -> {
                    if(faceData.mat().isEmpty()) return Mono.error(new NoFacesPresentedException("No Faces Presented"));
                    else if (faceData.mat().size()>1) return Mono.error(new MultipleFacesException("Multiple Faces Presented"));
                    else return Mono.just(faceData);
                })
                .handle((faceData, sink) -> {
                    try {
                        INDArray face = nativeImageLoader.asMatrix(faceData.mat().getFirst());
                        scalerNormalizer.transform(face);
                        DataSet objDataSet = new DataSet(face, Nd4j.create(new float[]{0,0}));
                        DataSet objFeaturized = transferLearningHelper.featurize(objDataSet);
                        INDArray featuresArray = objFeaturized.getFeatures();
                        long reshapeDimension=1;
                        for (long dimension : featuresArray.shape()) {
                            reshapeDimension *= dimension;
                        }
                        featuresArray = featuresArray.reshape(1,reshapeDimension);

                        sink.next(NLPFaceData.builder().face(featuresArray).build());
                    } catch (IOException e) {
                        sink.error(new ErrorParsingMat(e.getLocalizedMessage()));
                    }
                })
                .cast(NLPFaceData.class)
                .take(2)
                .collectList()
                .map(dataList -> {
                    double distance = euclideanDistance.run(dataList.get(0).face(),dataList.get(1).face());
                    return FaceRecognitionResult.builder()
                            .recognitionValue(distance)
                            .message(distance<Double.MAX_VALUE? "Match" : "Not Matched")
                            .build();
                });
    }

}
