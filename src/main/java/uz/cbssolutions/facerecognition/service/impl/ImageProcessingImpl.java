package uz.cbssolutions.facerecognition.service.impl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.enums.Mode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
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
import java.io.IOException;
import java.util.LinkedList;

@Service
@Slf4j
public class ImageProcessingImpl implements ImageProcessing {

    public final CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_alt.xml");
    private DataNormalization scalerNormalizer = new VGG16ImagePreProcessor();;
    private EuclideanDistance euclideanDistance = new EuclideanDistance();
    public final NativeImageLoader nativeImageLoader = new NativeImageLoader(224, 224, 3);
    private TransferLearningHelper transferLearningHelper;
    ComputationGraph objComputationGraph;

    public ImageProcessingImpl() {
        try {
            VGG16 objZooModel = VGG16.builder().build();
            Model vgg16 = objZooModel.initPretrained(PretrainedType.VGGFACE);
            objComputationGraph = (ComputationGraph) vgg16;
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
                        String file2 = "/home/ideasquarebased/img.jpg";
                        Imgcodecs imageCodecs = new Imgcodecs();
                        //Writing the image
                        imageCodecs.imwrite(file2, faceData.mat().getFirst());


                        INDArray face = nativeImageLoader.asMatrix(faceData.mat().getFirst());
                        scalerNormalizer.transform(face);


                        DataSet objDataSet = new DataSet(face, Nd4j.create(new float[]{0,0}));
                        DataSet objFeaturized = transferLearningHelper.featurize(objDataSet);
                        INDArray featuresArray = objFeaturized.getFeatures();


                        sink.next(NLPFaceData.builder().face(featuresArray).build());
                    } catch (IOException e) {
                        sink.error(new ErrorParsingMat(e.getLocalizedMessage()));
                    }
                })
                .cast(NLPFaceData.class)
                .take(2)
                .collectList()
                .map(dataList -> {
                    EuclideanDistance euclideanDistance = new EuclideanDistance();
                    double distance = dataList.get(0).face().distance2(dataList.get(1).face());
                    return FaceRecognitionResult.builder()
                            .recognitionValue(distance)
                            .message(distance<76_000? "Match" : "Not Matched")
                            .build();
                });
    }


}
