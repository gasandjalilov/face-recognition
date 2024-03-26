package uz.cbssolutions.facerecognition.service.impl;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import uz.cbssolutions.facerecognition.dto.FaceRecognitionResult;
import uz.cbssolutions.facerecognition.exception.ErrorParsingImage;
import uz.cbssolutions.facerecognition.helper.FaceFeatureTranslator;
import uz.cbssolutions.facerecognition.service.ImageProcessing;

import java.io.IOException;

@Slf4j
@RequiredArgsConstructor
@Service
public class DJLImageProcessor implements ImageProcessing {




    @Override
    public Mono<FaceRecognitionResult> recogniseFaces(Flux<FilePart> images) {
        return images.flatMap(filePart -> DataBufferUtils.join(filePart.content())
                .flatMap(dataBuffer -> {
                    try {
                        return Mono.just(ImageFactory.getInstance().fromInputStream(dataBuffer.asInputStream()));
                    } catch (IOException e) {
                        return Mono.error(new ErrorParsingImage("Could not parse image from InputStream"));
                    }
                })
                .flatMap(image -> {
                    try {
                        return Mono.just(predict(image));
                    } catch (Exception e) {
                        return Mono.error(new ErrorParsingImage("Could not predict image from Mono<Image>"));
                    }
                })
                )
                .take(2)
                .collectList()
                .map(floats -> {
                    var distance = calculSimilar(floats.get(0),floats.get(1));
                    return FaceRecognitionResult.builder()
                                    .recognitionValue(distance)
                                    .message(distance<79_000? "Match" : "Not Matched")
                                    .build();
                        }
                        );
    }


    public static float[] predict(Image img)
            throws IOException, ModelException, TranslateException {
        img.getWrappedImage();
        Criteria<Image, float[]> criteria =
                Criteria.builder()
                        .setTypes(Image.class, float[].class)
                        .optModelUrls("https://resources.djl.ai/test-models/pytorch/face_feature.zip")
                        .optModelName("face_feature") // specify model file prefix
                        .optTranslator(new FaceFeatureTranslator())
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch") // Use PyTorch engine
                        .build();

        try (ZooModel<Image, float[]> model = criteria.loadModel()) {
            Predictor<Image, float[]> predictor = model.newPredictor();
            return predictor.predict(img);
        }
    }

    public static double calculSimilar(float[] feature1, float[] feature2) {
        float ret = 0.0f;
        float mod1 = 0.0f;
        float mod2 = 0.0f;
        int length = feature1.length;
        for (int i = 0; i < length; ++i) {
            ret += feature1[i] * feature2[i];
            mod1 += feature1[i] * feature1[i];
            mod2 += feature2[i] * feature2[i];
        }
        return ((ret / Math.sqrt(mod1) / Math.sqrt(mod2) + 1) / 2.0f);
    }
}
