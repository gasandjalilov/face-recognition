package uz.cbssolutions.facerecognition.helper;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDArray;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Path;


public class FaceFeatureTranslator implements Translator<Image, float[]> {




    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
        Pipeline pipeline = new Pipeline();
        pipeline
                // .add(new Resize(160))
                .add(new ToTensor())
                .add(
                        new Normalize(
                                new float[] {127.5f / 255.0f, 127.5f / 255.0f, 127.5f / 255.0f},
                                new float[] {
                                        128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f
                                }));

        return pipeline.transform(new NDList(array));
    }

    /** {@inheritDoc} */
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        NDList result = new NDList();
        long numOutputs = list.singletonOrThrow().getShape().get(0);
        for (int i = 0; i < numOutputs; i++) {
            result.add(list.singletonOrThrow().get(i));
        }
        float[][] embeddings =
                result.stream().map(NDArray::toFloatArray).toArray(float[][]::new);
        float[] feature = new float[embeddings.length];
        for (int i = 0; i < embeddings.length; i++) {
            feature[i] = embeddings[i][0];
        }
        return feature;
    }
}
