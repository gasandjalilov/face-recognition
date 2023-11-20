package uz.cbssolutions.facerecognition.model;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

@Builder
public record NLPFaceData(
        INDArray face
) implements Serializable {
}
