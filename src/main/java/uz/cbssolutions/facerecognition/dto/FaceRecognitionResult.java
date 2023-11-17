package uz.cbssolutions.facerecognition.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;

import java.io.Serializable;

@Builder
public record FaceRecognitionResult(

        String message,
        @JsonProperty(value = "recognition-value")
        Double recognitionValue
) implements Serializable {
}
