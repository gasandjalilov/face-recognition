package uz.cbssolutions.facerecognition.model;

import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

import java.io.Serializable;
import java.util.List;

@RequiredArgsConstructor
@Builder
@Data
public class FaceData implements Serializable {
    private MatOfRect matOfRect;
    private List<Mat> mat;
}
