package uz.cbssolutions.facerecognition.model;

import lombok.Builder;
import lombok.Data;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

@Builder
public record FaceData(
        MatOfRect matOfRect,
        LinkedList<Mat> mat
) implements Serializable {

}
