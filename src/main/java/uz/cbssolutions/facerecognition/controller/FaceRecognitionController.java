package uz.cbssolutions.facerecognition.controller;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import uz.cbssolutions.facerecognition.dto.FaceRecognitionResult;
import uz.cbssolutions.facerecognition.service.ImageProcessing;

@RestController()
@RequestMapping(value = "/facerecognize")
public class FaceRecognitionController {


    private final ImageProcessing imageProcessing;

    public FaceRecognitionController(@Qualifier("DJLImageProcessor") ImageProcessing imageProcessing) {
        this.imageProcessing = imageProcessing;
    }

    @PostMapping
    public Mono<FaceRecognitionResult> upload(@RequestPart Flux<FilePart> fileParts) {
        return imageProcessing.recogniseFaces(fileParts);
    }
}
