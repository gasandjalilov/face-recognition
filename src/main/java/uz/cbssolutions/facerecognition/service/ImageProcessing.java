package uz.cbssolutions.facerecognition.service;

import org.springframework.http.codec.multipart.FilePart;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import uz.cbssolutions.facerecognition.dto.FaceRecognitionResult;

public interface ImageProcessing {

    Mono<FaceRecognitionResult> recogniseFaces(Flux<FilePart> images);
}
