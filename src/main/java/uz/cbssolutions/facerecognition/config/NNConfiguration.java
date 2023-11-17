package uz.cbssolutions.facerecognition.config;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;

import java.io.File;

@Configuration
public class NNConfiguration {

    public File neuralNetworkFile(){
        return new File();
    }
}
