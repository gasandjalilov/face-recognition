package uz.cbssolutions.facerecognition;

import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class FaceRecognitionApplication {

	public static void main(String[] args) {
		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		OpenCV.loadLocally();
		SpringApplication.run(FaceRecognitionApplication.class, args);
	}

}
