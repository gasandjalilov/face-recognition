package uz.cbssolutions.facerecognition.exception;

public class NoFacesPresentedException extends RuntimeException{

    public NoFacesPresentedException(String message) {
        super(message);
    }
}
