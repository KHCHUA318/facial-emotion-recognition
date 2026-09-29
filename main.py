import cv2 
import numpy as np
from keras.models import load_model

def main():
    # === Configuration ===
    # Path to your saved emotion-recognition model:
    model_path = r'C:\Users\Nigel\Downloads\Deep_learning model\FER\models\emotion_model_base.h5'
    # Define the class labels in the same order as the model was trained.
    # E.g., if during training you had folders or a generator with class_indices:
    #    class_indices = {'Angry':0, 'Disgust':1, ...}
    # ensure this list matches that order exactly.
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # ----------------------

    # Load the trained model
    model = load_model(model_path)
    # Inspect input shape:
    # model.input_shape is typically (None, height, width, channels)
    # For many emotion models: (None, 48, 48, 1) or (None, 64, 64, 1), etc.
    input_shape = model.input_shape  # e.g., (None, 48, 48, 1)
    if len(input_shape) != 4:
        raise ValueError(f"Expected model.input_shape to have 4 dims (None, H, W, C), got {input_shape}")
    _, H, W, C = input_shape
    print(f"Loaded model. Expected input size = ({H}, {W}, {C})")

    # Prepare face detector: OpenCV’s Haar cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar cascade from {face_cascade_path}")

    # Start webcam capture (0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame; exiting.")
            break

        # Convert to grayscale for face detection if model expects grayscale;
        # but for detection, Haar works on grayscale.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces: returns rectangles [x, y, w, h]
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # For each detected face, preprocess and predict emotion
        for (x, y, w, h) in faces:
            # Extract ROI
            if C == 1:
                # model expects grayscale
                face_roi = gray_frame[y:y+h, x:x+w]
                # Resize to target size
                face_resized = cv2.resize(face_roi, (W, H))
                # Normalize pixel values to [0,1]
                face_resized = face_resized.astype('float32') / 255.0
                # Expand dims to (H, W, 1)
                face_input = np.expand_dims(face_resized, axis=-1)
            else:
                # model expects color (3 channels)
                face_roi_color = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi_color, (W, H))
                face_resized = face_resized.astype('float32') / 255.0
                face_input = face_resized  # shape (H,W,3)

            # Create batch dimension: (1, H, W, C)
            face_input = np.expand_dims(face_input, axis=0)

            # Predict emotion probabilities
            preds = model.predict(face_input)
            # preds shape: (1, num_classes)
            pred_idx = np.argmax(preds[0])
            confidence = preds[0][pred_idx]
            label = class_labels[pred_idx] if pred_idx < len(class_labels) else str(pred_idx)

            # Draw rectangle and put label + confidence
            # Rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Prepare text: label + confidence
            text = f"{label}: {confidence*100:.1f}%"
            # Choose position: above the face rectangle (but ensure not negative y)
            text_y = y - 10 if y - 10 > 10 else y + 10
            cv2.putText(frame, text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Real-Time Facial Expression Recognition', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()