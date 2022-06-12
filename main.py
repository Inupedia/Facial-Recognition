import face_train
import face_capture
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def run(data, labels, label_ids, classification_method):

    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture(0)

    # perform PCA analysis
    variance_to_keep = 0.9  # in this case we want to keep 90% characteristics
    my_pca = PCA(n_components=variance_to_keep)
    model = my_pca.fit_transform(data)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_array = np.array(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        for (x, y, w, h) in faces:

            roi_gray = image_array[y:y + h, x:x + w]
            resized_array = cv2.resize(roi_gray, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
            reshaped_image = np.reshape(resized_array, (1, 150 * 150))
            # scale the dataset
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(reshaped_image)
            data_test_transform = my_pca.transform(data_scaled)

            name = face_train.get_name(model, labels, label_ids, classification_method, data_test_transform)
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_capture.start()
    print("Processing data...")
    data, labels, label_ids = face_train.create_model()
    classification_method = input("Which classification method you would like to perform? (svm/knn): ")
    print("Starting to detect your face from local image source, use CTRL + C to quit.")
    run(data, labels, label_ids, classification_method)
