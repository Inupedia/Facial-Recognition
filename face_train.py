import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageOps
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# get the current path and located the image directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "image")

# Cascade Classifier to determine the image face
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')

# read the image file and convert it to data (base on grayscale)
def decompose_image_to_data():
    current_id = 0
    label_ids = {} # associate the label with its id, for example {'pengju1': 1, 'pengju2': 2}
    data = []
    labels = []
    IMAGE_SIZE = 300 # smaller size has fewer components
    # check if there is image exists (through its extension jpg or png in this case)
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):

                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()

                # note every label in our dataset must be unique
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                current_image_id = label_ids[label]

                # convert image and add its face data to list
                pil_image = Image.open(path).convert("L")
                resized_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                image_array = np.array(resized_image, "uint8")
                # # by using haarcascade_frontalface_alt2 we can find the front face through picture
                # # note that we only need to save the recognizable image
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.05, minNeighbors=5)
                for x, y, w, h in faces:
                    roi = image_array[y: y + h, x: x + w]
                    resized_array = cv2.resize(roi, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
                    reshaped_img = np.reshape(resized_array, (1, 150 * 150))
                    data.extend(reshaped_img)
                    #np.concatenate(data, roi)
                    labels.append(current_image_id)
    return np.array(data), np.array(labels), label_ids

def pca_analysis(data):
    variance_to_keep = 0.9 # in this case we want to keep 90% characteristics
    my_pca = PCA(n_components = variance_to_keep)
    transform_data = my_pca.fit(data)
    return transform_data

def svm_classification(model, label, test_data):
    clf3 = SVC(C=3.0)
    clf3.fit(model, label)
    return clf3.predict(test_data)

def knn_classification(neighbor, model, label, test_data):
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(model, label)
    return knn.predict(test_data)

def get_name(model, label, label_ids, classification_method, test_data):

    labels = {v: k for k, v in label_ids.items()}

    result = ""
    if classification_method == "svm":
        result = svm_classification(model, label, test_data)
    elif classification_method == "knn":
        result = knn_classification(5, model, label, test_data) # our k-nearst neighbors is 5 in this case

    return labels[result[0]]

def create_model():
    data, labels, label_ids = decompose_image_to_data()
    # scale the dataset
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    # model = pca_analysis(data_scaled)
    return data_scaled, labels, label_ids

if __name__ == '__main__':
    data, labels, label_ids = decompose_image_to_data()
    # scale the dataset
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    transform_data = pca_analysis(data_scaled)
    #
    print(transform_data.explained_variance_ratio_)


