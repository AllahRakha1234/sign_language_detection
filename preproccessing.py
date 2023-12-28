import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def edge_detection(image):
    minValue = 70
    blur = cv2.GaussianBlur(image, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

def preprocess_and_save(input_dir, output_dir):
    images = []
    labels = []
    
    # Iterate over each class directory in the input directory
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        
        # Create the corresponding class directory in the output directory
        output_class_path = os.path.join(output_dir, class_dir)
        os.makedirs(output_class_path, exist_ok=True)

        # Iterate over each image in the current class directory
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)

            # Read the image using OpenCV
            img = cv2.imread(image_path, 0)

            # Apply edge detection preprocessing
            img = edge_detection(img)

            # Resize the image to (64, 64)
            img = cv2.resize(img, (64, 64))

            # Convert the image to a NumPy array
            img_array = np.array(img)

            # Normalize pixel values to the range [0, 1]
            img_array = img_array.astype('float32') / 255.0

            # Save the preprocessed image in the corresponding output directory
            output_image_path = os.path.join(output_class_path, image_file)
            np.save(output_image_path, img_array)

            # Collect labels for each image
            images.append(img)
            labels.append(class_dir)

    # Convert string labels to numerical labels
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(labels)

    # Convert numerical labels to one-hot encoding
    labels = to_categorical(numerical_labels)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

    # Save the processed data
    np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(output_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

input_directory = 'asl_alphabet_train/asl_alphabet_train'
output_directory = 'preprocessedOut'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)
print("-----------------")
preprocess_and_save(input_directory, output_directory)
