import cv2
from skimage import feature as detector
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import os
import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import shutil
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
import re
import json

def load_image_dataset(directory):
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


    # List of image file extensions to read
    image_exts = ['jpg', 'jpeg', 'png', 'bmp']
    i=0
    # Detect faces in all images in the directory
    faces_list = []
    label = []
    for ext in image_exts:
        for filename in glob.glob(os.path.join(directory, f'*.{ext}')):
            temp = os.path.splitext(os.path.basename(filename))[0]
            i = i+1
    #         print(i,'read image :',temp)
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]
            faces_list.append(face)
            label.append(temp)
            
    return faces_list,label
    
    


def preprocess_image(img):
    
    # Convert to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    

    
    
    # Apply Gaussian blur to remove noise
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply adaptive thresholding to binarize image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    

    
    return thresh




def create_test_images(image_dir, test_dir, num_test_images):
    # Remove the existing test directory if it exists
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Create the test directory
    os.makedirs(test_dir)

    # Get a list of all image file names in the directory
    image_files = os.listdir(image_dir)

    # Shuffle the list of image file names randomly
    random.shuffle(image_files)

    # Select the specified number of images as test images
    test_files = image_files[:num_test_images]

    # Iterate over the test image files and copy them to the test directory
    for image_file in test_files:
        # Get the full path of the image file
        image_path = os.path.join(image_dir, image_file)

        # Get the label (image name) without the file extension
        label = os.path.splitext(image_file)[0]

        # Define the new file name for the test image
        new_filename = label + '.jpg'

        # Define the full path of the test image file
        new_filepath = os.path.join(test_dir, new_filename)

        # Copy the test image file to the test directory
        shutil.copy(image_path, new_filepath)



def preprocess(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply histogram equalization
    eq = cv2.equalizeHist(blur)

    return eq

def preprocess_images(images, target_size):
    preprocessed_images = []
    for image in images:
        resized_image = cv2.resize(image, target_size)
        preprocessed_image = preprocess(resized_image)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images









def extract_image_features(preprocessed_face_images):
    hog = cv2.HOGDescriptor()
    hog_features = []

    for im in preprocessed_face_images:
        if im.shape[0] < 8 or im.shape[1] < 8:
            continue  # Skip images that are smaller than the cell size

        try:
            features = detector.hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8))
            hog_features.append(features)
        except ValueError:
            continue  # Skip images that cause a ValueError

    hog_features = np.array(hog_features)
    print(len(hog_features))
    
    return hog_features 
    

# test data create
image_dir = './img/'
test_dir = './test_images/'
num_test_images = 10

create_test_images(image_dir, test_dir, num_test_images)


directory = './img'

faces_list,label = load_image_dataset(directory)

# image resize define
target_size = (224, 224) 
preprocessed_face_images = preprocess_images(faces_list,target_size)



hog_features = extract_image_features(preprocessed_face_images)

hog_normalize = Normalizer(norm='l2')
hog_nor = hog_normalize.transform(hog_features)

# split data#  data into training and testing sets
X_train = hog_features
y_train = label 


test_dir = './test_images/'
X_test,y_test = load_image_dataset(test_dir)
X_test = preprocess_images(X_test,target_size)
X_test = extract_image_features(X_test)


print("Length of Train dataset",len(X_train))
print("Length of Test dataset",len(X_test))


y_test = np.array([re.findall(r'\d+', item)[0] for item in y_test])

y_train = np.array([re.findall(r'\d+', item)[0] for item in y_train])

# model train

# Train an SVM classifier on the training data
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict labels for the testing data
y_pred = svm.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM model Accuracy: {accuracy}")


# Now print to file
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": accuracy}, outfile)

# confusion matrix
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a figure and axes
plt.figure(figsize=(8, 6))
ax = plt.subplot()

# Plot the confusion matrix using a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)

# Set labels, title, and ticks
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(y_test)
ax.yaxis.set_ticklabels(y_pred)


plt.savefig('confusion_matrix.png')