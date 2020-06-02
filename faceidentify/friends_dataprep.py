#   Data Prepare
#   Modified by Jongha
#   Last Update: 2020.06.02

from os import listdir
from os.path import isdir
from matplotlib import pyplot
from keras.models import load_model
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
model = load_model('./models/facenet_keras.h5')
print(model.inputs)
print(model.outputs)

def extract_face(filename, req_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    image = image.convert('RGB')
    # convert to array
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)

    print(filename) # check file name
    if results == []:
        return np.zeros((160, 160, 3))
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # take abs value to avoid negatives
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize the face to required size by model
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    print("image before np", image)
    face_array = np.asarray(image)
    print("extract face produces", face_array.shape)
    return face_array

#plot all the faces in the images in this directory
def plot_images(folder, plot_h, plot_w):
    i = 1
    # enumerate files
    for filename in listdir(folder):
        # add image file to path
        path = folder + filename
        # call get face
        face = extract_face(path)
        if face != []:
            print(i, face.shape)
        # plot
            pyplot.subplot(plot_h, plot_w, i)
            pyplot.axis('off')
            pyplot.imshow(face)
        i += 1
    pyplot.show()

#load all the faces from images in this directory
def load_faces(direc):
    faces = list()
    #enumerate files
    for filename in listdir(direc):
        # add image file to path
        path = direc + filename
        # call get face
        face = extract_face(path)
        faces.append(face)
    
    return faces

#To run over train and val directories
def load_dataset(direc):
    x, y = list(), list()
    #for every class directory in this train/val directory
    for subdir in listdir(direc):

        path = direc + subdir + '/' # Define path
        #if it is a file and not a dir then skip
        if not isdir(path):
            continue
        #load all faces in the class directory (subdir)
        faces = load_faces(path)
        #create labels
        labels = [subdir for i in range(len(faces))]
        #summarize progress
        print('loaded %d examples for class: %s' %(len(faces), subdir))
        print(faces)
        x.extend(faces)
        y.extend(labels)
    return np.asarray(x), np.asarray(y)

trainX, trainy = load_dataset('dataset/train/')
print(trainX.shape, trainy.shape)
valX, valy = load_dataset('dataset/val/')
print(trainX.shape, trainy.shape)
np.savez_compressed('FriendsDataset.npz', trainX, trainy, valX, valy)