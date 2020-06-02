#   facenet
#   Modified by Jongha
#   Last Update: 2020.06.02

import numpy as np
from keras.models import load_model

model = load_model('./models/facenet_keras.h5')
print('model loaded')

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	print(face_pixels.shape)
    # transform face into one sample
    #expand dims adds a new dimension to the tensor
	samples = np.expand_dims(face_pixels, axis=0)
	print(samples.shape)
    # make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

data = np.load('FriendsDataset.npz')
trainX, trainy, valX, valy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, valX.shape, valy.shape)

#convert each face in the train set into an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)

#convert each face in the val set into an embedding
newValX = list()
for face_pixels in valX:
    embedding = get_embedding(model, face_pixels)
    newValX.append(embedding)
newValX = np.asarray(newValX)
print(newValX.shape)

np.savez_compressed('FriendsFaceEmbeddingData.npz', newTrainX, trainy, newValX, valy)

# from now on, we get face embeddings for each face image
# now we can use these to make predictions using an SVM. 
# Next, open SVMclassifier.