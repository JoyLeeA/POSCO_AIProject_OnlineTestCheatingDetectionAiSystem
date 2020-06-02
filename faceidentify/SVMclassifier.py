#   SVMclassifier
#   Modified by Jongha
#   Last Update: 2020.06.02

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice


data = np.load('FriendsFaceEmbeddingData.npz')
trainX, trainy, valX, valy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: training examples =', trainX.shape, 'val examples =', valX.shape[0])

#Build the SVM time

#normalize the input tensors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
valX = in_encoder.transform(valX)

#label encode turns categorical input to numerical input
#that means ali, aqsa, manaal, umair will turn into 0, 1, 2, 3
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
valy = out_encoder.transform(valy)

#fit the model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

#predict
yhat_train = model.predict(trainX)
yhat_val = model.predict(valX)

#accuracy
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(valy, yhat_val)

print('Accuracy: train = %.3f, test = %.3f' %(score_train*100, score_test*100))

selection = choice([i for i in range(valX.shape[0])])
random_face_pixels = valX[selection]
random_face_emb = valX[selection]
random_face_class = valy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
samples = random_face_emb.reshape(-1, random_face_emb.shape[0])
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])

# We can identify user's face for real time
# From now on, we can detect user's cheating