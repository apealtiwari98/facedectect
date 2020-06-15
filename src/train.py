#!/usr/bin/env python

from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from os import listdir
from matplotlib import pyplot
from os.path import isdir
from numpy import savez_compressed
from numpy import expand_dims
from numpy import load
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice
import argparse
import os
import pickle
import shutil
import zipfile
import split_folders


def extract_face(filename):
    face_array = []
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) < 1:
        print("No face found")
        return []
    i = 0
    for i in range(len(results)):
        x1,y1,width,height = results[i]['box']
        x1,y1 = abs(x1),abs(y1)
        x2,y2 = x1 + width , y1 + height
        face = pixels[y1:y2,x1:x2]
        image = Image.fromarray(face)
        #pyplot.axis('off')
        #pyplot.imshow(face)
        #pyplot.show()
        image = image.resize((160,160))
        face_array.append(asarray(image))
        i += 1
    return face_array


def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        if not filename.startswith('.'):
            path = directory + filename;
            face = extract_face(path)
            if face == []:
                continue
            faces.extend(face)
    return faces   


def load_dataset(directory):
    X,y = list(),list()
    print(directory)
    for subdir in listdir(directory):
        print(subdir)
        path = directory + '/' + subdir + '/'
        print(path)
        if not isdir(path):
            continue
        print("Loading label %s " % subdir )
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print (">Loaded %d examples for %s label" % (len(faces),subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X),asarray(y)

def get_embeddings(model,face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean,std = face_pixels.mean(),face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    samples = expand_dims(face_pixels,axis=0)
    yhat = model.predict(samples)
    return yhat[0]


if __name__ == '__main__':

    print("Inside train")
    
    #trainX,trainy = load_dataset(f'file://data/train')
    #testX,testy  = load_dataset(f'file://data/val')
    file = os.path.join(os.environ['SM_CHANNEL_TRAINING'] ,'dataset.zip')
    print(file)
#    download_dir = os.path.join(os.environ['SM_CHANNEL_TRAINING'] , 'trainset')
#    print(download_dir)
    dataset_dir = os.path.join(os.environ['SM_CHANNEL_TRAINING'] ,'dataset_dir')
    print(dataset_dir)
 
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
#    os.mkdir(os.path.join(os.environ['SM_CHANNEL_TRAINING'],'dataset'))
    
 #   os.mkdir(download_dir)

#os.mkdir(dataset_dir)
    with zipfile.ZipFile(file) as zip_ref :
        zip_ref.extractall(os.environ['SM_CHANNEL_TRAINING'])

    split_folders.ratio(os.path.join(os.environ['SM_CHANNEL_TRAINING'],'Top 50 Actor'),output=dataset_dir, seed= 1337, ratio =(.8,.2))    
    data_path = os.path.join(dataset_dir,'train')
    print("data path is %s " % data_path)
    val_path = os.path.join(dataset_dir,'val')
    print("val path is %s " % val_path)
    trainX,trainy = load_dataset(data_path)
    #trainX,trainy = load_dataset(os.environ['SM_CHANNEL_TRAINING'])
    #print(trainX.shape,trainy.shape)
    testX,testy = load_dataset(val_path)

    savez_compressed('/opt/ml/model/20_celebrities.npz',trainX,trainy,testX,testy)

    data = load('/opt/ml/model/20_celebrities.npz')
    trainX,trainy,testX,testy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
    print("loaded : " , trainX.shape,trainy.shape,testX.shape,testy.shape)

    model = load_model('facenet_keras.h5')
    print('Loaded model')
    newTrainX = list()
    for face_pixels in trainX:
        #print(face_pixels.shape)
        embedding = get_embeddings(model,face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embeddings(model,face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    savez_compressed('/opt/ml/model/20_celebrities_embeddings.npz',newTrainX,trainy,newTestX,testy)

    #data = load('20_celebrities.npz')
    #testX_faces = data['arr_2']

    data = load('/opt/ml/model/20_celebrities_embeddings.npz')
    trainX,trainy,testX,testy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
    print("Dataset : train=%d , test=%d" % (trainX.shape[0],testX.shape[0]))
    print("Dataset : train=%s , test=%s" % (trainX.shape,testX.shape))
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    with open('/opt/ml/model/labels.txt','w') as f:
        for item in out_encoder.classes_:
            f.write("%s\n" % item)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    model1 = SVC(kernel='linear',probability=True)
    model1.fit(trainX,trainy)
    filename = "/opt/ml/model/test_images_model.sav"
    pickle.dump(model1,open(filename,'wb'))
    #model = pickle.load(open(filename,'rb'))
    yhat_train = model1.predict(trainX)
    yhat_test = model1.predict(testX)
    score_train = accuracy_score(trainy,yhat_train)
    score_test = accuracy_score(testy,yhat_test)
    print("Accuracy train=%.3f  test=%.3f " % ((score_train*100 , score_test*100)))

