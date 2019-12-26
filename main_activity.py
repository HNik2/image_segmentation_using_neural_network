import traitement
import shutil
from collections import Counter
import numpy as np
import cv2
import os
import random
from shutil import copyfile

import _pickle as cPickle
model_file='keypoints_model/model_file'
training_folder = 'training'
test_folder = 'test'
model_dir = "keypoints_model"
points = 5

print("Bienvenu à l'application pour la reconnaissance d'objets avec le descripteur SIFT,\n")
chemain = input("Veuillez saisir le chemain de la dataset : ")
path_data = str(chemain)
traitement.split_dataset(path_data, 0.5)
traitement.descripteurTraining('training')
test_file = input("Veuillez entrer le chemain d'un fichier du dossier test (test\classe_name\image_name) : ")
test_path = str(test_file)
descriptors, classes = traitement.read_descriptor(model_file)
traitement.matchImage(test_path, descriptors, classes)




