import pickle
import shutil
from collections import Counter
import numpy as np
import cv2
import os
import random
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import _pickle as cPickle

chemain = "/home/nikam/Téléchargements/dataset"

def split_dataset(img_source_dir, train_size):
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists

    if not os.path.exists('training'):
        os.makedirs('training')
    if not os.path.exists('test'):
        os.makedirs('test')

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('training', subdir)
        validation_subdir = os.path.join('test', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0

        # Randomly assign an image to train or validation folder
        for filename in os.listdir(subdir_fullpath):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                fileparts = filename.split('.')

                if random.uniform(0, 1) <= train_size:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(train_subdir, str(train_counter) + '.' + fileparts[1]))
                    train_counter += 1
                else:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(validation_subdir, str(validation_counter) + '.' + fileparts[1]))
                    validation_counter += 1

        print('Copied ' + str(train_counter) + ' images to data/train/' + subdir)
        print('Copied ' + str(validation_counter) + ' images to data/validation/' + subdir)


#split_dataset(chemain, 0.5)

training_folder = 'training'
test_folder = 'test'
model_dir = "keypoints_model"
points = 5

def descripteurTraining(training_folder):
    file_name, file_extension = os.path.splitext("model_file")
    for path, dirs, files in os.walk(training_folder):
        for file in files:
            image_file = os.path.join(path, file)
            train_image = cv2.imread(image_file)
            train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            keys, descriptor = sift.detectAndCompute(train_image_gray, None)
            classe = os.path.basename(os.path.dirname(image_file))
            keys_descriptor = {'classe': classe, 'descriptor': descriptor}
            #print(keys_descriptor)
            with open(file_name, "ab") as fichier:
                fichier.write(cPickle.dumps(keys_descriptor))
    if not os.path.exists(model_dir) :
        os.mkdir(model_dir)
    else:
        shutil.rmtree(model_dir)
        os.mkdir(model_dir)

    shutil.move(file_name, model_dir)

#descripteurTraining(training_folder)


def read_descriptor(model_file):
    descriptorTab = []
    classeTab = []
    with open(model_file, 'rb') as f:
        while True:
            try:
                oneVar = cPickle.load(f)
                #index = cPickle.loads(f.read())
                descriptor = (oneVar.get('descriptor'))
                classe = (oneVar.get('classe'))

                #tab = [oneVar.get('descriptor'), oneVar.get('classe')]
                descriptorTab.append(descriptor)
                classeTab.append(classe)
            except:
                break
    return descriptorTab, classeTab

#descriptor_list, classe_list = read_descriptor(model_file)
#print(descriptor_list)
#print(classe_list)



def matchImage(file_name, descriptor_list, classe_list):

    test_image = cv2.imread(file_name)
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(points)
    keys1, descriptor1 = sift.detectAndCompute(test_image_gray, None)
    correct = 0
    total = 0
    k_flann = 2
    classe_vrai = os.path.basename(os.path.dirname(file_name))
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    good_matches = []
    if len(descriptor1) > k_flann:
        for id, value in enumerate(descriptor_list):
            c = 0

            matches = bf.knnMatch(value, descriptor1, 2)
            for m, n in matches:
                if m.distance < 0.5 * n.distance:
                    c += 1
            good_matches.append([classe_list[id], c])
        good_matches = sorted(good_matches, key=lambda x: x[1], reverse=True)
        k_nearest = good_matches[:10]
        #print(k_nearest)
        E = []
        for classe_trouve, b in k_nearest:
            E.append(classe_trouve)
        #print(E[0])

        if E[0] == classe_vrai:
            correct += 1
        total += 1
        #print(total)
        #print(correct)
        print('La classe vraie est ', classe_vrai, ' *****La classe predictie est ', E[0])
        print('Score de prediction : ', round((100 * correct/total), 2), ' %')
        cv2.imshow("image_test", test_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def classesNames(train_folder):
    folder = os.listdir(train_folder)
    tab = []
    for fichier in folder:
        tab.append(fichier)
    return tab


def matrixConfusion(test_folder, descriptor_list):
    confusion_matrice = []
    classe_predit = []
    classe_true = []
    correct = 0
    total = 0
    tabClasse = classesNames(training_folder)
    for path, dirs, files in os.walk(test_folder):
        ligne = np.zeros(len(tabClasse))
        for file in files:
            image_file = os.path.join(path, file)
            test_image = cv2.imread(image_file)
            test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create(points)
            keys1, descriptor1 = sift.detectAndCompute(test_image_gray, None)
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
            classe_vrai = os.path.basename(os.path.dirname(image_file))
            classe_true.append(classe_vrai)
            good_matches = []
            for id, value in enumerate(descriptor_list):
                c = 0
                matches = bf.knnMatch(value, descriptor1, 2)
                for m, n in matches:
                    if m.distance < 0.5 * n.distance:
                        c += 1
                good_matches.append([tabClasse[id], c])
            good_matches = sorted(good_matches, key=lambda x: x[1], reverse=True)
            k_nearest = good_matches[:10]

            for classe_trouve, b in k_nearest:
                classe_predit.append(classe_trouve)
            if classe_predit[0] == classe_vrai:
                correct += 1
            total += 1
            ligne[tabClasse.index(classe_predit[0])] += 1
        confusion_matrice.append(correct)
    score = round((100*correct/total),2)
    print(score)

    conf = confusion_matrix(classe_true, classe_predit)
    """sns.heatmap(conf, square=True, annot=True, cbar=False
                , xticklabels=list(tabClasse)
                , yticklabels=list(tabClasse)"""
    plt.xlabel('valeurs prédites')
    plt.ylabel('valeurs réelles');







