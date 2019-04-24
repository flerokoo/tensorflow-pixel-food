import numpy as np
import cv2
import os
import pickle
import random


def get_image_data(base):
    label_paths = os.listdir(base)
    data = []
    for i, label in enumerate(label_paths):        
        label_path = os.path.join(base, label)
        for file in os.listdir(label_path):
            image_data = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
            image_data = np.array(image_data).reshape((32, 32, 1))
            data.append((image_data, i));
    
    random.shuffle(data);

    X = []
    Y = []
    for (x, y) in data:
        X.append(x/255)    
        Y.append(y)            

    return (X, Y)




if __name__ == "__main__":  
    train_data = get_image_data("train_images")
    test_data = get_image_data("test_images")
    pickle.dump(train_data, open("train_data.pickle", "wb"))
    pickle.dump(test_data, open("test_data.pickle", "wb"))

# print(np.array(train_data).shape)
