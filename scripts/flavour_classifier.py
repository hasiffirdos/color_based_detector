import os
from scripts import knn_classifier
from scripts.color_classifier import get_color_palette



def get_flavour(img_path):

    if os.path.isfile(img_path) and (".jpg" in img_path or ".jpeg" in img_path or ".png" in img_path):
        line = get_color_palette(img_path=img_path, show_color_palette=False)
        line = line[:-1]

        with open('data_files/test.data', 'w') as myfile:
            myfile.write(line + '\n')
            print("test.")

    return knn_classifier.main('data_files/training_adv.data', 'data_files/test.data')



