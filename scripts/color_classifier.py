import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.vq import whiten, kmeans, vq
from colorclassifier import Classifier
from scripts.knn_classifier import main



def is_black(img_path,bboxes=None):
    image = cv2.imread(img_path)
    if bboxes != None:
        image = image[bboxes["top"]:bboxes["bottom"], bboxes["left"]:bboxes["right"]]
    w, h, c = image.shape
    image = image[int(0.22 * w):w - int(0.22 * w), int(0.22 * h):h - int(0.22 * h)]
    blue = int(np.average(image[:, :, 0]))
    green = int(np.average(image[:, :, 1]))
    red = int(np.average(image[:, :, 2]))

    classifier = Classifier(rgb=[red, green, blue])
    if classifier.get_name() == "black":
        return True
    else:
        return False


cmyk_scale = 100

def rgb_to_cmyk(r, g, b):
    r = int(r)
    g = int(g)
    b = int(b)

    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / 255.
    m = 1 - g / 255.
    y = 1 - b / 255.

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c * cmyk_scale, m * cmyk_scale, y * cmyk_scale, k * cmyk_scale


def get_color_palette(img_path=None, number_of_colors_per_line=3, show_color_palette=False, cmyk_out=True):
    image = cv2.imread(img_path)

    h, w, c = image.shape

    b, g, r = cv2.split(image)
    r = r.reshape((h * w))
    g = g.reshape((h * w))
    b = b.reshape((h * w))

    df = pd.DataFrame({'red': r, 'blue': b, 'green': g})

    df['scaled_red'] = whiten(df['red'])
    df['scaled_blue'] = whiten(df['blue'])
    df['scaled_green'] = whiten(df['green'])

    cluster_centers, distortion = kmeans(df[['scaled_red', 'scaled_green', 'scaled_blue']], number_of_colors_per_line)

    r_std, g_std, b_std = df[['red', 'green', 'blue']].std()

    idx, _ = vq(df[['scaled_red', 'scaled_green', 'scaled_blue']], cluster_centers)

    counts = np.bincount(idx)

    if show_color_palette:
        colors = []
        for cluster_center in cluster_centers:
            scaled_r, scaled_g, scaled_b = cluster_center
            colors.append((
                scaled_r * r_std / 255,
                scaled_g * g_std / 255,
                scaled_b * b_std / 255
            ))
        plt.title(img_path.split('/')[-2])
        container = []
        for i, c in enumerate(counts):
            container.append((c, colors[i]))

        container.sort(key=lambda x: x[0])
        col = []
        for i, c in container:
            col.append(c)
        plt.imshow([col])
        plt.show()
        print("count:", counts)

    container = []
    for i, c in enumerate(counts):
        container.append((c, cluster_centers[i]))

    container.sort(key=lambda x: x[0])
    # print("Results:", container)
    result = []
    for i, c in container:
        result.append(int(c[0] * r_std))
        result.append(int(c[1] * g_std))
        result.append(int(c[2] * b_std))
        # print([c[0] * r_std, c[1] * g_std, c[2] * b_std])
    if cmyk_out:
        cmyk_result = ''
        for ind in range(int(result.__len__() / 3)):
            r, g, b = result[(ind * 3):(ind * 3) + 3]
            c, m, y, k = rgb_to_cmyk(r, g, b)
            cmyk_result += str(c)
            cmyk_result += ','
            cmyk_result += str(m)
            cmyk_result += ','
            cmyk_result += str(y)
            cmyk_result += ','
            cmyk_result += str(k)
            cmyk_result += ','
            # print(cmyk_result)
        # print(cmyk_result)
        return cmyk_result
    return result

def color_histogram_of_cap_image(test_src_image):


    blue = int(np.average(test_src_image[:, :, 0]))
    green = int(np.average(test_src_image[:, :, 1]))
    red = int(np.average(test_src_image[:, :, 2]))

    c,m,y,k = rgb_to_cmyk(red,green,blue)

    feature_data = str(c)+','+ str(m)+','+ str(y)+','+str(k)
            # print(feature_data)

    with open('data_files/cc_test.data', 'w') as myfile:
        myfile.write(feature_data)
def get_color(img_path,bboxes=None):
    image = cv2.imread(img_path)
    if bboxes!= None:
        image = image[bboxes["top"]:bboxes["bottom"],bboxes["left"]:bboxes["right"]]
    color_histogram_of_cap_image(image)
    prediction = main('data_files/training.data', 'data_files/cc_test.data',vector_size=4)
    return prediction

def train_on_it(img_folder_path, tag=None, resultfile_path=None):
    for the_file in os.listdir(img_folder_path):
        file_path = os.path.join(img_folder_path, the_file)
        if os.path.isfile(file_path) and (".jpg" in file_path or ".jpeg" in file_path or ".png" in file_path):
            line = get_color_palette(img_path=file_path, show_color_palette=False, cmyk_out=True)

            tag = tag.split(' (#')[0]
            tag = tag.replace(' ', '_')
            line += tag
            with open(resultfile_path, 'a') as myfile:
                myfile.write(line + '\n')
                print(".")

if __name__ == "__main__":
    ROOT_DIR = './snack_cropped/'
    trainin_data_file = 'data_files/training_adv.data'
    i = 0
    for the_directoy in os.listdir(ROOT_DIR):
        if the_directoy == ".DS_Store":
            continue
        train_on_it(img_folder_path=os.path.join(ROOT_DIR, the_directoy),
                    tag=the_directoy,
                    resultfile_path=trainin_data_file)
