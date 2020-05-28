import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle
import csv

# defining global variable path
ch_path = "./coronahack/data/images"

def main(image_dim = 64):

    x,y = get_chestxray_covid(image_dim)

    x_train_plus = x[:200]
    y_train_plus = y[:200]

    x_test_plus = x[200:]
    y_test_plus = y[200:]

    x_train, y_train = create_dataset("train", image_dim)

    x_train = x_train + x_train_plus
    y_train = y_train + y_train_plus

    for _ in range(21):
        x_train_plus_distort = distort(x_train_plus)
        x_train = x_train + x_train_plus_distort
        y_train = y_train + y_train_plus

    print(len(x_train))
    print(len(y_train))

    pickle.dump(x_train, open( "./covid/" + str(image_dim) + "/x_train_d.p", "wb" ) )
    pickle.dump(y_train, open( "./covid/" + str(image_dim) + "/y_train_d.p", "wb" ) )

    x_test, y_test = create_dataset("test", image_dim)

    x_test = x_test + x_test_plus
    y_test = y_test + y_test_plus

    print(len(x_test))
    print(len(y_test))

    pickle.dump(x_test, open( "./covid/" + str(image_dim) + "/x_test.p", "wb" ) )
    pickle.dump(y_test, open( "./covid/" + str(image_dim) + "/y_test.p", "wb" ) )

def get_chestxray_covid(image_dim):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([(os.path.join("./chestxray/images", file),file) 
        for file in os.listdir("./chestxray/images")])
    
    print(len(image_files))

    resized_images = resize(image_files, image_dim)

    print(len(resized_images))

    labels = generate_labels_chestxray()

    x = []
    y = []

    for filename, label in labels.items():
        if filename in resized_images:
            y.append(label)
            x.append(resized_images[filename])

    # print(x[:3])
    # print(y[:3])

    # print(len(x))
    # print(len(y))

    return x,y

def distort(images):
    distorted_images = []
    for image in images:
        gauss_p = np.random.rand()
        if gauss_p > 0.3:
            blur_factor1 = np.random.randint(1,7)

            if blur_factor1 % 2 == 0:
                blur_factor1 += 1 # needs to be odd
            blur_factor2 = np.random.randint(1,7)
            if blur_factor2 % 2 == 0:
                blur_factor2 += 1 # needs to be odd
            image = cv2.GaussianBlur(image, (blur_factor1, blur_factor2), 0)

        flip_p = np.random.rand()
        if flip_p > 0.5:
            # horizontal flip
            image = cv2.flip(image,1)
        
        distorted_images.append(image)
    
    return distorted_images

def create_dataset(split, image_dim):
    image_tup = load_images(ch_path,split)
    
    resized_images = resize(image_tup, image_dim)

    labels = generate_labels(split)

    x = []
    y = []

    for filename, label in labels.items():
        if filename in resized_images:
            y.append(label)
            x.append(resized_images[filename])

    # print(x[:3])
    # print(y[:3])

    # print(len(x))
    # print(len(y))

    return x,y

def display_image(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

def resize(images, image_dim):
    # return dictionary from filename to image array
    img = [(cv2.imread(i[0], cv2.IMREAD_GRAYSCALE),i[1]) for i in images]
    print('Original size',img[0][0].shape)

    height = image_dim
    width = image_dim
    dim = (width, height)
    res_img = {}
    for i in range(len(img)):
        res = cv2.resize(img[i][0], dim, interpolation=cv2.INTER_LINEAR)
        res_img[img[i][1]] = res

    # display_image(res_img[img[i][1]])

    print(len(res_img))
    # # Checcking the size
    # print("RESIZED", res_img[1].shape)
    
    # # Visualizing one of the images in the array

    return res_img

def generate_labels(split):
    split = split.upper()
    labels = {}
    with open('./coronahack/metadata.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3] == split:
                labels[row[1]] = int(row[4] == "COVID-19")
            # print(row)
    return labels

def generate_labels_chestxray():
    labels = {}
    with open('./chestxray/metadata.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[4] == "COVID-19":
                labels[row[21]] = 1
            # print(row)
    return labels

def load_images(path,split):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([(os.path.join(path, split, file),file) 
        for file in os.listdir(path + "/" + split) if file.endswith('.jpeg')])
    
    print(len(image_files))
    return image_files

if __name__ == "__main__":
    main()

