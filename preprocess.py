import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle
import csv

# defining global variable path
ch_path = "./coronahack/data/images"

def main(image_dim = 128):
    x_train, y_train = create_dataset("train", image_dim)

    pickle.dump(x_train, open( "./coronahack/" + str(image_dim) + "/x_train.p", "wb" ) )
    pickle.dump(y_train, open( "./coronahack/" + str(image_dim) + "/y_train.p", "wb" ) )

    x_test, y_test = create_dataset("test", image_dim)

    pickle.dump(x_test, open( "./coronahack/" + str(image_dim) + "/x_test.p", "wb" ) )
    pickle.dump(y_test, open( "./coronahack/" + str(image_dim) + "/y_test.p", "wb" ) )

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

    print(x[:3])
    print(y[:3])

    print(len(x))
    print(len(y))

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

    print(len(res_img))
    # # Checcking the size
    # print("RESIZED", res_img[1].shape)
    
    # # Visualizing one of the images in the array

    return res_img

    no_noise = []
    for i in range(len(res_img)):
        blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)
        no_noise.append(blur)


    image = no_noise[1]
    display_image(image)

    return images

def generate_labels(split):
    split = split.upper()
    labels = {}
    with open('./coronahack/metadata.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3] == split:
                labels[row[1]] = int(row[2] == "Pnemonia")
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

