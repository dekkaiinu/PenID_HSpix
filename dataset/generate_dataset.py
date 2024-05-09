import os
import random
import json
import numpy as np
from tqdm import tqdm
import cv2
from hsitools.convert import nh9_to_array
from hsitools.correction import hsi_blur
from sklearn.mixture import GaussianMixture

LOAD_PATH = '/mnt/hdd1/datasets/hyperspectral/hyper_penguin/HSpenguins'

def generate_dataset():
    category = ['0373', '0143', '0346', '0166', '1556', '0566', '0126', '0473', '0456', '0146']
    date = ['20230623', '20230627']
    dataset = load_dataset(date_list=date, pen_id_list=category)
    train_pen_spectrum, train_pen_id, test_pen_spectrum, test_pen_id, val_pen_spectrum, val_pen_id = split_dataset(dataset=dataset, train_rate=0.8, random_seed=0)
    
    train_spectrum_array, train_target = list2array(train_pen_spectrum, train_pen_id)
    test_spectrum_array, test_target = list2array(test_pen_spectrum, test_pen_id)
    val_spectrum_array, val_target = list2array(val_pen_spectrum, val_pen_id)

    train_spectrum_array, train_target = sampling_data(train_spectrum_array, train_target, 50000)
    test_spectrum_array, test_target = sampling_data(test_spectrum_array, test_target, 5000)
    val_spectrum_array, val_target = sampling_data(val_spectrum_array, val_target, 5000)

    print(train_spectrum_array.shape)
    print(train_target.shape)
    print(test_spectrum_array.shape)
    print(test_target.shape)
    print(val_spectrum_array.shape)
    print(val_target.shape)

    np.save('train_feature5.npy', train_spectrum_array)
    np.save('train_target5.npy', train_target)
    np.save('test_feature5.npy', test_spectrum_array)
    np.save('test_target5.npy', test_target)
    np.save('validation_feature5.npy', val_spectrum_array)
    np.save('validation_target5.npy', val_target)


def load_dataset(date_list, pen_id_list):
    with open(os.path.join(LOAD_PATH, 'images', 'images.json'), 'r') as file:
        hsi_informations = json.load(file)
    
    mask_file_list = os.listdir(os.path.join(LOAD_PATH, 'anotation', 'mask_images'))

    skipline_mask_img = np.zeros((1080, 2048), dtype=np.uint8)
    for y in range(1080):
        for x in range(0, 2048, 3):
            skipline_mask_img[y, x] = 255

    dataset = []
    for hsi_information in tqdm(hsi_informations):
        img_id = hsi_information['image_id']
        date = hsi_information['date']
        if not (date in date_list):
            continue
        hsi = nh9_to_array(os.path.join(LOAD_PATH, 'images', 'HS', img_id + '.nh9'), 
                           height=1080, width=2048, spectral_dimension=151)
        penguins_data = []
        for mask_file in mask_file_list:
            if (mask_file == 'white') or (mask_file == 'black') or (mask_file == 'mask_images.json'):
                continue
            mask_id = mask_file.split('.')[0]
            pen_id = mask_id.split('_')[1]
            if (pen_id in pen_id_list) and (img_id == mask_id.split('_')[0]):
                mask_img = cv2.imread(os.path.join(LOAD_PATH, 'anotation', 'mask_images', mask_id + '.png'), cv2.COLOR_BGR2GRAY)
                mask_img = cv2.erode(mask_img, np.ones((3, 3), np.uint8), iterations=3)

                smooth_hsi = hsi_blur(hsi=hsi)
                pen_spectrum = hsi[(mask_img == 255) & (skipline_mask_img == 255)]
                smooth_spectrum = smooth_hsi[(mask_img == 255) & (skipline_mask_img == 255)]
                pen_spectrum = pick_white_area(pen_spectrum, smooth_spectrum)
                # pen_spectrum = noise_ref(spectrum=pen_spectrum, pix_num=5)
                # print(pen_spectrum.shape)
                pen_id_index = [pen_id_list.index(pen_id)]
                
                penguin_data = {'spectrum': pen_spectrum, 'pen_id': pen_id_index}
                penguins_data.append(penguin_data)
        if len(penguins_data) > 0:
            dataset.append({'img_id': img_id, 'data': penguins_data})
        dataset = sorted(dataset, key=lambda x: int(x['img_id']))
    return dataset

def pick_white_area(spectrum, ref_spectrum):
    spectrum = (spectrum - np.min(spectrum, axis=0)) / (np.max(spectrum, axis=0) - np.min(spectrum, axis=0))
    gmm = GaussianMixture(n_components=2)
    gmm.fit(spectrum)
    labels = gmm.predict(spectrum)

    spectrum_0 = ref_spectrum[labels == 0]
    spectrum_1 = ref_spectrum[labels == 1]

    if np.average(spectrum_0) > np.average(spectrum_1):
        white_area_spectrum = spectrum_0
    else:
        white_area_spectrum = spectrum_1
    return white_area_spectrum


def noise_ref(spectrum, pix_num=5):
    new_shape = (pix_num, spectrum.shape[0] // pix_num, spectrum.shape[1])
    reshape_spectrum = spectrum[:new_shape[1] * pix_num].reshape(new_shape)
    new_spectrum = np.mean(reshape_spectrum, axis=0)
    return new_spectrum


def split_dataset(dataset, train_rate, random_seed):
    random.seed(random_seed)
    random.shuffle(dataset)

    train_ratio = train_rate
    val_ratio = (1 - train_ratio) * 0.5

    total_samples = len(dataset)
    train_samples = int(train_ratio * total_samples)
    val_samples = int(val_ratio * total_samples)

    train_dataset = dataset[:train_samples]
    val_dataset = dataset[train_samples:train_samples+val_samples]
    test_dataset = dataset[train_samples+val_samples:]

    train_feature, train_target = open_dataset(train_dataset)
    val_feature, val_target = open_dataset(val_dataset)
    test_feature, test_target = open_dataset(test_dataset)
    return train_feature, train_target, test_feature, test_target, val_feature, val_target


def open_dataset(dataset):
    spectrums, pen_ids = [], []
    for hsi_data in dataset:
        for pen_data in hsi_data['data']:
            spectrums.append(pen_data['spectrum'])
            pen_ids.append(pen_data['pen_id'])
    return spectrums, pen_ids


def list2array(X_list, y_list):
    X = np.empty((0, X_list[0].shape[1]))
    batch_size = 2000
    for i in range(0, len(X_list), batch_size):
        X_batch = np.concatenate(X_list[i:i + batch_size], axis=0)
        X = np.concatenate([X, X_batch], axis=0)

    y_array = []
    for x, y in zip(X_list, y_list):
        for i in range(x.shape[0]):
            y_array.append(y)        
    y = np.array(y_array)
    return X, y


def sampling_data(features, labels, target_count, random_seed=0):
    unique_labels = np.unique(labels)
    balanced_features = []
    balanced_labels = []
    
    print(features.shape)
    print(labels.shape)
    print(unique_labels.shape)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.seed(random_seed)
        selected_indices = np.random.choice(label_indices, size=target_count, replace=False)
        balanced_features.extend(features[selected_indices])
        balanced_labels.extend([label] * target_count)

    return np.array(balanced_features), np.array(balanced_labels)


if __name__ == '__main__':
    generate_dataset()