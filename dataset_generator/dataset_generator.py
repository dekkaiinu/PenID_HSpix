import os
import random
import json
import pickle
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import cv2
import hsitools as hst


@hydra.main(version_base=None, config_path='cfg', config_name='config')
def main(cfg: DictConfig):
    save_folder = os.path.join(cfg.save_folder, cfg.save_file_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print('create_dataset running')
    dataset_path = create_dataset(cfg, save_folder)
    print('dataset_samplinmg running')
    dataset_path = dataset_sampling(dataset_path, cfg.classwize_data_num, cfg.reference_id_data, cfg.select_target, cfg.random_seed)
    print('split_dataset running')
    train_path, validation_path, test_path = split_dataset(dataset_path, save_folder, cfg.train_rate)


def create_dataset(cfg: DictConfig, save_folder):
    with open(cfg.mask_anotation_json, 'rb') as f:
        anotation_data_dict = json.load(f)
    
    save_path = os.path.join(save_folder, cfg.save_file_name + '.json')
    with open(save_path, 'w') as file:
        file.write('[')
        n_flag = False

    for index, anotation_data in enumerate(tqdm(anotation_data_dict)):
        if index == 0:
            hsi_path_stack = ''
        
        hsi_path = os.path.join(cfg.hsi_folder, anotation_data['image_id'] + cfg.binary_type)

        if hsi_path != hsi_path_stack:
            hsi = hst.convert.nh9_to_array(hsi_path)
            hsi_path_stack = hsi_path
            if cfg.blur:
                hsi = hst.correction.hsi_blur(hsi=hsi, kernel_size=cfg.blur)
        
        if cfg.mask_image == 'white':
            mask_img = cv2.imread(os.path.join(cfg.mask_folder, 'white', 'w' + anotation_data['mask_id'] + '.png'))
        elif cfg.mask_image == 'black':
            mask_img = cv2.imread(os.path.join(cfg.mask_folder, 'black', 'b' + anotation_data['mask_id'] + '.png'))
        else:
            mask_img = cv2.imread(os.path.join(cfg.mask_folder, anotation_data['mask_id'] + '.png'))
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        hs_pixels = hst.convert.extract_pixels_from_hsi_mask(hsi, mask_img)
        if cfg.mean:
            new_shape = (cfg.mean, hs_pixels.shape[0] // cfg.mean, hs_pixels.shape[1])
            reshaped_hs_pixels = hs_pixels[:new_shape[1] * cfg.mean].reshape(new_shape)
            hs_pixels = np.mean(reshaped_hs_pixels, axis=0) 

        for i in range(hs_pixels.shape[0]):
            append_hs_pixel = hs_pixels[i, :].tolist()
            data = {
                'target': anotation_data['category_id'],
                'data': append_hs_pixel,
                'meta_data': anotation_data['image_id']
            } 
            with open(save_path, 'a') as file:
                if n_flag:
                    file.write(',\n')
                else:
                    n_flag = True
                json.dump(data, file)
    with open(save_path, 'a') as file:
        file.write(']')
    return save_path


def dataset_sampling(dataset_path, classwize_data_num, ref_id_data_path, select_target, random_seed):
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
    random.seed(random_seed)
    random.shuffle(dataset)
    print(len(dataset))
    with open(ref_id_data_path, 'r') as file:
        ref_id_data = json.load(file)
        id_list = [item['category_id'] for item in ref_id_data]
        id_list = id_list[:select_target]
    
    sampling_dataset = []
    for id in id_list:
        select_id_data = [item for item in dataset if item['target'] == id]
        samling_data = select_id_data[:classwize_data_num]
        sampling_dataset.extend(samling_data)
    random.shuffle(sampling_dataset)
    print(len(sampling_dataset))

    with open(dataset_path, 'w') as file:
        json.dump(sampling_dataset, file)
    return dataset_path


def split_dataset(dataset_path, save_folder, train_rate):
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)

    total_samples = len(dataset)
    train_samples = int(train_rate * total_samples)
    val_samples = int((1 - train_rate) * 0.5 * total_samples)

    train_path = os.path.join(save_folder, 'train.json')
    validation_path = os.path.join(save_folder, 'validation.json')
    test_path = os.path.join(save_folder, 'test.json')

    train_dataset = dataset[:train_samples]
    validation_dataset = dataset[train_samples:train_samples+val_samples]
    test_dataset = dataset[train_samples+val_samples:]

    with open(train_path, mode='w') as file:
        json.dump(train_dataset, file)
    with open(validation_path, mode='w') as file:
        json.dump(validation_dataset, file)
    with open(test_path, mode='w') as file:
        json.dump(test_dataset, file)
    print(len(dataset))
    print(len(train_dataset))
    print(len(validation_dataset))
    print(len(test_dataset))

    return train_path, validation_path, test_path


if __name__ == '__main__':
    main()