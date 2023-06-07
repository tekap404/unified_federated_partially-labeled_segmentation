import os
import json
import shutil

def move_files(path):

    client_list = ['Client1', 'Client2', 'Client3']  # out-FL client do not need this step
    for client in client_list:
        path_c = path + client
        mode = ['val', 'test']
        for m in mode:
            path_m = os.path.join(path_c, m)
            type = ['img', 'label']
            for t in type:
                path_t = os.path.join(path_m, t)
                data_path = os.listdir(path_t)
                for data_p in data_path:
                    ori_path = os.path.join(path_t, data_p)
                    shutil.move(ori_path, ori_path.replace(f'/{m}/', '/train/'))

def read_jsons(json_path):

    json_c = json.load(open(json_path))

    mode = ['train', 'val', 'test']
    img, mask = [], []
    for m in mode:
        all = json_c[m]
        for data in all:
            img.append(data['image'].replace('../data/client_data', '.'))
            mask.append(data['mask'].replace('../data/client_data', '.'))

    return img, mask

def redivide_files(path):

    client_list = ['client_1', 'client_2', 'client_3']  # out-FL client do not need this step
    for client in client_list:

        json_path = f'./path_json/{client}.json'
        imgs_path, labels_path = read_jsons(json_path)

        for img_path, labels_path in zip(imgs_path, labels_path):
            try:
                shutil.move(img_path.replace(f'/val/', '/train/'), img_path)
                shutil.move(labels_path.replace(f'/val/', '/train/'), labels_path)
            except:
                try:
                    shutil.move(img_path.replace(f'/test/', '/train/'), img_path)
                    shutil.move(labels_path.replace(f'/test/', '/train/'), labels_path)
                except:
                    pass

if __name__ == '__main__':

    path = './'
    move_files(path)
    redivide_files(path)    # to normal distribution

    
