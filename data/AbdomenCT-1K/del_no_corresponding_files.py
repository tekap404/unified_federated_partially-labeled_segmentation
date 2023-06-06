from genericpath import exists
import os
import shutil

def del_img(data_path):

    train_img_path = data_path + 'imagesTr/'
    output_train_path_img = './AbdomenCT/no_respond_file/img'
    os.makedirs(output_train_path_img, exist_ok=True)

    train_img_path_list = os.listdir(train_img_path)
    train_img_path_list.sort(key=lambda x : int(x.split('.')[0].split('_')[1] ))

    count = 0
    len_train_list = len(train_img_path_list)
    
    for a in range(len_train_list-1, -1, -1):
        label_path = (train_img_path + train_img_path_list[a]).replace('imagesTr/', 'labelsTr/')
        if not exists(label_path):
            shutil.copy(train_img_path + train_img_path_list[a], output_train_path_img)
            os.remove(train_img_path + train_img_path_list[a])
            print(train_img_path + train_img_path_list[a])
            count+=1
        
    print(f'{count} imgs without mask')

def del_mask(data_path):

    train_label_path = data_path + 'labelsTr/'
    output_train_path_label = './AbdomenCT/no_respond_file/label'
    os.makedirs(output_train_path_label, exist_ok=True)

    train_label_path_list = os.listdir(train_label_path)
    train_label_path_list.sort(key=lambda x : int(x.split('.')[0].split('_')[1] ))

    count = 0
    len_train_list = len(train_label_path_list)
    
    for a in range(len_train_list-1, -1, -1):
        img_path = (train_label_path + train_label_path_list[a]).replace('labelsTr/', 'imagesTr/')
        if not exists(img_path):
            shutil.copy(train_label_path + train_label_path_list[a], output_train_path_label)
            os.remove(train_label_path + train_label_path_list[a])
            print(train_label_path + train_label_path_list[a])
            count+=1
        
    print(f'{count} masks without img')

# 61 imgs without mask, 6 masks without img
if __name__ == '__main__':
    data_path = './AbdomenCT/'

    del_img(data_path)
    del_mask(data_path)