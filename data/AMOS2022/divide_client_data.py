import os
import shutil

if __name__ == '__main__':
    data_path = './AMOS22/'

    train_img_path = data_path + 'imagesTr/'
    train_label_path = data_path + 'labelsTr/'

    train_img_path_list = os.listdir(train_img_path)
    for i in range(len(train_img_path_list)-1, -1, -1):
        cur = int(train_img_path_list[i].split('.')[0].split('_')[1])
        if cur > 500:   # delete all mri data
            del train_img_path_list[i]

    len_train_list = len(train_img_path_list) * 7 // 10
    train_label_path_list = os.listdir(train_label_path)

    # train
    output_train_path_img = data_path + 'Client3/train/img/'
    output_train_path_label = data_path + 'Client3/train/label/'
    os.makedirs(output_train_path_img, )
    os.makedirs(output_train_path_label)

    for a in range(len_train_list-1, -1, -1):
        shutil.copy(train_img_path + train_img_path_list[a], output_train_path_img)
        shutil.copy(train_label_path + train_img_path_list[a], output_train_path_label)
        del train_img_path_list[a]

    # val
    len_val_list = len(train_img_path_list) // 3
    output_val_path_img = data_path + 'Client3/val/img/'
    output_val_path_label = data_path + 'Client3/val/label/'
    os.makedirs(output_val_path_img)
    os.makedirs(output_val_path_label)

    for a in range(len_val_list-1, -1, -1):
        shutil.copy(train_img_path + train_img_path_list[a], output_val_path_img)
        shutil.copy(train_label_path + train_img_path_list[a], output_val_path_label)
        del train_img_path_list[a]

    # test
    len_test_list = len(train_img_path_list)
    output_test_path_img = data_path + 'Client3/test/img/'
    output_test_path_label = data_path + 'Client3/test/label/'
    os.makedirs(output_test_path_img)
    os.makedirs(output_test_path_label)

    for a in range(len_test_list-1, -1, -1):
        shutil.copy(train_img_path + train_img_path_list[a], output_test_path_img)
        shutil.copy(train_label_path + train_img_path_list[a], output_test_path_label)
        del train_img_path_list[a]