import os
import shutil

if __name__ == '__main__':
    data_path = './BTCV/'

    img_path = data_path + 'imagesTr/'
    label_path = data_path + 'labelsTr/'

    img_path_list = os.listdir(img_path)
    label_path_list = os.listdir(label_path)

    len_test_list = len(img_path_list)

    output_test_path_img = data_path + 'Client4/test/img/'
    output_test_path_label = data_path + 'Client4/test/label/'
    os.makedirs(output_test_path_img, exist_ok=True)
    os.makedirs(output_test_path_label, exist_ok=True)

    for a in range(len_test_list-1, -1, -1):
        shutil.copy(img_path + img_path_list[a], output_test_path_img)
        shutil.copy(label_path + img_path_list[a], output_test_path_label)
        del img_path_list[a]