import os
import shutil

data_path = './AbdomenCT/'

train_img_path = data_path + 'imagesTr/'
train_label_path = data_path + 'labelsTr/'

train_img_path_list = os.listdir(train_img_path)
total_len_train_list = 0
# filter large files to accelerate training
for a in range(len(train_img_path_list)-1, -1, -1):
    file_stats = os.stat(train_img_path + train_img_path_list[a])
    file_size = file_stats.st_size // (1024*1024)
    if file_size < 30:
        total_len_train_list += 1
    else:
        del train_img_path_list[a]
len_train_list = total_len_train_list * 7 // 10

train_label_path_list = os.listdir(train_label_path)

# train
output_train_path_img = data_path + 'Client2/train/img/'
output_train_path_label = data_path + 'Client2/train/label/'
os.makedirs(output_train_path_img, exist_ok=True)
os.makedirs(output_train_path_label, exist_ok=True)

for a in range(len_train_list-1, -1, -1):
    shutil.copy(train_img_path + train_img_path_list[a], output_train_path_img)
    shutil.copy(train_label_path + train_img_path_list[a], output_train_path_label)
    del train_img_path_list[a]

# val
len_val_list = len(train_img_path_list) // 3
output_val_path_img = data_path + 'Client2/val/img/'
output_val_path_label = data_path + 'Client2/val/label/'
os.makedirs(output_val_path_img, exist_ok=True)
os.makedirs(output_val_path_label, exist_ok=True)

for a in range(len_val_list-1, -1, -1):
    shutil.copy(train_img_path + train_img_path_list[a], output_val_path_img)
    shutil.copy(train_label_path + train_img_path_list[a], output_val_path_label)
    del train_img_path_list[a]


# test
len_test_list = len(train_img_path_list)
output_test_path_img = data_path + 'Client2/test/img/'
output_test_path_label = data_path + 'Client2/test/label/'
os.makedirs(output_test_path_img, exist_ok=True)
os.makedirs(output_test_path_label, exist_ok=True)

for a in range(len_test_list-1, -1, -1):
    shutil.copy(train_img_path + train_img_path_list[a], output_test_path_img)
    shutil.copy(train_label_path + train_img_path_list[a], output_test_path_label)
    del train_img_path_list[a]