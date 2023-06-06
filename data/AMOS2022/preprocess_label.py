import glob
import numpy as np

def process_label(path, subjectid):
    niblabel = np.load(path[subjectid])
    niblabel_new = np.zeros((5, niblabel.shape[0], niblabel.shape[1], niblabel.shape[2])).astype(np.int8)

    for i in range(16):
        if i == 2 or i == 3:
            niblabel_new[1, niblabel==i] = 1
        elif i == 6:
            niblabel_new[2, niblabel==i] = 1
        elif i == 1:
            niblabel_new[3, niblabel==i] = 1
        elif i == 10:
            niblabel_new[4, niblabel==i] = 1
        else:
            niblabel_new[0, niblabel==i] = 1

    np.save(path[subjectid], niblabel_new)

if __name__ == '__main__':
    # train
    train_path = './AMOS22-propressed/Client3/train/label'
    train_file_path = glob.glob(train_path + '/*')
    for subjectid in range(len(train_file_path)):
        process_label(train_file_path, subjectid)
        
    # val
    val_path = './AMOS22-propressed/Client3/val/label'
    val_file_path = glob.glob(val_path + '/*')
    for subjectid in range(len(val_file_path)):
        process_label(val_file_path, subjectid)

    # test
    test_path = './AMOS22-propressed/Client3/test/label'
    test_file_path = glob.glob(test_path + '/*')
    for subjectid in range(len(test_file_path)):
        process_label(test_file_path, subjectid)