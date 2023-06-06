import glob
import numpy as np

def process_label(path, subjectid):
    niblabel = np.load(path[subjectid])
    niblabel_new = np.zeros((5, niblabel.shape[0], niblabel.shape[1], niblabel.shape[2])).astype(np.int8)
    for i in range(14):
        if i == 2 or i == 3:
            niblabel_new[1, niblabel==i] = 1
        elif i == 6:
            niblabel_new[2, niblabel==i] = 1
        elif i == 1:
            niblabel_new[3, niblabel==i] = 1
        elif i == 11:
            niblabel_new[4, niblabel==i] = 1
        else:
            niblabel_new[0, niblabel==i] = 1
    np.save(path[subjectid], niblabel_new)


if __name__ == '__main__':
    # test
    test_path = './BTCV-propressed/Client4/test/label'
    test_file_path = glob.glob(test_path + '/*')
    for subjectid in range(len(test_file_path)):
        process_label(test_file_path, subjectid)
