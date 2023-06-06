import glob
import os

if __name__ == '__main__':
    base_path = './AbdomenCT/imagesTr'
    file_path = glob.glob(base_path + '/*')

    for subjectid in range(len(file_path)):
        oldname = file_path[subjectid]
        newname = file_path[subjectid].replace('_0000.', '.')
        os.rename(oldname, newname)