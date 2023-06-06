import glob
import os

base_path = './BTCV/imagesTr'
file_path= glob.glob(base_path + '/*')

for subjectid in range(len(file_path)):
    oldname= file_path[subjectid]
    newname= file_path[subjectid].replace('img00', 'btcv00')
    os.rename(oldname, newname)

base_path = './BTCV/labelsTr'
file_path= glob.glob(base_path + '/*')

for subjectid in range(len(file_path)):
    oldname= file_path[subjectid]
    newname= file_path[subjectid].replace('label00', 'btcv00')
    os.rename(oldname, newname)