import pandas as pd
import glob

if __name__ == '__main__':
    client_id = [1, 2, 3, 4]
    mode = ['train', 'val', 'test']

    client_list = []
    mode_list = []
    img_list = []
    label_list = []

    for cid in client_id:
        for mo in mode:
            if cid == 4 and (mo == 'train' or mo == 'val'):
                continue
            else:
                train_path = './Client' + str(cid) + '/' + mo + '/img'  # 必须是完整路径
                train_file_path= glob.glob(train_path + '/*')
                for subjectid in range(len(train_file_path)):
                    img_path = train_file_path[subjectid]
                    client_list.append(cid)
                    mode_list.append(mo)
                    img_list.append(img_path.replace('./', '../data/client_data/'))
                    label_list.append(img_path.replace('img', 'label').replace('./', '../data/client_data/'))

    frame = pd.DataFrame({'client_id':client_list, 'mode':mode_list, 'img_path':img_list, 'label_path':label_list})
    frame.to_csv("./data_path.csv", index=False, sep=',')