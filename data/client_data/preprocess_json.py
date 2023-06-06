import pandas as pd
import os
import json

if __name__ == '__main__':
    data_3d_info = pd.read_csv('./data_path.csv')
    client_id = ['1', '2', '3', '4']

    os.makedirs('./path_json/', exist_ok=True)

    for cid in client_id:
        if cid != '4':
            train_data, val_data, test_data = [], [], []
            train_df = data_3d_info[data_3d_info["mode"] == 'train']
            train_df = train_df[data_3d_info["client_id"] == int(cid)]
            val_df = data_3d_info[data_3d_info["mode"] == 'val']
            val_df = val_df[data_3d_info["client_id"] == int(cid)]
            test_df = data_3d_info[data_3d_info["mode"] == 'test']
            test_df = test_df[data_3d_info["client_id"] == int(cid)]
            
            for line in train_df.values:
                train_data.append({"image": line[2], "mask": line[3]})
            for line in val_df.values:
                val_data.append({"image": line[2], "mask": line[3]})
            for line in test_df.values:
                test_data.append({"image": line[2], "mask": line[3]})

            all_data = {"train": train_data, "val": val_data, "test": test_data}
        else:
            test_data = []
            test_df = data_3d_info[data_3d_info["mode"] == 'test']
            test_df = test_df[data_3d_info["client_id"] == int(cid)]
            
            for line in test_df.values:
                test_data.append({"image": line[2], "mask": line[3]})

            all_data = {"test": test_data}
        
        with open(os.path.join('./path_json/', f"client_{cid}.json"), 'w') as f:
            json.dump(all_data, f)