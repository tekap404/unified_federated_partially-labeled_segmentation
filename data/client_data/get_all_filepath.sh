mv ../WORD/WORD-propressed/Client1 .
mv ../AbdomenCT-1K/AbdomenCT-propressed/Client2 .
mv ../AMOS2022/AMOS22-propressed/Client3 .
mv ../BTCV/BTCV-propressed/Client4 .
python rearrange.py
python generate_csv.py
python preprocess_json.py