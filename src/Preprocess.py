import numpy as np
import pandas as pd
import json


if __name__ == '__main__':
    # df = pd.read_csv('../data/message.csv', header=0, delimiter="\t", usecols=[4, 8])
    df = pd.read_csv('../data/sample.csv', header=0, delimiter=None, usecols=[4, 6, 8])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # isnull = df.isnull().any(axis=1)
    # df2 = df[isnull]
    conversations = []

    for index, row in df.iterrows():
        # data.append(['{}:{}'.format(int(row['isSend']), row['content']), row['createTime']])

        conv = {}
        list = []
        for i in range(index-10, index):
            list.append([df.loc[i]['content'], int(df.loc[i]['isSend'])])
        conv['content'] = list
        conv['response'] = [row['content'], int(row['isSend'])]
        conversations.append(conv)

    print(conversations)
    with open('../data/sample.json', 'w', encoding='utf-8') as file:
        json.dump(conversations, file, ensure_ascii=False, indent=2)
