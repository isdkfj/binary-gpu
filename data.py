import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def process_binary(X):
    print('#instances: {}, #features: {}'.format(X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        s0 = np.sum(np.isclose(X[:, i], 0))
        s1 = np.sum(np.isclose(X[:, i], 1))
        if s0 + s1 == X.shape[0]:
            if s0 > s1:
                # swap 0 and 1 if there are more 0's
                X[:, i] = 1 - X[:, i]
                s0, s1 = s1, s0
            print('feature no.{} is binary, {}% are 1\'s'.format(i, s1 / X.shape[0] * 100))

def load_data(dname, path, SEED):
    if dname == 'bank':
        path = os.path.join(path, 'bank-marketing/bank-additional-full.csv')
        df = pd.read_csv(path, delimiter=';')
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, :-1]
        Y = df.values[:, -1].astype('int')
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    elif dname == 'credit':
        path = os.path.join(path, 'default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
        df = pd.read_csv(path, delimiter=',')
        df = df.drop(columns=['ID'])
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, :-1]
        Y = df.values[:, -1].astype('int')
        X[:, 1] -= 1
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    elif dname == 'mushroom':
        path = os.path.join(path, 'mushroom-classification/mushrooms.csv')
        df = pd.read_csv(path)
        df = df.drop(columns=['veil-type'])
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, 1:].astype('float')
        Y = df.values[:, 0].astype('int')
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    elif dname == 'nursery':
        path = os.path.join(path, 'nursery/nursery.csv')
        df = pd.read_csv(path)
        df['parents'] = df['parents'].map({'usual': 0, 'pretentious': 1, 'great_pret': 2})
        df['has_nurs'] = df['has_nurs'].map({'proper': 0, 'less_proper': 1, 'improper': 2, 'critical': 3, 'very_crit': 4})
        df['form'] = df['form'].map({'complete': 0, 'completed': 1, 'incomplete': 2, 'foster': 3})
        df['children'] = df['children'].map({'1': 1, '2': 2, '3': 3, 'more': 4})
        df['housing'] = df['housing'].map({'convenient': 0, 'less_conv': 1, 'critical': 2})
        df['finance'] = df['finance'].map({'convenient': 0, 'inconv': 1})
        df['social'] = df['social'].map({'nonprob': 0, 'slightly_prob': 1, 'problematic': 2})
        df['health'] = df['health'].map({'recommended': 0, 'priority': 1, 'not_recom': 2})
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, :-1].astype('float')
        Y = df.values[:, -1].astype('int')
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    elif dname == 'covertype':
        path = os.path.join(path, 'forest-cover-type-dataset/covtype.csv')
        df = pd.read_csv(path)
        for attr in df.columns:
            if df[attr].dtype == 'object':
                encoder= LabelEncoder().fit(df[attr])
                df[attr] = encoder.transform(df[attr])
        X = df.values[:, :-1].astype('float')
        Y = df.values[:, -1].astype('int') - 1
        process_binary(X)
        fake_label = np.random.randint(0, 2, (X.shape[0], 1))
        X = np.concatenate([X, fake_label], axis=1)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    min_X = np.min(train_X, axis=0)
    train_X -= min_X
    test_X -= min_X
    max_X = np.max(train_X, axis=0)
    train_X /= max_X
    test_X /= max_X
    return train_X, test_X, train_Y, test_Y
