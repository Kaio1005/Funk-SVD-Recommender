import pandas as pd
import numpy as np
import util
import sys

ratings = sys.argv[1]
targets = sys.argv[2]

df = pd.read_csv(ratings)

df[['UserId', 'ItemId']] = df['UserId:ItemId'].str.split(':', n=1, expand=True)

users = df['UserId'].unique()
items = df['ItemId'].unique()

rename_items = util.Maping()

rename_items.map_ids(df)

triples = []
for index, row in df.iterrows():
    triple = util.get_triple(row)
    triples.append(triple)

m = len(users)
n = len(items)

#diminuir fatores melhorou, aumentar epocas piorou
#melhor até agora: 5,40, alpha = 0.005
P, Q = util.SGD(triples, m, n, latent_factors=5, epochs=40, alpha = 0.005)

df_previsions = pd.read_csv(targets)

df_previsions[['UserId', 'ItemId']] = df_previsions['UserId:ItemId'].str.split(':', n=1, expand=True)

rename_items.map_to(df_previsions)

duples = []

for index, row in df_previsions.iterrows():
    duples.append((row['UserId_int'], row['ItemId_int']))

ratings = util.predict(duples, P, Q)

ratings = util.adjust_ratings (ratings)

previsions = pd.DataFrame()

previsions['UserId:ItemId'] = df_previsions['UserId:ItemId'].copy()
previsions['Rating'] = list(ratings.values())

previsions.to_csv("submission.csv", index=False)


