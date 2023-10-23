import numpy as np

class Maping:

    def __init__(self) -> None:
        self.user_map = {}
        self.item_map = {}

    def map_ids (self, df):
        users = df['UserId'].unique()
        items = df['ItemId'].unique()

        for i,user in enumerate(users):
            self.user_map[user] = i
        for i,item in enumerate(items):
            self.item_map[item] = i

        self.map_to(df)
    
    def map_to (self, df):
        df['UserId_int'] = df['UserId'].map(self.user_map)
        df['ItemId_int'] = df['ItemId'].map(self.item_map)
    
def get_triple (row):
    return(row['UserId_int'], row['ItemId_int'], row['Rating'])

def SGD (known_pairs, num_users, num_items, latent_factors = 100, alpha = 0.005, regularization_parameter = 0.02, epochs = 20):
    P = np.ones([num_users, latent_factors])
    Q = np.ones([latent_factors, num_items])
    #tentar inicializar P e Q com 5*rand/sqrt(5)
    errors = {}
    for i in range(epochs):
        for user_idx, item_idx, rating in known_pairs:
            errors[(user_idx,item_idx)] = rating - np.dot(P[user_idx], Q[:, item_idx])
            P_aux = P[user_idx] + alpha * (errors[(user_idx,item_idx)] * Q[:, item_idx] - regularization_parameter * P[user_idx])
            Q_aux = Q[:, item_idx] + alpha * (errors[(user_idx,item_idx)] * P[user_idx] - regularization_parameter * Q[:, item_idx])
            P[user_idx] = P_aux
            Q[:, item_idx] = Q_aux
    
    return (P,Q)

def predict (pairs, P, Q):
    predictions = {}
    for pair in pairs:
        predictions[pair] = np.dot(P[pair[0]], Q[:, pair[1]])
    
    return predictions

def adjust_ratings (ratings):
    for pair in ratings:
        if ratings[pair] > 5:
            ratings[pair] = 5
        elif ratings[pair] < 1:
            ratings[pair] = 1

    return ratings