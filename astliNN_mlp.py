from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import legacy

def astliNN_MLP(latent_dim=7):
    model_mlp = Sequential()
    model_mlp.add(Input(shape=14))
    model_mlp.add(Dense(161, activation='relu'))
    model_mlp.add(Dense(320, activation='relu'))
    model_mlp.add(Dense(511, activation='tanh'))
    model_mlp.add(Dense(latent_dim, activation='sigmoid'))
    
    model_mlp.compile(optimizer=legacy.Adam(), loss='mean_squared_error', 
                      metrics=[RootMeanSquaredError()])
    return model_mlp