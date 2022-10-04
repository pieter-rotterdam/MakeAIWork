import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

df = pd.read_csv ('./simulations/car/control_client/sonar.samples.csv', sep=' ', names=["dist1","dist2","dist3","angle"])

(df.head())

X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=8,
                                            max_value=256,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory=('./simulations/car/control_client/'),
    project_name='tuner25')

tuner.search_space_summary()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test))
