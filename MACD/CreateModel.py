import numpy as np
import keras_tuner as kt
import pandas as pd
import time
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pickle import dump


start_time = time.time()

def set_up_variables():
    maxLayers = 3
    # (optional but change recomended depending on input data) warning increasing amount will increase time taken to run.
    # Recomended values (length of the output values) - (length of the input values). Default value given is 512.
    maxUnits = 256
    # (optional but change recomended depending on input data) warning decreasing amount will increase time taken to run.
    # Recomended values depends on the maxUni ts used. Default value given is 32.
    unitDecriment = 32
    # (optional but change recomended depending on input data) warning increasing amount will increase time taken to run.
    # Recomended values 0.1 - 0.8. Default value given is 0.5.
    maxDropout = 0.4
    # (optional but change recomended depending on input data) warning decreasing amount will increase time taken to run.
    # Recomended values depends on the maxDropout used. Default value given is 0.1.
    dropoutDecriment = 0.1
    # (optional) Recomended values 1 - (length of the input values). Default value 128.
    batchSize = 128
    # (optional) warning increasing amount will increase time taken to run. Default value 32.
    searchMaxEpochs = 48
    # (optional) warning increasing amount will increase time taken to run. Default value 5.
    errorCheckLoopAmount = 5
    # (optional) warning increasing amount will increase time taken to run. Default value 32.
    predictEpochs = 4

    return maxLayers, maxUnits, unitDecriment, maxDropout, dropoutDecriment, batchSize, searchMaxEpochs, errorCheckLoopAmount, predictEpochs

def get_data(filePath):
    data = pd.read_csv(filePath, index_col=False)

    trainSplit = data.values[2500:, :]
    testSplit = data.values[:2500, :]
    train_act = data.values[2500:, :]
    test_act = data.values[:2500, :]

    train_x = trainSplit[:, :-1]
    scaler = StandardScaler().fit(train_x)
    train_x= scaler.transform(train_x)
    train_x = np.expand_dims(train_x, axis=1)
    train_y = train_act[:, -1:]
    test_x = testSplit[:, :-1]
    test_x= scaler.transform(test_x)
    test_x = np.expand_dims(test_x, axis=1)
    test_y = test_act[:, -1:]


    return train_x, train_y, test_x, test_y, scaler

def createModel(hp):
    """Builds and trains a neural network model using the imported hyperparameters and train data.

    Parameters
    ----------
    hp : object
        object that we pass to the model-building function, that allows us to define the space search of the hyperparameters
    
    Returns
    -------
    model : object
        Seqential model the neural network model that was trained using the x and y inputs with the set hyperparameters
    Object hp is an object that we pass to the model-building function, that allows us to define the space search of the hyperparameters
    return: Seqential model the neural network model that was trained using the x and y inputs with the set 
    hyperparameters.
    """
    model = Sequential()
    #Global variable hpValues used in model building to specify the min and max values as well as the steps to take.
    #First LSTM layer specifies the input shape. The last LSTM layer is to set return sequences to false so the model knows it will be the last LSTM layer.
    #Model can be changed depending on the data input however testing will need to be done to assure overfirring is not a problem.
    model.add(LSTM(hp.Int("inputUnit", min_value = unitDecriment, max_value = maxUnits, step = unitDecriment), return_sequences = True, input_shape = (train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(hp.Float('input_Dropout_rate', min_value = dropoutDecriment, max_value = maxDropout, step = dropoutDecriment)))
    for i in range(hp.Int("nLayers", 0, maxLayers)):
        model.add(LSTM(hp.Int(f'{i}lstmLayer', 0, max_value = maxUnits, step = unitDecriment), return_sequences = True))
        model.add(Dropout(hp.Float(f'{i}dropoutLayer', min_value = dropoutDecriment, max_value = maxDropout, step = dropoutDecriment)))
    model.add(LSTM(hp.Int('layer_2_neurons', min_value = 0, max_value = maxUnits, step = unitDecriment)))
    model.add(Dropout(hp.Float('end_Dropout_rate', min_value = dropoutDecriment, max_value = maxDropout, step = dropoutDecriment)))
    model.add(Dense(train_y.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ["accuracy"])

    return model

def createBasicModel():
    model = Sequential()
    model.add(Dense(units=128, input_shape = (train_x.shape[1], train_x.shape[2])))
    model.add(Dense(train_y.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ["accuracy"])

    return model

def hyperPerameterTweaking():
    """Uses hyperband tuner to build model and test depending on the model that manages to get the mse the lowest. 
    The the best models are accuracy checkedusing test data to ensure the model picked hasnt overfitted which leads to a low mse without being accurate.

    Parameters
    ----------
    trainX : Matrix/List
        inputs that are used to train the neural network
    trainY : Matrix/List
        inputs that will be used to error check the neural network whilst training
    testX : Matrix/List
        inputs that are used to predict values once the network has been trained

    Returns
    -------
    (tuner.get_best_hyperparameters(num_trials = self.errorCheckLoopAmount)[modelNo]) : List
        this is the best preforming model in both mse and percentage error
    tuner : object
        the hyperband tuner that is used to go through different hyperpareter values and saves the trials to the dictionary specified"""
    
    #Tuner instantiated with the objective of minimal mse (max_epochs, factor are both set to default values).
    tuner = kt.Hyperband(
        createModel, 
        objective = "accuracy",
        max_epochs = searchMaxEpochs,
        directory = f"MACD\\ml",
        project_name = "macd_tests")
    #Preforms the search for the best hyperparameters using the train data(epochs and batch size used holds default values).
    tuner.search(train_x, train_y, epochs = searchMaxEpochs, batch_size = batchSize)
        
    return tuner.get_best_hyperparameters(num_trials = errorCheckLoopAmount)[0], tuner

def tuned_model():
    #Tuning occures to obtain best hyperparameters using the given data given.
    bestHp, tuner = hyperPerameterTweaking()
    bestModel = tuner.hypermodel.build(bestHp)

    #Fits the a model and stores MSE history to then check the epoch that has the lowest MSE.
    bestEpochTest = bestModel.fit(train_x, train_y, epochs = searchMaxEpochs)
    history = bestEpochTest.history["accuracy"]
    bestEpoch = history.index(min(history)) + 1

    #Refits the model using the best epoch loop found previously and uses this model and test data to make predictions.
    bestModel.fit(train_x, train_y, epochs = bestEpoch)
    predictions = bestModel.predict(test_x)

    threshold = 0.5
    predictions = (predictions >= threshold).astype(int)
    return predictions, bestModel

def test_model():
    model = get_model()
    predictions = model.predict(test_x)

    threshold = 0.5
    predictions = (predictions >= threshold).astype(int)
    return predictions

def get_model():
    preModel = load_model("MACD\\ml\\MACD")
    return preModel

def test_basic_model():
    model = createBasicModel()
    print(model.summary())
    model.fit(train_x, train_y, epochs = searchMaxEpochs)
    predictions = model.predict(test_x)

    threshold = 0.5
    predictions = (predictions >= threshold).astype(int)
    return predictions

filePath = "MACD\\ml_data.csv"
maxLayers, maxUnits, unitDecriment, maxDropout, dropoutDecriment, batchSize, searchMaxEpochs, errorCheckLoopAmount, predictEpochs = set_up_variables()

#Fetches and formats train data and test data as well as the actual prices, then sets objects parameters to that data.
train_x, train_y, test_x, test_y, scaler= get_data(filePath)
# predictions = test_basic_model()
# predictions = np.squeeze(predictions, axis=2)
predictions = test_model()
# predictions, model = tuned_model()

#ac_true_signals
print(np.sum(test_y == 1))
#ac_false_signals
print(np.sum(test_y == 0))

#True Positives
print(f"True Positives: {np.sum((test_y == 1) & (predictions == 1))}")
#False Positives
print(f"False Positives: {np.sum((test_y == 0) & (predictions == 1))}")
#True Negatives
print(f"True Negatives: {np.sum((test_y == 0) & (predictions == 0))}")
#False Negatives
print(f"False Negatives: {np.sum((test_y == 1) & (predictions == 0))}")

# """
# HOW TO LOAD MODELS
# preModel = load_model("FYP\\Scripts\\MACD Short Term\\MachineLearningModel")
# predictions = use_model(preModel)
# """

# true_recall = np.sum((test_y == 1) & (predictions == 1))
# incorrect_true_recall = np.sum((test_y == 0) & (predictions == 1))

# print(np.sum(test_y == 1) / (np.sum(test_y == 0) + np.sum(test_y == 1)), np.sum((test_y == 1)))
# print(true_recall/(true_recall + incorrect_true_recall), np.sum((predictions == 1)))

# decision = input("Do you want to save the model:")

# if(decision == "yes"):
#     name = input("name your model: ")
#     model.save(f"MACD\\ml\\{name}")
#     dump(scaler, open(f'MACD\\ml\\{name}_scaler.pkl', 'wb'))
# else:
#     print("ok")

# end_time = time.time()

# print(f"Elapsed Time: {end_time - start_time} seconds")
