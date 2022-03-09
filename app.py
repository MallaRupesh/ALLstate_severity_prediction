import pandas as pd
import numpy as np
import pickle
import streamlit as stl
import numpy as np
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
from bs4 import BeautifulSoup
# from werkzeug import filename
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from joblib import dump, load
import time
import itertools
import warnings
warnings.filterwarnings("ignore")





###################################################


def final_fun_1(X):
    "The function which transforms all the raw input into predictions"
    try :
        X = X.drop(['loss'], axis=1)
    except :
        X=X

    start = time.time()
    train_data = X.copy()

    top_cat_feats = "cat80,cat79,cat87,cat57,cat101,cat12,cat81,cat7,cat89,cat10,cat1,cat72,cat2,cat94,cat103,cat111,cat114,cat11,cat53,cat106,cat9,cat13,cat38,cat100,cat105,cat44,cat108,cat75,cat109,cat90,cat116,cat6,cat5,cat25".split(
        ',')

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    def encode(string):
        '''Using unicode encoding to encode the categorical variables , this encoding uses relative position of the alphabet to encode the categorical variables'''
        r = 0
        length = len(str(string))
        for i in range(length):
            # unicode of the alphabet - unicode of first letter
            # +1 to give maintain non zero postion
            # *26 for equating all the alphabets to a level as 26 is total number of alphabets
            # To the power of the position of the charcode
            r += (ord(str(string)[i]) - ord('A') + 1) * 26 ** (length - i - 1)
        return r

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    def mungeskewed(train, numeric_feats):
        '''This function checks for skewness in the categorical features and applies box-cox transformation'''
        ntrain = train.shape[0]

        # Calculating the skewness on the entire data's features
        skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
        # seperating the features which have higher than 0.25 skewness
        skewed_feats = skewed_feats[skewed_feats > 0.25]
        skewed_feats = skewed_feats.index

        # Transforming all the highly skewed variables with BOX-cox
        # Data leakage is avoided by checking the skewness on train_data only and skipping the test data
        for feats in skewed_feats:
            train[feats] = train[feats] + 1
            train[feats], lam = boxcox(train[feats])
        return train, ntrain

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    numeric_feats = [x for x in train_data.columns[1:-1] if 'cont' in x]
    categorical_feats = [x for x in train_data.columns[1:-1] if 'cat' in x]
    train_test, ntrain = mungeskewed(train_data, numeric_feats)
    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--

    train = train_test.iloc[:ntrain, :].copy()
    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--

    mimx_scale_data = load(open('min_max_scale.pkl', 'rb'))

    def min_max_scaler(data, var):
        #     print("initiated")
        scaled_data = []
        for i in (range(0, len(data))):
            X_std = (float(data[var].iloc[i]) - float(mimx_scale_data[var].min())) / (
                        float(mimx_scale_data[var].max()) - float(mimx_scale_data[var].min()))

            scaled_data.append(float(X_std))
        return scaled_data

    # Referenced from Ali's script (https://www.kaggle.com/aliajouz/allstate-claims-severity/singel-model-lb-1117)

    train["cont1"] = np.sqrt(min_max_scaler(train, "cont1"))
    train["cont4"] = np.sqrt(min_max_scaler(train, "cont4"))
    train["cont5"] = np.sqrt(min_max_scaler(train, "cont5"))
    train["cont8"] = np.sqrt(min_max_scaler(train, "cont8"))
    train["cont10"] = np.sqrt(min_max_scaler(train, "cont10"))
    train["cont11"] = np.sqrt(min_max_scaler(train, "cont11"))
    train["cont12"] = np.sqrt(min_max_scaler(train, "cont12"))

    train["cont6"] = np.log(min_max_scaler(train, "cont6"))
    train["cont7"] = np.log(min_max_scaler(train, "cont7"))
    train["cont9"] = np.log(min_max_scaler(train, "cont9"))
    train["cont13"] = np.log(min_max_scaler(train, "cont13"))
    train["cont14"] = (np.maximum(train["cont14"] - 0.179722, 0) / 0.665122) ** 0.25

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--

    #     print('Combining Columns')

    for comb in (itertools.combinations(top_cat_feats, 2)):
        feat = comb[0] + "_" + comb[1]
        train[feat] = train[comb[0]] + train[comb[1]]
        train[feat] = train[feat].apply(encode)

    #     print('Encoding columns')
    for col in (categorical_feats):
        train[col] = train[col].apply(encode)

    ss = load(open('tot_data_scale.pkl', 'rb'))
    train[numeric_feats] = ss.fit_transform(train[numeric_feats].values)

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    auto_res = load(open('auto_scaler.pkl', 'rb'))
    train1 = train_data.drop(['id'], axis=1).copy()
    k = []
    #     enc_dict={}
    #     list(auto_res.values)
    for col in train1.select_dtypes(include=['object']).columns:
        enc = auto_res[str(col)]
        train1[col] = enc.transform(train1[col])

    encoder = load_model('encoder1.h5', compile=False)

    encoder.run_eagerly = True
    X_train_encode = encoder.predict(train1)

    train_final = np.concatenate((train.drop(['id'], axis=1), X_train_encode), axis=1)

    # -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--
    d_train = xgb.DMatrix(train_final)
    modell = load(open('xgb_model.pkl', 'rb'))

    predictions = modell.predict(d_train)
    shift = 200
    print(f'Time: {time.time() - start}')

    return np.exp(predictions) - shift


###################################################




stl.title("All State Insurance Prediction")
stl.write("Input the data to return their respective predictions :")

filename = stl.text_input('Enter a file (in .txt) path:')

if(stl.button('Submit filename')):
	
   
	try:
	    with open(filename) as input:

	        # stl.text(input.read())
	        stl.text("File loaded")
	        
	except FileNotFoundError:
	    stl.error('File not found.')

	data = pd.read_csv(filename, sep=',')

# number_of_preds = stl.text_input('Enter the number of predictions needed :')

number_of_preds = stl.slider("Select the level", 1, 100)
  

stl.text('Selected: {}'.format(number_of_preds ))

if(stl.button('Submit number of preds')):
	number_of_preds=int(number_of_preds)
	data = pd.read_csv(filename, sep=',')

	
	stl.text("The input data is :")
	stl.text(data[:number_of_preds])


	stl.text("The predictions of the data are :")
	stl.text(final_fun_1(data[:number_of_preds]))






# for i in range (0,int(number_of_preds)):








