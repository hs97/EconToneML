import pandas as pd
import datetime
from keras.models import model_from_json
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

def to_seconds(x):
    """
    This function converts the time formatting for video start/end time into seconds
    """
    if pd.notna(x):
        x = x.replace(' ', '')
        duration = datetime.datetime.strptime(x, "%H:%M:%S")
        timedelta = duration - datetime.datetime(1900, 1, 1)
        seconds = timedelta.total_seconds()
        return float(seconds)
    else:
        return None
    
def evaluate(model_dir, model_name, json_name, lb_path, X, continuous, outcome, i):
    """ 
    This function loads the model and evaluate
    """
    json_file = open(model_dir + json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_dir + model_name)
    # evaluate loaded model on test data
    preds = loaded_model.predict(X, 
                                 batch_size=32, 
                                 verbose=1)
    if not continuous:
        lb = LabelEncoder()
        lb.classes_ = np.load(lb_path, allow_pickle=True)
        if outcome == 'emotion':
            cols = [emotion + f'_{i}' for emotion in lb.inverse_transform((range(preds.shape[1])))]
            return pd.DataFrame(preds, columns=cols)
        else:    
            preds1 = preds.argmax(axis=1)
            predictions = (lb.inverse_transform((preds1)))
            return pd.DataFrame(predictions, columns=[outcome + f'_{i}'])
    else:
        return pd.DataFrame(np.ravel(preds.T), columns=[outcome + f'_{i}'])

def choose_gender_val(row, varname):
    """
    This function chooses which column depending on the gender labels
    """
    return row[f"{row['gender']}_{varname}"]