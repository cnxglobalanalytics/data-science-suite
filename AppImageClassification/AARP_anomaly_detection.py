import os.path
import pandas as pd
import numpy as np
import joblib
from ImageClassification.settings import BASE_DIR,MODELS_PATH

#path=r"C:\Users\RNALAB\Documents\Tuhin_Majumder\AppImageClassification\April_1_Demo\Quantitative"
#file_path=path.replace("\\","/")

def read_file(file_path):

    data_raw = pd.read_excel(file_path)

    data_raw=data_raw.rename(columns={'CALLS_HANDLED':'Calls Handled','SCHEDULE_ADHERENCE_REVISED':'Schedule Adherence Revised','AVG_HOLD_TIME':'Avg Hold Time','STAFFED_HRS':'Staffed Hrs'})

    data_raw_cols=[col.strip() for col in data_raw.columns]
    data_raw.columns=data_raw_cols

    data_raw['Aux Total Min']=data_raw["AUX_TIME_HRS"]*60

    data_raw=data_raw[data_raw["Calls Handled"]>0]

    data_raw=data_raw[(data_raw["Calls Handled"]> data_raw["Calls Handled"].quantile(0.05)) &
                        (data_raw["Calls Handled"]< data_raw["Calls Handled"].quantile(0.99))]
    
    data_raw["Aux Total Min"]=data_raw["Aux Total Min"].abs()

    data_raw=data_raw.replace([np.inf,-np.inf],0)      

    return data_raw

def aarp_iforest(data_raw):

    data_raw = read_file(data_raw)   

    X_features_selected= data_raw[['Avg Hold Time',"Aux Total Min",'Schedule Adherence Revised','Calls Handled','Staffed Hrs','AGENT_PRODUCTIVITY']].copy()
    X_features_selected.isnull().sum()
    X_features_selected.fillna(0, inplace=True)

    model = joblib.load(MODELS_PATH +'/Anomaly_Detection/Isolation_Forest/aarp_iforest.obj')
    data_raw["Sample Score"] = model.score_samples(X_features_selected)
    data_raw["IForest Risk Category"] = data_raw["Sample Score"].apply(lambda x : "Category 1:Low Risk" if x >= data_raw["Sample Score"].quantile(.60) else ("Category 2:Moderate Risk" if x >= data_raw["Sample Score"].quantile(.30) else("Category 3:High Risk" if x >= data_raw["Sample Score"].quantile(.05) else "Category 4:Very High Risk")))
    
    excel_filename = "IForest_Prediction" + ".xlsx"
    data_raw.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    return True, df_filepath

def aarp_onesvm(data_raw):

    data_raw = read_file(data_raw)   

    X_features_selected= data_raw[['Avg Hold Time',"Aux Total Min",'Schedule Adherence Revised','Calls Handled','Staffed Hrs','AGENT_PRODUCTIVITY']].copy()
    X_features_selected.isnull().sum()
    X_features_selected.fillna(0, inplace=True)

    model = joblib.load(MODELS_PATH +'/Anomaly_Detection/One_Class_SVM/aarp_svm.obj')
    data_raw["Sample Score"] = model.score_samples(X_features_selected)
    data_raw["One Class SVM Risk Category"] = data_raw["Sample Score"].apply(lambda x : "Category 1:Low Risk" if x >= data_raw["Sample Score"].quantile(.60) else ("Category 2:Moderate Risk" if x >= data_raw["Sample Score"].quantile(.30) else("Category 3:High Risk" if x >= data_raw["Sample Score"].quantile(.05) else "Category 4:Very High Risk")))
    
    excel_filename = "IForest_Prediction" + ".xlsx"
    data_raw.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    return True, df_filepath
    