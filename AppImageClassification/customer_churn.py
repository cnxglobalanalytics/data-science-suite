#import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler
import joblib
from ImageClassification.settings import BASE_DIR,MODELS_PATH

cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

imp_cols = ['Contract','PhoneService','OnlineSecurity','TechSupport','PaperlessBilling']

mapp = {1:'Yes',0:'No'}
map_Contract = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
map_PhoneService = {'No': 0, 'Yes': 1}
map_OnlineSecurity = {'No': 0, 'Yes': 1, 'No internet service': 2}
map_TechSupport = {'No': 0, 'Yes': 1, 'No internet service': 2}
map_PaperlessBilling = {'Yes': 0, 'No': 1}

#le = joblib.load('C:/Users/RNALAB/Documents/churn_le')
le = LabelEncoder()
#std = joblib.load('churn_std')
model = joblib.load(MODELS_PATH + '/customer_churn/churn_pred_clf')

def read_csv_data(csv_filename):

    if csv_filename.split(".")[-1] == 'csv':
        data = pd.read_csv(csv_filename)
        return data
    elif csv_filename.split(".")[-1] == 'xlsx':
        data = pd.read_excel(csv_filename)
        return data

def churn_pred(data):

    df = read_csv_data(data)
    cleanDF = df[imp_cols]

    #if 'Churn' in df.columns:

        #df = df.drop(['Churn'], axis=1)

    #cleanDF = df.drop(['customerID'], axis=1)
    cleanDF.Contract = cleanDF.Contract.map(map_Contract)
    cleanDF.PhoneService = cleanDF.PhoneService.map(map_PhoneService)
    cleanDF.OnlineSecurity = cleanDF.OnlineSecurity.map(map_OnlineSecurity)
    cleanDF.TechSupport = cleanDF.TechSupport.map(map_TechSupport)
    cleanDF.PaperlessBilling = cleanDF.PaperlessBilling.map(map_PaperlessBilling)
    #x = std.transform(cleanDF)
    predictions = model.predict(cleanDF)

    df["Churn_Predicted"] = pd.Series(predictions).apply(lambda x: mapp[x])
    df.drop(["Churn"],axis=1,inplace=True)

    excel_filename = "Churn_Prediction" + ".xlsx"
    df.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    #pbi_file = "https://app.powerbi.com/groups/me/apps/376a79f5-38ed-4add-9a36-ba55160dd364/reports/d7593690-748d-4da1-b8b8-6cbffe6c6bc3/ReportSection?ctid=599e51d6-2f8c-4347-8e59-1f795a51a98c"
    # df.to_excel(BASE_DIR+'/media/' + excel_filename)

    #df_pbi_file = '/media/' + pbi_file
    # print(d['file_name'])
    return True, df_filepath


