import pandas as pd
from sklearn.preprocessing import LabelEncoder
#import shap
from sklearn.ensemble import IsolationForest
from ImageClassification.settings import BASE_DIR

contamination = 0.03

def read_csv_data(csv_filename):
    if csv_filename.split(".")[-1] =='csv':
        data = pd.read_csv(csv_filename)
        return data
    elif csv_filename.split(".")[-1] =='xlsx':
        data = pd.read_excel(csv_filename)
        return data

def anomaly_detect(csv_data):
    data = read_csv_data(csv_data)
    rndrng_Prvdr_geo_lvl_map = {'National': 0, 'State': 1}
    HCPCS_Drug_Ind_map = {'N': 0, 'Y': 1}
    Place_Of_Srvc_map = {'F': 0, 'O': 1}
    target_map = {1: 'Normal/Non_Anomalous Datapoint', -1: 'Anomalous Datapoint'}
    data.Rndrng_Prvdr_Geo_Lvl = data.Rndrng_Prvdr_Geo_Lvl.map(rndrng_Prvdr_geo_lvl_map)
    data.HCPCS_Drug_Ind = data.HCPCS_Drug_Ind.map(HCPCS_Drug_Ind_map)
    data.Place_Of_Srvc = data.Place_Of_Srvc.map(Place_Of_Srvc_map)

    la1 = LabelEncoder()
    la2 = LabelEncoder()
    la3 = LabelEncoder()

    data.drop('Rndrng_Prvdr_Geo_Cd', axis=1, inplace=True)
    data.Rndrng_Prvdr_Geo_Desc = la1.fit_transform(data.Rndrng_Prvdr_Geo_Desc)
    data.HCPCS_Cd = la2.fit_transform(data.HCPCS_Cd)
    data.HCPCS_Desc = la3.fit_transform(data.HCPCS_Desc)

    data['Avg_Sbmtd_Chrg_more_than_Alowd'] = data.Avg_Sbmtd_Chrg - data.Avg_Mdcr_Alowd_Amt
    #data['claim_id'] = data.index
    data['Avg_Pymt_Amt_more_than_Stdzd_Amt'] = data.Avg_Mdcr_Pymt_Amt - data.Avg_Mdcr_Stdzd_Amt
    data['Avg_Alowd_Amt_more_than_Pymt_Amt'] = data.Avg_Mdcr_Alowd_Amt - data.Avg_Mdcr_Pymt_Amt
    data['No_of_srvcs_taken_by_ben'] = data.Tot_Srvcs - data.Tot_Bene_Day_Srvcs
    data['Tot_Srvcs/Tot_Rndrng_Prvdrs'] = data['Tot_Srvcs'] / data['Tot_Rndrng_Prvdrs']
    data['Tot_Srvcs/Tot_Benes'] = data['Tot_Srvcs'] / data['Tot_Benes']
    data['Avg_Sbmtd_Chrg/Tot_Benes'] = data['Avg_Sbmtd_Chrg'] / data['Tot_Benes']
    data['Avg_Sbmtd_Chrg/Tot_Srvcs'] = data['Avg_Sbmtd_Chrg'] / data['Tot_Srvcs']

    X = data.copy()

    clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=contamination,
                          max_features=1.0, bootstrap=False, random_state=42, verbose=0)
    clf.fit(X)
    pred = clf.predict(X)
    data['anomaly'] = pred
    data['anomaly_score'] = clf.decision_function(X)

    data["Anomaly_Predicted"] = pd.Series(pred).apply(lambda x: target_map[x])


    excel_filename = "Anomaly_Detection_Prediction" + ".xlsx"
    data.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    pdf_filename = "Ex-AI_Feature_Importance_Insights.pdf"
    # df.to_excel(BASE_DIR+'/media/' + excel_filename)

    df_pdf_filepath = '/media/' + pdf_filename

    return True, df_filepath,df_pdf_filepath



