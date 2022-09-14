import joblib
import pandas as pd
from ImageClassification.settings import BASE_DIR,MODELS_PATH

rf = joblib.load(MODELS_PATH + "/Emp_Attrition_pred/rf_clf_emp_attrition")
target_map = {1:'Yes',0:'No'}
imp_cols_categorical = ['OverTime']
imp_cols_numerical = ['Age','MonthlyRate','TotalWorkingYears','YearsAtCompany']

def read_csv_data(csv_filename):
    if csv_filename.split(".")[-1] =='csv':
        data = pd.read_csv(csv_filename)
        return data
    elif csv_filename.split(".")[-1] =='xlsx':
        data = pd.read_excel(csv_filename)
        return data

def attrition_pred(data):

    attrition = read_csv_data(data)

    categorical = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']

    numerical = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']

    attrition_cat = attrition[imp_cols_categorical]
    attrition_cat = pd.get_dummies(attrition_cat)
    attrition_num = attrition[imp_cols_numerical]
    attrition_final = pd.concat([attrition_num, attrition_cat], axis=1)

    rf_predictions = rf.predict(attrition_final)
    attrition["Attrition_Predicted"] = pd.Series(rf_predictions).apply(lambda x: target_map[x])


    excel_filename = "Employee_Attrition_Prediction" + ".xlsx"
    attrition.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    return True, df_filepath



