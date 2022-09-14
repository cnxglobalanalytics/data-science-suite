import numpy
import matplotlib.pyplot as plt
import pandas
from tensorflow.keras.models import load_model
import joblib
from ImageClassification.settings import BASE_DIR,MODELS_PATH
from matplotlib.backends.backend_pdf import PdfPages
# fix random seed for reproducibility
numpy.random.seed(7)
look_back = 1
model = load_model(MODELS_PATH + '/time_series_lstm/lstm_model.h5')
scaler = joblib.load(MODELS_PATH + '/time_series_lstm/Minmax_scaler')

def read_csv_data(csv_filename):

    data = pandas.read_csv(csv_filename)

    dataframe = pandas.read_csv(csv_filename, usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    dataset = scaler.fit_transform(dataset)
    return dataset,data['Month']


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def ts_lstm(csv_file):

    data = pandas.DataFrame()
    test,time_test = read_csv_data(csv_file)
    testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    shape_0 = time_test.shape[0]
    testPredict = model.predict(testX)
    # invert predictions
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    data['Timestamp'] = time_test[:(shape_0-4)]
    pass_actual = testY[0].tolist()[:(shape_0-4)]
    data['Passengers(Actual_Values)'] = [int(i) for i in pass_actual]

    pass_forecasted = testPredict[:, 0].tolist()[:(shape_0-4)]
    data['Passengers(Forcasted_Values)'] = [int(i) for i in pass_forecasted]
    data['Passengers(Forcasted_Values)'] = data['Passengers(Forcasted_Values)'].shift(-1)
    data = data.iloc[:-1 , :]


    excel_filename = "Time_Series_Forecasting" + ".xlsx"
    data.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    with PdfPages(BASE_DIR + "/media/timeseries_forecast.pdf") as pdf:
        data_plot = data.copy()
        data_plot.set_index('Timestamp', inplace=True)
        plt.rcParams['figure.figsize'] = (22, 10)
        plt.plot(data_plot['Passengers(Actual_Values)'].iloc[100:120])
        plt.plot(data_plot['Passengers(Forcasted_Values)'].iloc[100:120])
        plt.legend()
        # plt.show()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

    #pdf_filename = "timeseries_forecast.pdf"
    # df.to_excel(BASE_DIR+'/media/' + excel_filename)

    #df_pdf_filepath = '/media/' + pdf_filename


    return True, df_filepath