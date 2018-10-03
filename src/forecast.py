import os
import sys
from pathlib import Path

root = Path(os.path.join(os.getcwd().split("src")[0], "src"))
if root not in sys.path:
    sys.path.append(str(root))

from sk import sk_chart
import numpy as np
import pandas as pd
from pdb import set_trace
from data.data_handler import DataHandler
from statsmodels.tsa.arima_model import ARIMA


class TimeSeriesAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def moving_window(dframe, frame=4):
        """
        Moving window train test generator

        Parameters
        ----------
        dataframe: pandas.core.frame.DataFrame
            A dataframe of time series data
        frame: int
            Window size

        Yields
        ------
        pandas.core.frame.DataFrame:
            Training window 
        pandas.core.frame.DataFrame:
            Testing window
        """

        for i in range(len(dframe) - frame - 1):
            yield dframe.iloc[:i + frame],\
            dframe.iloc[i + frame:i + frame + 1]

    @staticmethod
    def mag_abs_err(actual, forecasted):
        """
        Compute the magnitude of absolute error of actual and forecasted

                N
                =====
                \              |~      |
            MAE =  >    P   *  |Y  - Y |
                /        i     | i    i|
                =====
                i = 1
        
        Parameters
        ----------
        actual: array_like
            Actual values
        
        forecasted: array_like
            Forecaster values
        
        Returns
        float
            MAE value
        """
        actual = np.array(actual)
        forecasted = np.abs(np.array(forecasted))
        return np.sum(np.abs(forecast - actual))/len(actual)


    def bug_count(self):

        data_handler = DataHandler(data_path=root.joinpath('data'))
        files = data_handler.get_data()
        for proj, data in files.items():
            actual   = ["Actual"]
            forecast = ["Forecast"]
            date = ['Date']
            for train, test in self.moving_window(data, frame=4):
                model = ARIMA(train, order=(1, 1, 0), freq='W-MON')
                model_fit = model.fit(disp=1)
                start, end = test.index[0], test.index[-1]
                actual.extend(test.values.ravel())
                forecast.append(int(abs(model_fit.forecast()[0])))
                date.append(test.index.strftime("%Y-%m-%d").values[0])

            for d, a, p in zip(date, actual, forecast):
                print(d,a,p,sep=",")
            
            set_trace()


if __name__ == "__main__":
    ts = TimeSeriesAnalysis()
    bugs = ts.bug_count()
