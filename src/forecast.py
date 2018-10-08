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
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

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
            MAE =  >      P   *  |Y  - Y |
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

    @staticmethod
    def is_stationary(dframe, min_pval = 0.05):
        """
        Apply the augmented dickey-fuller test to check for stationarity 
        of the timeseries data.
        
        Parameters
        ----------
        dframe : pandas.core.frame.DataFrame
            Timeseries data as a dataframe
        min_pval : float, optional
            The minimum require p-value to reject the null hypothesis (the 
            default is 0.05).
        
        Returns
        -------
        bool:
            True if stationary. False otherwise.
        
        Notes
        -----
        Null hypothesis: The data is non-stationary.

        If p<0.05, then we reject the null hypothesis, the data is stationary. 
        Otherwise, it is non-stationary.
        """
        values = dframe.values.ravel()
        adf_test = adfuller(values, autolag=None)
        return adf_test[1] < min_pval
    
    @staticmethod
    def detrend_series(dframe):
        """
        Apply second differential to make data stationary
        
        Parameters
        ----------
        dframe : pandas.core.frame.DataFrame
            Timeseries data as a dataframe
        
        Returns
        -------
        pandas.core.frame.DataFrame
            Detrended timeseries data as a dataframe
        """
        dy = 1.2
        dval = list(abs(diff(diff(drame.values))/dy)/dy)
        dframe[dframe.columns[0]] = dval
        return dframe

    def bug_count(self):

        data_handler = DataHandler(data_path=root.joinpath('data'))
        files = data_handler.get_data()
        all_results = dict()
        for proj, data in files.items():
            col_name = ['Date', 'Actual', 'ARIMA', 'NAIVE']
            results = []
            actual = 0
            for train, test in self.moving_window(data, frame=24):
                try:
                    p, d, q = 4, 1, 4
                    
                    # if not self.is_stationary(train):
                    #     train = self.detrend_series(train)

                    arima = ARIMA(train, order=(p, d, q), freq='W-MON')
                    arima_fit = arima.fit(disp=0)

                    # Find start and end time stamps
                    start, end = test.index[0], test.index[-1]
                    
                    # Save date, actual, and forecast
                    prev_actual = actual
                    actual = test.values.ravel()[0]
                    
                    forecast_arima = int(abs(arima_fit.forecast()[0]))
                    forecast_naive = prev_actual
                    date = test.index.strftime("%Y-%m-%d").values[0]
                    results.append([date, actual, forecast_arima, forecast_naive])
                except:
                    X = np.arange(len(train.values)+1)
                    X = np.reshape(X, (len(X), 1))
                    y = train.values
                    model = LinearRegression()
                    model.fit(X[:-1], y)
                    prev_actual = actual
                    actual = test.values.ravel()[0]
                    forecast_arima = int(abs(model.predict(X[-1].reshape(1, -1))[0]))
                    forecast_naive = prev_actual
                    date = test.index.strftime("%Y-%m-%d").values[0]
                    results.append([date, actual, forecast_arima, forecast_naive])



            results = pd.DataFrame(results, columns=col_name).set_index('Date')
            results.to_csv(root.joinpath('results',proj+".csv"))
        
        return all_results


if __name__ == "__main__":
    ts = TimeSeriesAnalysis()
    bugs = ts.bug_count()
