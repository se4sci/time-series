"""
A data handler to read, write, and process data
"""
import os
import sys
import pandas as pd
from glob2 import glob
from pdb import set_trace
from collections import OrderedDict

from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))

if root not in sys.path:
    sys.path.append(str(root))


class DataHandler:
    def __init__(self, data_path=root.joinpath("data")):
        """
        A Generic data handler class

        Parameters
        ----------
        data_path = <pathlib.PosixPath>
            Path to the data folder.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Read data as pandas and return a dictionary of time series data

        Returns
        -------
        all_data: dict
            A dictionary of data with key-project_name, value-list of file
            level metrics
        """

        all_data = OrderedDict()
        projects = [Path(proj) for proj in glob(str(self.data_path.joinpath("*"))) if Path(proj).is_dir()]

        for project in projects:
            files = []
            
            # Read all csv files and save them as a list in files
            for ver in glob(str(project.joinpath("*.csv"))):
                files.extend(pd.read_csv(ver, usecols=['time', 'buggy']).values.tolist())
            
            # Create a pandas dataframe from the csv sorted by datetime
            df = pd.DataFrame(files, columns=['Time', 'Bugs']).sort_values(by='Time').reset_index(drop=True)
            
            # Convert time to Pandas DateTime format
            df['Time'] = pd.to_datetime(df['Time']) 
            
            # Group bug counts by week starting on monday
            df = df.reset_index().set_index('Time').groupby(
                [pd.Grouper(freq='W-MON')])["Bugs"].sum().astype(int).reset_index()
            
            df = df.set_index('Time')
            # Save the data to dictionary
            all_data.update(OrderedDict({project.name: df}))

        return all_data


if __name__ == "__main__":
    dh = DataHandler()
    dh.get_data()
