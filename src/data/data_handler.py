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
    def __init__(self, data_path=root.parent.joinpath("labeled_commits")):
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
        Read data as pandas and return a dictionary of data

        Returns
        -------
        all_data: dict
            A dictionary of data with key-project_name, value-list of file
            level metrics
        """

        all_data = OrderedDict()
        projects = [Path(proj) for proj in glob(str(self.data_path.joinpath("*"))) if Path(proj).is_dir()]

        for project in projects:
            all_data.update(OrderedDict({project.name: OrderedDict({Path(ver).name: pd.read_csv(ver, usecols=['time', 'buggy']) for ver in glob(str(project.joinpath("*.csv")))})}))
        
        
        return all_data


if __name__ == "__main__":
    dh = DataHandler()
    dh.get_data()