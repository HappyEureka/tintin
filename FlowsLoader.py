import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import asyncio
from .DataLoader import DataLoader

class FlowsLoader(DataLoader):
    def __init__(self, outcome=None):
        self.time_col = "performed"
        self.feature_col = "measure_name"
        self.value_col = "value"
        self.selected_cols = ['pat_mrn_id', self.time_col, self.feature_col, 'flowsheet_name', 'value']
        super().__init__(outcome, self.time_col, self.feature_col,self.value_col, self.selected_cols)

    async def _read(self):
        if self.data is None:
            self.print_msg(self.WARNING,"Init: reading flows...")
            #self.data = pd.read_csv("filtered_flows.csv", usecols=self.selected_cols)
            self.data = pd.read_csv("pulse.csv", usecols=self.selected_cols)
            self.data['pat_mrn_id'] = self.data['pat_mrn_id'].astype("int64")
            self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
            self.print_msg(self.SUCCESS, "Done: finished reading flows.")
    
    @DataLoader.beforeAfter
    def split(self):
        pos_outcome = "Deceased"
        self.split_patients(pos_outcome)
        self.pos = self.pos.drop_duplicates(subset=['pat_mrn_id'], keep='first')
        self.neg = self.neg.drop_duplicates(subset=['pat_mrn_id'], keep='first')
        self.data = self.data.dropna(subset=[self.value_col])
        self.split_data(self.data)
         
    def view_all_patient_data_hist(self):
        assert not self.selected_data is None, "Error: selected data is None!"
        grouped = self.selected_data.groupby(['pat_mrn_id']).value.count()
        grouped.hist(bins=100)
        plt.show()
        return grouped.describe()

    def view_patient_data_hist(self, mrn):
        data = self.selected_data
        data = data[data.pat_mrn_id == mrn]
        plt.figure(figsize=(7, 3))
        plt.scatter(data.performed, data.value, s=5)
        plt.show()
        plt.figure(figsize=(7, 3))
        sns.kdeplot(data=data, x="performed")
        plt.show()

    def view_patient_last_k_hours(self, mrn):
        pass
        
    def read(self):
        self.task_read = asyncio.create_task(self._read())

    def load(self):
        pass