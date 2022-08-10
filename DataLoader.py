import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self, outcome=None, time_col=None, feature_col=None, value_col=None, selected_cols=None):
        # init vars
        self.outcome = outcome
        self.time_col = time_col
        self.feature_col = feature_col
        self.value_col = value_col
        self.selected_cols = selected_cols
        
        # reading task
        self.task_read = None
        
        # colors
        self.SUCCESS = '\033[92m'
        self.WARNING = '\033[93m'
        self.ERROR = '\033[91m'
        
        # original data
        ## patient
        self.pos = None
        self.neg = None
        self.all = None
        ## data
        self.pos_data = None 
        self.neg_data = None
        self.data = None
        
        # selected data
        ## patient
        self.selected_pos = None 
        self.selected_neg = None
        self.selected_all = None 
        ## data
        self.selected_pos_data = None 
        self.selected_neg_data = None
        self.selected_data = None
        
    # read patient data
    def read_patients(self):
        self.pos = pd.read_csv("static_positive.csv")
        self.neg = pd.read_csv("static_negative.csv")
        self.pos[['pat_mrn_id', 'csn']] = self.pos[['pat_mrn_id', 'csn']].astype("int64")
        self.neg[['pat_mrn_id', 'csn']] = self.neg[['pat_mrn_id', 'csn']].astype("int64")
        self.pos[['admit_time', 'discharge_time']] = self.pos[['admit_time', 'discharge_time']].apply(pd.to_datetime)
        self.neg[['admit_time', 'discharge_time']] = self.neg[['admit_time', 'discharge_time']].apply(pd.to_datetime)
        self.all = pd.concat([self.pos, self.neg]).copy()
    
    # split patient by outcome, binary case only
    def split_patients(self, pos_outcome):
        self.pos = self.all[self.all[self.outcome] == pos_outcome]
        self.neg = self.all[self.all[self.outcome] != pos_outcome]
    
    # split data by patient
    def split_data(self, data):
        self.pos_data = data[data.pat_mrn_id.isin(self.pos.pat_mrn_id)]
        self.neg_data = data[data.pat_mrn_id.isin(self.neg.pat_mrn_id)]

    # selected random number of positive and negative patients
    def select_random_patients(self, size=100):
        if self.pos is None or self.neg is None:
            self.read_patients()
        pos_index = np.random.randint(0, high=len(self.pos), size=size)
        neg_index = np.random.randint(0, high=len(self.neg), size=size)
        self.pos = self.pos.iloc[pos_index, :]
        self.neg = self.neg.iloc[neg_index, :]
        return self.pos, self.neg

    # print relevant info before and after when a method is called
    def beforeAfter(function):
        def wrapper(*args, **kwargs):
            def find_len(data):
                return "None" if data is None else len(data)
            def show_message(self, position):
                self.print_msg(self.SUCCESS, position)
                self.print_msg(self.WARNING, 
                    f"data: {find_len(self.data)}, selected_data: {find_len(self.selected_data)}\n" +
                    f"pos: {find_len(self.pos)}, neg: {find_len(self.neg)}\n" + 
                    f"selected_pos_data: {find_len(self.selected_pos_data)}, selected_neg_data: {find_len(self.selected_neg_data)}")
            self = args[0]
            print(function.__name__)
            show_message(self, "Before:") 
            ret = function(*args, **kwargs)
            show_message(self, "After:")
            return ret
        return wrapper
    
    @beforeAfter 
    def select_data_by_most_freq_patient(self, size=100):
        def _group(data):
            return data.groupby(['pat_mrn_id'])[self.feature_col]
        grouped_pos = _group(self.pos_data).count().reset_index()
        grouped_neg = _group(self.neg_data).count().reset_index()
        selected_pos_mrns = grouped_pos.sort_values(by=[self.feature_col], ascending=False)[:size]['pat_mrn_id']
        selected_neg_mrns = grouped_neg.sort_values(by=[self.feature_col], ascending=False)[:size]['pat_mrn_id']
        self.selected_pos_data = self.data[self.data.pat_mrn_id.isin(selected_pos_mrns)]
        self.selected_neg_data = self.data[self.data.pat_mrn_id.isin(selected_neg_mrns)]
        self.selected_data = pd.concat([self.selected_pos_data, self.selected_neg_data]) 

    @beforeAfter
    def filter_by_features(self, features):
        self.data = self.data[self.data.measure_name.isin(features)]

    @beforeAfter
    def filter_by_patients(self, patient_ids):
        self.data = self.data[self.data.pat_mrn_id.isin(patient_ids)]

    @beforeAfter
    def filter_data_by_patient_hospitalization(self):
        def duration_filter(data, pat):
            dur = pat[['pat_mrn_id', 'admit_time', 'discharge_time', self.outcome]]
            data = pd.merge(data, dur, on=['pat_mrn_id'], how="inner")
            data['duration'] = (data.discharge_time - data.admit_time).dt.total_seconds() / (60 * 60)
            return data[(data[self.time_col] >= data.admit_time) & (data[self.time_col] <= data.discharge_time)]
        self.selected_pos_data = duration_filter(self.selected_pos_data, self.pos)
        self.selected_neg_data = duration_filter(self.selected_neg_data, self.neg)
        del self.selected_data 
        self.selected_data = pd.concat([self.selected_pos_data, self.selected_neg_data])

    @staticmethod
    def print_msg(level, msg):
        print(level + msg + "\033[0m")
    
    @abstractmethod
    def read(self):
        pass
    
    @abstractmethod
    def split(self, pos_outcome):
        pass
        
    @abstractmethod
    def load(self):
        pass

    def view_freq_features(self, data, n_rows=20):
        view = data.groupby([self.feature_col]).count().reset_index().sort_values(by=[self.value_col], ascending=False).head(n_rows)
        return view
    
    def __repr__(self):
        return f"time_col: {self.time_col}\nfeature_col: {self.feature_col}\nselected_cols: {self.selected_cols}"

    beforeAfter = staticmethod(beforeAfter)