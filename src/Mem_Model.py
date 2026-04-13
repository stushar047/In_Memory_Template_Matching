import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from src.allf import *
import json

with open('config.json', 'r') as file:
    # Load the JSON data
    data = json.load(file)

class Mem_Model:
    
    """
    Create a memristive model with the input data of voltage and resistance and predicting current
    
    Input:
    
    file, V_column, R_column, I_column, P_column
    
    Returns
    
    Model with a tuple with (r,a,b,c,d) where with a given r, I = f(V, a, b)
    
    """
    
    def __init__(self,file, V_column, R_column, I_column, P_column):
        
        self.mem_data = [file, V_column, R_column, I_column, P_column]
        
        self.R_list = []
        
        self.clip_V = data["clip_V"]
        
        self.clip = False
        
    def curve_fitting(self, func, V, I): 
        
        """
        Create a line fit with V and I so that, I = a*V + b
        
        return a,b
        """
    
        popt, pcov = curve_fit(func, V, I)

        a,b=popt

        return a,b

    def func(self, x, a, b):

        return a*x + b
        
    def read_data(self):
        
        if self.mem_data[0].split(".")[1]=="xlsx":
            
            mem = pd.read_excel(self.mem_data[0])
            
            print(mem)
            
        elif self.mem_data[0].split(".")[1]=="csv":
        
            mem = pd.read_csv(self.mem_data[0],header=None)
            
            self.mem_ = mem.rename(columns={mem.columns[0]:self.mem_data[1],mem.columns[1]:self.mem_data[2],mem.columns[2]:self.mem_data[3],mem.columns[3]:self.mem_data[4]})
        
        self.mem_.iloc[:,2]=self.mem_.iloc[:,2]*-1
        
    def clip_data(self):
        
        self.read_data()
        
        self.clip = True

        self.mem_ = self.mem_[np.abs(self.mem_[self.mem_data[1]])<=self.clip_V]
        
    def model_creation(self, clip_=False, R_list=[]):
        """
        Create a model such that, I = a(r)*V + b(r) and P = c(r)*V + d(r)
        
        """
        
        Model =[];
        
        self.R_list = R_list
        
        if not clip_:
            
            self.read_data()
            
        else:
            
            self.clip_data()

        for r in self.R_list:

            RI=self.mem_[self.mem_[self.mem_data[2]]==r]

            a, b = self.curve_fitting(self.func,RI[self.mem_data[1]].values,RI[self.mem_data[3]].values)

            c, d = self.curve_fitting(self.func,RI[self.mem_data[1]].values,RI[self.mem_data[4]].values)

            Model.append((r,a,b,c,d))

        self.model = np.array(Model)