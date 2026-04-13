import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
import os
import sys
from src.Mem_model import *
from src.Template_matching import *
import json

# sys.path.append(os.path.join(os.getcwd(), 'src'))

with open('data.json', 'r') as file:
    # Load the JSON data
    data = json.load(file)

mem_model_fitter = Mem_model(data['device_file'], data["V_column"], data["R_column"], data["I_column"], data["P_column"])

mem_model_fitter.model_creation(clip_ = data["clip_"], R_list = np.arange(5,21)*1000)

tm = Template_matching(mem_model_fitter.model)

tm.template_maching(NL=[0, 0.1, 0.2])