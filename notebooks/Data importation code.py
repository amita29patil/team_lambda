#Data importation code
import pandas as pd
import os

#Get current working directory (.\\notebooks)
cwd = os.getcwd()

#Get parent folder to access (.\\Data)
parent = os.path.dirname(cwd)

#Get file path to load into environment
file_path = os.path.join(parent,"Data","insert excel file needed here.xlsx")

#Load data into environment using the file path variable with appropriate relative path.
df= pd.read_excel(file_path)