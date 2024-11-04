# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 08:31:19 2024

@author: Naresh
"""
import numpy as np 
import pickle


loaded_model = pickle.load(open("C:/Users/Naresh/OneDrive/Desktop/energy prediction project/trained_model.sav", 'rb'))

input_data = (18.87,52.08,1005.25,99.19)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print("Total Energy Prediction",prediction)