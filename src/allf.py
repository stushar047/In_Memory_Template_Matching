import numpy as np
import pandas as pd
import cv2

def w2Current(w,model):
    
    r, a, b, c, d = (model[:,0]//1000).astype("int"), model[:, 1], model[:, 2], model[:, 3], model[:, 4]

    G_H, G_L, mu, W_H =1/(np.min(r)*1e3), 1/(np.max(r)*1e3), np.min(w), np.max(w)

    alpha=(G_H-G_L)/(W_H+abs(mu))

    G=alpha*(w+abs(mu))+G_L

    R = ((1e-3/G).astype('int')).astype('str')

    dict_Ra_map={}

    for (key,value) in zip(r,a):

        dict_Ra_map[str(key)]=value

    dict_Rb_map={}

    for (key,value) in zip(r,b):

        dict_Rb_map[str(key)]=value
        
    dict_Rc_map={}

    for (key,value) in zip(r,c):

        dict_Rc_map[str(key)]=value
        
    dict_Rd_map={}

    for (key,value) in zip(r,d):

        dict_Rd_map[str(key)]=value

    R_df=pd.DataFrame(R)
    
    return R_df.replace(dict_Ra_map).values.reshape(-1,), R_df.replace(dict_Rb_map).values.reshape(-1,), R_df.replace(dict_Rc_map).values.reshape(-1,),R_df.replace(dict_Rd_map).values.reshape(-1,),R

def ADC (input_value, min_input = -0.150, max_input = 0.150, bits=16):

    # Calculate ratio
    ratio = (input_value - min_input) / (max_input - min_input)

    # Scale ratio to output range
    output_value = int(ratio * 2**bits - 2**(bits-1))

    return output_value

def Data2Voltage(w, pixel_min = -255, pixel_max=255, v_max =0.75, v_min = -0.75, num_levels=32):
    
    V_ = (w - np.mean(w)).astype('int') #image pixel substracting mean
    
    V_, sma= (V_*(v_max - v_min))/(pixel_max-pixel_min), np.sum(np.abs(V_)) # sum_of_mean_abs, volage
    
    quantized_V = np.linspace(v_min, v_max, num_levels)

    # Find the closest quantized value for each element in V
    quantized_V_values = np.array([quantized_V[np.argmin(np.abs(quantized_V - value))] for value in V_])
    
    return sma,quantized_V_values

def Data2Voltage2(w, pixel_min = -255, pixel_max=255, v_max =0.75, v_min = -0.75, num_levels=32):
    
    V_ = (w - np.mean(w)) #image pixel substracting mean
    
    V_, sma= (V_*(v_max - v_min))/(pixel_max-pixel_min), np.sqrt(np.sum(V_**2)) # sum_of_mean_abs, volage
    
    quantized_V = np.linspace(v_min, v_max, num_levels)

    # Find the closest quantized value for each element in V
    quantized_V_values = np.array([quantized_V[np.argmin(np.abs(quantized_V - value))] for value in V_])
    
    return sma,quantized_V_values


"Data mapped to conductance"

def Data2Current(Data, Model):

    Data_ = Data - np.mean(Data)
    
    Data = Data_/np.sqrt(np.sum(Data_**2))

    I_a, I_b, p_c, p_d, R = w2Current(np.abs(Data), Model)

    sign = []

    for d in Data:

        if d<0:

            sign.append(-1)
        else:

            sign. append(1) 

    sign = np.array(sign)

    return I_a*sign, I_b*sign, p_c, p_d, R
    
def mem_pcc(I_a, I_b, p_c, p_d, Data_sample,n=16,noise_level=0):

    " Data mapped to -1 to +1 range: V"

    sma, quantized_V_values = Data2Voltage(Data_sample,num_levels=n)
    
    quantized_V_values = add_gaussian_noise(quantized_V_values,noise_level)
    
    "Current for the V,G using equation"

    I_sign = sum(add_gaussian_noise(I_a * quantized_V_values + I_b, noise_level))
    
    IV_signq = ADC(I_sign)/np.round((sma/2**20) * 2**20).astype('int')
    
#     IV_signq = to_fixed(I_sign,43)/to_fixed(sma,43)
    
    e = (p_c*quantized_V_values + p_d)*2e-6
    
    E = sum(add_gaussian_noise(e, noise_level))
    
    return I_sign, np.float32(IV_signq), E, sma

def mem_pcc3(I_a, I_b, p_c, p_d, Data_sample,n=16,noise_level=0):

    " Data mapped to -1 to +1 range: V"

    sma, quantized_V_values = Data2Voltage(Data_sample,num_levels=n)
    
    quantized_V_values = add_gaussian_noise(quantized_V_values,noise_level)
    
    "Current for the V,G using equation"

    I_sign = sum(add_gaussian_noise(I_a * quantized_V_values + I_b, noise_level))
    
    IV_signq = ADC(I_sign)/np.round((sma/2**20) * 2**20).astype('int')
    
#     IV_signq = to_fixed(I_sign,43)/to_fixed(sma,43)
    
    e = (p_c*quantized_V_values + p_d)*2e-6
    
    E = sum(add_gaussian_noise(e, noise_level))
    
    return I_sign, np.float32(IV_signq), E, sma, quantized_V_values

def mem_pcc2(I_a, I_b, p_c, p_d, Data_sample,n=16,noise_level=0):

    " Data mapped to -1 to +1 range: V"

    sma, quantized_V_values = Data2Voltage2(Data_sample,num_levels=n)
    
    quantized_V_values = add_gaussian_noise(quantized_V_values,noise_level)

    "Current for the V,G using equation"

    I_sign = sum(add_gaussian_noise(I_a * quantized_V_values + I_b, noise_level))
    
    IV_signq = ADC(I_sign)/np.round((sma/2**20) * 2**20).astype('int')
    
    e = (p_c*quantized_V_values + p_d)*2e-6
    
    E = sum(add_gaussian_noise(e, noise_level))
    
    return I_sign, np.float32(IV_signq), E, sma

def mem_pcc3(I_a, I_b, p_c, p_d, Data_sample,n=16,noise_level=0):

    " Data mapped to -1 to +1 range: V"

    sma, quantized_V_values = Data2Voltage2(Data_sample,num_levels=n)
    
    quantized_V_values = add_gaussian_noise(quantized_V_values,noise_level)

    "Current for the V,G using equation"

    I_sign = sum(add_gaussian_noise(I_a * quantized_V_values + I_b, noise_level))
    
    IV_signq = ADC(I_sign)
    
    e = (p_c*quantized_V_values + p_d)*2e-6
    
    E = sum(add_gaussian_noise(e, noise_level))
    
    return I_sign, np.float32(IV_signq), E, sma

def soft_pcc(Data_template, Data_sample):

    Data_1 = Data_template - np.mean(Data_template)

    Data_2 = Data_sample - np.mean(Data_sample)

    return np.sum(Data_1* Data_2)/(np.sqrt(sum(Data_1**2))*np.sqrt(sum(Data_2**2)))

def soft_pcc_approximate(Data_template, Data_sample):

    Data_1 = Data_template - np.mean(Data_template)

    Data_2 = Data_sample - np.mean(Data_sample)

    return np.sum(Data_1* Data_2)/(np.sum(np.abs(Data_1))*np.sum(np.abs(Data_1)))

def scale_values(vector, old_min, old_max, new_min, new_max):
    
    scaled_vector = []
    
    for value in vector:
        
        scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        
        scaled_vector.append(scaled_value)
        
    return scaled_vector


def add_gaussian_noise(matrix, noise_level, size = 100):
    
    np.random.seed(42)

    STD  = np.abs(matrix*noise_level)

    noisy=[]

    for Mean,Standard_deviation in zip(matrix,STD):

        # creating a normal distribution data
        noisy.append(np.random.normal(Mean, Standard_deviation, size)[np.random.choice(size)])

    return np.array(noisy)


def to_fixed(f, e):
    
    # Scale the floating-point number
    scaled_number = f * (2 ** e)
    
    # Round to the nearest integer
    rounded_number = int(round(scaled_number))
    
    # Ensure the result is non-negative
    if rounded_number < 0:
        
        # Take the 2's complement
        rounded_number = abs(rounded_number)
        
        rounded_number = ~rounded_number
        
        rounded_number = rounded_number + 1
    
    return 

# Resize function
def resize_image(image, scale_factor):
    height, width = image.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)