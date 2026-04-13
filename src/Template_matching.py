import os
import cv2
import numpy as np
from src.allf import *

class Template_matching:
    
    """
    Read a JSON file and perform template matching

    Input Parameters:
    model : The memristive model with V,R input and Current Output.
    
    img_direc: Directory containing the image
    
    temp_img: The template image
    
    method: The method of PCC compuatation

    Attributes:
    Result_dict: dict file with the result
    
    """
    
    def __init__(self, model, img_direc = "PeanutBottle/", temp_img = "template/template2.jpg"):
    
        self.img_direc = img_direc 

        self.temp_img = temp_img
        
        self.model = model
        
        self.method =  "Approximate"

        self.Result_dict={"Img":[], "mcorr":[], "mbox":[],"scorr":[], "sbox":[],"scorr2":[], "sbox2":[],"mse":[],"mse2":[], "E":[],"All_E":[], "All_I":[], "save":[],"savedE": []}

    def dataset_creation(self):
        
        """
        Dataset creation

        Attributes:
        img_list: the list of name of the image of dataset

        full_img_list: the list of the image of dataset

        template: template image grayscale

        Data_template: Reshape the template to a vector

        temp_row, temp_col: row and column of the template image

        """
        
        self.img_list = [os.path.join(self.img_direc, i) for i in os.listdir(self.img_direc)]

        # Desired scale factor
        scale_factor = 1.0  # For example, 0.25 means resize to 25% of the original size

        # Load and resize images
        self.full_img_list = [resize_image(cv2.imread(im, 0), scale_factor) for im in self.img_list]

        # Load and resize template image
        self.template = resize_image(cv2.imread(f"{self.temp_img}", 0), scale_factor)
        
#         self.full_img_list = [cv2.imread(im, 0)[::4,::4] for im in self.img_list]

#         self.template= cv2.imread(f"{self.temp_img}", 0)[::4,::4]

        self.Data_template = self.template.reshape(-1,)
        
        self.temp_row, self.temp_col = self.template.shape       
        
    def template_data(self):
        
        """
        Dataset creation

        Create the model of the template image

        Attributes:

        I_a, I_a = Model's current coefficient of the template image which will return I given V,R

        p_c, p_d = Model's power coefficient of the template image which will return power given V,R

        R = Template data presentation in R

        """
        
        self.I_a, self.I_b, self.p_c, self.p_d, self.R = Data2Current(self.Data_template,self.model)
        
    def pcc_computation(self, full_image, row_stride=20, col_stride =20, nl=0):
        
        """
        PCC Compuation using image, template and noise level nl

        Input Parameters:
        
        full_image: The image which PCC needs to be computed with template
        
        row_stride, col_stride
        
        nl: noise level

        Attributes:

        I, SMA, E, Corr_template, Corr_template_soft, Corr_template_soft

        """
        
        iter_row, iter_col = ((full_image.shape[0]-self.temp_row)//row_stride), (full_image.shape[1]-self.temp_col)//col_stride

        self.Corr_template, self.Corr_template_soft, self.Corr_template_soft2=np.zeros((iter_row+1, iter_col+1)), np.zeros((iter_row+1, iter_col+1)),np.zeros((iter_row+1, iter_col+1))

        self.I, self.SMA, self.E =np.zeros((iter_row+1, iter_col+1)),np.zeros((iter_row+1, iter_col+1)), np.zeros((iter_row+1, iter_col+1))

        for ir in range(iter_row+1):

            for ic in range(iter_col+1):

                Data_sample = full_image[ir*col_stride:ir*col_stride+self.temp_row, ic*col_stride:ic*col_stride+self.temp_col].reshape(-1,)

                if self.method == "Approximate":

                    self.I[ir,ic], self.Corr_template[ir,ic], self.E[ir,ic], self.SMA[ir,ic]=mem_pcc(self.I_a, self.I_b, self.p_c, self.p_d, Data_sample, n=4, noise_level=nl)

                elif self.method == "Exact":

                    self.I[ir,ic], self.Corr_template[ir,ic], self.E[ir,ic], self.SMA[ir,ic]=mem_pcc2(self.I_a, self.I_b, self.p_c, self.p_d, Data_sample, n=4, noise_level=nl)

                if nl==0:

                    self.Corr_template_soft[ir,ic]=soft_pcc(self.Data_template, Data_sample)

                    self.Corr_template_soft2[ir,ic]=soft_pcc_approximate(self.Data_template, Data_sample)
                    
        flattened_SAD = self.SMA.flatten()
        
        flattened_E_ = self.E.flatten()
        
        self.save = len(flattened_SAD[flattened_SAD<3.5e5])/len(flattened_SAD)
        
        self.saved_E = np.sum(flattened_E_[flattened_SAD<3.5e5])
            
    def calculate_metrics(self, nl=0, row_stride=20, col_stride =20):
        
        """
        Caculate corr metrics
        
        """
        
        if nl==0:

                self.min_corr, self.max_corr = np.min(self.Corr_template), np.max(self.Corr_template)

                self.min_corrs, self.max_corrs = np.min(self.Corr_template_soft), np.max(self.Corr_template_soft)

                self.min_corrs2, self.max_corrs2 = np.min(self.Corr_template_soft2), np.max(self.Corr_template_soft2)

            
        self.Corr_template_norm = scale_values(self.Corr_template,self.min_corr,self.max_corr,self.min_corrs,self.max_corrs)

        self.Corr_template_norm2 = scale_values(self.Corr_template,self.min_corr,self.max_corr,self.min_corrs2,self.max_corrs2)


        self.mse, self.mse2 = np.mean((self.Corr_template_norm - self.Corr_template_soft)**2),np.mean((self.Corr_template_norm2 - self.Corr_template_soft2)**2)

        
        self.mcorr, max_index = np.max(self.Corr_template), np.argmax(self.Corr_template)

        idx_2d = np.unravel_index(max_index, self.Corr_template.shape)

        self.my0, self.my1, self.mx0, self.mx1 = idx_2d[0]*col_stride,idx_2d[0]*col_stride+self.temp_row, idx_2d[1]*col_stride,idx_2d[1]*col_stride+self.temp_col


        self.scorr, max_index = np.max(self.Corr_template_soft), np.argmax(self.Corr_template_soft)

        idx_2d = np.unravel_index(max_index, self.Corr_template_soft.shape)

        self.sy0, self.sy1, self.sx0, self.sx1 = idx_2d[0]*col_stride,idx_2d[0]*col_stride+self.temp_row, idx_2d[1]*col_stride,idx_2d[1]*col_stride+self.temp_col


        self.scorr2, max_index = np.max(self.Corr_template_soft2), np.argmax(self.Corr_template_soft2)

        idx_2d = np.unravel_index(max_index, self.Corr_template_soft2.shape)

        self.say0, self.say1, self.sax0, self.sax1 = idx_2d[0]*col_stride,idx_2d[0]*col_stride+self.temp_row, idx_2d[1]*col_stride,idx_2d[1]*col_stride+self.temp_col
        
            
    def store_result(self,im):
        
        """
        Storing results
        
        """
        
        self.Result_dict["Img"].append(im)

        self.Result_dict["mcorr"].append(np.max(self.Corr_template_norm))

        self.Result_dict["E"].append(np.sum(self.E))
        
        self.Result_dict["All_E"].append(self.E.flatten())
        
        self.Result_dict["All_I"].append(self.I.flatten())

        self.Result_dict["mbox"].append([self.my0, self.my1, self.mx0, self.mx1])

        self.Result_dict["scorr"].append(self.scorr)

        self.Result_dict["sbox"].append([self.sy0, self.sy1, self.sx0, self.sx1])

        self.Result_dict["scorr2"].append(self.scorr2)

        self.Result_dict["sbox2"].append([self.say0, self.say1, self.sax0, self.sax1])

        self.Result_dict["mse"].append(self.mse)

        self.Result_dict["mse2"].append(self.mse2)
        
        self.Result_dict["save"].append(self.save)
        
        self.Result_dict["savedE"].append(self.saved_E)
            
    def template_matching(self, NL):
        
        """
        Template matching
        
        """
        
        self.dataset_creation()
        
        self.template_data()

        for (im,full_image) in zip(self.img_list, self.full_img_list):

            for nl in NL:

                self.pcc_computation(full_image, nl=nl)
                
                self.calculate_metrics(nl=nl)
                
                self.store_result(im)


# import os
# import cv2
# import numpy as np
# from multiprocessing import Pool, cpu_count
# from src.allf import *

# class Template_matching:
    
#     """
#     Read a JSON file and perform template matching

#     Input Parameters:
#     model : The memristive model with V,R input and Current Output.
    
#     img_direc: Directory containing the image
    
#     temp_img: The template image
    
#     method: The method of PCC computation

#     Attributes:
#     Result_dict: dict file with the result
    
#     """
    
#     def __init__(self, model, img_direc="PeanutBottle/", temp_img="template/template2.jpg"):
    
#         self.img_direc = img_direc 
#         self.temp_img = temp_img
#         self.model = model
#         self.method = "Approximate"
#         self.Result_dict = {"Img": [], "mcorr": [], "mbox": [], "scorr": [], "sbox": [], "scorr2": [], "sbox2": [], "mse": [], "mse2": [], "E": []}

#     def dataset_creation(self):
        
#         """
#         Dataset creation

#         Attributes:
#         img_list: the list of name of the image of dataset
#         full_img_list: the list of the image of dataset
#         template: template image grayscale
#         Data_template: Reshape the template to a vector
#         temp_row, temp_col: row and column of the template image
#         """
        
#         self.img_list = [os.path.join(self.img_direc, i) for i in os.listdir(self.img_direc)]
#         scale_factor = 1.0  # Example scale factor

#         self.full_img_list = [resize_image(cv2.imread(im, 0), scale_factor) for im in self.img_list]
#         self.template = resize_image(cv2.imread(f"{self.temp_img}", 0), scale_factor)
#         self.Data_template = self.template.reshape(-1,)
#         self.temp_row, self.temp_col = self.template.shape       

#     def template_data(self):
        
#         """
#         Create the model of the template image

#         Attributes:
#         I_a, I_b = Model's current coefficient of the template image
#         p_c, p_d = Model's power coefficient of the template image
#         R = Template data presentation in R
#         """
        
#         self.I_a, self.I_b, self.p_c, self.p_d, self.R = Data2Current(self.Data_template, self.model)

#     def pcc_computation(self, full_image, row_stride=20, col_stride=20, nl=0):
        
#         """
#         PCC Computation using image, template and noise level nl

#         Input Parameters:
#         full_image: The image which PCC needs to be computed with template
#         row_stride, col_stride
#         nl: noise level

#         Attributes:
#         I, SMA, E, Corr_template, Corr_template_soft, Corr_template_soft2
#         """
        
#         iter_row, iter_col = (full_image.shape[0] - self.temp_row) // row_stride, (full_image.shape[1] - self.temp_col) // col_stride

#         Corr_template = np.zeros((iter_row + 1, iter_col + 1))
#         Corr_template_soft = np.zeros((iter_row + 1, iter_col + 1))
#         Corr_template_soft2 = np.zeros((iter_row + 1, iter_col + 1))
#         I = np.zeros((iter_row + 1, iter_col + 1))
#         SMA = np.zeros((iter_row + 1, iter_col + 1))
#         E = np.zeros((iter_row + 1, iter_col + 1))

#         for ir in range(iter_row + 1):
#             for ic in range(iter_col + 1):
#                 Data_sample = full_image[ir * col_stride:ir * col_stride + self.temp_row, ic * col_stride:ic * col_stride + self.temp_col].reshape(-1,)

#                 if self.method == "Approximate":
#                     I[ir, ic], Corr_template[ir, ic], E[ir, ic], SMA[ir, ic] = mem_pcc(self.I_a, self.I_b, self.p_c, self.p_d, Data_sample, n=32, noise_level=nl)
#                 elif self.method == "Exact":
#                     I[ir, ic], Corr_template[ir, ic], E[ir, ic], SMA[ir, ic] = mem_pcc2(self.I_a, self.I_b, self.p_c, self.p_d, Data_sample, n=32, noise_level=nl)

#                 if nl == 0:
#                     Corr_template_soft[ir, ic] = soft_pcc(self.Data_template, Data_sample)
#                     Corr_template_soft2[ir, ic] = soft_pcc_approximate(self.Data_template, Data_sample)

#         return Corr_template, Corr_template_soft, Corr_template_soft2, I, SMA, E

#     def calculate_metrics(self, Corr_template, Corr_template_soft, Corr_template_soft2, I, SMA, E, nl=0, row_stride=20, col_stride=20):
        
#         """
#         Calculate correlation metrics
        
#         """
        
#         if nl == 0:
#             min_corr = np.min(Corr_template)
#             max_corr = np.max(Corr_template)
#             min_corrs = np.min(Corr_template_soft)
#             max_corrs = np.max(Corr_template_soft)
#             min_corrs2 = np.min(Corr_template_soft2)
#             max_corrs2 = np.max(Corr_template_soft2)

#         Corr_template_norm = scale_values(Corr_template, min_corr, max_corr, min_corrs, max_corrs)
#         Corr_template_norm2 = scale_values(Corr_template, min_corr, max_corr, min_corrs2, max_corrs2)

#         mse = np.mean((Corr_template_norm - Corr_template_soft) ** 2)
#         mse2 = np.mean((Corr_template_norm2 - Corr_template_soft2) ** 2)

#         mcorr = np.max(Corr_template)
#         max_index = np.argmax(Corr_template)
#         idx_2d = np.unravel_index(max_index, Corr_template.shape)
#         my0, my1, mx0, mx1 = idx_2d[0] * col_stride, idx_2d[0] * col_stride + self.temp_row, idx_2d[1] * col_stride, idx_2d[1] * col_stride + self.temp_col

#         scorr = np.max(Corr_template_soft)
#         max_index = np.argmax(Corr_template_soft)
#         idx_2d = np.unravel_index(max_index, Corr_template_soft.shape)
#         sy0, sy1, sx0, sx1 = idx_2d[0] * col_stride, idx_2d[0] * col_stride + self.temp_row, idx_2d[1] * col_stride, idx_2d[1] * col_stride + self.temp_col

#         scorr2 = np.max(Corr_template_soft2)
#         max_index = np.argmax(Corr_template_soft2)
#         idx_2d = np.unravel_index(max_index, Corr_template_soft2.shape)
#         say0, say1, sax0, sax1 = idx_2d[0] * col_stride, idx_2d[0] * col_stride + self.temp_row, idx_2d[1] * col_stride, idx_2d[1] * col_stride + self.temp_col

#         return mcorr, mse, mse2, my0, my1, mx0, mx1, scorr, sy0, sy1, sx0, sx1, scorr2, say0, say1, sax0, sax1, E

#     def store_result(self, im, mcorr, mse, mse2, my0, my1, mx0, mx1, scorr, sy0, sy1, sx0, sx1, scorr2, say0, say1, sax0, sax1, E):
        
#         """
#         Storing results
        
#         """
        
#         self.Result_dict["Img"].append(im)
#         self.Result_dict["mcorr"].append(mcorr)
#         self.Result_dict["E"].append(np.sum(E))
#         self.Result_dict["mbox"].append([my0, my1, mx0, mx1])
#         self.Result_dict["scorr"].append(scorr)
#         self.Result_dict["sbox"].append([sy0, sy1, sx0, sx1])
#         self.Result_dict["scorr2"].append(scorr2)
#         self.Result_dict["sbox2"].append([say0, say1, sax0, sax1])
#         self.Result_dict["mse"].append(mse)
#         self.Result_dict["mse2"].append(mse2)
    
#     def process_image(self, args):
#         """
#         Process an image and compute metrics
        
#         Args:
#         args (tuple): (image_path, full_image, noise_level)
        
#         Returns:
#         result (tuple): Contains all relevant results to be stored
#         """
#         im, full_image, nl = args
        
#         print(im)

#         Corr_template, Corr_template_soft, Corr_template_soft2, I, SMA, E = self.pcc_computation(full_image, nl=nl)
#         mcorr, mse, mse2, my0, my1, mx0, mx1, scorr, sy0, sy1, sx0, sx1, scorr2, say0, say1, sax0, sax1, E = self.calculate_metrics(
#             Corr_template, Corr_template_soft, Corr_template_soft2, I, SMA, E, nl=nl
#         )
#         return (im, mcorr, mse, mse2, my0, my1, mx0, mx1, scorr, sy0, sy1, sx0, sx1, scorr2, say0, say1, sax0, sax1, E)
    
#     def template_matching(self, NL):
        
#         """
#         Template matching
        
#         """
        
#         self.dataset_creation()
#         self.template_data()

#         # Prepare arguments for multiprocessing
#         args_list = [(im, full_image, nl) for im, full_image in zip(self.img_list, self.full_img_list) for nl in NL]

#         # Create a Pool of worker processes
#         with Pool(processes=cpu_count()) as pool:
#             results = pool.map(self.process_image, args_list)

#         # Store results
#         for result in results:
#             im, mcorr, mse, mse2, my0, my1, mx0, mx1, scorr, sy0, sy1, sx0, sx1, scorr2, say0, say1, sax0, sax1, E = result
#             self.store_result(im, mcorr, mse, mse2, my0, my1, mx0, mx1, scorr, sy0, sy1, sx0, sx1, scorr2, say0, say1, sax0, sax1, E)