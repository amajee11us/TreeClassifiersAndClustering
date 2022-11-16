from PIL import Image
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os

class KMeans:
    '''
    Class to perform K-Means Clustering for Image Compression Task.
    '''

    def __init__(self, image_path, k):
        self.image_path = image_path
        self.k = k
        self.rgb = []
        self.centroids =[]
        self.datapoint_cluster = {}
        self.final = []
        self.width=0
        self.height =0

        # calculate the RGB values from input image 
        self.__rgb_calc()
        self.o_size = os.path.getsize(self.image_path)

    def __rgb_calc(self):
        '''
        Calculate the final RGB values of the image from the path input to the class
        '''
        im = Image.open(self.image_path) 
        pix = im.load()
        self.width,self.height = im.size
        self.rgb = [pix[i,j] for i in range(self.width) for j in range(self.height)]
        self.final = np.copy(self.rgb)
        
    def __initialize_Centroids(self):
        '''
        Random initialization of cluster centroids.
        '''        
        centroids=[]
        for value in range(self.k):
            c = np.random.uniform(min(self.rgb),max(self.rgb))
            self.centroids.append(c)        
    
    def kmeans(self,iterations, image_name="tmp"):
        '''
        Compute K-Means.
        '''
        self.datapoint_cluster=[]
        cluster ={}
        final_rgb = np.copy(self.rgb)

        output_c_ratios = []

        print("Start Clustering ...")

        for iteration in tqdm(range(iterations)):
            # Init centroids
            self.__initialize_Centroids()

            # Compute K-Means and assign pixels
            for pos, pixel in enumerate(self.rgb):
                errors = {}
                for point in self.centroids:
                    error = self.SSE(pixel,point)
                    errors[error]=point
                closest_cluster = errors.get(min(errors.keys()))
                final_rgb[pos] = closest_cluster
            
            # Decompress image and calculate the compression ratio
            output_c_ratios.append(self.__store_and_return_ratio(
                                            final_rgb, 
                                            image_name, 
                                            iteration)            
                                    )
        print("Done.")
        
        return output_c_ratios            

    def __store_and_return_ratio(self, rgb_array, image_name="tmp", seed=0):
        '''
        Convert the array format to JPG format
        '''
        im = Image.new("RGB",(self.height, self.width))
        data = rgb_array
        data = [tuple(x) for x in data]
        im.putdata(data)
        file_store_path = "data/compressed_{}_k{}_seed{}.jpg".format(image_name, str(self.k), seed)
        im.save(file_store_path)
        
        calculated_size = os.path.getsize(file_store_path)
        c_ratio = round(self.o_size/calculated_size, 3)
        return c_ratio

    def SSE(self,x,y):        
        sum = np.square(y[0]-x[0]) + np.square(y[1]-x[1])+np.square(y[2]-x[2])
        return np.sqrt(sum)
        