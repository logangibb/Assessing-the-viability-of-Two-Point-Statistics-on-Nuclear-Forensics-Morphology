'''
This creates the target csv for the folder full of SEM images and txt files of MAMA data
'''

import csv
import numpy as np
import os

#Filename and path
direct = r'C:\Users\u0944665\Box\2 Point Statistics\test'
name_save = 'test.csv'
filename_save = os.path.join(direct,name_save)

#These four are the column number for mama data.
pixarea = 12
vectper = 14
circ = 30
ellipse = 18

#Converting mama data's distances to pixel lengths.
convert_553 = 1024/55.3
convert_100000x = 1024/3.064
convert_30000x = 1024/4.27
convert_25000x = 1024/5.119
convert_20000x = 1024/6.399
convert_15000x = 2048/8.5333
convert_613 = 1024/6.13
convert_u3o8 = 1024/6.13
convert_uo4 = 1024/3.06
convert = 0

previous_filename = 0
dataname = ''


for filename in os.listdir(direct):
    #Based on the file name, it chooses the pixel conversion
    if('30000x' in filename): 
        convert = convert_30000x
    elif('100000x' in filename):
        convert = convert_100000x
    elif('55.3um' in filename):
        convert = convert_553
    elif('25000x' in filename):
        convert = convert_25000x
    elif('20000x' in filename):
        convert = convert_20000x
    elif('15000x' in filename):
        convert = convert_15000x
    elif('6.13um' in filename):
        convert = convert_613
    elif('~' in filename and 'U3O8' in filename):
        convert = convert_u3o8
    elif('~' in filename and 'UO4' in filename):
        convert = convert_uo4
    else:
        print('Something went wrong')
        print(filename)
    
    if(filename[-4:]!='.tif' and filename[-4:]!= '.csv'): #checks to make sure it is not an image or csv file
        filepath = os.path.join(direct,filename) 
        file = np.genfromtxt(filepath,skip_header=1) #gets the mama data
        pix = np.mean(file[:,pixarea]*convert**2) #converts area to pixel area
        vect = np.mean(file[:,vectper]*convert) #converts vector perimeter to pixel perimetere
        circle = np.mean(file[:,circ]) 
        ellipseratio = np.mean(file[:,ellipse])
        with open(filename_save, 'a', newline='') as savefile: #saves to a csv file
            save = csv.writer(savefile)
            save.writerow([filename, pix, vect,circle,ellipseratio])
    else:
        print('Something didnot save')
        print(filename)













