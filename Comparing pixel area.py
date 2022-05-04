'''
Material Informatics Final
2 point Statistics on SEM Nuclear Particles.
'''

#%%
'''
Importing useful files for loading data
'''
import PIL
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import umap
import hdbscan
import cv2

from pymks import (
    plot_microstructures,
    PrimitiveTransformer,
    TwoPointCorrelation,
    FlattenTransformer
)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def evaluate(target, pred):
    r2 = r2_score(target, pred)
    mae = mean_absolute_error(target, pred)
    rmse = mean_squared_error(target, pred, squared=False)
    return r2,mae,rmse

#%%
#Load in the data
#directories for the train,val, and test sets
train_location = r'C:\Users\u0944665\Box\2 Point Statistics\Manual\Train' #directory
val_location = r'C:\Users\u0944665\Box\2 Point Statistics\Manual\Val'
test_location = r'C:\Users\u0944665\Box\2 Point Statistics\Manual\Test'

def loadimages(directory,target_name):
    '''
    directory is the directory to the location of images and the .csv file with all the targets
    target_name is the name of the .csv file of the targets
    This functions will load in all the images, while croping them at the same time.
    splits will determine how many split on the x and y axis there are. For example, splits =3 will give you 4 smaller images from 1 image.
    Returns the cropped images, and the targets associated with them.
    '''
    targetfile = os.path.join(directory,target_name)
    target_name =[]
    target_value = []
    with open(targetfile,newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            target_value.append(row[1:])
            target_name.append(row[0])
    target_value = np.array(target_value)
    target_name = np.array(target_name)
    left = 0
    top = 0
    right = 1024
    bottom = 880
    splits = 2
    vertical = np.linspace(top,bottom,splits)
    horizontal = np.linspace(left,right,splits)
    images=[]
    target = []
    name = []
    for filename in os.listdir(directory):
    	# load image
        try:
            img_data = PIL.Image.open(directory +'\\' + filename) #load in the image
            img_data = img_data.convert(mode='L')
            for vert in range(len(vertical)-1): #splits the image into 4 parts
                for hor in range(len(horizontal)-1):
                    im = img_data.crop((horizontal[hor],vertical[vert],horizontal[hor+1],vertical[vert+1])) #crop the image
                    im = np.asarray(im) #convert image to np array
                    images.append(im) #append to train_images list
                    temp = target_value[target_name==filename[:-4]]
                    target.append(temp[0])
                    temp2 = target_name[target_name==filename[:-4]]
                    name.append(temp2[0])
            
        except Exception:
            pass
    target =np.array(target, dtype=float)    
    images = np.array(images) #converts the list to a numpy array
    name = np.array(name)
    return images, target



#%%
'''
These are all the function for converting from grayscale to black and white
'''
def cutoff(images,cut_off,inverted=False):
    '''
    Convert to black and white using a cutoff value
    Example: 0.5 means that half way between the max and min of an image all the imagepoints above will be coverted to white,
    while everything below will be coverted to black
    
    '''
    type_thresh = cv2.THRESH_BINARY
    if(inverted):
        type_thresh = cv2.THRESH_BINARY_INV
    
    cutoff_images = np.zeros_like(images)
    i =0
    for im in images:
        maxi = np.amax(im)
        mini = np.amin(im)
        medi = (maxi - mini)*cut_off + mini
        ret, cutoff_images[i] = cv2.threshold(im,medi,1,type_thresh)
        i +=1
    return cutoff_images

def adaptive_cutoff(images,type_thresh =cv2.ADAPTIVE_THRESH_MEAN_C,block=51,subtr=2):
    '''
    
    '''
    adapted_images = np.zeros_like(images)
    for i, im in enumerate(images):
        adapted_images[i] = cv2.adaptiveThreshold(im,1,type_thresh,cv2.THRESH_BINARY,block,subtr)
    return adapted_images
    
def otsu(images,type_thresh=cv2.THRESH_BINARY):
    # Otsu's thresholding
    otsu_images = np.zeros_like(images)
    for i, img in enumerate(images):
        ret, otsu_images[i] = cv2.threshold(img,0,1, type_thresh+cv2.THRESH_OTSU)
    return otsu_images


def water_shed(images):
    watershed_images = np.zeros_like(images)
    for i, img in enumerate(images):
        # noise removal
        ret4,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((13,13),np.uint8)
        opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),1,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers +1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        img = PIL.Image.fromarray(np.uint8(img)).convert('RGB')
        img = np.asarray(img)
        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]

        markers = np.where(markers<=1,0,markers)
        markers = np.where(markers!=0,1,markers)
        
        watershed_images[i]=markers
    return watershed_images



#%%
'''
This is where twopoint is preformed on the black and white images
'''

def twopoint(b_w_images,bound=False,cut=10):
    '''
    Process black and white images using two point statistics.
    Bounds is if the the boundary is periodic or not. For me, it is not
    Cut is the largest vector that can be drawn
    '''
    data_disc = PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(b_w_images)
    data_corr = TwoPointCorrelation(
        periodic_boundary=bound,
        cutoff=cut,
        correlations=[(0, 0), (0, 1)]
    ).transform(data_disc)
    return data_corr

def convert_to_numpy(bw_images,bounds=False,cut=10):
    size = len(bw_images[:,0,0])
    chunk = 30
    leftover = (size)%chunk
    image_array = np.array([])
    indexing = np.arange(0,size,30,dtype=int)
    if(indexing[-1]!=chunk+leftover):
        indexing = np.append(indexing,size)
    for i in range(len(indexing)-1):
        twopoint_images = twopoint(bw_images[indexing[i]:indexing[i+1],:,:],bounds,cut)
        good_images = twopoint_images[:,:,:,0]
        nparray = np.array(good_images)
        if(i==0):
            image_array = nparray.copy()
        else:
            image_array = np.append(image_array,nparray,axis=0)
    return image_array


#%%
'''
Preprocess all the two point images for machine learning
'''
def preprocess(images,b_w_function='cutoff',cut_off=0.5, invert=False, type_t= cv2.ADAPTIVE_THRESH_MEAN_C,block=51,sub=2, bounds=False,cuts=10):
    '''
    culmination of the all the functions.
    will returned fully preprocessed data for machine learning
    images are to be grayscale
    b_w_function is the function used to turn images from grayscale to black and white
    cut_off is decimal percent of the image turning to black or white
    bounds is the periodisity, so True or False
    cut is the largest vector drawn in the twopoint function
    target_col = the csv file has 4 cols, 0=pixel area, 1= vector perimeter, 2= circularity, 3= elispe aspect ratio
    '''
    if (b_w_function=='None'):
        size = np.size(images[0,:,:])
        flatten = images.reshape(size,-1)
        return flatten
    elif(b_w_function=='cutoff'):
        images = cutoff(images,cut_off,invert)
    elif(b_w_function=='adaptive'):
        images = adaptive_cutoff(images,type_t,block,sub)
    elif(b_w_function=='otsu'):
        images =  otsu(images,type_t)
    else:
        images = water_shed(images)
    images = convert_to_numpy(images,bounds,cuts)    
    return images

#%%
def umap_plotting(images,labels):
    '''

    Parameters
    ----------
    images : Array of images processed using two point, and flatten

    Returns
    -------
    umaping projection on a 2-d plane of all images.

    '''
    embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42).fit_transform(images)
    

    plt.scatter(embedding[:, 0],
                embedding[:, 1],
                c=labels,
                s=1,
                alpha=0.5)
    plt.title('UMAP Clustering of the data')
    plt.show()
    
#%%
def percent_pixel(processed_images,percent=0.5):
    y_predicted = []
    for i, img in enumerate(processed_images):
        mini = np.amin(img)
        maxi = np.amax(img)
        thresh = (maxi-mini)*percent+mini
        c = np.ones_like(img)
        c = np.where(img >=thresh,1,0)
        y_predicted.append(np.sum(c))
    y_predicted = np.array(y_predicted)
    return y_predicted

#%%
def test_val(train_images,train_y,val_images,val_y,bottom =0.001, top = 0.999, spacing = 1000):
    percent = np.linspace(bottom,top,spacing)
    score =[]
    for p in percent:
        temp = percent_pixel(train_images,p)
        mae = mean_absolute_error(train_y, temp)
        score.append(mae)
    score = np.array(score)
    best_percent = np.amin(percent[score==np.amin(score)])
    temp = percent_pixel(val_images,best_percent)
    r2,mae, rmse = evaluate(val_y,temp)
    return r2,mae,rmse, best_percent


#%%
#actually load the data in

images_test, y_test = loadimages(test_location,'test.csv')
images_val, y_val = loadimages(val_location,'val.csv')
images_train, y_train = loadimages(train_location,'train.csv')
y_test = y_test[:,0]
y_val = y_val[:,0]
y_train = y_train[:,0]
#%%
adaptive_mean_images_test = preprocess(images_test,'adaptive',0.5,False,cv2.ADAPTIVE_THRESH_MEAN_C,51,2,False,100)
adaptive_mean_images_val = preprocess(images_val,'adaptive',0.5,False,cv2.ADAPTIVE_THRESH_MEAN_C,51,2,False,100)
adaptive_mean_images_train = preprocess(images_train,'adaptive',0.5,False,cv2.ADAPTIVE_THRESH_MEAN_C,51,2,False,100)


adaptive_gauss_images_test = preprocess(images_test,'adaptive',0.5,False,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2,False,100)
adaptive_gauss_images_val = preprocess(images_val,'adaptive',0.5,False,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2,False,100)
adaptive_gauss_images_train = preprocess(images_train,'adaptive',0.5,False,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2,False,100)



threshold_images_test = preprocess(images_test,'cutoff',0.5,False,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2,False,100)
threshold_images_val = preprocess(images_val,'cutoff',0.5,False,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2,False,100)
threshold_images_train = preprocess(images_train,'cutoff',0.5,False,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2,False,100)

otsu_images_test = preprocess(images_test,'otsu',0.5,False,cv2.THRESH_BINARY,51,2,False,100)
otsu_images_val = preprocess(images_val,'otsu',0.5,False,cv2.THRESH_BINARY,51,2,False,100)
otsu_images_train = preprocess(images_train,'otsu',0.5,False,cv2.THRESH_BINARY,51,2,False,100)

watershed_images_test = preprocess(images_test,'watershed',0.5,False,cv2.THRESH_BINARY,51,2,False,100)
watershed_images_val = preprocess(images_val,'watershed',0.5,False,cv2.THRESH_BINARY,51,2,False,100)
watershed_images_train = preprocess(images_train,'watershed',0.5,False,cv2.THRESH_BINARY,51,2,False,100)

#%%
threshold_r2,threshold_mae,threshold_rmse, threshold_percent = test_val(threshold_images_train,y_train,threshold_images_val,y_val,.49,.5,100)
adaptive_mean_r2,adaptive_mean_mae, adaptive_mean_rmse,adaptive_mean_percent = test_val(adaptive_mean_images_train,y_train,adaptive_mean_images_val,y_val,0.1,0.2,100)
adaptive_gauss_r2,adaptive_gauss_mae,adaptive_gauss_rmse, adaptive_gauss_percent = test_val(adaptive_gauss_images_train,y_train,adaptive_gauss_images_val,y_val,0.06,0.1,100)
otsu_r2,otsu_mae,otsu_rmse, otsu_percent = test_val(otsu_images_train,y_train,otsu_images_val,y_val,0.49,.5,100)
watershed_r2,watershed_mae,watershed_rmse, watershed_percent = test_val(watershed_images_train,y_train,watershed_images_val,y_val,0.89,0.9,100)


#%%
#plotting for different algorithms
algorithms =["Threshold 50%","adaptive mean", "adaptive gauss", "otsu", "watershed"]
error_algo = [[threshold_r2, threshold_mae,threshold_rmse],
               [adaptive_mean_r2,adaptive_mean_mae, adaptive_mean_rmse],
               [adaptive_gauss_r2,adaptive_gauss_mae,adaptive_gauss_rmse],
               [otsu_r2,otsu_mae,otsu_rmse],
               [watershed_r2,watershed_mae,watershed_rmse]
               ]
distance = np.arange(len(error_algo))
error_algo = np.array(error_algo).transpose()

color =['b','g','r']
label =['r^2','mae','rmse']
i=0
for e in error_algo:
    plt.bar(distance +i*0.25,e,color=color[i],width=0.25,label=label[i])
    i+=1
    
plt.xticks(ticks=distance+0.25,labels=algorithms,rotation=45)
plt.ylabel('Error')
plt.title('Error vs algorithm')
plt.legend()
plt.show()
#%%
'''
Plot examples of highlighting the area of the average two particles
'''
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharex=True,sharey=True)
ax1.axes.xaxis.set_ticks([])
ax1.axes.yaxis.set_ticks([])

maxi1 = np.amax(threshold_images_test[0,:,:])
mini1 = np.amin(threshold_images_test[0,:,:])
thresh1 = (maxi1-mini1)*threshold_percent+mini1
ax1.imshow(np.where(threshold_images_test[0,:,:]>=thresh1,1,0),'gray')
ax1.set_title('Threshold 50%')

maxi2 = np.amax(adaptive_mean_images_test[0,:,:])
mini2 = np.amin(adaptive_mean_images_test[0,:,:])
thresh2 = (maxi2-mini2)*adaptive_mean_percent+mini2
ax2.imshow(np.where(adaptive_mean_images_test[0,:,:]>=thresh2,1,0),'gray')
ax2.set_title('Adaptive Mean')

maxi3 = np.amax(adaptive_gauss_images_test[0,:,:])
mini3 = np.amin(adaptive_gauss_images_test[0,:,:])
thresh3 = (maxi3-mini3)*adaptive_gauss_percent+mini3
ax3.imshow(np.where(adaptive_gauss_images_test[0,:,:]>=thresh3,1,0),'gray')
ax3.set_title('Adaptive Gauss')

maxi4 = np.amax(otsu_images_test[0,:,:])
mini4 = np.amin(otsu_images_test[0,:,:])
thresh4 = (maxi4-mini4)*otsu_percent+mini4
ax4.imshow(np.where(otsu_images_test[0,:,:]>=thresh4,1,0),'gray')
ax4.set_title('Otsu')

maxi5 = np.amax(watershed_images_test[0,:,:])
mini5 = np.amin(watershed_images_test[0,:,:])
thresh5 = (maxi5-mini5)*watershed_percent+mini5
ax5.imshow(np.where(watershed_images_test[0,:,:]>=thresh5,1,0),'gray')
ax5.set_title('Watershed')

plt.tight_layout()
plt.show()

#%%
'''
Plot examples of highlighting the area of the average two particles
'''
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
ax1.axes.xaxis.set_ticks([])
ax1.axes.yaxis.set_ticks([])

ax2.axes.xaxis.set_ticks([])
ax2.axes.yaxis.set_ticks([])

ax3.axes.xaxis.set_ticks([])
ax3.axes.yaxis.set_ticks([])

ax4.axes.xaxis.set_ticks([])
ax4.axes.yaxis.set_ticks([])

ax5.axes.xaxis.set_ticks([])
ax5.axes.yaxis.set_ticks([])

ax6.axes.xaxis.set_ticks([])
ax6.axes.yaxis.set_ticks([])

pred = percent_pixel(threshold_images_val,threshold_percent)
ax1.plot(pred,y_val,'.k')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
ax1.set_title('Threshold 50%')

pred = percent_pixel(adaptive_mean_images_val,threshold_percent)
ax2.plot(pred,y_val,'.k')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
ax2.set_title('Adaptive Mean')

pred = percent_pixel(adaptive_gauss_images_val,threshold_percent)
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')
ax3.plot(pred,y_val,'.k')
ax3.set_title('Adaptive Gauss')

pred = percent_pixel(otsu_images_val,threshold_percent)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('True')
ax4.plot(pred,y_val,'.k')
ax4.set_title('Otsu')

pred = percent_pixel(watershed_images_val,threshold_percent)
ax5.set_xlabel('Predicted')
ax5.set_ylabel('True')
ax5.plot(pred,y_val,'.k')
ax5.set_title('watershed')

plt.tight_layout()
plt.show()


