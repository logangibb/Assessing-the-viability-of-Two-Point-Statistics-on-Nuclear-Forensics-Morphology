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
import dask.array as da

from pymks import (
    plot_microstructures,
    PrimitiveTransformer,
    TwoPointCorrelation,
    FlattenTransformer
)

#%%
#Load in the data
#directories for the train,val, and test sets
train_location = r'C:\Users\u0944665\Box\2 Point Statistics\Train' #directory
val_location = r'C:\Users\u0944665\Box\2 Point Statistics\Val'
test_location = r'C:\Users\u0944665\Box\2 Point Statistics\Test'

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
    chunk = 10
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
def percent_pixel(processed_images,y_mama,percent=0.5):
    y_predicted = []
    for i, img in enumerate(processed_images):
        temp = img.compute()
        print(temp)
        mini = np.amin(temp)
        maxi = np.amax(temp)
        thresh = (maxi-mini)*percent+mini
        c = np.ones_like(temp)
        c = np.where(img >=thresh,1,0)
        y_predicted.append(np.sum(c))
    y_predicted = np.array(y_predicted)
    difference = np.abs(y_mama-y_predicted)/y_mama
    return difference

#%%
def test_val(train_images,train_y,val_images,val_y,bottom =0.001, top = 0.999, spacing = 10000):
    percent = np.linspace(bottom,top,spacing)
    difference_train_mean =[]
    difference_train_std = []
    for p in percent:
        temp = percent_pixel(train_images,train_y,p)
        difference_train_mean.append(np.mean(temp))
        difference_train_std.append(np.std(temp))
    
    difference_train_mean = np.array(difference_train_mean)
    difference_train_std = np.array(difference_train_std)
    best_percent = percent[difference_train_mean==np.amin(difference_train_mean)]
    best_train_mean = difference_train_mean[percent==best_percent]
    best_train_std = difference_train_std[percent==best_percent]
    temp = percent_pixel(val_images,val_y,best_percent)
    val_mean = np.mean(temp)
    val_std = np.std(temp)
    return val_mean, val_std, best_percent

#%%
#actually load the data in

images_test, y_test = loadimages(test_location,'test.csv')
images_val, y_val = loadimages(val_location,'val.csv')
images_train, y_train = loadimages(train_location,'train.csv')
y_test = y_test[:,0]
y_val = y_val[:,0]
y_train = y_train[:,0]

#%%
'''
Convert to black and white images
'''
adaptive_mean_bwimages_test = adaptive_cutoff(images_test,cv2.ADAPTIVE_THRESH_MEAN_C,51,2)
adaptive_mean_bwimages_val = adaptive_cutoff(images_val,cv2.ADAPTIVE_THRESH_MEAN_C,51,2)
adaptive_mean_bwimages_train = adaptive_cutoff(images_train,cv2.ADAPTIVE_THRESH_MEAN_C,51,2)

adaptive_gauss_bwimages_test = adaptive_cutoff(images_test,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2)
adaptive_gauss_bwimages_val = adaptive_cutoff(images_val,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2)
adaptive_gauss_bwimages_train = adaptive_cutoff(images_train,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2)

threshold_bwimages_test =cutoff(images_test,0.5,False)
threshold_bwimages_val=cutoff(images_val,0.5,False)
threshold_bwimages_train=cutoff(images_train,0.5,False)

otsu_bwimages_test = otsu(images_test,cv2.THRESH_BINARY)
otsu_bwimages_val = otsu(images_val,cv2.THRESH_BINARY)
otsu_bwimages_train = otsu(images_train,cv2.THRESH_BINARY)

watershed_bwimages_test = water_shed(images_test)
watershed_bwimages_val = water_shed(images_val)
watershed_bwimages_train = water_shed(images_train)

#%%
'''
Plot an example of the black and white images
'''
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharex=True,sharey=True)
fig.suptitle('Coversion of grayscale to black and white')
ax1.axes.xaxis.set_ticks([])
ax1.axes.yaxis.set_ticks([])
ax1.imshow(images_test[0,:,:], 'gray')
ax1.set_title('Orginial Image')
ax2.imshow(threshold_bwimages_test[0,:,:],'gray')
ax2.set_title('Threshold 50%')
ax3.imshow(adaptive_mean_bwimages_test[0,:,:], 'gray')
ax3.set_title('Adaptive Mean')
ax4.imshow(adaptive_gauss_bwimages_test[0,:,:],'gray')
ax4.set_title('Adaptive Gauss')
ax5.imshow(otsu_bwimages_test[0,:,:],'gray')
ax5.set_title('Otsu')
ax6.imshow(watershed_bwimages_test[0,:,:],'gray')
ax6.set_title('Watershed')

plt.tight_layout()
plt.show()
#%%
'''
Now do two point statistics
'''
bound = False
cut = 100
chunksize = 1
adaptive_mean_2ptimages_test = twopoint(adaptive_mean_bwimages_test,bound,cut)
adaptive_mean_2ptimages_val = twopoint(adaptive_mean_bwimages_val,bound,cut)
adaptive_mean_2ptimages_train = twopoint(adaptive_mean_bwimages_train,bound,cut)

adaptive_mean_2ptimages_test =adaptive_mean_2ptimages_test[:,:,:,0]
adaptive_mean_2ptimages_test = da.rechunk(adaptive_mean_2ptimages_test,(chunksize,201,201))
adaptive_mean_2ptimages_val = adaptive_mean_2ptimages_val[:,:,:,0]
adaptive_mean_2ptimages_val = da.rechunk(adaptive_mean_2ptimages_val,(chunksize,201,201))
adaptive_mean_2ptimages_train = adaptive_mean_2ptimages_train[:,:,:,0]
adaptive_mean_2ptimages_train = da.rechunk(adaptive_mean_2ptimages_train,(chunksize,201,201))

adaptive_gauss_2ptimages_test = twopoint(adaptive_gauss_bwimages_test,bound, cut)
adaptive_gauss_2ptimages_val = twopoint(adaptive_gauss_bwimages_val,bound, cut)
adaptive_gauss_2ptimages_train = twopoint(adaptive_gauss_bwimages_train,bound, cut)

adaptive_gauss_2ptimages_test = adaptive_gauss_2ptimages_test[:,:,:,0]
adaptive_gauss_2ptimages_test = da.rechunk(adaptive_gauss_2ptimages_test,(chunksize,201,201))
adaptive_gauss_2ptimages_val = adaptive_gauss_2ptimages_val[:,:,:,0]
adaptive_gauss_2ptimages_val = da.rechunk(adaptive_gauss_2ptimages_val,(chunksize,201,201))
adaptive_gauss_2ptimages_train = adaptive_gauss_2ptimages_train[:,:,:,0]
adaptive_gauss_2ptimages_train = da.rechunk(adaptive_gauss_2ptimages_train,(chunksize,201,201))

threshold_2ptimages_test = twopoint(threshold_bwimages_test,bound,cut)
threshold_2ptimages_val = twopoint(threshold_bwimages_val,bound,cut)
threshold_2ptimages_train = twopoint(threshold_bwimages_train,bound,cut)

threshold_2ptimages_test = threshold_2ptimages_test[:,:,:,0]
threshold_2ptimages_test = da.rechunk(threshold_2ptimages_test,(chunksize,201,201))
threshold_2ptimages_val = threshold_2ptimages_val[:,:,:,0]
threshold_2ptimages_val = da.rechunk(threshold_2ptimages_val,(chunksize,201,201))
threshold_2ptimages_train = threshold_2ptimages_train[:,:,:,0]
threshold_2ptimages_train = da.rechunk(threshold_2ptimages_train,(chunksize,201,201))

otsu_2ptimages_test = twopoint(otsu_bwimages_test,bound,cut)
otsu_2ptimages_val = twopoint(otsu_bwimages_val,bound,cut)
otsu_2ptimages_train = twopoint(otsu_bwimages_train,bound,cut)

otsu_2ptimages_test = otsu_2ptimages_test[:,:,:,0]
otsu_2ptimages_test = da.rechunk(otsu_2ptimages_test,(chunksize,201,201))
otsu_2ptimages_val = otsu_2ptimages_val[:,:,:,0]
otsu_2ptimages_val = da.rechunk(otsu_2ptimages_val,(chunksize,201,201))
otsu_2ptimages_train = otsu_2ptimages_train[:,:,:,0]
otsu_2ptimages_train = da.rechunk(otsu_2ptimages_train,(chunksize,201,201))

watershed_2ptimages_test = twopoint(watershed_bwimages_test,bound,cut)
watershed_2ptimages_val = twopoint(watershed_bwimages_val,bound,cut)
watershed_2ptimages_train = twopoint(watershed_bwimages_train,bound,cut)

watershed_2ptimages_test = watershed_2ptimages_test[:,:,:,0]
watershed_2ptimages_test = da.rechunk(watershed_2ptimages_test,(chunksize,201,201))
watershed_2ptimages_val = watershed_2ptimages_val[:,:,:,0]
watershed_2ptimages_val = da.rechunk(watershed_2ptimages_val,(chunksize,201,201))
watershed_2ptimages_train = watershed_2ptimages_train[:,:,:,0]
watershed_2ptimages_train = da.rechunk(watershed_2ptimages_train,(chunksize,201,201))


#%%
'''
Plot examples of two point statistics transforms
'''
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharex=True,sharey=True)
fig.suptitle('Two point transform')
ax1.axes.xaxis.set_ticks([])
ax1.axes.yaxis.set_ticks([])
im1 = ax1.imshow(threshold_2ptimages_test[0,:,:])
ax1.set_title('Threshold 50%')
cbar1 = plt.colorbar(im1, ax=ax1)
im2 = ax2.imshow(adaptive_mean_2ptimages_test[0,:,:])
ax2.set_title('Adaptive Mean')
cbar2 = plt.colorbar(im2, ax=ax2)
im3 = ax3.imshow(adaptive_gauss_2ptimages_test[0,:,:])
ax3.set_title('Adaptive Gauss')
cbar3 = plt.colorbar(im3, ax=ax3)
im4= ax4.imshow(otsu_2ptimages_test[0,:,:])
ax4.set_title('Otsu')
cbar4 = plt.colorbar(im4, ax=ax4)
im5 = ax5.imshow(watershed_2ptimages_test[0,:,:])
ax5.set_title('Watershed')
cbar5 = plt.colorbar(im5, ax=ax5)

plt.tight_layout()
plt.show()





