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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import umap
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB



from pymks import (
    PrimitiveTransformer,
    TwoPointCorrelation,
)

#%%
#Load in the data
#directories for the train,val, and test sets
direct = r'C:\Users\u0944665\Box\MODE-Nizinski-Train-Images' #directory
def loadimages(directory):
    '''
    loads in the target images and aranges the target class based on the files name.
    '''
    left = 0
    top = 0
    right = 1024
    bottom = 880
    # load all images in a directory
    images = []
    target = []
    for filename in os.listdir(directory):
    	# load image
        if('50000x' in filename):
            loc = os.path.join(directory,filename)
            img = PIL.Image.open(loc,formats=('TIFF','JPEG')) #load in the image
            img = img.crop((left,top,right,bottom))
            if(img.mode == 'I;16'):
                img = np.asarray(img)
                img = np.uint8(np.array(img)/256)
            elif(img.mode=='L'):
                img = np.asarray(img)
            else:
                img = img.convert(mode='L')
                img = np.asarray(img)
            images.append(img)
            target_name = ''
            #
            if('ADU' in filename):
                target_name ='ADU-'
            elif('AUC' in filename):
                target_name = 'AUC-'
            elif('MDU' in filename):
                target_name = 'MDU-'
            elif('SDU' in filename):
                target_name = 'SDU-'
            else:
                target_name = 'UO4-'
            if('UO3' in filename):
                target_name = target_name+'UO3'
            elif('U3O8' in filename):
                target_name = target_name +'U3O8'
            else:
                target_name = target_name + 'UO2'
            target.append(target_name)
    images = np.array(images)
    target = np.array(target)
    return images, target

images, target = loadimages(direct)

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
    s = data_corr.shape
    res = (10,s[1],s[2],s[3])
    data_corr = data_corr.rechunk(res)
    return data_corr

def convert_to_numpy(bw_images,bounds=False,cut=10):
    '''
    Converts a dask array to a numpy array. The PYMKS is written for dask arrays.
    This is by far the slowest step in the code.
    '''
    size = len(bw_images[:,0,0])
    chunk = 1
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
'''
performing a two point on data
'''
bounds = False
cut =100
threshold_images = preprocess(images,'cutoff',0.5,False,cv2.THRESH_BINARY,51,2,bounds, cut)
adaptive_mean_images = preprocess(images,'adaptive',0.5,False,cv2.ADAPTIVE_THRESH_MEAN_C,51,2,bounds,cut)
adaptive_gauss_images = preprocess(images,'adaptive',0.5,False,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,51,2,bounds,cut)
otsu_images = preprocess(images,'otsu',0.5,False,cv2.THRESH_BINARY,51,2,bounds,cut)
watershed_images = preprocess(images,'watershed',0.5,False,cv2.THRESH_BINARY,51,2,bounds,cut)
#%%
'''
Plot a umap imbedding
'''

def umap_plotting(images, ax, title):
    '''

    Parameters
    ----------
    images : Array of images processed using two point, and flatten

    Returns
    -------
    umaping projection on a 2-d plane of all images.

    '''
    embedding = umap.UMAP(
        n_neighbors=10,
        min_dist=0.0,
        n_components=2,
        random_state=42).fit_transform(images)
    ax.scatter(embedding[:,0],
                embedding[:,1],
                s=0.1,
                )

    ax.set_title(title)



size = np.size(threshold_images[:,0,0])
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharex=True,sharey=True)
fig.suptitle('UMAP')
ax1.axes.xaxis.set_ticks([])
ax1.axes.yaxis.set_ticks([])

umap_plotting(threshold_images.reshape(size,-1),  ax1, 'Threshold')

umap_plotting(adaptive_mean_images.reshape(size,-1),  ax2, 'Adaptive Mean')

umap_plotting(adaptive_gauss_images.reshape(size,-1),  ax3, 'Addaptive Gauss')

umap_plotting(otsu_images.reshape(size,-1),  ax4, 'Otsu')

umap_plotting(watershed_images.reshape(size,-1),  ax5, 'Watershed')

plt.tight_layout()
plt.show()

#%%
'''
Trying different classifications
Following https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
'''
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix


algo =['Threshold 50%', 'Adaptive Mean', 'Adaptive Gauss', 'Otsu', 'Watershed']
data = [threshold_images.reshape(size,-1), adaptive_mean_images.reshape(size,-1),adaptive_gauss_images.reshape(size,-1),otsu_images.reshape(size,-1),watershed_images.reshape(size,-1)]
#data = [threshold_data, adaptive_mean_data,adaptive_gauss_data,otsu_data,watershed_data]
names = [
    "SVM",
    "Random Forest",
]
n_jobs = -1

parms_svm = {'C': uniform(),
             "kernel":['linear','poly','rbf','sigmoid'],
             }

parms_rfc = {"n_estimators":randint(1,1000),
              "max_depth": randint(1,1000)
              }

parms = [parms_svm,parms_rfc]
#parms = [parms_svm,parms_rfc]
  

classifiers = [
    SVC(),
    RandomForestClassifier(n_jobs=n_jobs),
]
accuracy = []
best_parameters = []
confusion = []

for al, da in zip(algo,data):
    print(al)
    X_train, X_test, y_train, y_test = train_test_split(da,target,test_size = 0.3,random_state=42)
    score =[]

    for i, clf in enumerate(classifiers):
        print(names[i])
        random_cv = RandomizedSearchCV(clf,parms[i],n_iter=25)
        search=random_cv.fit(X_train,y_train)
        best_p = search.best_params_
        best_parameters.append(best_p)
        score.append(search.score(X_test,y_test))
        y_pred = search.predict(X_test)
        confusion.append({al+names[i]:confusion_matrix(y_test, y_pred)})
    score = np.array(score)
    
    accuracy.append(score)
    
accuracy= np.array(accuracy)

#%%
'''
Making a table with the best accuracies
'''
plt.table(cellText=accuracy,rowLabels=algo, colLabels=names, loc='center')
plt.axis('tight')
plt.axis('off')
plt.savefig(r'C:\Users\u0944665\OneDrive\Spring 2022\Material Infomatics\Final\accuracies',dpi=1200,bbox_inches='tight')







#%%
'''
Plot the best confusion matrix
'''
temp = confusion[4]

best_confusion = temp['Adaptie GaussSVM']

plt.imshow(best_confusion,cmap='viridis')
plt.xticks([])
plt.yticks([])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on the Test set')
for (j,i),label in np.ndenumerate(best_confusion):
    plt.text(i,j,label,ha='center',va='center')
    plt.text(i,j,label,ha='center',va='center')

plt.show()




#%%
'''
Make a table of the number of different types of images
'''
oxide_size = 3
oxide_labels = ['U3O8','UO3','UO2']
starting_material_size = 5
starting_labels = ['ADU','AUC','MDU','SDU','UO4']

distribution = np.zeros((oxide_size,starting_material_size), dtype=int)

for o, oxide in enumerate( oxide_labels):
    for s, start in enumerate(starting_labels):
        distribution[o,s] = np.sum(target==start+'-'+oxide)
            
plt.table(cellText=distribution,rowLabels=oxide_labels, colLabels=starting_labels, loc='center')
plt.axis('tight')
plt.axis('off')
plt.savefig(r'C:\Users\u0944665\OneDrive\Spring 2022\Material Infomatics\Final\distribution',dpi=1200,bbox_inches='tight')



