import PIL
import os
import numpy as np
import matplotlib.pyplot as plt

loc = r'C:\Users\u0944665\Box\MODE-Nizinski-Train-Images' #directory


def loadimages(directory):
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

images, target  = loadimages(loc)


#%%
'''
Plot an example of the different train groups
'''
fig, ax = plt.subplots(5,3,sharex=True,sharey=True,figsize=(15,15))
#fig.suptitle('Coversion of grayscale to black and white')
ax[0,0].axes.xaxis.set_ticks([])
ax[0,0].axes.yaxis.set_ticks([])

starting = ['ADU-','AUC-','MDU-','SDU-','UO4-']
oxide = ['UO3','U3O8','UO2']
size = np.zeros((5,3))
name = np.empty((5,3),dtype=object)
#%%
for i, s in enumerate(starting):
    for j, o in enumerate(oxide):
        temp = images[target == s+o]
        size[i,j] = np.sum(target == s+o)
        name[i,j] = (s+o)
        ax[i,j].imshow(temp[0],'gray')
        ax[i,j].set_title(s+o)

plt.tight_layout()
plt.show()































