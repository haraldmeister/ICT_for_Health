# -*- coding: utf-8 -*-
"""
@author: monica
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.set_printoptions(precision=2)# use only two decimal digits when printing numbers
plt.close('all')# close previously opened pictures
#filein='medium_risk_8.jpg';# file to be analyzed
filein='low_risk_4.jpg';# file to be analyzed
im_or = mpimg.imread(filein)
# im_or is Ndarray 583 x 584 x 3 unint8 
# plot the image, to check it is correct:
plt.figure()
plt.imshow(im_or)
plt.title('original image')
#plt.draw()
plt.pause(0.1)
#%% reshape the image from 3D to 2D
N1,N2,N3=im_or.shape # note: N3 is 3, the number of elementary colors, i.e. red, green ,blue
# im_or(i,j,1) stores the amount of red for the pixel in position i,j
# im_or(i,j,2) stores the amount of green for the pixel in position i,j
# im_or(i,j,3) stores the amount of blue for the pixel in position i,j
# we resize the original image
im_2D=im_or.reshape((N1*N2,N3))# im_2D has N1*N2 rows and N3 columns
# pixel in position i.j goes to position k=(i-1)*N2+j)
# im_2D(k,1) stores the amount of red of pixel k 
# im_2D(k,2) stores the amount of green of pixel k 
# im_2D(k,3) stores the amount of blue of pixel k 
# im_2D is a sequence of colors, that can take 2^24 different values
Nr,Nc=im_2D.shape
#%% get a simplified image with only Ncluster colors
# number of clusters/quantized colors we want to have in the simpified image:
Ncluster=3
# instantiate the object K-means:
kmeans = KMeans(n_clusters=Ncluster, random_state=0)
# run K-means:
kmeans.fit(im_2D)
# get the centroids (i.e. the 3 colors). Note that the centroids
# take real values, we must convert these values to uint8
# to properly see the quantized image
kmeans_centroids=kmeans.cluster_centers_.astype('uint8')
# copy im_2D into im_2D_quant
im_2D_quant=im_2D.copy()
for kc in range(Ncluster):
    quant_color_kc=kmeans_centroids[kc,:]
    # kmeans.labels_ stores the cluster index for each of the Nr pixels
    # find the indexes of the pixels that belong to cluster kc
    ind=(kmeans.labels_==kc)
    # set the quantized color to these pixels
    im_2D_quant[ind,:]=quant_color_kc
im_quant=im_2D_quant.reshape((N1,N2,N3))
plt.figure()
plt.imshow(im_quant,interpolation=None)
plt.title('image with quantized colors')
#plt.draw()
plt.pause(0.1)
#%% Find the centroid of the main mole

#%% Preliminary steps to find the contour after the clustering
# 
# 1: find the darkest color found by k-means, since the darkest color
# corresponds to the mole:
centroids=kmeans_centroids
sc=np.sum(centroids,axis=1)
i_col=sc.argmin()# index of the cluster that corresponds to the darkest color
# 2: define the 2D-array where in position i,j you have the number of
# the cluster pixel i,j belongs to 
im_clust=kmeans.labels_.reshape(N1,N2)
# plt.matshow(im_clust)
# 3: find the positions i,j where im_clust is equal to i_col
# the 2D Ndarray zpos stores the coordinates i,j only of the pixels
# in cluster i_col
zpos=np.argwhere(im_clust==i_col)
# 4: ask the user to write the number of objects belonging to
# cluster i_col in the image with quantized colors

N_spots_str=input("How many distinct dark spots can you see in the image? ")
N_spots=int(N_spots_str)
# 5: find the center of the mole
if N_spots==1:
    center_mole=np.median(zpos,axis=0).astype(int)
else:
    # use K-means to get the N_spots clusters of zpos
    kmeans2 = KMeans(n_clusters=N_spots, random_state=0)
    kmeans2.fit(zpos)
    centers=kmeans2.cluster_centers_.astype(int)
    # the mole is in the middle of the picture:
    center_image=np.array([N1//2,N2//2])
    center_image.shape=(1,2)
    d=np.zeros((N_spots,1))
    for k in range(N_spots):
        d[k]=np.linalg.norm(center_image-centers[k,:])
    center_mole=centers[d.argmin(),:]    
# 6: take a subset of the image that includes the mole
cond=True
area_old=0
step=10# each time the algorithm increases the area by 2*step pixels 
# horizontally and vertically
c0=center_mole[0]
c1=center_mole[1]
im_sel=(im_clust==i_col)# im_sel is a boolean NDarray with N1 rows and N2 columns
im_sel=im_sel*1# im_sel is now an integer NDarray with N1 rows and N2 columns
while cond:
    subset=im_sel[c0-step:c0+step+1,c1-step:c1+step+1]
    area=np.sum(subset)
    if area>area_old:
        step=step+10
        area_old=area
        cond=True
    else:
        cond=False
        # subset is the serach area
plt.matshow(subset)
