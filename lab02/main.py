import scipy
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
filein="./lab2_moles/medium_risk_8.jpg"
fileout="medium_risk_8"

# Open the image and show it
im_or= mpimg.imread(filein)
plt.figure()
plt.imshow(im_or)
plt.title('original image')
plt.savefig(fileout+".pdf",bbox_inches='tight')
plt.show()

# The image is reshaped from 3D to 2D
[N1,N2,N3]=im_or.shape
im_2D=im_or.reshape((N1*N2,N3))
[Nr,Nc]=im_2D.shape

# KMeans algorithm is applied to the image
Ncluster=3
kmeans=KMeans(n_clusters=Ncluster, random_state=0)
kmeans.fit(im_2D)

#In a new image clusters are shown
imm=im_2D.copy()
kmeans_centroids=kmeans.cluster_centers_.astype("uint8")
for kc in range(Ncluster): # For every cluster found
    ind=(kmeans.labels_==kc) # Find the cluster in the image
    imm[ind,:]=kmeans_centroids[kc,:] #Save the cluster in the new image

#The new image is reshaped in 3D and shown
imm_sci=imm.reshape((N1,N2,N3))
plt.figure()
plt.imshow(imm_sci,interpolation=None)
plt.savefig(fileout+"_centroids"+".pdf",bbox_inches='tight')
plt.title("result of scikit kmeans")
plt.show()

#The centroids are evaluated and the one with
#the lowest RGB color scale is taken.
centroids=kmeans_centroids
sc=np.sum(centroids,axis=1)
i_col=sc.argmin()
im_clust=kmeans.labels_.reshape(N1,N2)
zpos=np.argwhere(im_clust==i_col) # The mole pixels indexes are saved
N_spots_str=input("How many distinct dark spots you can see in the image?")
N_spots=int(N_spots_str)

#If only one centroid is seen, that is for sure the one describing the mole
if N_spots==1:
    center_mole=np.median(zpos,axis=0).astype(int)
else: #otherwise, we reapply the KMeans algorithm to select the cluster corresponding to the mole
    kmeans2= KMeans(n_clusters=N_spots, random_state=0)
    kmeans2.fit(zpos)
    centers=kmeans2.cluster_centers_.astype(int)

    center_image=np.array([N1//2,N2//2])
    center_image.shape=(1,2)
    d=np.zeros((N_spots,1))
    for k in range(N_spots):
        d[k]=np.linalg.norm(center_image-centers[k,:])
    center_mole=centers[d.argmin(),:]

cond=True
area_old=0
step=10

c0=center_mole[0] #Centroid x coordinate
c1=center_mole[1] #Centroid y coordinate
im_sel=(im_clust==i_col) #We select in the cluster image map, the one with the lower RGB
im_sel=im_sel*1

# We define the mole and we calculate its area (in pixels)
while cond:
    subset=im_sel[c0-step:c0+step+1,c1-step:c1+step+1]
    area=np.sum(subset)
    if area>area_old:
        step=step+10
        area_old=area
        cond=True
    else:
        cond=False
plt.matshow(subset) # The area of the mole is plotted
plt.savefig(fileout+"_central_mole"+".pdf",bbox_inches='tight')


