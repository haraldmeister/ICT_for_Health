import scipy
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def filter_background(image,mask,dir,Nr,Nc):
    if dir=='down':
        for j in range(1,Nc-1):
            for i in range(1,Nr-1):
                subim=np.array(((image[i-1,j-1],image[i-1,j],image[i-1,j+1]),
                                (image[i,j-1],image[i,j],image[i,j+1]),
                                (image[i+1,j-1],image[i+1,j],image[i+1,j+1])),dtype=int)
                result=subim==mask
                N_pic=np.sum((result[0,1],result[1,0],result[1,1],result[1,2],result[2,1]))
                N_backg=5-N_pic
                if N_pic>=N_backg:
                    image[i][j]=1
                else:
                    image[i][j]=0
    elif dir=='right':
        for i in range(1,Nr-1):
            for j in range(1,Nc-1):
                subim=np.array(((image[i-1,j-1],image[i-1,j],image[i-1,j+1]),
                                (image[i,j-1],image[i,j],image[i,j+1]),
                                (image[i+1,j-1],image[i+1,j],image[i+1,j+1])),dtype=int)
                result=subim==mask
                N_pic=np.sum((result[0,1],result[1,0],result[1,1],result[1,2],result[2,1]))
                N_backg=5-N_pic
                if N_pic>=N_backg:
                    image[i][j]=1
                else:
                    image[i][j]=0
    elif dir=='up':
        for j in range(1,Nc-1):
            for i in reversed(range(1,Nr-1)):
                subim=np.array(((image[i-1,j-1],image[i-1,j],image[i-1,j+1]),
                                (image[i,j-1],image[i,j],image[i,j+1]),
                                (image[i+1,j-1],image[i+1,j],image[i+1,j+1])),dtype=int)
                result=subim==mask
                N_pic=np.sum((result[0,1],result[1,0],result[1,1],result[1,2],result[2,1]))
                N_backg=5-N_pic
                if N_pic>=N_backg:
                    image[i][j]=1
                else:
                    image[i][j]=0
    elif dir=='left':
        for i in reversed(range(1,Nr-1)):
            for j in reversed(range(1,Nc-1)):
                subim=np.array(((image[i-1,j-1],image[i-1,j],image[i-1,j+1]),
                                (image[i,j-1],image[i,j],image[i,j+1]),
                                (image[i+1,j-1],image[i+1,j],image[i+1,j+1])),dtype=int)
                result=subim==mask
                N_pic=np.sum((result[0,1],result[1,0],result[1,1],result[1,2],result[2,1]))
                N_backg=5-N_pic
                if N_pic>=N_backg:
                    image[i][j]=1
                else:
                    image[i][j]=0
    return

def polish_image(image):
    [Nr,Nc]=image.shape
    mask=np.array(((0,1,0),(1,1,1),(0,1,0)),dtype=int)
    for i in ['down','right','up','left']:
        filter_background(image,mask,i,Nr,Nc)

    #Now the image holes should be filled
    #Define a new image and save the original image's background
    image3=np.zeros([Nr,Nc],dtype=int)
    for i in range(0,Nr):
        for j in range(0,Nc):
            if image[i][j]==1:
                break
            else:
                image3[i][j]=1
    for i in range(0,Nr):
        for j in reversed(range(0,Nc)):
            if image[i][j]==1:
                break
            else:
                image3[i][j]=1

    # Invert the new image to show holes
    holes_image=1*np.logical_not(image3)
    # Fill the original image holes
    final_image=1*np.logical_or(image,holes_image)
    return final_image

def find_contour(image):
    [Nr,Nc]=image.shape
    image3=np.zeros([Nr,Nc],dtype=int)
    for j in range(0,Nc):
        for i in range(0,Nr):
            if image[i][j]==1:
                image3[i][j]=1
                break
    for i in range(0,Nr):
        for j in range(0,Nc):
            if image[i][j]==1:
                image3[i][j]=1
                break
    for j in range(0,Nc):
        for i in reversed(range(0,Nr)):
            if image[i][j]==1:
                image3[i][j]=1
                break
    for i in range(0,Nr):
        for j in reversed(range(0,Nc)):
            if image[i][j]==1:
                image3[i][j]=1
                break
    perimeter=np.sum(image3)
    return perimeter

plt.close('all')
filein="./lab2_moles/medium_risk_1.jpg"
fileout="medium_risk_1"

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
Ncluster=4
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
subset=polish_image(subset)
plt.matshow(subset) # The area of the mole is plotted
plt.savefig(fileout+"_central_mole_polished"+".pdf",bbox_inches='tight')

area_polished=np.sum(subset)
per=find_contour(subset)

print("Area mole with hole = %d\n" % area)
print("Area mole without holes = %d\n" % area_polished)
print("Perimeter mole= %d\n\n" % per)

ideal_circle_per= 2*np.sqrt(area_polished*np.pi)
ratio=float(per/ideal_circle_per)
print("Perimeter ideal circle having same area = %f\n" % ideal_circle_per)
print("Ratio perimeter mole and ideal circle = %f" %ratio)


