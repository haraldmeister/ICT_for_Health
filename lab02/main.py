from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

class Mole():
    def __init__(self,filein,fileout):

        np.set_printoptions(precision=2)# use only two decimal digits when printing numbers
        plt.close('all')# close previously opened pictures
        #filein='medium_risk_8.jpg';# file to be analyzed
        im_or = mpimg.imread(filein)
        # im_or is Ndarray 583 x 584 x 3 unint8

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
        Ncluster=4
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
        plt.imshow(im_quant)
        plt.pause(0.5)
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

        self.filein=fileout
        self.im_quant=im_quant
        self.image_or=im_or
        self.image=subset
        self.dim=subset.shape
        self.binary_im=np.zeros(self.dim,dtype=int)
        self.image_filtered=np.zeros(self.dim,dtype=int)
        self.image_contour=np.zeros(self.dim,dtype=int)
        self.mask=np.array(((0,1,0),(1,1,1),(0,1,0)),dtype=int)
        self.perimeter=0
        self.area=0

    def polish(self):
        #[Nr,Nc]=image.shape
        #mask=np.array(((0,1,0),(1,1,1),(0,1,0)),dtype=int)
        for i in ['down','right','up','left']:
            self.filter_background(i)

        #Now the image holes should be filled
        #Define a new image and save the original image's background
        self.find_background()

        # Invert the new image to show holes
        holes_image=1*np.logical_not(self.image_filtered)
        # Fill the original image holes
        self.image_filtered=1*np.logical_or(self.binary_im,holes_image)
        return

    def filter_background(self,dir):
        [Nr,Nc]=self.dim
        if dir=='down':
            for j in range(1,Nc-1):
                for i in range(1,Nr-1):
                    subim=np.array(((self.image[i-1,j-1],self.image[i-1,j],self.image[i-1,j+1]),
                                    (self.image[i,j-1],self.image[i,j],self.image[i,j+1]),
                                    (self.image[i+1,j-1],self.image[i+1,j],self.image[i+1,j+1])),dtype=int)
                    result=subim==self.mask
                    N_pic=np.sum((result[0,1],result[1,0],result[1,1],result[1,2],result[2,1]))
                    N_backg=5-N_pic
                    if N_pic>=N_backg:
                        self.binary_im[i][j]=1
                    else:
                        self.binary_im[i][j]=0
        elif dir=='right':
            for i in range(1,Nr-1):
                for j in range(1,Nc-1):
                    subim=np.array(((self.image[i-1,j-1],self.image[i-1,j],self.image[i-1,j+1]),
                                    (self.image[i,j-1],self.image[i,j],self.image[i,j+1]),
                                    (self.image[i+1,j-1],self.image[i+1,j],self.image[i+1,j+1])),dtype=int)
                    result=subim==self.mask
                    N_pic=np.sum((result[0,1],result[1,0],result[1,1],result[1,2],result[2,1]))
                    N_backg=5-N_pic
                    if N_pic>=N_backg:
                        self.binary_im[i][j]=1
                    else:
                        self.binary_im[i][j]=0
        elif dir=='up':
            for j in range(1,Nc-1):
                for i in reversed(range(1,Nr-1)):
                    subim=np.array(((self.image[i-1,j-1],self.image[i-1,j],self.image[i-1,j+1]),
                                    (self.image[i,j-1],self.image[i,j],self.image[i,j+1]),
                                    (self.image[i+1,j-1],self.image[i+1,j],self.image[i+1,j+1])),dtype=int)
                    result=subim==self.mask
                    N_pic=np.sum((result[0,1],result[1,0],result[1,1],result[1,2],result[2,1]))
                    N_backg=5-N_pic
                    if N_pic>=N_backg:
                        self.binary_im[i][j]=1
                    else:
                        self.binary_im[i][j]=0
        elif dir=='left':
            for i in reversed(range(1,Nr-1)):
                for j in reversed(range(1,Nc-1)):
                    subim=np.array(((self.image[i-1,j-1],self.image[i-1,j],self.image[i-1,j+1]),
                                    (self.image[i,j-1],self.image[i,j],self.image[i,j+1]),
                                    (self.image[i+1,j-1],self.image[i+1,j],self.image[i+1,j+1])),dtype=int)
                    result=subim==self.mask
                    N_pic=np.sum((result[0,1],result[1,0],result[1,1],result[1,2],result[2,1]))
                    N_backg=5-N_pic
                    if N_pic>=N_backg:
                        self.binary_im[i][j]=1
                    else:
                        self.binary_im[i][j]=0
        return

    def find_background(self):
        for i in range(0,self.dim[0]):
            for j in range(0,self.dim[1]):
                if self.binary_im[i][j]==1:
                    break
                else:
                    self.image_filtered[i][j]=1
        for i in range(0,self.dim[0]):
            for j in reversed(range(0,self.dim[1])):
                if self.binary_im[i][j]==1:
                    break
                else:
                    self.image_filtered[i][j]=1
        return

    def find_contour(self):
        [Nr,Nc]=self.dim
        for j in range(0,Nc):
            for i in range(0,Nr):
                if self.image_filtered[i][j]==1:
                    self.image_contour[i][j]=1
                    break
        for i in range(0,Nr):
            for j in range(0,Nc):
                if self.image_filtered[i][j]==1:
                    self.image_contour[i][j]=1
                    break
        for j in range(0,Nc):
            for i in reversed(range(0,Nr)):
                if self.image_filtered[i][j]==1:
                    self.image_contour[i][j]=1
                    break
        for i in range(0,Nr):
            for j in reversed(range(0,Nc)):
                if self.image_filtered[i][j]==1:
                    self.image_contour[i][j]=1
                    break
        self.perimeter=np.sum(self.image_contour)
        return

    def find_area(self):
        self.area=np.sum(self.image_filtered)
        return

    def print_result(self):
        print("Mole %s\n" %self.filein)
        print("Area mole = %d\n" % self.area)
        print("Perimeter mole= %d\n\n" % self.perimeter)

        ideal_circle_per= 2*np.sqrt(self.area*np.pi)
        ratio=float(self.perimeter/ideal_circle_per)
        print("Perimeter ideal circle having same area = %f\n" % ideal_circle_per)
        print("Ratio perimeter mole and ideal circle = %f" %ratio)

    def plot_image(self):
        # plot the image, to check it is correct:
        fileout=self.filein
        plt.imshow(self.image_or)
        plt.title('original image')
        plt.savefig(fileout+".pdf",bbox_inches='tight')


    def plot_centroids(self):
        fileout=self.filein
        plt.imshow(self.im_quant)
        plt.title('Centroids output of K-Means')
        plt.savefig(fileout+"_centroids"+".pdf",bbox_inches='tight')


    def plot_binary_im(self):
        fileout=self.filein
        plt.imshow(self.image)
        plt.title('Binary image')
        plt.savefig(fileout+"_central_mole"+".pdf",bbox_inches='tight')


    def plot_binary_im_polished(self):
        fileout=self.filein
        plt.imshow(self.image_filtered)
        plt.title('Binary image polished')
        plt.savefig(fileout+"_central_mole_polished"+".pdf",bbox_inches='tight')


filein=["./lab2_moles/low_risk_1.jpg","./lab2_moles/low_risk_2.jpg",
        "./lab2_moles/low_risk_3.jpg","./lab2_moles/low_risk_3_h.jpg",
        "./lab2_moles/low_risk_3_s.jpg","./lab2_moles/low_risk_4.jpg",
        "./lab2_moles/low_risk_5.jpg","./lab2_moles/low_risk_6.jpg",
        "./lab2_moles/low_risk_7.jpg","./lab2_moles/low_risk_8.jpg",
        "./lab2_moles/low_risk_9.jpg","./lab2_moles/low_risk_10.jpg",
        "./lab2_moles/low_risk_11.jpg","./lab2_moles/medium_risk_1.jpg",
        "./lab2_moles/medium_risk_2.jpg","./lab2_moles/medium_risk_3.jpg",
        "./lab2_moles/medium_risk_3_h.jpg","./lab2_moles/medium_risk_3_s.jpg",
        "./lab2_moles/medium_risk_4.jpg","./lab2_moles/medium_risk_5.jpg",
        "./lab2_moles/medium_risk_6.jpg","./lab2_moles/medium_risk_7.jpg",
        "./lab2_moles/medium_risk_8.jpg","./lab2_moles/medium_risk_9.jpg",
        "./lab2_moles/medium_risk_10.jpg","./lab2_moles/medium_risk_11.jpg",
        "./lab2_moles/medium_risk_12.jpg","./lab2_moles/medium_risk_13.jpg",
        "./lab2_moles/medium_risk_14.jpg","./lab2_moles/medium_risk_15.jpg",
        "./lab2_moles/medium_risk_16.jpg","./lab2_moles/melanoma_1.jpg",
        "./lab2_moles/melanoma_2.jpg","./lab2_moles/melanoma_3.jpg",
        "./lab2_moles/melanoma_4.jpg","./lab2_moles/melanoma_5.jpg",
        "./lab2_moles/melanoma_6.jpg","./lab2_moles/melanoma_7.jpg",
        "./lab2_moles/melanoma_8.jpg","./lab2_moles/melanoma_9.jpg",
        "./lab2_moles/melanoma_10.jpg","./lab2_moles/melanoma_11.jpg",
        "./lab2_moles/melanoma_12.jpg","./lab2_moles/melanoma_13.jpg",
        "./lab2_moles/melanoma_14.jpg","./lab2_moles/melanoma_15.jpg",
        "./lab2_moles/melanoma_16.jpg","./lab2_moles/melanoma_17.jpg",
        "./lab2_moles/melanoma_18.jpg","./lab2_moles/melanoma_19.jpg",
        "./lab2_moles/melanoma_20.jpg","./lab2_moles/melanoma_21.jpg",
        "./lab2_moles/melanoma_22.jpg","./lab2_moles/melanoma_23.jpg",
        "./lab2_moles/melanoma_24.jpg","./lab2_moles/melanoma_25.jpg",
        "./lab2_moles/melanoma_26.jpg","./lab2_moles/melanoma_27.jpg"]

fileout=["low_risk_1","low_risk_2","low_risk_3","low_risk_3_h",
         "low_risk_3_s","low_risk_4","low_risk_5","low_risk_6","low_risk_7",
         "low_risk_8","low_risk_9","low_risk_10","low_risk_11",
         "medium_risk_1","medium_risk_2","medium_risk_3","medium_risk_3_h",
         "medium_risk_3_s","medium_risk_4","medium_risk_5","medium_risk_6",
         "medium_risk_7","medium_risk_8","medium_risk_9","medium_risk_10",
         "medium_risk_11","medium_risk_12","medium_risk_13","medium_risk_14",
         "medium_risk_15","medium_risk_16","melanoma_1","melanoma_2","melanoma_3",
         "melanoma_4","melanoma_5","melanoma_6","melanoma_7","melanoma_8","melanoma_9",
         "melanoma_10","melanoma_11","melanoma_12","melanoma_13","melanoma_14",
         "melanoma_15","melanoma_16","melanoma_17","melanoma_18","melanoma_19",
         "melanoma_20","melanoma_21","melanoma_22","melanoma_23","melanoma_24",
         "melanoma_25","melanoma_26","melanoma_27"]

moles=[]
for i in range(0,len(filein)):
    moles.append(Mole(filein[i],fileout[i]))
    moles[i].plot_image()
    moles[i].plot_centroids()
    moles[i].plot_binary_im()
    moles[i].polish()
    moles[i].plot_binary_im_polished()
    moles[i].find_contour()
    moles[i].find_area()
    moles[i].print_result()

