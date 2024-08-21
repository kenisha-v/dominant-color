import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('chihiro.jpg') #takes in image in RGB 
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
original_shape = im.shape
print(im.shape)

plt.imshow(im)
plt.show()


all_pixels = im.reshape((800*1200,3))
print(all_pixels.shape)


from sklearn.cluster import KMeans
dominant_colors = 4
km = KMeans(n_clusters = dominant_colors)
km.fit(all_pixels)
centers = km.cluster_centers_
centers = np.array(centers,dtype = 'uint8')
 


i = 1
plt.figure(0,figsize=(8,2))
colors = []

for each_col in centers:
    plt.subplot(1,4,i)
    i+=1

    colors.append(each_col)

    #color swatch

    a = np.zeros((100,100,3), dtype = 'uint8')
    a[:,:,:] = each_col
    plt.imshow(a)

plt.show()




new_img = np.zeros((800*1200,3), dtype = 'uint8')
print(new_img.shape)
km.labels_
for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]

new_img = new_img.reshape((original_shape))
plt.imshow(new_img)
plt.show()