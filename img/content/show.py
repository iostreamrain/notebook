from skimage import data,io 

import matplotlib.pyplot as plt 

plt.figure(num='风格图片',figsize=(8,5)) 

img = io.imread_collection("*.jpg") 

for i in range(len(img)): 

    plt.subplot(3,8,i+1) 

    plt.imshow(img[i]) 
plt.show()
