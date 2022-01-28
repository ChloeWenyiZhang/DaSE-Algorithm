from PIL import Image
import numpy as np

def getImage(path):
    image = Image.open(path)
    image = img.convert("L")
    # PIL中图像的size是（宽，高）
    width = image.size[0]   # width取size的第一个值
    height = image.size[1]   # height取第二个
    img = image.getdata()
    # 为了避免溢出，这里对数据进行一个缩放，缩小100倍
    img = np.array(img).reshape(height,width)/255
    # 查看原图的话，需要还原数据
    new_img = Image.fromarray(img*255)
    new_img.show()
    return img

def pca(img, k):
    n_samples,n_features = img.shape
    mean = np.array([np.mean(img[:,i]) for i in range(n_features)])
    normal_img = img - mean
    matrix_ = np.dot(np.transpose(normal_img),normal_img)   # 协方差矩阵
    eig_val,eig_vec = np.linalg.eig(matrix_)
    #print(matrix_.shape)
    #print(eig_val)
    eigIndex = np.argsort(eig_val)
    eigVecIndex = eigIndex[:-(k+1):-1]
    feature = eig_vec[:,eigVecIndex]
    new_data = np.dot(normal_img,feature)
    # 将降维后的数据映射回原空间
    compressed_img = np.dot(new_data,np.transpose(feature))+ mean
    # print(compressed_img)
    newImage = Image.fromarray(compressed_img*255)
    newImage.show()
    return compressed_img

def error(img, compressed_img):
    sum1 = 0
    sum2 = 0
    D = img - compressed_img
    for i in range(img.shape[0]):
        sum1 += np.dot(img[i],img[i])
        sum2 += np.dot(D[i], D[i])
    error = sum2/sum1
    print(error)

if __name__ == '__main__':
    path = './Images/beach/beach00.tif'
    img = loadImage(path)
    k = 30
    compressed_img = pca(img,k)
    error(img,compressed_img)