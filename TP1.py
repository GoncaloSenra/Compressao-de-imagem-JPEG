import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as clr


def encoder():
    pass

def decoder():
    pass

def colorMapEnc(img):
    
    #Encoder
    cmRed = clr.LinearSegmentedColormap.from_list('red', [(0,0,0), (1,0,0)], 256)
    cmGreen = clr.LinearSegmentedColormap.from_list('green', [(0,0,0), (0,1,0)], 256)
    cmBlue = clr.LinearSegmentedColormap.from_list('blue', [(0,0,0), (0,0,1)], 256)

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    showImageColormap(R, cmRed)
    showImageColormap(G, cmGreen)
    showImageColormap(B, cmBlue)
    
    
    
    #Decoder
    [nl, nc, ch] = img.shape
    imgRec = np.zeros((nl, nc, ch))
    
    
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    showImage(imgRec.astype(np.uint8))
    
    

    
def showImageColormap(auxColormap1,auxColormap2):
    plt.figure()
    plt.axis("off")
    plt.imshow(auxColormap1, auxColormap2)
    
    
def showImage(img):
    plt.figure()
    plt.axis("off")
    plt.imshow(img, cmap = "gray")
    #print(img.shape)
    
    
def main():
    
    #Ler imagens
    img1 = plt.imread('imagens/barn_mountains.bmp')
    img2 = plt.imread('imagens/logo.bmp')    
    img3 = plt.imread('imagens/peppers.bmp')

    #Mostrar imagens
    showImage(img1)
    colorMapEnc(img1)
    #showImage(img2)
    #showImage(img3)
   #plt.close('all')
    
    

if __name__ == '__main__':
    main()
