import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as clr


def encoder():
    pass

def decoder():
    pass

def colorMapEnc(img):
    print(img.shape)
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
    
def padding(img):

    xp = img
    shape = img.shape
    
    n = max(shape[0], shape[1])

    while n % 32 != 0:
        n += 1

    nl = n - shape[0]
    nc = n - shape[1]
    
    ll = img[shape[0] - 1, :] [np.newaxis, :]
    
    #print(ll, cc)
    
    repl = ll.repeat(nl, axis = 0) 
    

    xp = np.vstack([img, repl]) 

    cc = xp[:, shape[1] -1] [:, np.newaxis]
    repc = cc.repeat(nc, axis = 1)

    xt = np.hstack([xp, repc])      

    print(xt.shape)
    showImage(xt)
    
    return xt


def padding_inv(pad, shape):
    nl = pad.shape[0] - shape[0]
    nc = pad.shape[1] - shape[1]
    
    ipad = pad[:-nl,:-nc]
    showImage(ipad)
    
    

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
    #showImage(img1)
    #colorMapEnc(img1)
    #showImage(img2)
    #showImage(img3)
    #plt.close('all')
    pad = padding(img1)
    padding_inv(pad, img1.shape)
    

if __name__ == '__main__':
    main()
