import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as clr

def encoder(img):
    
    #3) Color maps RGB
    colorMapEnc(img)
    
    #4) Padding
    padding(img)

    #5) Convert to YCbCr
    ycbcr = convert_ycbcr(img)

    

def decoder(img, shape):
    pass

def colorMapEnc(img):
    
    cmRed = clr.LinearSegmentedColormap.from_list('red', [(0,0,0), (1,0,0)], 256)
    cmGreen = clr.LinearSegmentedColormap.from_list('green', [(0,0,0), (0,1,0)], 256)
    cmBlue = clr.LinearSegmentedColormap.from_list('blue', [(0,0,0), (0,0,1)], 256)

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    

    showImageColormap(R, "R", cmRed)
    showImageColormap(G, "G", cmGreen)
    showImageColormap(B, "B", cmBlue)
    
    
    
    #Decoder
    [nl, nc, ch] = img.shape
    imgRec = np.zeros((nl, nc, ch))
    
    
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    showImageColormap(imgRec.astype(np.uint8), "RGB")
    
def padding(img):

    xp = img
    shape = img.shape
    
    n = max(shape[0], shape[1])

    while n % 32 != 0:
        n += 1

    nl = n - shape[0]
    nc = n - shape[1]
    
    ll = img[shape[0] - 1, :] [np.newaxis, :]
    
    
    repl = ll.repeat(nl, axis = 0) 
    

    xp = np.vstack([img, repl]) 

    cc = xp[:, shape[1] -1] [:, np.newaxis]
    repc = cc.repeat(nc, axis = 1)

    xt = np.hstack([xp, repc])      

    showImageColormap(xt, "Padding")
    
    return xt


def padding_inv(pad, shape):
    nl = pad.shape[0] - shape[0]
    nc = pad.shape[1] - shape[1]
    
    ipad = pad[:-nl,:-nc]
    showImageColormap(ipad, "Reversed padding")

def convert_ycbcr(img):
    mat = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

    y = np.dot(img, mat.T)
    #y = np.zeros_like(img)
    #y = mat[0,0] * img[:,:,0] + mat[0,1] * img[:,:,1] + mat[0,2]  * img[:,:,2]
    
    y[:,:,0] = y[:,:,0] + 0
    y[:,:,1] = y[:,:,1] + 128
    y[:,:,2] = y[:,:,2] + 128
    

    showImageColormap(y.astype(np.uint8), "YCbCr")
    
    showImageColormap(y[:,:,0].astype(np.uint8), "Y", "gray")
    
    showImageColormap(y[:,:,1].astype(np.uint8), "Cb")
    
    showImageColormap(y[:,:,2].astype(np.uint8), "Cr")
    
    return y

def convert_rgb(img):
    mat = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

    matI = np.linalg.inv(mat)
    
    img[:,:,0] = img[:,:,0] - 0
    img[:,:,1] = img[:,:,1] - 128
    img[:,:,2] = img[:,:,2] - 128

    y = np.dot(img, matI.T)
    

    #y[y>255] = 255
    #y[y<0] = 0
    
    showImageColormap(y.astype(np.uint8), "RGB")
 
    
#3.3) 
def showImageColormap(auxColormap1, title = None, auxColormap2 = None):
    plt.figure()
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.imshow(auxColormap1, auxColormap2)

    
def main():
    
    #Ler imagens
    img1 = plt.imread('imagens/barn_mountains.bmp')
    img2 = plt.imread('imagens/logo.bmp')    
    img3 = plt.imread('imagens/peppers.bmp')

    #Escolher imagem
    imgx = img1

    encoder(imgx)
    decoder(imgx, imgx.shape)

if __name__ == '__main__':
    main()
