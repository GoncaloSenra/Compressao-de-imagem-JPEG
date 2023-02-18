import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as clr

"""
#SAMPLING (6)
 4:1:2 Cr media de 4 valores e Cb media de 2 valores
 4:2:0 Ambos os canais sao comprimmidos de igual forma (2:1), mas na Horizontal e na Vertical
 import cv2
 cv2.resize(matriz, fatores de compressao, tipo de interpolacao (linear, cubica, etc))
                       (4:2:2)              (4:2:0)
       300x300         150(H):300(V)        150(H):150(V)
       
no decoder: fazer o contrario se foi reduzido a metade passar para o dobro (operaçao destrutiva)
#DCT (7)
usar dct sobre os canais que queremos
"""


def encoder(img):
    #print(img.shape)
    #3) Color maps RGB
    R, G, B , cmRed, cmGreen, cmBlue = colorMapEnc(img)
    
        
    #4) Padding
    padding_img , l ,c= padding(img)
    padding(R, cmRed)
    padding(G, cmGreen)
    padding(B, cmBlue)
    

    #5) Convert to YCbCr
    ycbcr = convert_ycbcr(padding_img)
    
    
    return ycbcr, l, c 
    

def decoder(img, line, col):
    
    imgRGB  = convert_rgb(img)
    
    padding_inv(imgRGB, line, col)


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

    """
    #Decoder
    [nl, nc, ch] = img.shape
    imgRec = np.zeros((nl, nc, ch))
    
    
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    showImageColormap(imgRec.astype(np.uint8), "RGB")
    """
    return R, G, B, cmRed, cmGreen, cmBlue
    

def padding(img, colormap = None):

    xp = img
    shape = img.shape
    col = shape[1]
    line = shape[0]


    nl = 32 - line % 32
    nc = 32 - col % 32
    
    ll = img[shape[0] - 1, :] [np.newaxis, :]
    
    
    repl = ll.repeat(nl, axis = 0) 
    

    xp = np.vstack([img, repl]) 

    cc = xp[:, shape[1] -1] [:, np.newaxis]
    repc = cc.repeat(nc, axis = 1)

    xt = np.hstack([xp, repc])      

    showImageColormap(xt, "Padding", colormap)
    
    #print(xt.shape)
    
    return xt, line, col



def padding_inv(padding_img, line, col):
    nl = padding_img.shape[0] - line
    nc = padding_img.shape[1] - col
    
    un_padding = padding_img[:-nl,:-nc]
    showImageColormap(un_padding, "Reversed padding")
    #print(ipad.shape)

def convert_ycbcr(img):
    mat = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

    y = np.dot(img, mat.T)
    #y = np.zeros_like(img)
    #y = mat[0,0] * img[:,:,0] + mat[0,1] * img[:,:,1] + mat[0,2]  * img[:,:,2]
    
    #y[:,:,0] = y[:,:,0] + 0
    y[:,:,1] = y[:,:,1] + 128
    y[:,:,2] = y[:,:,2] + 128
    

    #showImageColormap(y, "YCbCr")
    
    showImageColormap(y[:,:,0], "Y", "gray")
    
    showImageColormap(y[:,:,1], "Cb", "gray")
    
    showImageColormap(y[:,:,2], "Cr", "gray")
    
    return y

def convert_rgb(img):
    mat = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

    matI = np.linalg.inv(mat)
    
    img[:,:,0] = img[:,:,0] - 0
    img[:,:,1] = img[:,:,1] - 128
    img[:,:,2] = img[:,:,2] - 128

    rgb_img = np.dot(img, matI.T)
    

    rgb_img[rgb_img>255] = 255
    rgb_img[rgb_img<0] = 0
    
    rgb_img = np.round(rgb_img).astype(np.uint8)
    
    showImageColormap(rgb_img, "RGB")
    
    return rgb_img
 
    
#3.3) 
def showImageColormap(auxColormap1, title = None, auxColormap2 = None):
    plt.figure()
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.imshow(auxColormap1, auxColormap2)
    plt.show()

    
def main():
    
    #Ler imagens
    img1 = plt.imread('imagens/barn_mountains.bmp')
    img2 = plt.imread('imagens/logo.bmp')    
    img3 = plt.imread('imagens/peppers.bmp')

    #Escolher imagem
    imgx = img1

    out, line, col = encoder(imgx)
    decoder(out , line, col)

if __name__ == '__main__':
    main()
