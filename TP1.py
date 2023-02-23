import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as clr
import cv2
from scipy.fftpack import dct, idct


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

factor = (4, 2, 1)

def encoder(img):
    
    #3) Color maps RGB
    R, G, B , cmRed, cmGreen, cmBlue = colorMapEnc(img)
        
    #4) Padding
    #padding_img , l , c= padding(img)
    R_padding, l, c = padding(R, cmRed)
    G_padding, l, c = padding(G, cmGreen)
    B_padding, l, c = padding(B, cmBlue)
    

    #5) Convert to YCbCr
    y, cb, cr = convert_ycbcr(R_padding, G_padding, B_padding)
    cb, cr = down_sampling(cb, cr)

    y_dct, cb_dct, cr_dct = dct_convert(y, cb, cr)

    return y_dct, cb_dct, cr_dct, l, c 
    

def decoder(y, cb, cr, line, col):
    
    cmRed = clr.LinearSegmentedColormap.from_list('red', [(0,0,0), (1,0,0)], 256)
    cmGreen = clr.LinearSegmentedColormap.from_list('green', [(0,0,0), (0,1,0)], 256)
    cmBlue = clr.LinearSegmentedColormap.from_list('blue', [(0,0,0), (0,0,1)], 256)
    
    y, cb, cr = dct_invert(y, cb, cr)

    cb, cr = up_sampling(cb, cr)

    R_dec, G_dec, B_dec = convert_rgb(y, cb, cr)
    
    
    R_upad = padding_inv(R_dec, line, col)
    showImageColormap(R_upad, "R_decode", cmRed)
    
    G_upad = padding_inv(G_dec, line, col)
    showImageColormap(R_upad, "G_decode", cmGreen)
    
    B_upad = padding_inv(B_dec, line, col)
    showImageColormap(R_upad, "B_decode", cmBlue)
    
    
    join_channels(R_upad, G_upad, B_upad, line, col)
    

def dct_convert(y_d, cb_d, cr_d):

    y_dct = dct(dct(y_d, norm="ortho").T, norm="ortho").T
    cb_dct = dct(dct(cb_d, norm="ortho").T, norm="ortho").T
    cr_dct = dct(dct(cr_d, norm="ortho").T, norm="ortho").T

    plt.figure()
    plt.title("DCT y")
    plt.imshow(np.log(np.abs(y_dct) + 0.0001))
    plt.figure()
    plt.title("DCT cb")
    plt.imshow(np.log(np.abs(cb_dct) + 0.0001))
    plt.figure()
    plt.title("DCT cr")
    plt.imshow(np.log(np.abs(cr_dct) + 0.0001))
    
    #showImageColormap(y_dct, "DCT Y", "gray")
    #showImageColormap(cb_dct, "DCT CB", "gray")
    #showImageColormap(cr_dct, "DCT CR", "gray")

    return y_dct, cb_dct, cr_dct

def dct_invert(y_dct, cb_dct, cr_dct):

    y_idct = idct(idct(y_dct, norm="ortho").T, norm="ortho").T
    cb_idct = idct(idct(cb_dct, norm="ortho").T, norm="ortho").T
    cr_idct = idct(idct(cr_dct, norm="ortho").T, norm="ortho").T

    showImageColormap(y_idct, "INVERT DCT y", "gray")
    showImageColormap(cb_idct, "INVERT DCT cb", "gray")
    showImageColormap(cr_idct, "INVERT DCT cr", "gray")
    
    return y_idct, cb_idct, cr_idct

def down_sampling(cb, cr):
    height, width = cr.shape[:2]
    xcb = 0
    xcr = 0
    aux = 0
    cb_d = 0
    cr_d = 0

    if factor[2] != 0:
        aux = 1
        xcb = factor[0] / factor[2]
        xcr = factor[0] / factor[1]
    else:
        xcb = factor[0] / factor[1]
        xcr = factor[0] / factor[1]

    # Specify the new dimensions for the downscaled image
    
    if aux == 1:
        width_cb = int(width / xcb)
        width_cr = int(width / xcr)

        cb_d = cv2.resize(cb, (width_cb, height), interpolation=cv2.INTER_AREA)
        cr_d = cv2.resize(cr, (width_cr, height), interpolation=cv2.INTER_AREA)
    
    else:
        height_cb = int(height / xcb)
        width_cb = int(width / xcb)
        
        height_cr = int(height / xcr)
        width_cr = int(width / xcr)

        cb_d = cv2.resize(cb, (width_cb, height_cb), interpolation=cv2.INTER_AREA)
        cr_d = cv2.resize(cr, (width_cr, height_cr), interpolation=cv2.INTER_AREA)
    
    

    # Use cv2.resize() to downsample the image
    
    showImageColormap(cb_d, "down_cb", "gray")
    showImageColormap(cr_d, "down_cr", "gray")

    return cb_d, cr_d

    
def up_sampling(cb_d , cr_d):

    hr, wr = cr_d.shape[:2]
    hb, wb = cb_d.shape[:2]
    xcb = 0
    xcr = 0
    aux = 0
    cb = 0
    cr = 0

    if factor[2] != 0:
        aux = 1
        xcb = factor[0] / factor[2]
        xcr = factor[0] / factor[1]
    else:
        xcb = factor[0] / factor[1]
        xcr = factor[0] / factor[1]

    
    if aux == 1:
        width_cb = int(wb * xcb)
        width_cr = int(wr * xcr)

        cb = cv2.resize(cb_d, (width_cb, hb), interpolation=cv2.INTER_AREA)
        cr = cv2.resize(cr_d, (width_cr, hr), interpolation=cv2.INTER_AREA)
    
    else:
        height_cb = int(hb * xcb)
        width_cb = int(wb * xcb)
        
        height_cr = int(hr * xcr)
        width_cr = int(wr * xcr)

        cb = cv2.resize(cb_d, (width_cb, height_cb), interpolation=cv2.INTER_AREA)
        cr = cv2.resize(cr_d, (width_cr, height_cr), interpolation=cv2.INTER_AREA)
    

    showImageColormap(cb, "up_cb", "gray")
    showImageColormap(cr, "up_cr", "gray")

    return cb, cr

def join_channels(R, G, B, line, col):
    
    
    imgRec = np.zeros((line, col, 3))
    
    
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    
    
    showImageColormap(imgRec.astype(np.uint8), "Decoded image")
    

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
    
    
    return xt, line, col



def padding_inv(padding_img, line, col):
    nl = (padding_img.shape[0] - line)
    nc = (padding_img.shape[1] - col)

    un_padding = padding_img[:-nl,:-nc]
    
    
    #showImageColormap(un_padding, "Reversed padding")
    
    return un_padding
    #print(ipad.shape)



def convert_ycbcr(R, G, B):
    mat = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

    #y = np.dot(img, mat.T)
    
    y = mat[0,0] * R + mat[0,1] * G + mat[0,2] * B
    cb = (mat[1,0] * R + mat[1,1] * G + mat[1,2] * B) + 128 
    cr = (mat[2,0] * R + mat[2,1] * G + mat[2,2] * B) + 128
    #y = np.zeros_like(img)
    #y = mat[0,0] * img[:,:,0] + mat[0,1] * img[:,:,1] + mat[0,2]  * img[:,:,2]
    
    #y[:,:,0] = y[:,:,0] + 0
    #y[:,:,1] = y[:,:,1] + 128
    #y[:,:,2] = y[:,:,2] + 128
    

    #showImageColormap(y, "YCbCr")
    
    showImageColormap(y, "Y", "gray")
    
    showImageColormap(cb, "Cb", "gray")
    
    showImageColormap(cr, "Cr", "gray")
    
    return y, cb, cr

def convert_rgb(y, cb, cr):
    mat = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])

    matI = np.linalg.inv(mat)
    
    R = matI[0,0] * y + matI[0,1] *(cb -128) + matI[0,2] * (cr -128)
    G = matI[1,0] * y + matI[1,1] *(cb -128) + matI[1,2] * (cr -128)
    B = matI[2,0] * y + matI[2,1] *(cb -128) + matI[2,2] * (cr -128)
    #rgb_img = np.dot(img, matI.T)
    #rgb_img = matI[0,0] * y + matI[0,1] * (cb -128) + matI[0,2] * (cr -128) 

   
    
    B[B>255] = 255
    B[B<0] = 0
    
    R[R>255] = 255
    R[R<0] = 0
    
    G[G>255] = 255
    G[G<0] = 0
    
    R = np.round(R).astype(np.uint8)
    G = np.round(G).astype(np.uint8)
    B = np.round(B).astype(np.uint8)
    
    #showImageColormap(R, "R_conver_rgb")
    #showImageColormap(G, "G_conver_rgb")
    #showImageColormap(B, "B_conver_rgb")
    
    return R, G, B
 
    
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

    y, cb, cr, line, col = encoder(imgx)
    decoder(y,cb ,cr ,line, col)

if __name__ == '__main__':
    main()
