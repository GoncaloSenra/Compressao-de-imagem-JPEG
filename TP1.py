import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as clr
import cv2
import math
from scipy.fftpack import dct, idct

# Fator de sub-amostragem
factor = (4, 2, 2)

# Dimensao dos blocos DCT
blocos = 8

# Matriz de conversao YCbCr
mat = np.array([[0.299, 0.587, 0.114], 
                [-0.168736, -0.331264, 0.5], 
                [0.5, -0.418688, -0.081312]])

# Matriz de quantizacao de Y
quant_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

# Matriz de quantizacao de CbCr
quant_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

# Fator de qualidade na quantizaçao
quality = 10

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

    #6) DownSampling
    cb, cr = down_sampling(cb, cr)

    # DCT SEM BLOCOS
    
    y_dct = dct_convert(y)

    plt.figure()
    plt.title("DCT y")
    plt.imshow(np.log(np.abs(y_dct) + 0.0001), "gray")

    cb_dct = dct_convert(cb)

    plt.figure()
    plt.title("DCT cb")
    plt.imshow(np.log(np.abs(cb_dct) + 0.0001), "gray")

    cr_dct = dct_convert(cr)

    plt.figure()
    plt.title("DCT cr")
    plt.imshow(np.log(np.abs(cr_dct) + 0.0001), "gray")
    

    # DCT BLOCOS 8x8
    yb_dct = dct_blocks(blocos, y)

    plt.figure()
    plt.title("DCT y 8x8")
    plt.imshow(np.log(np.abs(yb_dct) + 0.0001), "gray")

    cbb_dct = dct_blocks(blocos, cb)
    
    plt.figure()
    plt.title("DCT cb 8x8")
    plt.imshow(np.log(np.abs(cbb_dct) + 0.0001), "gray")
    
    crb_dct = dct_blocks(blocos, cr)

    plt.figure()
    plt.title("DCT cr 8x8")
    plt.imshow(np.log(np.abs(crb_dct) + 0.0001), "gray")

    # QUANTIZACAO

    # Calcular matriz de quantizaçao consoante o fator de qualidade escolhido
    aux_quant_y, aux_quant_cbcr = calc_quality()

    y_quant = quantization(blocos, yb_dct, 0, aux_quant_y, aux_quant_cbcr)
    plt.figure()
    plt.title("DCT y 8x8 QUANTIZACAO")
    plt.imshow(np.log(np.abs(y_quant) + 0.0001), "gray")

    cb_quant = quantization(blocos, cbb_dct, 1, aux_quant_y, aux_quant_cbcr)
    plt.figure()
    plt.title("DCT cb 8x8 QUANTIZACAO")
    plt.imshow(np.log(np.abs(cb_quant) + 0.0001), "gray")
    
    cr_quant = quantization(blocos, crb_dct, 1, aux_quant_y, aux_quant_cbcr)
    plt.figure()
    plt.title("DCT cr 8x8 QUANTIZACAO")
    plt.imshow(np.log(np.abs(cr_quant) + 0.0001), "gray")

    y_dpcm = DPCM (blocos, y_quant)
    plt.figure()
    plt.title("y DPCM")
    plt.imshow(np.log(np.abs(y_dpcm) + 0.0001), "gray")
    
    cb_dpcm = DPCM (blocos, cb_quant)
    plt.figure()
    plt.title("cb DPCM")
    plt.imshow(np.log(np.abs(cb_dpcm) + 0.0001), "gray")
    
    cr_dpcm = DPCM (blocos, cr_quant)
    plt.figure()
    plt.title("cr DPCM")
    plt.imshow(np.log(np.abs(cr_dpcm) + 0.0001), "gray")

    return y_dpcm, cb_dpcm, cr_dpcm, l, c 

def decoder(y_dpcm, cb_dpcm, cr_dpcm, line, col):
    
    cmRed = clr.LinearSegmentedColormap.from_list('red', [(0,0,0), (1,0,0)], 256)
    cmGreen = clr.LinearSegmentedColormap.from_list('green', [(0,0,0), (0,1,0)], 256)
    cmBlue = clr.LinearSegmentedColormap.from_list('blue', [(0,0,0), (0,0,1)], 256)

    y_quant = invert_DPCM(blocos, y_dpcm)
    cb_quant = invert_DPCM(blocos, cb_dpcm)
    cr_quant = invert_DPCM(blocos, cr_dpcm)

    aux_quant_y = 0
    aux_quant_cbcr = 0

    aux_quant_y, aux_quant_cbcr = calc_quality()

    y = invert_quantization(blocos, y_quant, 0, aux_quant_y, aux_quant_cbcr)
    plt.figure()
    plt.title("INVERT QUANTIZATION y")
    plt.imshow(np.log(np.abs(y) + 0.0001), "gray")
    cb = invert_quantization(blocos, cb_quant, 1, aux_quant_y, aux_quant_cbcr)
    plt.figure()
    plt.title("INVERT QUANTIZATION cb")
    plt.imshow(np.log(np.abs(cb) + 0.0001), "gray")
    cr = invert_quantization(blocos, cr_quant, 1, aux_quant_y, aux_quant_cbcr)
    plt.figure()
    plt.title("INVERT QUANTIZATION cr")
    plt.imshow(np.log(np.abs(cr) + 0.0001), "gray")

    # INVERT DCT SEM BLOCOS
    """
    y = dct_invert(y)
    showImageColormap(y, "INVERT DCT y", "gray")
    cb = dct_invert(cb)
    showImageColormap(cb, "INVERT DCT cb", "gray")
    cr = dct_invert(cr)
    showImageColormap(cr, "INVERT DCT cr", "gray")
    """
    # INVERT DCT BLOCOS 8x8

    y = inverse_dct_blocks(blocos, y)
    showImageColormap(y, "INVERT DCT y", "gray")
    cb = inverse_dct_blocks(blocos, cb)
    showImageColormap(cb, "INVERT DCT cb", "gray")
    cr = inverse_dct_blocks(blocos, cr)
    showImageColormap(cr, "INVERT DCT cr", "gray")

    # UPSAMPLING    
    cb, cr = up_sampling(cb, cr)

    R_dec, G_dec, B_dec = convert_rgb(y, cb, cr)
    
    
    R_upad = padding_inv(R_dec, line, col)
    showImageColormap(R_upad, "R_decode", cmRed)
    
    G_upad = padding_inv(G_dec, line, col)
    showImageColormap(R_upad, "G_decode", cmGreen)
    
    B_upad = padding_inv(B_dec, line, col)
    showImageColormap(R_upad, "B_decode", cmBlue)
    
    
    join_channels(R_upad, G_upad, B_upad, line, col)
    return R_upad, G_upad, B_upad

def DPCM(num, img):
    lin, col = img.shape
    temp = 0

    h = lin / num
    w = col / num

    for i in range(int(h)):
        for j in range(int(w)):
            block = img[i*num:(i+1)*num, j*num:(j+1)*num]

            aux = block[0, 0] - temp
            temp = block[0, 0]
            block[0, 0] = aux
            
    return img

def invert_DPCM(num, img):
    lin, col = img.shape
    temp = 0

    h = lin / num
    w = col / num

    for i in range(int(h)):
        for j in range(int(w)):
            block = img[i*num:(i+1)*num, j*num:(j+1)*num]

            block[0, 0] = block[0, 0] + temp
            temp = block[0, 0]
           
    return img


def calc_quality():
    aux_quality = 0
    aux_quant_y = 0
    aux_quant_cbcr = 0

    if quality >= 50:
        aux_quality = (100 - quality) / 50
    elif quality == 0:
        aux_quality = 50
    elif quality < 50:
        aux_quality = 50 / quality

    if quality == 100:
        aux_quant_y = np.ones_like(quant_y)
        aux_quant_cbcr = np.ones_like(quant_cbcr)
    else:
        aux_quant_y = quant_y * aux_quality
        aux_quant_cbcr = quant_cbcr * aux_quality
    aux_quant_y[aux_quant_y > 255] = 255
    aux_quant_y[aux_quant_y < 1] = 1
    aux_quant_y = np.round(aux_quant_y).astype(np.uint8)
    aux_quant_cbcr[aux_quant_cbcr > 255] = 255
    aux_quant_cbcr[aux_quant_cbcr < 1] = 1
    aux_quant_cbcr = np.round(aux_quant_cbcr).astype(np.uint8)

    return aux_quant_y, aux_quant_cbcr

def quantization(num, img, aux, aux_quant_y, aux_quant_cbcr):
    lin, col = img.shape
    temp = np.zeros_like(img)
    aux_quality = 0
    aux_quant = 0

    h = lin / num
    w = col / num

    for i in range(int(h)):
        for j in range(int(w)):
            block = img[i*num:(i+1)*num, j*num:(j+1)*num]

            if aux == 0:
                block_dct = np.round(block / aux_quant_y)
            else:
                block_dct = np.round(block / aux_quant_cbcr)

            temp[i*num:(i+1)*num, j*num:(j+1)*num] = block_dct

    return temp.astype(int)

def invert_quantization(num, img, aux, aux_quant_y, aux_quant_cbcr):
    lin, col = img.shape
    temp = np.zeros_like(img)
    aux_quality = 0
    aux_quant = 0

    h = lin / num
    w = col / num

    for i in range(int(h)):
        for j in range(int(w)):
            block = img[i*num:(i+1)*num, j*num:(j+1)*num]

            if aux == 0:
                block_dct = block * aux_quant_y
            else:
                block_dct = block * aux_quant_cbcr

            temp[i*num:(i+1)*num, j*num:(j+1)*num] = block_dct

    return temp.astype(float)

def dct_blocks(num, img):
    lin, col = img.shape

    h = lin / num
    w = col / num

    for i in range(int(h)):
        for j in range(int(w)):
            block = img[i*num:(i+1)*num, j*num:(j+1)*num]

            block_dct = dct_convert(block)

            img[i*num:(i+1)*num, j*num:(j+1)*num] = block_dct

    return img

def inverse_dct_blocks(num, img):
    lin, col = img.shape

    h = lin / num
    w = col / num

    for i in range(int(h)):
        for j in range(int(w)):
            block = img[i*num:(i+1)*num, j*num:(j+1)*num]

            block_dct = dct_invert(block)

            img[i*num:(i+1)*num, j*num:(j+1)*num] = block_dct

    return img

def dct_convert(block):

    block_dct = dct(dct(block, norm="ortho").T, norm="ortho").T

    return block_dct

def dct_invert(block):

    block_idct = idct(idct(block, norm="ortho").T, norm="ortho").T
    
    return block_idct

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
  
def exe10(R, G, B, line, col, imgOriginal):
    imgRec = np.zeros((line, col, 3))
    
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    
    #Imagem das diferencas

    
    #MSE
    auxMSE = imgOriginal - imgRec
    auxMSE = auxMSE.astype(np.float64)
    MSE = np.sum((auxMSE)**2) / (line*col)
    print("MSE value -> " + str(MSE))
    
    #RMSE
    RMSE = math.sqrt(MSE)
    print("RMSE value -> " + str(RMSE))
    
    #SNR
    P = np.sum((imgOriginal.astype(np.float64))**2) / (line*col)
    SNR = 10*(math.log10(P / MSE))
    print("SNR value -> " + str(SNR))
    
    #PSNR
    PSNR = 10*(math.log10(((imgOriginal.max()).astype(np.float64)**2)/MSE))
    print("PSNR value -> " + str(PSNR))
    
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
    R_rec, G_rec, B_rec = decoder(y,cb ,cr ,line, col) #alt
    exe10(R_rec, G_rec, B_rec, line, col, imgx) ##alt

if __name__ == '__main__':
    main()
