#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import integrate
from matplotlib import pyplot as plt
from PIL import Image
from time import time




## vars ##
PLOT = True
plot_type = "no_sep"
#plot_type = "sep"
var = [10, 8/3, 30.0] # sigma, beta, rho
#var = [20, 50, 8] # sigma, beta, rho

#X_initials = [5, -7, 16]
X_initials = [5, -7, 12]




# helper func
# Function for converting decimal to binary
def float_bin(my_number, places = 3):
    my_whole, my_dec = str(my_number).split(".")
    my_whole = int(my_whole)
    res = (str(bin(my_whole))+".").replace('0b','')
 
    for x in range(places):
        my_dec = str('0.')+str(my_dec)
        temp = '%1.20f' %(float(my_dec)*2)
        my_whole, my_dec = temp.split(".")
        res += my_whole
    return res
 
 
 
def IEEE754(n) :
    # identifying whether the number
    # is positive or negative
    sign = 0
    if n < 0 :
        sign = 1
        n = n * (-1)
    p = 30
    # convert float to binary
    dec = float_bin (n, places = p)
 
    dotPlace = dec.find('.')
    onePlace = dec.find('1')
    # finding the mantissa
    if onePlace > dotPlace:
        dec = dec.replace(".","")
        onePlace -= 1
        dotPlace -= 1
    elif onePlace < dotPlace:
        dec = dec.replace(".","")
        dotPlace -= 1
    mantissa = dec[onePlace+1:]
 
    # calculating the exponent(E)
    exponent = dotPlace - onePlace
    exponent_bits = exponent + 127
 
    # converting the exponent from
    # decimal to binary
    exponent_bits = bin(exponent_bits).replace("0b",'')
 
    mantissa = mantissa[0:23]
 
    # the IEEE754 notation in binary    
    final = str(sign) + exponent_bits.zfill(8) + mantissa
 
    # convert the binary to hexadecimal
    hstr = '0x%0*X' %((len(final) + 3) // 4, int(final, 2))
    return final #(hstr, final)



def lorentz_sys(X, t0, sigma=var[0], beta=var[1], rho=var[2]):
    """
    Compute the time-derivative of a Lorentz system 
        sigma * (y - x)
        x * (rho - z) - y
        x * y - beta * z
    
    X = [x, y, z]
    
    """
    return [sigma*(X[1]-X[0]), X[0]*(rho-X[2])-X[1], X[0]*X[1]-beta*X[2]]

########## 

def generate_trajs(X_init=X_initials, t=np.linspace(0,20,300*300), plot=True):

    # Solve for the trajectories of x,y,z into df X_t
    X_t = pd.DataFrame([integrate.odeint(lorentz_sys, X_init, t)][0], columns=["X", "Y", "Z"])
    
    X_t["encr_R"] = pd.Series([ int(IEEE754(i * 10**13)) % 256 for i in X_t["X"].values ])
    X_t["encr_G"] = pd.Series([ int(IEEE754(i * 10**13)) % 256 for i in X_t["Y"].values ])
    X_t["encr_B"] = pd.Series([ int(IEEE754(i * 10**13)) % 256 for i in X_t["Z"].values ])
    #print(X_t.head())
    
    # plot types
    if plot and plot_type == "sep":
            
        plt.figure(1)
        
        # plot x vs t
        plt.subplot(1,3,1)
        plt.plot(t, X_t.X)
        plt.grid()
        plt.suptitle("x vs t")
        plt.title(f"for Lorenz system [sigma, beta, r] = {[ '%.2f' % v for v in var ]}\nand initial [x, y, z] = {['%.1f' % i for i in X_init]}", fontsize=8)
        plt.xlabel("t")
        plt.ylabel("x")
        
        # plot y vs t
        plt.subplot(1,3,2)
        plt.plot(t, X_t.Y)
        plt.grid()
        plt.suptitle("y vs t")
        plt.title(f"for Lorenz system [sigma, beta, r] = {[ '%.2f' % v for v in var ]}\nand initial [x, y, z] = {['%.1f' % i for i in X_init]}", fontsize=8)
        plt.xlabel("t")
        plt.ylabel("y")
        
        # plot x vs z
        plt.subplot(1,3,3)
        plt.plot(t, X_t.Z)
        plt.grid()
        plt.suptitle("z vs t")
        plt.title(f"for Lorenz system [sigma, beta, r] = {[ '%.2f' % v for v in var ]}\nand initial [x, y, z] = {['%.1f' % i for i in X_init]}", fontsize=8)
        plt.xlabel("t")
        plt.ylabel("z")
        
        plt.savefig('imgs/trajs.png', bbox_inches='tight')

    
    elif plot and plot_type == "no_sep":
        plt.figure(1, (10,8))
        plt.plot(t, X_t.X, label="x")
        plt.plot(t, X_t.Y, label="y")
        plt.plot(t, X_t.Z, label="z")
        plt.grid()
        plt.suptitle("x,y,z vs t")
        plt.title(f"for Lorenz system [sigma, beta, r] = {[ '%.2f' % v for v in var ]}\nand initial x,y,z = {['%.1f' % i for i in X_init]}", fontsize=8)
        plt.xlabel("t")
        plt.ylabel("x,y,z")
        plt.legend()
        plt.savefig('imgs/trajs.png', bbox_inches='tight')
    
    
    if plot:
        plt.figure(10)
        ax = plt.axes(projection='3d')
        ax.scatter3D(X_t.encr_R[:500], X_t.encr_G[:500], X_t.encr_B[:500]);
        plt.grid()
        plt.suptitle("Pseudorandom numbers generated from Lorenz system")
        plt.xlabel("R")
        plt.ylabel("G")
        ax.set_zlabel('B')
        plt.savefig('imgs/PRNs.png', bbox_inches='tight')

    return X_t



def read_image(filename="SgtPeppers.png", plot=True):
    img = Image.fromarray(np.asarray(Image.open(filename)))
    
    if plot:
        plt.figure(2, (14,6))
        plt.subplot(1,2,1)
        plt.title(f'Image {filename}')
        plt.imshow(img)
        
        hist = img.histogram()
        plt.subplot(1,2,2)
        plt.title(f'Histogram of {filename}')
        plt.grid()
        for i in range(0,256):
            plt.bar(i, hist[i    ], color='red')
            plt.bar(i, hist[i+256], color='green')
            plt.bar(i, hist[i+512], color='blue')
        
        plt.savefig('imgs/initial.png', bbox_inches='tight')
        
    return img



def convert_pix(rgb_pix, factor):
    #for 
    #new_rgb_pix = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    
    new_rgb_pix = [ p^int(f) for p,f in zip(rgb_pix, factor) ]
    #print(rgb_pix, factor, new_rgb_pix)
    return tuple(new_rgb_pix)



def encrypt_image(df, img):
    np_img = img.load()
    
    for col in range(1,img.height):
        for row in range(1,img.width):
            np_img[row,col] = convert_pix(np_img[row,col], list(df.iloc[row*col][3:]))
        print(f"Encrypting... {row}, {col}")
    
    if PLOT:
        plt.figure(3, (14,6))
        plt.subplot(1,2,1)
        plt.title('Encrypted image')
        plt.imshow(img)
        
        hist = img.histogram()
        plt.subplot(1,2,2)
        plt.title('Histogram of encrypted image')
        plt.grid()
        for i in range(0,256):
            plt.bar(i, hist[i    ], color='red')
            plt.bar(i, hist[i+256], color='green')
            plt.bar(i, hist[i+512], color='blue')
        
        plt.savefig('imgs/encrypt.png', bbox_inches='tight')
        
    return np_img



def decrypt_image(df, encr_img):
    
    for col in range(1,300):
        for row in range(1,300):
            encr_img[row,col] = convert_pix(encr_img[row,col], list(df.iloc[row*col][3:]))
        print(f"Decrypting... {row}, {col}")
    
    if PLOT:
        plt.figure(4, (14,6))
        plt.subplot(1,2,1)
        plt.title('Decrypted image')
        plt.imshow(img)
        
        hist = img.histogram()
        plt.subplot(1,2,2)
        plt.title('Histogram of decrypted image')
        plt.grid()
        for i in range(0,256):
            plt.bar(i, hist[i    ], color='red')
            plt.bar(i, hist[i+256], color='green')
            plt.bar(i, hist[i+512], color='blue')
        
        plt.savefig('imgs/decrypt.png', bbox_inches='tight')
    
    return encr_img



def alter_encrypt_image(encr_img):
    
    for col in range(100,140):
        for row in range(130,190):
            encr_img[row,col] = (0,0,0)
        print(f"Altering encrypted... {row}, {col}")
    
    if PLOT:
        plt.figure(6, (14,6))
        plt.subplot(1,2,1)
        plt.title('Altered encrypted image')
        plt.imshow(img)
        
        hist = img.histogram()
        plt.subplot(1,2,2)
        plt.title('Histogram of altered encrypted image')
        plt.grid()
        for i in range(0,256):
            plt.bar(i, hist[i    ], color='red')
            plt.bar(i, hist[i+256], color='green')
            plt.bar(i, hist[i+512], color='blue')
        
        plt.savefig('imgs/alter_encrypt.png', bbox_inches='tight')
    
    return encr_img



if __name__=="__main__":
    t_start = time()
    key = generate_trajs(plot=PLOT)
    #key.to_csv("test_data.csv", index=False)
    #key = pd.read_csv("test_data.csv")
    t1 = time()
    img = read_image(plot=PLOT)
    t2 = time()
    encr_img = encrypt_image(key, img)
    t3 = time()
    #alt_encr_img = alter_encrypt_image(encr_img)
    #decr_img = decrypt_image(key, alt_encr_img)
    decr_img = decrypt_image(key, encr_img)
    t4 = time()
    
    print(f"""
          Load Lorenz trajectories: {round(t1-t_start,3)}s
          Read image: {round(t2-t1,3)}s
          Encrypt image: {round(t3-t2,3)}s
          Decrypt image: {round(t4-t3,3)}s
          ---------------------------
          TOTAL ELAPSED TIME: {round(t4-t_start,3)}s
          """)
    
    