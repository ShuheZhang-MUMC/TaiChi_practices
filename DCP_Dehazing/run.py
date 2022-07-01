import taichi as ti
import cv2

import numpy as np
import tkinter as tk
from tkinter import filedialog

import class_image as CI

ti.init(arch = ti.cpu);
# read image
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()


img = np.array(cv2.imread(file_path))/255
img = np.flipud(img)
img = img.transpose(1,0,2);
[m,n,c] = img.shape
img = img[:,:,[2,1,0]]


# win_size = 2.0;
fog_img = CI.ti_image(img,5)
fog_img.get_darkchannel(2);
fog_img.guided_filter_darkchannel();



gui = ti.GUI('Window Title', res = (m, n))


is_selecting = False
is_slide_able = False
is_other_gui = False

selecting = gui.button('selecting')

while gui.running:
    
    for e in gui.get_events(gui.PRESS):
        if e.key == selecting:
            is_selecting = True
            
    
    if is_selecting:
        gui.get_event()        
        if (ti.GUI.MOVE):
            x0 = m*gui.get_cursor_pos()[0]
            y0 = n*gui.get_cursor_pos()[1]
            fog_img.gui_show_raw_image(gui,int(x0),int(y0))
        else:
            gui.set_image(fog_img.img)
        
        if (gui.is_pressed(ti.GUI.LMB)):
            dx = True
        else:
            dx = False
            push = False
        
    
        if dx and (not push):
            push = True
            x0 = int(m*gui.get_cursor_pos()[0])
            y0 = int(n*gui.get_cursor_pos()[1])
            r = fog_img.img[x0,y0][0]
            g = fog_img.img[x0,y0][1]
            b = fog_img.img[x0,y0][2]
        
            print("global color are : ",
                  int(r*100)/100,int(g*100)/100,int(b*100)/100)
            
            if not is_other_gui:
                gui2 = ti.GUI("Dehazed image", res=(m, n))
                is_other_gui = True
            fog_img.gui_show_dehazed(gui2,r,g,b,0.95,re_dehaze = True)
            gui2.show()
            if not is_slide_able:
                radius = gui.slider('Windows', 1, 25, step=1)
                ratios = gui.slider('Ratio', 0, 100, step=1)
                radius_old = radius.value;
                ratios_old = ratios.value;
                is_slide_able = True
            is_selecting = False 
            
    else:
        gui.set_image(fog_img.img)
    
    if is_slide_able:
        if radius_old != radius.value:
            fog_img.get_darkchannel(radius.value);
            fog_img.guided_filter_darkchannel();
            fog_img.gui_show_dehazed(gui2,r,g,b,ratios.value/100,re_dehaze = True)
            radius_old = radius.value
    
        if ratios_old != ratios.value:
            fog_img.gui_show_dehazed(gui2,r,g,b,ratios.value/100,re_dehaze = True)
            ratios_old = ratios.value
    
    
    if is_other_gui and gui2.running:
        fog_img.gui_show_dehazed(gui2,0,0,0,0.95,re_dehaze = False)    
        gui2.show()
    else:
        is_other_gui = False
    gui.show()