import taichi as ti

@ti.data_oriented
class ti_image():
    def __init__(self,img,win):
        self.x,self.y,self.c = img.shape
        
        self.img          = ti.Vector.field(3, ti.f32, shape = (self.x,self.y))
        self.dehazed_img  = ti.Vector.field(3, ti.f32, shape = (self.x,self.y))
        self.dark_channel = ti.field(ti.f32, shape = (self.x,self.y))
        self.g_dark       = ti.field(ti.f32, shape = (self.x,self.y))
        self.trans        = ti.field(ti.f32, shape = (self.x,self.y))

        # self.win = win
        self.pad_image = ti.Vector.field(3, ti.f32, shape = (self.x + win,self.y + win))

        self.pad_gray = ti.field(ti.f32, shape = (self.x + win,self.y + win))
        self.pad_dark = ti.field(ti.f32, shape = (self.x + win,self.y + win))
        self.temp_var = ti.Vector.field(4,ti.f32, shape = (self.x + win,self.y + win))
        
        self.draw_img = ti.Vector.field(3, ti.f32, shape = (self.x,self.y))
        
        self.img.from_numpy(img)

    @ti.kernel
    def get_darkchannel(self,win:ti.f32):
        # pad image
        for x,y in ti.ndrange(self.x + win,self.y + win):
            if (x < self.x) and (y < self.y):
                self.pad_image[x,y][0] = self.img[x,y][0]
                self.pad_image[x,y][1] = self.img[x,y][1]
                self.pad_image[x,y][2] = self.img[x,y][2]
            else:
                self.pad_image[x,y][0] = 100
                self.pad_image[x,y][1] = 100
                self.pad_image[x,y][2] = 100
            
            
        # dark channel 
        for x in ti.ndrange(self.x):
            for y in ti.ndrange(self.y): 
                local_min = 100.0
                for px,py in ti.ndrange(win,win):                      
                        this_mean = min(self.pad_image[x + px,y + py][0],
                                        self.pad_image[x + px,y + py][1],
                                        self.pad_image[x + px,y + py][2])
                        if local_min > this_mean:
                            local_min = this_mean    
                self.dark_channel[x,y] = local_min
       
                
            
    @ti.kernel    
    def guided_filter_darkchannel(self): 
        # filtering dark channel using guided filter
        # guide image is raw image in gray-scale
        win = 3.0
        for x,y in ti.ndrange(self.x,self.y):
            if (x < self.x) and (y < self.y):
                self.pad_gray[x,y] = (self.img[x,y][0] + self.img[x,y][1] + self.img[x,y][2]) / 3
                self.pad_dark[x,y] = self.dark_channel[x,y]
            else:
                self.pad_gray[x,y] = 1;
                self.pad_dark[x,y] = 1;
            
        # calcu meang
        for x,y in ti.ndrange(self.x,self.y):
            local_avg = ti.Vector([0.0,0.0,0.0,0.0])
            for px,py in ti.ndrange(win,win):
                local_avg[0] += self.pad_gray[x + px,y + py]
                local_avg[1] += self.pad_dark[x + px,y + py]
                local_avg[2] += (self.pad_gray[x + px,y + py] * 
                                 self.pad_gray[x + px,y + py])
                local_avg[3] += (self.pad_gray[x + px,y + py] * 
                                 self.pad_dark[x + px,y + py])      
            self.temp_var[x,y][0] = local_avg[0]/(win ** 2) 
            self.temp_var[x,y][1] = local_avg[1]/(win ** 2) 
            self.temp_var[x,y][2] = local_avg[2]/(win ** 2) 
            self.temp_var[x,y][3] = local_avg[3]/(win ** 2) 
             
        # calcu var_g  
        for x,y in ti.ndrange(self.x,self.y):
            v1 = self.temp_var[x,y][2] - self.temp_var[x,y][0] * self.temp_var[x,y][0] 
            v2 = self.temp_var[x,y][3] - self.temp_var[x,y][0] * self.temp_var[x,y][1] 
            self.temp_var[x,y][2] = v2/(v1 + 0.0001) 
            self.temp_var[x,y][3] = (self.temp_var[x,y][1] - 
                                     self.temp_var[x,y][2] * 
                                     self.temp_var[x,y][0]) 
            
        for x,y in ti.ndrange(self.x,self.y):
            local_avg = ti.Vector([0.0,0.0])
            for px,py in ti.ndrange(win,win):
                local_avg[0] += self.temp_var[x + px,y + py][2]
                local_avg[1] += self.temp_var[x + px,y + py][3]
            self.temp_var[x,y][0] = local_avg[0]/(win ** 2) 
            self.temp_var[x,y][1] = local_avg[1]/(win ** 2) 

        
        for x,y in ti.ndrange(self.x,self.y):
            self.g_dark[x,y] = self.temp_var[x,y][0] * self.pad_gray[x,y] + self.temp_var[x,y][1]
    
        
    @ti.kernel
    def get_transmission(self,a:ti.f32,val:ti.f32):
        for x,y in ti.ndrange(self.x,self.y):
            self.trans[x,y] = val * (1.0 - a * self.g_dark[x,y])
         
    @ti.kernel
    def draw_target(self,pos_x:ti.i32,pos_y:ti.i32):
        for x,y in ti.ndrange(self.x,self.y):
            self.draw_img[x,y][0] = self.img[x,y][0]
            self.draw_img[x,y][1] = self.img[x,y][1]
            self.draw_img[x,y][2] = self.img[x,y][2]
            
        for con in range(self.x):
            self.draw_img[con,pos_y][0] = 1.0
            self.draw_img[con,pos_y][1] = 1.0
            self.draw_img[con,pos_y][2] = 1.0
            
        for con in range(self.y):
            self.draw_img[pos_x,con][0] = 1.0
            self.draw_img[pos_x,con][1] = 1.0
            self.draw_img[pos_x,con][2] = 1.0
    
    @ti.kernel
    def draw_dehazed(self,r:ti.f32,g:ti.f32,b:ti.f32):    
        for x,y in ti.ndrange(self.x,self.y):
            self.dehazed_img[x,y][0] = ((self.img[x,y][0] - r)/(self.trans[x,y] + 0.00001) + r);
            self.dehazed_img[x,y][1] = ((self.img[x,y][1] - g)/(self.trans[x,y] + 0.00001) + g);
            self.dehazed_img[x,y][2] = ((self.img[x,y][2] - b)/(self.trans[x,y] + 0.00001) + b);         

        
    def gui_show_raw_image(self,gui,pos_x,pos_y):
        self.draw_target(pos_x,pos_y)
        gui.set_image(self.draw_img)
        
    def gui_show_dehazed(self,gui,r,g,b,a,re_dehaze):
        if re_dehaze:
            self.get_transmission(a,(r+g+b)/3)
            self.draw_dehazed(r,g,b)
        gui.set_image(self.dehazed_img)    
    
        # print(self.fog_color[0],self.fog_color[1],self.fog_color[2])
        
        











         