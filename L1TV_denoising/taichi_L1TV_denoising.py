import taichi as ti
import cv2
import numpy as np



@ti.data_oriented
class Img_L1TV:
    def __init__(self,img,lamda):
        [m,n] = img.shape;
        
        self.size = ti.Vector([m,n])
        self.lamda = lamda
        self.img_field_o = ti.field(ti.f32, shape = (m,n))
        self.img_field_s = ti.field(ti.f32, shape = (m,n))
        
        self.dx = ti.field(ti.f32, shape = (m,n));
        self.dy = ti.field(ti.f32, shape = (m,n));
        
        self.dxT = ti.field(ti.f32, shape = (m,n));
        self.dyT = ti.field(ti.f32, shape = (m,n));
        
        self.grad = ti.field(ti.f32, shape = (m,n));
        self.mom1 = ti.field(ti.f32, shape = (m,n));
        self.mom2 = ti.field(ti.f32, shape = (m,n));

        # init
        self.img_field_o.from_numpy(img)
        self.img_field_s.from_numpy(img)
    
        
    @ti.func
    def sign(self,v):
        c = 0;
        if v > 0:
            c = 1;
        if v < 0:
            c = -1

        return c;   

    @ti.kernel # 计算图像x方向导数
    def compute_dx(self):
        m = self.size[0];
        n = self.size[1];
        for y in range(m-1):
            for x in range(n-1):
                self.dx[y,x] = self.img_field_o[y,x] - self.img_field_o[y,x+1];
   
    @ti.kernel # 计算图像y方向导数
    def compute_dy(self):
        m = self.size[0];
        n = self.size[1];
        for y in range(m-1):
            for x in range(n-1):
                self.dy[y,x] = self.img_field_o[y,x] - self.img_field_o[y+1,x];        
    

    
    @ti.kernel # TV 正则化           
    def get_TV_term(self):
        m = self.size[0];
        n = self.size[1];
        for y in range(m-1):
            for x in range(n-1):
                self.dx[y,x] = self.sign(self.dx[y,x]);
                self.dy[y,x] = self.sign(self.dy[y,x]);
    
    @ti.kernel           
    def compute_dyT(self):
        m = self.size[0];
        n = self.size[1];
        for x in range(n-1):
            y = m;
            while y > 1:
                y = y - 1;
                self.dyT[y,x] = self.dy[y,x] - self.dy[y - 1,x]; 
                
    @ti.kernel
    def compute_dxT(self):
        m = self.size[0];
        n = self.size[1];
        for y in range(m-1):
            x = n;
            while x > 1:
                x = x - 1;
                self.dxT[y,x] = self.dx[y,x] - self.dx[y,x-1];
    
    @ti.kernel
    def updata_o(self):
        m = self.size[0];
        n = self.size[1];
        for y in range(m-1):
            for x in range(n-1):  
                self.grad[y,x] = (self.sign(self.img_field_o[y,x] - self.img_field_s[y,x]) 
                                  + self.lamda * (self.dxT[y,x] + self.dyT[y,x]));                
                self.mom1[y,x] = 0.9 *  self.mom1[y,x] + (1 - 0.9) *  self.grad[y,x];
                self.mom2[y,x] = 0.99 * self.mom2[y,x] + (1 - 0.99) * self.grad[y,x] ** 2;                
                self.img_field_o[y,x] = (self.img_field_o[y,x] - 
                                         0.01/(self.mom2[y,x] ** 0.5 + 0.0001) 
                                         * (0.9 * self.mom1[y,x] + (1 - 0.9) *  self.grad[y,x]));
    
    def update(self):
        self.compute_dx();
        self.compute_dy();
        self.get_TV_term();
        self.compute_dxT();
        self.compute_dyT();
        
        self.updata_o();
        
                
                
                
    def display(self,gui):
        gui.set_image(self.img_field_o);
              

def salt_pepper_noise(img,snr):
    h=img.shape[0]
    w=img.shape[1]
    img1=img.copy()
    sp=h*w   # 计算图像像素点个数
    NP=int(sp*(1-snr))   # 计算图像椒盐噪声点个数
    for i in range (NP):
        randx=np.random.randint(1,h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy=np.random.randint(1,w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random()<=0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx,randy]=0
        else:
            img1[randx,randy]=1
    return img1
        
import time
if __name__ == "__main__":
    ti.init(arch = ti.cpu);
    img = np.array(cv2.imread("Lena512.png"),'float');
    img = img[:,:,1]/255;
    img = np.rot90(img,-1);
    new_noise = salt_pepper_noise(img,0.3);
    [m,n] = img.shape;  
    
    img_L1TV = Img_L1TV(new_noise,0.8);
    
    max_step = 10000;
    gui = ti.GUI("Implicit Mass Spring System", res=(m, n))
    while gui.running:
        img_L1TV.update();
        img_L1TV.display(gui);
        gui.show();
    


                    
