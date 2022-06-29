# 使用TaiChi实现的 L1-TV，针对椒盐噪声（脉冲噪声）的图像降噪

### L1-TV 求解以下优化问题： <br> $$ \{\bf{o}} = \arg \min {\left\|| {{\bf{o}} - {\bf{s}}} \right\||_1} + {\left\|| {\nabla {\bf{o}}} \right\||_1}\ $$ <br> 其中 o 为待求降噪图像，s 为已知原始图像。该损失函数对 **o** 的导数为: <br>$$\frac{{\partial E}}{{\partial {\bf{o}}}} = {\rm{sign}}\left( {{\bf{o}} - {\bf{s}}} \right) + {\nabla ^T}\left[ {{\rm{sign}}\left( {\nabla {\bf{o}}} \right)} \right]$$  本程序使用 Nesterov-accelerated Adaptive Moment Estimation (NAdam) 梯度下降法解决该优化问题。![]()



