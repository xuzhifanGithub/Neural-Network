#pragma once

#include<stdio.h>
#include <iostream>
#include<stdlib.h>
#include<vector>
#include<string>
#include <random>  

struct Tensor2D {
    int h;//高（行）
    int w;//宽（列）
    float** data;
};

std::mt19937& get_generator();//伪随机数生成器
Tensor2D Tensor2D_init(int h, int w);//初始化tensor，值默认正态分布均值0，方差1
Tensor2D Tensor2D_init(int h, int w, float value);//初始化tensor，值均为自定义value
void Tensor2D_display(Tensor2D tensor,bool printValue = true);//打印tensor大小和打印值(默认)
Tensor2D Tensor2D_copy(Tensor2D tensor);//复制
void Tensor2D_copy(Tensor2D des,Tensor2D src);//复制
void Tensor2D_setZero(Tensor2D tensor);//tensor值置0
void Tensor2D_freeTensor(Tensor2D tensor);//释放tensor内存
