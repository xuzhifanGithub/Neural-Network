#pragma once

#include<stdio.h>
#include <iostream>
#include<stdlib.h>
#include<vector>
#include<string>
#include <random>  

struct Tensor2D {
    int h;//�ߣ��У�
    int w;//���У�
    float** data;
};

std::mt19937& get_generator();//α�����������
Tensor2D Tensor2D_init(int h, int w);//��ʼ��tensor��ֵĬ����̬�ֲ���ֵ0������1
Tensor2D Tensor2D_init(int h, int w, float value);//��ʼ��tensor��ֵ��Ϊ�Զ���value
void Tensor2D_display(Tensor2D tensor,bool printValue = true);//��ӡtensor��С�ʹ�ӡֵ(Ĭ��)
Tensor2D Tensor2D_copy(Tensor2D tensor);//����
void Tensor2D_copy(Tensor2D des,Tensor2D src);//����
void Tensor2D_setZero(Tensor2D tensor);//tensorֵ��0
void Tensor2D_freeTensor(Tensor2D tensor);//�ͷ�tensor�ڴ�
