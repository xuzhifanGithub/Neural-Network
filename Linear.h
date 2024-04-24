#pragma once
#include"Tensor2D.h"

//���Բ�
struct FC {
    Tensor2D W;//Ȩ�ؾ���
    Tensor2D B;//ƫ�þ���
    int in_features;//������������
    int out_features;//�����������
};


FC Linear_init(int in_features, int out_features);//��ʼ�����Բ�
FC Linear_init(int in_features, int out_features, float value);//��ʼ�����Բ�Ϊ�Զ���ֵ
void Linear_display(FC fc,bool bias = true);//��ӡ���Բ�Ȩ�ؾ����ƫ��
Tensor2D Linear_forward(Tensor2D tensor_in, FC fc_w, bool bias = true);//���Բ�ǰ�򴫲�
FC Linear_copy(FC fc);//�������Բ�
Tensor2D Linear_backward(Tensor2D pre_det, FC fc_w);//���Բ㷴�򴫲�
void Linear_update_grid(FC fc_w,Tensor2D grid,Tensor2D input,float learning_rate = 0.01);//�����ݶ�
float Linear_loss(Tensor2D y_out, Tensor2D y_true); //������ʧSGD
Tensor2D Linear_loseBack(Tensor2D y_out, Tensor2D y_true);//����loss�����ֵ�ݶ�

void Linear_actfun_forward(Tensor2D input,std::string actfunName);//�����ǰ�򴫲�
void Linear_actfun_backward(std::string actfunName,Tensor2D grid, Tensor2D tensor_out);//��������򴫲�
float Linear_sigmoid(float a);
void Linear_sigmoid(Tensor2D tensor);
float Linear_sigmoid_back(float a);//sigmoid�ݶ�(���ڷ��򴫲�)
void Linear_sigmoid_back(Tensor2D grid, Tensor2D tensor_out);//sigmoid�ݶ�(���ڷ��򴫲�)

float Linear_relu(float a);
void Linear_relu(Tensor2D tensor);
float Linear_relu_back(float a);//relu�ݶ�(���ڷ��򴫲�)
void Linear_relu_back(Tensor2D grid, Tensor2D tensor_out);//relu�ݶ�(���ڷ��򴫲�)

float Linear_tanh(float a);
void Linear_tanh(Tensor2D tensor);
float Linear_tanh_back(float a);
void Linear_tanh_back(Tensor2D grid, Tensor2D tensor_out);

