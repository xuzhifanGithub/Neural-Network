#pragma once
#include"Tensor2D.h"

//线性层
struct FC {
    Tensor2D W;//权重矩阵
    Tensor2D B;//偏置矩阵
    int in_features;//输入特征数量
    int out_features;//输出特征数量
};


FC Linear_init(int in_features, int out_features);//初始化线性层
FC Linear_init(int in_features, int out_features, float value);//初始化线性层为自定义值
void Linear_display(FC fc,bool bias = true);//打印线性层权重矩阵和偏置
Tensor2D Linear_forward(Tensor2D tensor_in, FC fc_w, bool bias = true);//线性层前向传播
FC Linear_copy(FC fc);//复制线性层
Tensor2D Linear_backward(Tensor2D pre_det, FC fc_w);//线性层反向传播
void Linear_update_grid(FC fc_w,Tensor2D grid,Tensor2D input,float learning_rate = 0.01);//更新梯度
float Linear_loss(Tensor2D y_out, Tensor2D y_true); //计算损失SGD
Tensor2D Linear_loseBack(Tensor2D y_out, Tensor2D y_true);//计算loss对输出值梯度

void Linear_actfun_forward(Tensor2D input,std::string actfunName);//激活函数前向传播
void Linear_actfun_backward(std::string actfunName,Tensor2D grid, Tensor2D tensor_out);//激活函数反向传播
float Linear_sigmoid(float a);
void Linear_sigmoid(Tensor2D tensor);
float Linear_sigmoid_back(float a);//sigmoid梯度(用于反向传播)
void Linear_sigmoid_back(Tensor2D grid, Tensor2D tensor_out);//sigmoid梯度(用于反向传播)

float Linear_relu(float a);
void Linear_relu(Tensor2D tensor);
float Linear_relu_back(float a);//relu梯度(用于反向传播)
void Linear_relu_back(Tensor2D grid, Tensor2D tensor_out);//relu梯度(用于反向传播)

float Linear_tanh(float a);
void Linear_tanh(Tensor2D tensor);
float Linear_tanh_back(float a);
void Linear_tanh_back(Tensor2D grid, Tensor2D tensor_out);

