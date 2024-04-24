#include"Linear.h"



//初始化线性层
FC Linear_init(int in_features,int out_features) {
    FC fc_init;
    fc_init.in_features = in_features;
    fc_init.out_features = out_features;
    fc_init.W = Tensor2D_init(in_features, out_features);
    fc_init.B = Tensor2D_init(1, out_features,0.0); 
    return fc_init;
}

FC Linear_init(int in_features, int out_features, float value) {
    FC fc_init;
    fc_init.in_features = in_features;
    fc_init.out_features = out_features;
    fc_init.W = Tensor2D_init(in_features, out_features,value);
    fc_init.B = Tensor2D_init(1, out_features,value);
    return fc_init;
}


void Linear_display(FC fc,bool bias){
    int m = fc.in_features;
    int n = fc.out_features;
    printf("\nLinear size:[%d,%d]\nweight tensor:\n", m, n);
    printf("[");
    for (int i = 0; i < m; ++i) {
        printf("[");
        for (int j = 0; j < n; ++j) {
            printf("%f", fc.W.data[i][j]);
            if (j != n - 1)printf(",");
        }
        printf("]");
        //if(i != m - 1)printf(",");
        if (i != m - 1)printf("\n");
    }
    printf("]\n");

    if (bias == true) {
        printf("bias:\n");
        printf("[");
        for (int j = 0; j < n; ++j) {
            printf("%f", fc.B.data[0][j]);
            if (j != n - 1)printf(",");
        }
        printf("]");
    }
}

Tensor2D Linear_forward(Tensor2D tensor_in, FC fc_w, bool bias ) {
    Tensor2D tensor = Tensor2D_init(tensor_in.h, fc_w.out_features);
    //Linear_display(fc_out);
    //矩阵乘法
    if (tensor_in.w != fc_w.in_features) {
        printf("\nLinear_forward error:size error!\n");
    }
    for (int i = 0; i < tensor.h; ++i) {
        for (int j = 0; j < tensor.w; ++j) {
            float sum = 0;
            //相同边
            for (int m = 0; m < tensor_in.w; ++m) {
                //printf("(%f,%f)", fc_in.w[i][m], fc_w.w[m][j]);
                sum += tensor_in.data[i][m] * fc_w.W.data[m][j];
            }
            //printf("**%f",sum);
            if (bias == true)tensor.data[i][j] = sum + fc_w.B.data[0][j];
            else tensor.data[i][j] = sum;
        }
    }

    return tensor;
}


FC Linear_copy(FC fc) {
    FC fc_init = Linear_init(fc.in_features, fc.out_features);
    Tensor2D_copy(fc_init.W,fc.W);
    Tensor2D_copy(fc_init.B,fc.B);
    return fc_init;
}

Tensor2D Linear_backward(Tensor2D pre_det, FC fc_w) {
    //Tensor2D cur_det = Tensor2D_init(1, fc_w.in_features,0.0);
    Tensor2D ret_det = Tensor2D_init(1, fc_w.in_features, 0.0);

    for (int i = 0; i < fc_w.in_features; i++) {
        for (int k = 0; k < pre_det.w; ++k) {
            //cur_det.data[0][i] += fc_w.W.data[i][k] * pre_det.data[0][k];
            ret_det.data[0][i] += fc_w.W.data[i][k] * pre_det.data[0][k];
        }
        //cur_det.data[0][i] *= cur_in.data[0][i];
    }

    return ret_det;
}

void Linear_update_grid(FC fc_w, Tensor2D grid, Tensor2D input, float learning_rate ) {
    for (int i = 0; i < fc_w.W.h; ++i) {
        for (int j = 0; j < fc_w.W.w; ++j) {
            fc_w.W.data[i][j] -= learning_rate * (input.data[0][i] * grid.data[0][j]);
        }
    }
    //更新b
    for (int j = 0; j < fc_w.W.w; ++j) {
        fc_w.B.data[0][j] -= learning_rate * (grid.data[0][j]);
    }
}


float Linear_loss(Tensor2D y_out, Tensor2D y_true) {
    float loss = 0;
    for (int i = 0; i < y_out.w; ++i) {
        loss += pow(y_out.data[0][i] - y_true.data[0][i], 2);
    }
    loss /= y_true.w;
    return loss;

}

Tensor2D Linear_loseBack(Tensor2D y_out, Tensor2D y_true) {
    Tensor2D tensor = Tensor2D_init(1, y_out.w,0.0);

    for (int i = 0; i < y_out.w; ++i) {
        tensor.data[0][i] += 2*(y_out.data[0][i] - y_true.data[0][i]) / y_out.w;
    }

    return tensor;
}

float Linear_sigmoid(float a) {
    return 1 / (1 + exp(-a));
}

void Linear_sigmoid(Tensor2D tensor) {
    for (int i = 0; i < tensor.h; ++i) {
        for (int j = 0; j < tensor.w; ++j) {
            tensor.data[i][j] = 1 / (1 + exp(-tensor.data[i][j]));
        }
    }
}

float Linear_sigmoid_back(float a) {
    return a * (1 - a);
}

void Linear_sigmoid_back(Tensor2D grid, Tensor2D tensor_out) {
    for (int i = 0; i < grid.h; ++i) {
        for (int j = 0; j < grid.w; ++j) {
            grid.data[i][j] *= (tensor_out.data[i][j]) * (1 - tensor_out.data[i][j]);
        }
    }
}

float Linear_relu(float a) {
    return a > 0 ? a : 0;
}
void Linear_relu(Tensor2D tensor) {
    for (int i = 0; i < tensor.h; ++i) {
        for (int j = 0; j < tensor.w; ++j) {
            if (tensor.data[i][j] < 0)tensor.data[i][j] = 0;
        }
    }
}

float Linear_relu_back(float a) {
    return a > 0 ? 1 : 0;
}

void Linear_relu_back(Tensor2D grid, Tensor2D tensor_out) {
    for (int i = 0; i < grid.h; ++i) {
        for (int j = 0; j < grid.w; ++j) {
            if (tensor_out.data[i][j] <= 0) {
                grid.data[i][j] *= 0;
            }
        }
    }
}

float Linear_tanh(float a) {
    return (exp(a) - exp(-a)) / (exp(a) + exp(-a));
}

void Linear_tanh(Tensor2D tensor) {
    for (int i = 0; i < tensor.h; ++i) {
        for (int j = 0; j < tensor.w; ++j) {
            tensor.data[i][j] = (exp(tensor.data[i][j]) - exp(-tensor.data[i][j]))
                / (exp(tensor.data[i][j]) + exp(-tensor.data[i][j]));
        }
    }
}

float Linear_tanh_back(float a) {
    return (1 - a * a);
}

void Linear_tanh_back(Tensor2D grid, Tensor2D tensor_out) {
    for (int i = 0; i < grid.h; ++i) {
        for (int j = 0; j < grid.w; ++j) {
            grid.data[i][j] *= (1 - tensor_out.data[i][j] * tensor_out.data[i][j]);
        }
    }
}

void Linear_actfun_forward(Tensor2D input, std::string actfunName)
{
    if (actfunName == "sigmoid") {
        Linear_sigmoid(input);
    }
    else if (actfunName == "relu") {
        Linear_relu(input);
    }
    else if (actfunName == "tanh") {
        Linear_tanh(input);
    }
    else {
        ;
    }

}

void Linear_actfun_backward(std::string actfunName,Tensor2D grid, Tensor2D tensor_out) {
    if (actfunName == "sigmoid") {
        Linear_sigmoid_back(grid, tensor_out);
    }
    else if (actfunName == "relu") {
        Linear_relu_back(grid, tensor_out);
    }
    else if (actfunName == "tanh") {
        Linear_tanh_back(grid, tensor_out);
    }
    else {
        ;
    }
}