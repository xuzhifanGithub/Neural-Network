#include"net.h"



void Net::init() {
    netDepth = 3;
    net = (FC*)malloc(sizeof(FC) * netDepth);
    actfun.resize(netDepth, "no");
    output = (Tensor2D*)malloc(sizeof(Tensor2D) * (netDepth + 1));
    grid = (Tensor2D*)malloc(sizeof(Tensor2D) * (netDepth + 1));
    have_grid = (bool*)malloc(sizeof(bool));
    *have_grid = false;
    have_output = (bool*)malloc(sizeof(bool));
    *have_output = false;

    int inputsize = 3;
    net[0] = Linear_init(inputsize, inputsize); actfun[0] = "relu";
    net[1] = Linear_init(inputsize, inputsize); actfun[1] = "relu";
    net[2] = Linear_init(inputsize, 1); actfun[2] = "no";

}



void Net::display(bool printWB) {
    printf("\nNet info:\nnet depth:%d.\n", netDepth);
    for (int i = 0; i < netDepth; ++i) {
        printf("net[%d]: in_features:%d , out_features:%d\n", i, net[i].in_features, net[i].out_features);
        if (printWB == true) {
            printf("weigth:\n");
            Tensor2D_display(net[i].W);
            printf("bias:\n");
            Tensor2D_display(net[i].B);
        }

        //printf("\n");
        printf("actfun:%s\n\n", actfun[i].c_str());
    }
}

void Net::save_mode() {
    printf("\nsave model start.[path:model.dat].");

    FILE* modelPtr;
    modelPtr = fopen("model.dat", "w+");
    //打开文件
    if (modelPtr == NULL) {
        perror("Error opening file(save model)");
        exit(0);
    }
    //保存netDepth
    fprintf(modelPtr, "%d ", netDepth);
    //保存网络w，b
    for (int i = 0; i < netDepth; ++i)
    {
        fprintf(modelPtr, "%d ", net[i].in_features);
        fprintf(modelPtr, "%d ", net[i].out_features);
        for (int m = 0; m < net[i].in_features; m++) {
            for (int n = 0; n < net[i].out_features; n++) {
                fprintf(modelPtr, "%f ", net[i].W.data[m][n]);
            }
        }

        for (int n = 0; n < net[i].out_features; n++) {
            fprintf(modelPtr, "%f ", net[i].B.data[0][n]);
        }
        fprintf(modelPtr, "%s ", actfun[i].c_str());
    }
    //关闭文件
    fclose(modelPtr);
    printf("save model successful.\n");
}

void Net::save_mode(std::string path) {
    printf("\nsave model start.[path:%s].",path.c_str());

    FILE* modelPtr;
    modelPtr = fopen(path.c_str(), "w+");
    //打开文件
    if (modelPtr == NULL) {
        perror("Error opening file(save model)");
        exit(0);
    }
    //保存netDepth
    fprintf(modelPtr, "%d ", netDepth);
    //保存网络w，b
    for (int i = 0; i < netDepth; ++i)
    {
        fprintf(modelPtr, "%d ", net[i].in_features);
        fprintf(modelPtr, "%d ", net[i].out_features);
        for (int m = 0; m < net[i].in_features; m++) {
            for (int n = 0; n < net[i].out_features; n++) {
                fprintf(modelPtr, "%f ", net[i].W.data[m][n]);
            }
        }

        for (int n = 0; n < net[i].out_features; n++) {
            fprintf(modelPtr, "%f ", net[i].B.data[0][n]);
        }
        //fputs(actfun[i].c_str(), modelPtr);
        fprintf(modelPtr, "%s ", actfun[i].c_str());
    }
    //关闭文件
    fclose(modelPtr);
    printf("save model successful.\n");
}

void Net::load_mode() {
    printf("\nload model start.[model.dat].");

    FILE* modelPtr;
    modelPtr = fopen("model.dat", "r");
    //打开文件
    if (modelPtr == NULL) {
        perror("Error opening file(load model)");
        exit(0);
    }
    //保存netDepth
    fscanf(modelPtr, "%d ", &netDepth);
    net = (FC*)malloc(sizeof(FC) * netDepth);
    actfun.resize(netDepth, "no");
    output = (Tensor2D*)malloc(sizeof(Tensor2D) * (netDepth + 1));
    grid = (Tensor2D*)malloc(sizeof(Tensor2D) * (netDepth + 1));
    have_grid = (bool*)malloc(sizeof(bool));
    *have_grid = false;
    have_output = (bool*)malloc(sizeof(bool));
    *have_output = false;

    //保存网络w，b
    for (int i = 0; i < netDepth; ++i)
    {
        fscanf(modelPtr, "%d ", &net[i].in_features);
        fscanf(modelPtr, "%d ", &net[i].out_features);
        net[i] = Linear_init(net[i].in_features, net[i].out_features);

        for (int m = 0; m < net[i].in_features; m++) {
            for (int n = 0; n < net[i].out_features; n++) {
                fscanf(modelPtr, "%f ", &net[i].W.data[m][n]);
            }
        }

        for (int n = 0; n < net[i].out_features; n++) {
            fscanf(modelPtr, "%f ", &net[i].B.data[0][n]);
        }
        char* cString = (char*)malloc(sizeof(char) * 15);
        fscanf(modelPtr, "%s ", cString);
        actfun[i] = cString;
    }
    //关闭文件
    fclose(modelPtr);
    printf("load model end.\n");
}

void Net::load_mode(std::string path) {
    printf("\nload model start:[path:%s].",path.c_str());

    FILE* modelPtr;
    modelPtr = fopen(path.c_str(), "r");
    //打开文件
    if (modelPtr == NULL) {
        perror("Error opening file(load model)");
        exit(0);
    }
    //保存netDepth
    fscanf(modelPtr, "%d ", &netDepth);
    net = (FC*)malloc(sizeof(FC) * netDepth);
    actfun.resize(netDepth, "no");
    output = (Tensor2D*)malloc(sizeof(Tensor2D) * (netDepth + 1));
    grid = (Tensor2D*)malloc(sizeof(Tensor2D) * (netDepth + 1));
    have_grid = (bool*)malloc(sizeof(bool));
    *have_grid = false;
    have_output = (bool*)malloc(sizeof(bool));
    *have_output = false;

    //保存网络w，b
    for (int i = 0; i < netDepth; ++i)
    {
        fscanf(modelPtr, "%d ", &net[i].in_features);
        fscanf(modelPtr, "%d ", &net[i].out_features);
        net[i] = Linear_init(net[i].in_features, net[i].out_features);

        for (int m = 0; m < net[i].in_features; m++) {
            for (int n = 0; n < net[i].out_features; n++) {
                fscanf(modelPtr, "%f ", &net[i].W.data[m][n]);
            }
        }

        for (int n = 0; n < net[i].out_features; n++) {
            fscanf(modelPtr, "%f ", &net[i].B.data[0][n]);
        }
        char* cString = (char*)malloc(sizeof(char) * 15);
        fscanf(modelPtr, "%s ", cString);
        actfun[i] = cString;
    }
    //关闭文件
    fclose(modelPtr);
    printf("load model successful.\n");
}


void Net::freeOutput() {
    for (int i = 0; i < netDepth + 1; ++i) {
        Tensor2D_freeTensor(output[i]);
    }
}

void Net::freeGrid() {
    for (int i = 0; i < netDepth + 1; ++i) {
        Tensor2D_freeTensor(grid[i]);
    }
}

Tensor2D Net::forward(Tensor2D input,bool bias) {
    if (*have_output == true) freeOutput();
    else  *have_output = true;

    output[0] = Tensor2D_copy(input);
    for (int i = 1; i < netDepth + 1; ++i) {
        output[i] = Linear_forward(output[i - 1], net[i - 1], bias);
        Linear_actfun_forward(output[i], actfun[i - 1]);
    }
    return output[netDepth];
}

void Net::backward(Tensor2D y_out, Tensor2D y_true , float learning_rate) {
    if (*have_grid == true) freeGrid();
    else *have_grid = true;

    //计算loss梯度
    grid[netDepth] = Linear_loseBack(output[netDepth], y_true);
    if (netDepth - 1 >= 0) {
        Linear_actfun_backward(actfun[netDepth - 1], grid[netDepth], output[netDepth]);
    }

    //计算各个参数梯度
    for (int k = netDepth; k >= 1; k--) {
        grid[k - 1] = Linear_backward(grid[k], net[k - 1]);
        if (k - 2 >= 0) {
            Linear_actfun_backward(actfun[k - 2], grid[k - 1], output[k - 1]);
        }
    }

    //更新梯度
    for (int k = 0; k < netDepth; k++) {
        Linear_update_grid(net[k], grid[k + 1], output[k], learning_rate);
    }
}