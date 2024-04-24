#include"Tensor2D.h"

//伪随机数生成器
std::mt19937& get_generator() {
    thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}


//tensor值置0
void Tensor2D_setZero(Tensor2D tensor) {
    int m = tensor.h;
    int n = tensor.w;
 
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            tensor.data[i][j] = 0.0;
        }
    }
}

//初始化
Tensor2D Tensor2D_init(int h, int w) {
    Tensor2D tensor;
    tensor.h = h;
    tensor.w = w;
    tensor.data = (float**)malloc(sizeof(float*) * h);
    std::mt19937& gen = get_generator();

    // 生成标准正态分布的随机数，mean为均值，stddev为标准差  
    std::normal_distribution<> d(0.0, 1.0); // 标准正态分布，均值为0，标准差为1  

    for (int i = 0; i < h; ++i) {
        tensor.data[i] = (float*)malloc(sizeof(float) * w);
    }

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            tensor.data[i][j] = d(gen);
        }
    }
    return tensor;
}

//初始化
Tensor2D Tensor2D_init(int h, int w, float value) {
    Tensor2D tensor;
    tensor.h = h;
    tensor.w = w;
    tensor.data = (float**)malloc(sizeof(float*) * h);
    for (int i = 0; i < h; ++i) {
        tensor.data[i] = (float*)malloc(sizeof(float) * w);
    }

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            tensor.data[i][j] = value;
        }
    }
    return tensor;
}

//显示tensor信息
void Tensor2D_display(Tensor2D tensor, bool printValue) {
    int m = tensor.h;
    int n = tensor.w;
    printf("\nTensor size:[%d,%d]\n", m, n);

    if (printValue) {
        printf("[");
        for (int i = 0; i < m; ++i) {
            printf("[");
                for (int j = 0; j < n; ++j) {
                    printf("%f", tensor.data[i][j]);
                    if (j != n - 1)printf(",");
                }
                if(i!=m-1)printf("]\n");
        }
        printf("]\n");
    }
    


}

//复制tensor
Tensor2D Tensor2D_copy(Tensor2D tensor) {
    Tensor2D temp = Tensor2D_init(tensor.h, tensor.w);
    for (int i = 0; i < tensor.h; ++i) {
        for (int j = 0; j < tensor.w; ++j) {
            temp.data[i][j] = tensor.data[i][j];
        }
    }
    return temp;
}

//复制tensor
void Tensor2D_copy(Tensor2D des, Tensor2D src) {
    for (int i = 0; i < des.h; ++i) {
        for (int j = 0; j < des.w; ++j) {
            des.data[i][j] = src.data[i][j];
        }
    }
}

//释放tensor内存
void Tensor2D_freeTensor(Tensor2D tensor) {
    for (int i = 0; i < tensor.h; ++i) {
        free(tensor.data[i]);
    }
    free(tensor.data);
}

