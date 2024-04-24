#include"Tensor2D.h"

//α�����������
std::mt19937& get_generator() {
    thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}


//tensorֵ��0
void Tensor2D_setZero(Tensor2D tensor) {
    int m = tensor.h;
    int n = tensor.w;
 
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            tensor.data[i][j] = 0.0;
        }
    }
}

//��ʼ��
Tensor2D Tensor2D_init(int h, int w) {
    Tensor2D tensor;
    tensor.h = h;
    tensor.w = w;
    tensor.data = (float**)malloc(sizeof(float*) * h);
    std::mt19937& gen = get_generator();

    // ���ɱ�׼��̬�ֲ����������meanΪ��ֵ��stddevΪ��׼��  
    std::normal_distribution<> d(0.0, 1.0); // ��׼��̬�ֲ�����ֵΪ0����׼��Ϊ1  

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

//��ʼ��
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

//��ʾtensor��Ϣ
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

//����tensor
Tensor2D Tensor2D_copy(Tensor2D tensor) {
    Tensor2D temp = Tensor2D_init(tensor.h, tensor.w);
    for (int i = 0; i < tensor.h; ++i) {
        for (int j = 0; j < tensor.w; ++j) {
            temp.data[i][j] = tensor.data[i][j];
        }
    }
    return temp;
}

//����tensor
void Tensor2D_copy(Tensor2D des, Tensor2D src) {
    for (int i = 0; i < des.h; ++i) {
        for (int j = 0; j < des.w; ++j) {
            des.data[i][j] = src.data[i][j];
        }
    }
}

//�ͷ�tensor�ڴ�
void Tensor2D_freeTensor(Tensor2D tensor) {
    for (int i = 0; i < tensor.h; ++i) {
        free(tensor.data[i]);
    }
    free(tensor.data);
}

