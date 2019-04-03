#ifndef WCNN_H
#define WCNN_H
#include <vector>
#include <iostream>
using namespace std;

typedef   vector<double>  Tensor1f;
typedef   vector<vector<double>>  Tensor2f;
typedef   vector<vector<vector<double>>>  Tensor3f;
typedef   vector<vector<vector<vector<double>>>>  Tensor4f;
//定义WMat结构体
class CDataBlob{
public:
    //公共变量
    int batchsize;

    Tensor2f dense_float;//全连接数据
    //dense格式
    int out_features;
    int in_features;

    Tensor4f data_float;//图像数据
    //宽高
    int height;
    int width;
    //图像数据格式
    int channels;
    //权重格式
    int outc;
    int inc;

public:
    //默认构造函数
    CDataBlob() {
        this->width = 0;
        this->height = 0;
        this->channels = 0;
        this->batchsize=0;
        this->outc=0;
        this->inc=0;
        this->data_float.clear();
    }
    //nn.Line
    CDataBlob(int out ,int in) {
        this->data_float.clear();
        this->batchsize=this->out_features=out;
        this->in_features=in;
        this->dense_float.resize(out);
        for(auto k=0;k<out;k++)
        {
            this->dense_float[k].resize(in);
        }
    }
    //nn.Cov
    CDataBlob(int n,int c,int h,int w)
    {
        this->data_float.clear();
        this->width=w;
        this->height=h;
        this->inc=this->channels=c;
        this->outc=this->batchsize=n;
        this->data_float.resize(n);
        for(auto k=0;k<n;k++)
        {
            this->data_float[k].resize(c);
            for(auto i=0 ;i<c;i++)
            {
                this->data_float[k][i].resize(h);
                for(auto j=0;j<h;j++){
                    this->data_float[k][i][j].resize(w);
                }
            }
        }
    }

bool setDataFromImage( )
{

}


};

//bias参数
typedef vector<double> Cbias;

//卷积参数
typedef struct covFilters {
    int stride_sH;
    int stride_sW;
    int padding_padH;
    int padding_padW;
    int dilation_dH;
    int dilation_dW;
    int groups;
}covFilters;

//池化参数
typedef struct maxFilters {
    int stride_sH;
    int stride_sW;
    int padding_padH;
    int padding_padW;
    int dilation_dH;
    int dilation_dW;
    int kernel_size_kH;
    int kernel_size_kW;
    bool return_indices;
    bool ceil_mode;
}maxFilters;

//定义常见的计算
//已经实现
bool Conv2d(CDataBlob *inputData, CDataBlob *weight,Cbias *bias,covFilters *filters, CDataBlob *outputData);
bool ReLU(CDataBlob *inputData,int typeL,CDataBlob *outputData);
//待实现
bool MaxPool2d(CDataBlob *inputData,maxFilters *filters, CDataBlob *outputData);
bool Linear(CDataBlob *inputData, CDataBlob *weight,Cbias *bias, CDataBlob *outputData);


bool ConvTranspose2d();
bool BatchNorm2d();
bool InstanceNorm2d();
bool UpsamplingNearest2d();
bool UpsamplingBilinear2d();
bool RNN();
bool LSTM();
bool GRU();
bool Embedding();
bool Sigmoid();


#endif // WCNN_H
