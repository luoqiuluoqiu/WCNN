#include "wcnn.h"
#include<QDebug>
bool convolution(CDataBlob *inputData, CDataBlob *weight, covFilters *filters, CDataBlob *outputData)
{
    //hout=(h+2*p-(d*(k-1)+1))/s+1
    //原始数据进行扩边
    CDataBlob input(inputData->batchsize,inputData->channels,inputData->height+filters->padding_padH*2,inputData->width+filters->padding_padW*2);
    for(auto h=0;h<inputData->batchsize;h++)
        for(auto i=0;i<inputData->channels;i++)
            for(auto j=0;j<inputData->height;j++)
                for(auto k=0;k<inputData->width;k++)
                    input.data_float[h][i][j+filters->padding_padH][k+filters->padding_padW]=inputData->data_float[h][i][j][k];

    //扩充卷积权重 dilation参数  out,in/group,kh,kw
    CDataBlob nweight(weight->outc,weight->inc,filters->dilation_dH*(weight->height-1)+1,filters->dilation_dW*(weight->width-1)+1);
    for(auto h=0;h<weight->outc;h++)
        for(auto i=0;i<weight->inc;i++)
            for(auto j=0;j<weight->height;j++)
                for(auto k=0;k<weight->width;k++)
                    nweight.data_float[h][i][j+j*(filters->dilation_dH-1)][k+k*(filters->dilation_dW-1)]=weight->data_float[h][i][j][k];

    //卷积运算  group
    for(auto h=0;h<outputData->batchsize;h++)
        for(auto i=0;i<outputData->channels;i++)
            for(auto j=0;j<outputData->height;j++)
                for(auto k=0;k<outputData->width;k++)
                {
                    for(auto ni=0;ni<nweight.inc;ni++)
                        for(auto nj=0;nj<nweight.height;nj++)
                            for(auto nk=0;nk<nweight.width;nk++)
                            {
                                outputData->data_float[h][i][j][k]+=nweight.data_float[i][ni][nj][nk]*input.data_float[h][ni][j*filters->stride_sH+nj][k*filters->stride_sW+nk];
                            }
                }
    return  true;
}

bool Conv2d(CDataBlob *inputData, CDataBlob *weight, Cbias *bias, covFilters *filters, CDataBlob *outputData)
{
    int inPutgroupNum=inputData->channels/filters->groups;
    int weightgroupNum=weight->outc/filters->groups;
    int outputDatagroupNum=outputData->channels/filters->groups;
    //分组进行求卷积
    for(auto ni =0;ni<filters->groups;ni++)
    {
        CDataBlob tmpinput(inputData->batchsize,inPutgroupNum,inputData->height,inputData->width);
        CDataBlob tmpweigth(weightgroupNum,weight->inc,weight->height,weight->width);
        CDataBlob tmpoutputData(outputData->batchsize,outputDatagroupNum,outputData->height,outputData->width);

        for(auto h=0;h<tmpinput.batchsize;h++)
            for(auto i=0;i<tmpinput.channels;i++)
                for(auto j=0;j<tmpinput.height;j++)
                    for(auto k=0;k<tmpinput.width;k++)
                        tmpinput.data_float[h][i][j][k]=inputData->data_float[h][ni*inPutgroupNum+i][j][k];

        for(auto h=0;h<tmpweigth.outc;h++)
            for(auto i=0;i<tmpweigth.inc;i++)
                for(auto j=0;j<tmpweigth.height;j++)
                    for(auto k=0;k<tmpweigth.width;k++)
                        tmpweigth.data_float[h][i][j][k]=weight->data_float[ni*weightgroupNum+h][i][j][k];

        convolution(&tmpinput,&tmpweigth,filters,&tmpoutputData);

        for(auto h=0;h<tmpoutputData.batchsize;h++)
            for(auto i=0;i<tmpoutputData.channels;i++)
                for(auto j=0;j<tmpoutputData.height;j++)
                    for(auto k=0;k<tmpoutputData.width;k++)
                        outputData->data_float[h][ni*outputDatagroupNum+i][j][k]=tmpoutputData.data_float[h][i][j][k]+bias->at(ni*outputDatagroupNum+i);
    }
    return  true;
}

bool ReLU(CDataBlob *inputData, CDataBlob *outputData)
{

}

bool ReLU(CDataBlob *inputData, int typeL, CDataBlob *outputData)
{
    if(typeL==0){
        for(auto h=0;h<inputData->batchsize;h++)
            for(auto i=0;i<inputData->channels;i++)
                for(auto j=0;j<inputData->height;j++)
                    for(auto k=0;k<inputData->width;k++)
                    {
                        if (inputData->data_float[h][i][j][k]<0){
                            outputData->data_float[h][i][j][k]=0;
                        }
                        else{
                            outputData->data_float[h][i][j][k]=inputData->data_float[h][i][j][k];
                        }
                    }
    }
    else if (typeL==1) {
        for(auto j=0;j<inputData->out_features;j++)
            for(auto k=0;k<inputData->in_features;k++)
            {
                if (inputData->dense_float[j][k]<0){
                    outputData->dense_float[j][k]=0;
                }
                else{
                    outputData->dense_float[j][k]=inputData->dense_float[j][k];
                }
            }
    }
}
