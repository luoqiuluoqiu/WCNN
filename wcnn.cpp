#include "wcnn.h"
#include<QDebug>

//卷积实现
bool convolution(CDataBlob *inputData, CDataBlob *weight, covFilters *filters, CDataBlob *outputData)
{
    //hout=(h+2*p-(d*(k-1)+1))/s+1
    //原始数据进行扩边
    CDataBlob input(inputData->batchsize, inputData->channels, inputData->height + filters->padding_padH * 2, inputData->width + filters->padding_padW * 2);
    for (auto h = 0; h < inputData->batchsize; h++)
        for (auto i = 0; i < inputData->channels; i++)
            for (auto j = 0; j < inputData->height; j++)
                for (auto k = 0; k < inputData->width; k++)
                    input.data_float[h][i][j + filters->padding_padH][k + filters->padding_padW] = inputData->data_float[h][i][j][k];

    //扩充卷积权重 dilation参数  out,in/group,kh,kw
    CDataBlob nweight(weight->outc, weight->inc, filters->dilation_dH*(weight->height - 1) + 1, filters->dilation_dW*(weight->width - 1) + 1);
    for (auto h = 0; h < weight->outc; h++)
        for (auto i = 0; i < weight->inc; i++)
            for (auto j = 0; j < weight->height; j++)
                for (auto k = 0; k < weight->width; k++)
                    nweight.data_float[h][i][j + j*(filters->dilation_dH - 1)][k + k*(filters->dilation_dW - 1)] = weight->data_float[h][i][j][k];

    //卷积运算
    for (auto h = 0; h < outputData->batchsize; h++)
        for (auto i = 0; i < outputData->channels; i++)
            for (auto j = 0; j < outputData->height; j++)
                for (auto k = 0; k < outputData->width; k++)
                {
                    for (auto ni = 0; ni < nweight.inc; ni++)
                        for (auto nj = 0; nj < nweight.height; nj++)
                            for (auto nk = 0; nk < nweight.width; nk++)
                            {
                                outputData->data_float[h][i][j][k] += nweight.data_float[i][ni][nj][nk] * input.data_float[h][ni][j*filters->stride_sH + nj][k*filters->stride_sW + nk];
                            }
                }
    return  true;
}

bool Conv2d(CDataBlob *inputData, CDataBlob *weight, Cbias *bias, covFilters *filters, CDataBlob *outputData)
{
    int inPutgroupNum = inputData->channels / filters->groups;
    int weightgroupNum = weight->outc / filters->groups;
    int outputDatagroupNum = outputData->channels / filters->groups;
    //分组进行求卷积
    for (auto ni = 0; ni < filters->groups; ni++)
    {
        CDataBlob tmpinput(inputData->batchsize, inPutgroupNum, inputData->height, inputData->width);
        CDataBlob tmpweigth(weightgroupNum, weight->inc, weight->height, weight->width);
        CDataBlob tmpoutputData(outputData->batchsize, outputDatagroupNum, outputData->height, outputData->width);

        for (auto h = 0; h < tmpinput.batchsize; h++)
            for (auto i = 0; i < tmpinput.channels; i++)
                for (auto j = 0; j < tmpinput.height; j++)
                    for (auto k = 0; k < tmpinput.width; k++)
                        tmpinput.data_float[h][i][j][k] = inputData->data_float[h][ni*inPutgroupNum + i][j][k];

        for (auto h = 0; h < tmpweigth.outc; h++)
            for (auto i = 0; i < tmpweigth.inc; i++)
                for (auto j = 0; j < tmpweigth.height; j++)
                    for (auto k = 0; k < tmpweigth.width; k++)
                        tmpweigth.data_float[h][i][j][k] = weight->data_float[ni*weightgroupNum + h][i][j][k];

        convolution(&tmpinput, &tmpweigth, filters, &tmpoutputData);

        for (auto h = 0; h < tmpoutputData.batchsize; h++)
            for (auto i = 0; i < tmpoutputData.channels; i++)
                for (auto j = 0; j < tmpoutputData.height; j++)
                    for (auto k = 0; k < tmpoutputData.width; k++)
                        outputData->data_float[h][ni*outputDatagroupNum + i][j][k] = tmpoutputData.data_float[h][i][j][k] + bias->at(ni*outputDatagroupNum + i);
    }
    return  true;
}


//MaxPool实现
template <typename T> static T maxArray(const vector<vector<T>>& data)
{
    vector<T> maxvec;
    maxvec.reserve(data.size());
    auto it = data.cbegin();
    for (it; it != data.cend(); ++it)
    {
        auto big = std::max_element((*it).cbegin(), (*it).cend());
        maxvec.push_back(*big);
    }
    //vector<T>::iterator
    auto itval = std::max_element(maxvec.begin(), maxvec.end());
    return *itval;
}


template <typename T> vector<vector<T>> dilatdKernel(const vector<vector<T>> & kernel, int dh, int dw)
{
    if (kernel.size() == 0 || kernel.at(0).size() == 0)
        return kernel;
    if (dh == 1 && dw == 1)
        return kernel;

    int hk0 = kernel.size();
    int wk0 = kernel.at(0).size();
    vector<vector<T>> dkernel(hk0*dh, vector<T>(dw*wk0, 0.0));
    for (auto id = 0; id < hk0; ++id)
    {
        if (dw == 1)
            dkernel.at(id*dh) = kernel.at(id);
        else
        {
            vector<T> vec(dw*wk0, 0);
            for (auto it = 0; it < kernel.at(id).size(); ++it)
            {
                vec.at(it*dw) = kernel.at(id).at(it);
            }
            dkernel.at(id*dh) = vec;
        }
    }
    return dkernel;
}

template<typename T> _CDataBlob<T> padingTensor(const _CDataBlob<T> & input, int th, int bh, int lw, int rw, T value = 0)
{
    if ((th + bh + lw + rw) == 0)
    {
        return input;
    }

    _CDataBlob<T> newinput(input.batchsize, input.channels, input.height + th + bh, input.width + lw + rw);
    for (auto b = 0; b < newinput.batchsize; ++b)
    {
        for (auto c = 0; c < newinput.channels; ++c)
        {
            vector<T> vec(newinput.height, value);
            for (auto h = 0; h < newinput.height; ++h)
            {
                newinput.data_float[b][c][h] = vector<T>(newinput.height, value);
                if (h >= th && h < newinput.height - bh)
                {
                    std::copy(input.data_float[b][c][h - th].begin(), input.data_float[b][c][h - th].end(), newinput.data_float[b][c][h].begin() + lw);
                }
            }
        }
    }
    return newinput;
}

template<typename T> _CDataBlob<T> padingTensor(const _CDataBlob<T> & input, int pH, int pW, T value=0)
{
    if (pH == 0 && pW == 0)
    {
        return input;
    }
    return padingTensor(input, pH, pH, pW, pW, value);
}

bool MaxPool2d(CDataBlob *inputData, maxFilters *filters, CDataBlob *outputData)
{

    int kW = filters->kernel_size_kW;
    int kH = filters->kernel_size_kH;
    int padW = filters->padding_padW;
    int padH = filters->padding_padH;
    int sH = filters->stride_sH;
    int sW = filters->stride_sW;
    int dW = filters->dilation_dW;
    int dH = filters->dilation_dH;
    bool ceil_mode = filters->ceil_mode;

    int inputWidth = inputData->width;
    int inputHeight = inputData->height;
    int nInputPlane = inputData->channels; // number of channels (or colors)
    int outputWidth;
    int outputHeight;

    if (ceil_mode)
    {
        outputWidth = (int)(ceil((float)(inputWidth + 2 * padW - dW*(kW - 1) - 1) / sW)) + 1;
        outputHeight = (int)(ceil((float)(inputHeight + 2 * padH - dH*(kH - 1)) / sH)) + 1;
    }
    else
    {
        outputWidth = (int)(floor((float)(inputWidth + 2 * padW - dW*(kW - 1) - 1) / sW)) + 1;
        outputHeight = (int)(floor((float)(inputHeight + 2 * padH - dH*(kH - 1)) / sH)) + 1;
    }

    CDataBlob borderInput = padingTensor(*inputData, padH, padW);
    vector<vector<double>> kernel = dilatdKernel(vector<vector<double>>(kH, vector<double>(kW, 1.0)), dH, dW);
    outputData->setSize(inputData->batchsize, nInputPlane, outputHeight, outputWidth);
    CDataBlob& output = *outputData;
    int hNum = (borderInput.height - kH*dH) / sH + 1;
    int wNum = (borderInput.width - kW*dW) / sW + 1;
    int hMod = (borderInput.height - kH*dH) % sH;
    int wMod = (borderInput.width - kW*dW) % sW;
    if (ceil_mode)
    {
        if ((hMod + wMod) != 0)
        {
            borderInput = padingTensor(borderInput, 0, kH*dH - (hMod + 1), 0, kW*dW - (wMod + 1));
            hNum = (borderInput.height - kH*dH) / sH + 1;
            wNum = (borderInput.width - kW*dW) / sW + 1;
            hMod = (borderInput.height - kH*dH) % sH;
            wMod = (borderInput.width - kW*dW) % sW;
        }
    }
    qDebug()<<borderInput.data_float;
    for (auto b = 0; b < output.batchsize; ++b)
    {
        for (auto c = 0; c < output.channels; ++c)
        {
            try
            {
                vector<vector<double>> plane = borderInput.data_float[b][c];
                for (auto h = 0; h < outputHeight; ++h)
                {
                    for (auto w = 0; w < outputWidth; ++w)
                    {
                        //vector<vector<double>> subInput;
                        vector<vector<double>> multis;
                        multis.reserve(kH*dH);
                        int rowNum = (h*sH + kH*dH);// <= outputHeight ? (h*sH + kH*dH) : outputHeight;
                        for (auto id = h*sH; id < rowNum; ++id)
                        {
                            //TODO
                            if (h <= hNum && w <= wNum)
                            {
                                vector<double> vec;
                                vec.reserve(kW*dW);
                                vec.insert(vec.end(), plane[id].begin() + w*sW, plane[id].begin() + w*sW + kW*dW);
                                //subInput.push_back(vec);
                                vector<double> mult;
                                mult.reserve(kW*dW);
                                std::transform(kernel[id - h*sH].begin(), kernel[id - h*sH].end(), vec.begin(), std::back_inserter(mult), std::multiplies<double>());
                                multis.push_back(mult);
                            }
                        }

                        auto val = maxArray(multis);
                        //std::cout << "maxValue: " << val << ", b: " << b << ", c: " << c << ", h: " << h << ", w: " << w << std::endl;
                        output.data_float[b][c][h][w] = val;
                    }
                }
            }
            catch (const std::exception& ex)
            {
                std::cout << ex.what() << std::endl;
            }
        }
    }
    return true;
}


//Relu实现
bool ReLU(CDataBlob *inputData, int typeL, CDataBlob *outputData)
{
    if (typeL == 0) {
        for (auto h = 0; h < inputData->batchsize; h++)
            for (auto i = 0; i < inputData->channels; i++)
                for (auto j = 0; j < inputData->height; j++)
                    for (auto k = 0; k < inputData->width; k++)
                    {
                        if (inputData->data_float[h][i][j][k] < 0) {
                            outputData->data_float[h][i][j][k] = 0;
                        }
                        else {
                            outputData->data_float[h][i][j][k] = inputData->data_float[h][i][j][k];
                        }
                    }
    }
    else if (typeL == 1) {
        for (auto j = 0; j < inputData->out_features; j++)
            for (auto k = 0; k < inputData->in_features; k++)
            {
                if (inputData->dense_float[j][k] < 0) {
                    outputData->dense_float[j][k] = 0;
                }
                else {
                    outputData->dense_float[j][k] = inputData->dense_float[j][k];
                }
            }
    }
    return true;
}

//Linear实现
bool Linear(CDataBlob *inputData, CDataBlob *weight, Cbias *bias, CDataBlob *outputData)
{
    for (auto i = 0; i < outputData->batchsize; i++)
        for (auto j = 0; j < outputData->in_features; j++) {
            for (auto k = 0; k < inputData->in_features; k++)
            {
                outputData->dense_float[i][j] += inputData->dense_float[i][k] * weight->dense_float[j][k];
            }
            outputData->dense_float[i][j] += bias->at(j);
        }
    return true;
}
