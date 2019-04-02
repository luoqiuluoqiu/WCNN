#ifndef WCNN_H
#define WCNN_H
#include <vector>
#include <iostream>
#include <algorithm>
using std::vector;

template <typename _Tp>
using _vector = vector<_Tp>;

typedef   _vector<double>  Tensor1f;
typedef   _vector<vector<double>>  Tensor2f;
typedef   _vector<vector<vector<double>>>  Tensor3f;
typedef   _vector<vector<vector<vector<double>>>>  Tensor4f;
//定义WMat结构体

template <typename Tp> class _CDataBlob
{
public:
	//公共变量
	int batchsize;

	vector<vector<Tp>> dense_float;//full connect
	//dense format
	int out_features;
	int in_features;

	vector<vector<vector<vector<Tp>>>> data_float; //min-batch image data
	int height;
	int width;
	//min-batch image data format
	int channels;
	//weight format
	int outc;
	int inc;

public:
	//默认构造函数
	_CDataBlob() 
	{
		this->width = 0;
		this->height = 0;
		this->channels = 0;
		this->batchsize = 0;
		this->outc = 0;
		this->inc = 0;
		this->data_float.clear();
	}
	//nn.Line
	_CDataBlob(int out, int in)
	{
		this->data_float.clear();
		this->batchsize = this->out_features = out;
		this->in_features = in;
		this->dense_float.resize(out);
		for (auto k = 0; k < out; k++)
		{
			this->dense_float[k].resize(in);
		}
	}
	//nn.Cov
	_CDataBlob(int n, int c, int h, int w)
	{
		this->data_float.clear();
		this->width = w;
		this->height = h;
		this->inc = this->channels = c;
		this->outc = this->batchsize = n;
		this->data_float.resize(n);
		for (auto k = 0; k < n; k++)
		{
			this->data_float[k].resize(c);
			for (auto i = 0; i < c; i++)
			{
				this->data_float[k][i].resize(h);
				for (auto j = 0; j < h; j++)
				{
					this->data_float[k][i][j].resize(w);
				}
			}
		}
	}

	vector<int> shape()
	{
		vector<int> _shape{this->batchsize, this->channels, this->height, this->width};
		return _shape;
	}

	int shape(int dim)
	{
		switch (dim)
		{
		case 0:
			return this->batchsize;
		case 1:
			return this->channels;
		case 2:
			return this->height;
		case 3:
			return this->width;
		default:
			return -1;
		}
	}

	void setSize(int n, int c, int h, int w)
	{
		this->data_float.clear();
		this->width = w;
		this->height = h;
		this->inc = this->channels = c;
		this->outc = this->batchsize = n;
		this->data_float.resize(n);
		for (auto k = 0; k < n; k++)
		{
			this->data_float[k].resize(c);
			for (auto i = 0; i < c; i++)
			{
				this->data_float[k][i].resize(h);
				for (auto j = 0; j < h; j++)
				{
					this->data_float[k][i][j].resize(w);
				}
			}
		}
	}

	bool setDataFromImage()
	{
		return true;
	}


};

typedef _CDataBlob<double> CDataBlob;

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
	maxFilters() = default;
	maxFilters(
		int _kernel_size_kH,
		int _kernel_size_kW,
		int _stride_sH,
		int _stride_sW,
		int _padding_padH,
		int _padding_padW,
		int _dilation_dH=1,
		int _dilation_dW=1,
		bool _return_indices = false,
		bool _ceil_mode = false) :
		kernel_size_kH(_kernel_size_kH), kernel_size_kW(_kernel_size_kW), stride_sH(_stride_sW), stride_sW(_stride_sW), padding_padH(_padding_padH), padding_padW(_padding_padW), dilation_dH(_dilation_dH), dilation_dW(_dilation_dW),  return_indices(_return_indices), ceil_mode(_ceil_mode)
	{}
}maxFilters;


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
	vector<T>::iterator itval = std::max_element(maxvec.begin(), maxvec.end());
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


//定义常见的计算
//已经实现
bool Conv2d(CDataBlob *inputData, CDataBlob *weight, Cbias *bias, covFilters *filters, CDataBlob *outputData);
bool ReLU(CDataBlob *inputData, int typeL, CDataBlob *outputData);
//待实现
bool MaxPool2d(CDataBlob *inputData, maxFilters *filters, CDataBlob *outputData);
bool Linear(CDataBlob *inputData, CDataBlob *weight, Cbias *bias, CDataBlob *outputData);


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
