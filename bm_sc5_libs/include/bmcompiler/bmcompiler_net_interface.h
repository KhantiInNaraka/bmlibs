#ifndef BMCOMPILER_NET_INTERFACE_H_
#define BMCOMPILER_NET_INTERFACE_H_

#include "bmcompiler_net.h"
#include "bmcompiler_common.h"

namespace bmcompiler {

void common_proc_tensor_info(
    vector<int>&       __tensor_shape,
    string&    __tensor_name,
    const int*     tensor_shape,
    int    tensor_shape_dim,
    const char*    tensor_name
  );

void common_proc_io_info(
    vector<int>&       bottom_shape,
    vector<int>&       top_shape,
    string&    bottom_name,
    string&    top_name,
    const int*     input_shape,
    int    input_shape_dim,
    const char*    input_name,
    const int*     output_shape,
    int    output_shape_dim,
    const char*    output_name
  );

void bmcompiler_net_inf_reorg(const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int stride, bool reverse, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_priorbox(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_, const void* top_data,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    /* for save umodel */
    int min_len = 0, const float* min_size=NULL, int max_len = 0, const float* max_size=NULL,
    int ratio_len= 0, const float* aspect_ratio=NULL,
    int flip = 1, int clip=0, int var_len= 0, const float*  variance=NULL,
    int img_h=0, int img_w=0, float step_h=0, float step_w=0, float offset=0);

void bmcompiler_net_inf_permute(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    const int* permute_order, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_reverse(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    const int axis, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_normalize( const string layer_name,
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    bool across_spatial_, bool channel_shared_, const void* scale_data,
    float eps_, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_coeff = DTYPE_FP32);

void bmcompiler_net_inf_flatten(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    bmcompiler_net* p_bmcompiler_net, const vector<int>& raw_param, DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_reshape(
    const char*    input_name,
    const int*     input_shape,
    int    input_dims,
    const char*    output_name,
    const int*     new_shape,
    int    new_dims,
    DATA_TYPE_T    dtype,
    bmcompiler_net*    p_bmcompiler_net);

void bmcompiler_net_inf_biasadd(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    const string& layer_name_, const void* bias,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32,
    DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int scale = 1,
    int rshift = 0
    );

void bmcompiler_net_inf_active(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_, const int active_type,
    bmcompiler_net* p_bmcompiler_net, DATA_TYPE_T in_dtype = DTYPE_FP32,
    DATA_TYPE_T out_dtype = DTYPE_FP32,
    float input_scale = 1.0, float output_scale = 1.0);

void bmcompiler_net_inf_cpu(
    int                 input_num,
    const char* const*  input_names,         /* input1 name, input2 name...*/
    const int* const*   input_shapes,        /* input1 shape, input2 shape */
    const int*          input_dims,    /* input1 dim, input2 dim,... */
    const int*          input_dtypes,
    int                 output_num,
    const char* const*  output_names,
    const int* const*   output_shapes,
    const int*          output_dims,
    const int*          output_dtypes,
    int                 op_type,
    const void*         layer_param,        /* bmnetc --> cpu.so, not parse in compiler */
    int                 param_size,
    bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_deconv(
    const string layer_name,
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string top_name_,
    int kh, int kw, int dh, int dw,
    int pad_h_up, int pad_h_down, int pad_w_left, int pad_w_right, int groups,
    int stride_h, int stride_w, const void* weight, const void* bias,
    bool using_bias, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_weight = DTYPE_FP32, DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int rshift_num = 0);

void bmcompiler_net_inf_deconv_v2(
    const vector<int>& bottom_shape_,
    const vector<int>& top_shape_,
    const vector<string>& bottom_name_,
    const string top_name_,
    int kh, int kw, int dh, int dw,
    int pad_h_up, int pad_h_down, int pad_w_left, int pad_w_right, int groups,
    int stride_h, int stride_w, const void* weight, const void* bias,
    bool using_bias, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_weight = DTYPE_FP32, DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int rshift_num = 0);

void bmcompiler_net_inf_conv(
    const vector<int>& bottom_shape_,
    const vector<int>& top_shape_,
    const string& bottom_name_,
    const string& top_name_,
    const string& layer_name_,
    const float* weight,
    const float* bias,
    int kh_, int kw_, int groups,
    int pad_h_up, int pad_h_down, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, int dh, int dw, bool have_bias,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_weight = DTYPE_FP32, DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int rshift_num = 0,
    bool use_winograd = false
  );

void bmcompiler_net_inf_conv_v2(
    const vector<int>& bottom_shape_,
    const vector<int>& top_shape_,
    const vector<string>& bottom_name_,
    const string& top_name_,
    const float* weight,
    const float* bias,
    int kh_, int kw_, int groups,
    int pad_h_up, int pad_h_down, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, int dh, int dw, bool have_bias,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_weight = DTYPE_FP32, DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int rshift_num = 0,
    bool use_winograd = false
  );

void bmcompiler_net_inf_crop(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& shape_name_, const string& top_name_,
    vector<int>& offsets, u32 crop_mask,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_pooling(
    const vector<int>& bottom_shape_, const vector< vector<int> >& top_shape_,
    const string& bottom_name_, const vector<string>& top_name_,
    int kernel_h_, int kernel_w_,
    int up_pad_h_, int down_pad_h_, int left_pad_w_, int right_pad_w_,
    int stride_h_, int stride_w_, bool is_avg_pooling, bool avg_pooling_mode,
    bool is_global_pooling, int out_ceil_mode, const string& layer_name_, const float* mask_coeff_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32);

void bmcompiler_net_inf_pooling3d(
    const vector<int>& bottom_shape_, const vector< vector<int> >& top_shape_,
    const string& bottom_name_, const vector<string>& top_name_,
    int kernel_t_, int kernel_h_, int kernel_w_,
    int front_pad_t_,  int back_pad_t_,
    int up_pad_h_,  int down_pad_h_,
    int left_pad_w_, int right_pad_w_,
    int stride_t_, int stride_h_, int stride_w_,
    bool is_avg_pooling, bool avg_pooling_mode,
    bool is_global_pooling, int out_ceil_mode, const string& layer_name_, const float* mask_coeff_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32);

void bmcompiler_net_inf_shufflechannel(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int group_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_rpnproposal(
    const vector<vector<int> >& bottom_shape_, vector<int>& top_shape_,
    const vector<string>& bottom_name_, const string& top_name_,
    int feat_stride_, int base_size_,int min_size_,int pre_nms_topN_,int post_nms_topN_,float nms_thresh_,
    float score_thresh_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, float scale_val = 0);

void bmcompiler_net_inf_interp(
   const int*    input_shape,
   int     input_shape_dim,
   const char*   input_name,
   int     shape_is_fixed,
   // if shape is fixed, shape data should be given
   const int*    output_shape,
   int     output_shape_dim,
   // if shape is not fixed, its name should be given
   // which must be a shape tensor that be added already
   const char*   output_shape_name,
   const char*   output_name,
   int     pad_bag,
   int     pad_end,
   int     align_corners,
   int     half_pixel_centers,
   int     platform_sp,
   DATA_TYPE_T dtype,
   bmcompiler_net* p_bmcompiler_net
   );

void bmcompiler_net_inf_roipooling(
    const vector<vector<int> >& bottom_shape_, vector<int>& top_shape_,
    const vector<string>& bottom_name_, const string& top_name_,
    int pooled_h_, int pooled_w_,float spatial_scale_,int roi_nums,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_psroipooling(
    const vector<vector<int> >& bottom_shape_, vector<int>& top_shape_,
    const vector<string>& bottom_name_, const string& top_name_,
    int output_dim, int group_size,float spatial_scale_,int roi_nums,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32);


void bmcompiler_net_inf_pooling_tf(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int kernel_h_, int kernel_w_,
    int up_pad_h_, int down_pad_h_, int left_pad_w_, int right_pad_w_,
    int stride_h_, int stride_w_, bool is_avg_pooling,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_adaptivepooling(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int pooled_h_, int pooled_w_,
    bmcompiler_net* p_bmcompiler_net);

void bmcompiler_net_inf_stride_slice(
    const char*   input_name,
    const int*    input_shape,
    int     input_shape_dim,
    const char*   output_name,
    const int*    begin_index,
    const int*    end_index,
    const int*    strides,
    int     index_size,
    int     begin_mask,
    int     end_mask,
    int     shrink_axis_mask,
    int     new_axis_mask,
    int     ellipsis_mask,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_dropout(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_upsample(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int size, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32);

void bmcompiler_net_inf_fc_v2(
    const vector<vector<int>>& bottom_shapes_, vector<int>& top_shape_,
    const vector<string>& bottom_names_, const string& top_name_,
    const vector<DATA_TYPE_T> bottom_dtypes_, const DATA_TYPE_T top_dtype_,
    bool has_weight, const void* weight,
    bool weight_col_is_in_neuron_num,
    bool has_bias, const void* bias,
    bmcompiler_net* p_bmcompiler_net,
    int rshift_num = 0,
    float perlayer_bias = 0,
    const BmQuantizeInfo* quantize_info = nullptr,
    int axis = 1
  );

void bmcompiler_net_inf_fc(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    const string& layer_name_,
    int num_input_neuron, int num_output_neuron,
    const void* weight, const void* bias,
    bool have_bias, bool weight_col_is_in_neuron_num,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_weight = DTYPE_FP32, DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int rshift_num = 0
  );
void bmcompiler_net_inf_fc_weight(
    const vector<vector<int>>& bottom_shapes_, vector<int>& top_shape_,
    const vector<string>& bottom_names_, const string& top_name_,
    const string& layer_name_,
    const int num_input_neuron,
    const void* bias,
    bool have_bias, bool weight_col_is_in_neuron_num,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_weight = DTYPE_FP32, DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int rshift_num = 0
  );

/*
void bmcompiler_net_inf_data(
    const vector<int>& top_shape_, const string& top_name_,
    bmcompiler_net* p_bmcompiler_net);
*/
void bmcompiler_net_inf_batchnorm(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    const string& layer_name_,
    const void* mean_param, const void* variance_param,
    float scale_ma, float eps_, int NormMethod,
    bmcompiler_net* p_bmcompiler_net,
    int is_var_need_calc = 0,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_mean = DTYPE_FP32, DATA_TYPE_T dtype_var = DTYPE_FP32,
    int* rshift_num = NULL
  );

void bmcompiler_net_inf_scale(
    const vector<vector<int> >& bottom_shape_, vector<int>& top_shape_,
    const vector<string>& bottom_name_, const string& top_name_,
    const string& layer_name_,
    const void* scale_factor, const void* bias_,
    int num_axes, int axis_, bool have_bias,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    void *rshift_num = NULL
  );

void bmcompiler_net_inf_eltwise(
    const vector<vector<int> >& bottom_shape_, vector<int>& top_shape_,
    const vector<string>& bottom_name_, const string& top_name_,
    int op_, const float* coeff_,
    bmcompiler_net* p_bmcompiler_net,
    const vector<DATA_TYPE_T>& dtype_i = vector<DATA_TYPE_T>(),
    DATA_TYPE_T dtype_o = DTYPE_FP32,
    const vector<int>& rshift_num = vector<int>()
  );

void bmcompiler_net_inf_mulshift(
    const vector<int>& bottom_shape_, const vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int mul_value, int rshift_num,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i,
    DATA_TYPE_T dtype_o
  );

void bmcompiler_net_inf_concat(
    const vector<vector<int> >& bottom_shape_, vector<int>& top_shape_,
    const vector<string>& bottom_name_, const string& top_name_,
    int concat_axis_,
    bmcompiler_net* p_bmcompiler_net,
    const vector<DATA_TYPE_T>& dtype_i = vector<DATA_TYPE_T>(),
    DATA_TYPE_T dtype_o = DTYPE_FP32,
    const vector<int>& rshift_num = vector<int>(),
    const vector<int>& scale_val = vector<int>()
  );

void bmcompiler_net_inf_multiregion(
    const vector<vector<int> >& bottom_shape_, vector<int>& top_shape_,
    const vector<string>& bottom_name_, const string& top_name_, int input_num,
    int classes, int coords, int nums, const int* Activate_parm,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_lrn(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    float alpha_, int size_, float beta_, float k_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    float scale_in = 0, float scale_out = 0
  );

void bmcompiler_net_inf_prelu(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    const string& layer_name_,
    bool channel_shared_, const void* slope_val,
    bmcompiler_net* p_bmcompiler_net,
    int rshift_num = 0,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_coeff = DTYPE_FP32
  );

void bmcompiler_net_inf_relu(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    float negative_slope, float upper_limit,
    bmcompiler_net* p_bmcompiler_net,
    int rshift_num = 0,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_softmax(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int inner_num_, int outer_num_, int softmax_dim_, bool log,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32,
    float scale_val = 0
  );

void bmcompiler_net_inf_split(
    const vector<int>& bottom_shape_, const vector<vector<int> >& top_shape_,
    const string& bottom_name_, const vector<string>& top_name_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32
  );

void bmcompiler_net_inf_lstm(
    const vector<vector<int> >& bottom_shape_,
    const vector<vector<int> >& top_shape_,
    const vector<string>& bottom_name_,
    const vector<string>& top_name_,
    const string& layer_name_,
    int batch_num, int time_num,
    int input_dim, int output_dim,
    int user_define_cont,
    int with_x_static, int expose_hiden,
    const float* x_weight_, const float* x_bias_,
    const float* x_static_weight_,
    const float* h_weight_,
    bmcompiler_net* p_bmcompiler_net
  );

void bmcompiler_net_inf_pad(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    const int paddings_[4][2],
    float pad_val_,
    int pad_mode_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T in_data_type = DTYPE_FP32,
    DATA_TYPE_T out_data_type  = DTYPE_FP32
  );

void bmcompiler_net_inf_arg(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int axis,
    int method,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_in = DTYPE_FP32,
    DATA_TYPE_T dtype_out = DTYPE_FP32
  );

void bmcompiler_net_inf_upsamplemask(
    const vector< vector<int> >& bottom_shape_,
    const vector<int>& top_shape_,
    const vector<string>& bottom_name_,
    const string& top_name_,
    const string& layer_name_,
    const float* mask_coeff_,
    bmcompiler_net* p_bmcompiler_net
  );

void bmcompiler_net_inf_topk(
    const vector<int>& bottom_shape_, vector<vector<int> >& top_shape_,
    const string& bottom_name_, const vector<string>& top_name_,
    int k, int dim,
    bmcompiler_net* p_bmcompiler_net,
    bool descending = true,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32, DATA_TYPE_T dtype_index = DTYPE_FP32
  );

void bmcompiler_net_inf_split_tf(
    const vector<int>& bottom_shape_, const vector<vector<int> >& top_shape_,
    const string& bottom_name_, const vector<string>& top_name_,
    int shape_dim, int axis, const int* split_size, int split_num,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_split_dyn(
        const char* input_name,
        const char* size_name,
        const char* const* output_names,
        int output_num,
        int axis, bmcompiler_net* p_bmcompiler_net);

void bmcompiler_net_inf_reduce(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int reduce_method,
    DATA_TYPE_T data_type,
    bmcompiler_net* p_bmcompiler_net
  );

void bmcompiler_net_inf_broadcast_binary(
    const char* a_name,
    const int* a_shape,
    int a_dims,
    int a_is_coeff,
    const void* a_data,
    const char* b_name,
    const int* b_shape,
    int b_dims,
    int b_is_coeff,
    const void* b_data,
    const char* o_name,
    int binary_op,
    bmcompiler_net* p_bmcompiler_net,
    const vector<DATA_TYPE_T>& dtype_i = vector<DATA_TYPE_T>(2, DTYPE_FP32),
    DATA_TYPE_T dtype_o = DTYPE_FP32,
    const vector<int>& scale = vector<int>(),
    const vector<int>& rshift_num = vector<int>());

void bmcompiler_net_inf_eltwise_binary(
    const char* a_name,
    int a_is_coeff,
    const void* a_data,
    const char* b_name,
    int b_is_coeff,
    const void* b_data,
    const char* o_name,
    const int* shape,
    int  dims,
    int binary_op,
    bmcompiler_net* p_bmcompiler_net,
    const vector<DATA_TYPE_T>& dtype_i = vector<DATA_TYPE_T>(),
    DATA_TYPE_T dtype_o = DTYPE_FP32,
    const vector<int>& sacle = vector<int>(),
    const vector<int>& rshift_num = vector<int>()
    );

/* this function is only applicable for fix8b case */
void bmcompiler_net_inf_eltwise_binary_ex(
        const char* a_name,
        const char* b_hi8_name,
        const char* b_lo8_name,
        const char* o_name,
        const int* shape,
        int  dim,
        int binary_op,
        bmcompiler_net* p_bmcompiler_net,
        const vector<int>& input_sign,
        const vector<int>& scale,
        const vector<int>& rshift_num,
        int out_sign);

void bmcompiler_net_inf_const_binary(
    const char* a_name,
    const char* o_name,
    const int* shape,
    int dims,
    float b_value,
    int binary_op,
    int inversed,
    bmcompiler_net* p_bmcompiler_net,
    const vector<DATA_TYPE_T>& dtype_i = vector<DATA_TYPE_T>(),
    DATA_TYPE_T dtype_o = DTYPE_FP32,
    const vector<int>& sacle = vector<int>(),
    const vector<int>& rshift_num = vector<int>()
    );

void bmcompiler_net_inf_tile(
    const char* input_name,
    const int* input_shape,
    const int input_dim,
    const int is_coeff,
    const float* data,
    const int coeff_is_fixed,
    const char* coeff_name,
    const int* tile_coeff,
    const char* output_name,
    bmcompiler_net* p_bmcompiler_net,
    int type = 0,
    DATA_TYPE_T dtype_i = DTYPE_FP32,
    DATA_TYPE_T dtype_o = DTYPE_FP32);

void bmcompiler_net_inf_embedding(
    const char *coeff_name,
    int coeff_len,
    const float *coeff_data,
    const char *shape_name,
    int padding_idx,
    const char *output_name,
    DATA_TYPE_T data_type,
    bmcompiler_net* p_bmcompiler_net);

void bmcompiler_net_inf_expand(
    const char* input_name,
    const int* input_shape,
    const int input_dim,
    const int  is_coeff,
    const float* data,
    const int output_shape_is_fixed,
    const char* output_shape_name,
    const int* output_shape,
    int output_dim,
    const char* output_name,
    DATA_TYPE_T data_type,
    bmcompiler_net* p_bmcompiler_net,
    int type = 0
    );

void bmcompiler_net_inf_coeff_layer(
    const char* name,
    const int * shape,
    const void *data,
    const int dims,
    DATA_TYPE_T data_type,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_const_data_layer_v2(
    const char* output_name,
    const int * shape,
    const void *data,
    const int input_dim,
    DATA_TYPE_T data_type,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_const_data_layer(
    const char* input_name,
    const char* output_name,
    const int * shape,
    const void *data,
    const int input_dim,
    DATA_TYPE_T data_type,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_select(
    const char* cond_name,
    const int*  cond_shape,
    const char* s0_name,
    const int   s0_is_const,
    const float s0_value,
    const char* s1_name,
    const int   s1_is_const,
    const float s1_value,
    const char* output_name,
    const int*  shape,
    const int   dims,
    bmcompiler_net* p_bmcompiler_net,
    int scalea = 0, int nshifta = 0,
    int scaleb = 0, int nshiftb = 0,
    DATA_TYPE_T dtype_in = DTYPE_FP32, DATA_TYPE_T dtype_s0 = DTYPE_FP32,
    DATA_TYPE_T dtype_s1 = DTYPE_FP32, DATA_TYPE_T dtype_out = DTYPE_FP32
    );

void bmcompiler_net_inf_where(
        const char* cond_name,
        const char* output_name,
        const int*  in_shape,
        const int*  out_shape,
        const int   in_dims,
        const int   out_dims,
        DATA_TYPE_T dtype_in,
        DATA_TYPE_T dtype_out,
        bmcompiler_net* p_bmcompiler_net
        );

void  bmcompiler_net_inf_masked_select(
          const char* input_name,
          DATA_TYPE_T dtype_in,
          const int* in_shape,
          const int in_dims,
          bool in_is_coeff,
          const void * in_data,
          const char* mask_name,
          DATA_TYPE_T dtype_mask,
          const int* mask_shape,
          const int mask_dims,
          bool mask_is_coeff,
          const void * mask_data,
          const char* output_name,
          DATA_TYPE_T dtype_out,
          const int* out_shape,
          const int out_dims,
          bmcompiler_net* p_bmcompiler_net,
          bool bcast_from_begin = false
          );

void bmcompiler_net_inf_output(
    const vector<int>& bottom_shape_,
    const vector<int>& top_shape_,
    const string& bottom_name_,
    const string& top_name_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32
    );

void bmcompiler_net_inf_output_v2(
    const char* bottom_name_,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_shape_op(
    const char* in0_name,
    const char* in1_name,
    const int binary_op,
    const char* out_name,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_shape_pack(
    const char* const* shape_names,
    const int shape_num,
    const int axis,
    const char* pack_name,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_shape_ref(
    const char* input_name,
    const int*  input_shape,
    const int   input_dim,
    DATA_TYPE_T input_dtype,
    const char* shape_name,
    bmcompiler_net* p_bmcompiler_net,
    bool input_is_coeff = false
    );

void bmcompiler_net_inf_rank(
    const char* input_name,
    const int*  input_shape,
    const int   input_dim,
    DATA_TYPE_T input_dtype,
    const char* rank_name,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_squeeze(
    const char*   input_name, //must exist already
    const int*    input_shape,
    const int     input_dim,
    DATA_TYPE_T   input_dtype,
    const int*    axis_list,
    const int     axis_num, //0 means removal of all '1' dims
    const char*   output_name,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_expand_ndims(
    const char*   input_name, //must exist already
    const int*    input_shape,
    const int     input_dim,
    const void*   input_data,
    DATA_TYPE_T   input_dtype,
    const int     axis,
    const int     ndims,
    const char*   output_name,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_shape_assign(
    const char* input_name,
    const int*  input_shape,
    const int   input_dim,
    DATA_TYPE_T input_dtype,
    const char* shape_name,
    const char* output_name,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_shape_addn(
    const char* const* input_names,
    const int input_num,
    const char* output_name,
    bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_shape_reorder(
    const char* shape_name,
    const int* shape_order,
    const int order_num,
    const char* output_name,
    bmcompiler_net * p_bmcompiler_net
);

void bmcompiler_net_inf_ref_crop(
    const char* input_name,
    const int*  input_shape,
    const int   input_dim,
    DATA_TYPE_T input_dtype,
    const char* crop_name,
    const char* output_name,
    bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_ref_pad(
    const char* input_name,
    const int*  input_shape,
    const int   input_dim,
    DATA_TYPE_T input_dtype,
    const char* pad_name,
    const int   pad_mode,
    const float pad_value,
    const char* output_name,
    bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_conv_weight(
    const vector< vector<int> >& bottom_shape_,
    const vector<int>& top_shape_,
    const vector<string>& bottom_name_,
    const string& top_name_,
    const string& layer_name_,
    const float* bias,
    int groups,
    int pad_h_up, int pad_h_down, int pad_w_left, int pad_w_right,
    int stride_h, int stride_w, int dh, int dw, bool have_bias,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_weight = DTYPE_FP32, DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int rshift_num = 0,
    bool use_winograd = false
  );

void bmcompiler_net_inf_transpose(
    const char* input_name,
    const int*  input_shape,
    int         input_dims,
    DATA_TYPE_T input_dtype,
    const char* output_name,
    const char* order_name,
    const int*  order_data,
    bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_reduce_full(
    const char* input_name,
    const char* output_name,
    const int*  input_shape,
    int  input_dims,
    int  reduce_method,
    int  need_keep_dims,
    const int* axis_list,
    int  axis_num,
    DATA_TYPE_T input_dtype,
    bmcompiler_net* p_bmcompiler_net,
    float input_scale = 1.0,
    float output_scale = 1.0
);

void bmcompiler_net_inf_batch2space(
  const char* input_name,
  const int*  input_shape,
  int     input_dim,
  int     block_is_dynamic,
  const char* block_name,  //if dynamic, must be an existed shape tensor name
  const int*  block_sizes, //if not dynamic, must be valid--{hblock,wblock}
  int     crop_is_dynamic,
  const char* crop_name,    //if dynamic, must be an existed shape tensor name
  const int*  crop_sizes,   //if not dynamic, must be valid--{ht_crop, hb_crop, wl_crop, wr_crop}
  const char* output_name,
  bmcompiler_net* p_bmcompiler_net,
  DATA_TYPE_T dtype_i = DTYPE_FP32,
  DATA_TYPE_T dtype_o = DTYPE_FP32
);

void bmcompiler_net_inf_space2batch(
  const char* input_name,
  const int*  input_shape,
  int     input_dim,
  int     block_is_dynamic,
  const char* block_name,  //if dynamic, must be an existed shape tensor name
  const int*  block_sizes, //if not dynamic, must be valid--{hblock,wblock}
  int     pad_is_dynamic,
  const char* pad_name,    //if dynamic, must be an existed shape tensor name
  const int*  pad_sizes,   //if not dynamic, must be valid--{ht_pad, hb_pad, wl_pad, wr_pad}
  const char* output_name,
  bmcompiler_net* p_bmcompiler_net,
  DATA_TYPE_T dtype_i = DTYPE_FP32,
  DATA_TYPE_T dtype_o = DTYPE_FP32
);

void bmcompiler_net_inf_cumsum(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int dim,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32
);

void bmcompiler_net_inf_slice_like(
    const vector<vector<int> >& bottom_shape_, vector<int>& top_shape_,
    const vector<string>& bottom_name_, const string& top_name_,
    const int* axis, int axis_num,
    bmcompiler_net* p_bmcompiler_net,
    const vector<DATA_TYPE_T>& dtype_i = vector<DATA_TYPE_T>(),
    DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_stride_calculate(
    const vector<vector<int>>& bottom_shape_,
    const vector<int>& top_shape_,
    const vector<string>& bottom_name_,
    const string& top_name_,
    const vector <DATA_TYPE_T>& dtype_i,
    DATA_TYPE_T dtype_o,
    const vector<int>& offset,
    const vector<int>& a_stride,
    const vector<int>& b_stride,
    int op_,
    int a_is_const,
    int b_is_const,
    float const_val,
    DATA_TYPE_T const_dtype,
    int a_is_coeff,
    int b_is_coeff,
    const void* a_data,
    const void* b_data,
    int result_add,
    bmcompiler_net* p_bmcompiler_net
  );
 
void bmcompiler_net_inf_channel_shift(
    const vector<int>& bottom_shape_,
    const vector<int>& top_shape_,
    const string& bottom_name_,
    const string& top_name_,
    int shift_dir_,
    int shift_num_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_constant_fill(
    const char* shape_name,
    const char* output_name,
    const void* filled_value,
    int type_len,
    DATA_TYPE_T dtype,
    bmcompiler_net* p_bmcompiler_net
  );

void bmcompiler_net_inf_dtype_convert(
    const vector<int>& bottom_shape_,
    const vector<int>& top_shape_,
    const string& bottom_name_,
    const string& top_name_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32,
    DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_batch_matmul(
    const char*   input0_name,
    const int*    input0_shape,
    int       input0_shape_dim,
    int       input0_is_const,
    const float*  input0_data,

    const char*   input1_name,
    const int*    input1_shape,
    int       input1_shape_dim,
    int       input1_is_const,
    const float*  input1_data,

    const char*   output_name,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32
);

void bmcompiler_net_inf_interleave(
    const vector<vector<int>>& bottom_shape_,
    vector<int>& top_shape_,
    const vector<string>& bottom_name_,
    const string& top_name_,
    int axis_,
    int step_,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32,
    DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_shape_range(
    const char* begin_name,
    const char* delta_name,
    const char* end_name,
    const char* out_name,
    bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_shape_tile(
    const char* input_name,
    const int*  tile_coeff,
    int     tile_len,
    const char* output_name,
    bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_shape_reverse(
    const char* input_name,
    int     axis,
    const char* output_name,
    bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_shape_expand_ndims(
    const char* input_name,
    int     axis,
    int     expand_num,
    const char* output_name,
    bmcompiler_net* p_bmcompiler_net
);

//input is int32, then output is float32
//input is float32, then output is int32
void bmcompiler_net_inf_shape_cast(
  const char* input_name,
  const char* output_name,
  DATA_TYPE_T dtype,
  bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_shape_reshape(
  const char* input_name,
  const char* new_shape_name,
  const char* output_name,
  bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_shape_reduce(
  const char* input_name,
  const int*  axis_list,
  int     axis_num,
  int     keep_dims,  //true or false
  int     reduce_method,
  const char* output_name,
  bmcompiler_net* p_bmcompiler_net
);

void bmcompiler_net_inf_priorbox_cpu(
  const vector<int>& bottom_shape_,
  const vector<int>& top_shape_,
  const string& bottom_name_,
  const string& top_name_,
  const void* top_data,
  float*   min_sizes,
  int     real_min_size,
  float*   max_sizes,
  int     real_max_size,
  float*   aspect_ratios,
  int     real_spect_size,
  float*   variance,
  int     real_variance_size,
  int     num_priors,
  int     img_w,
  int     img_h,
  float   step_w,
  float   step_h,
  float   offset,
  float   thTop,
  int     bottom_0_width,
  int     bottom_0_height,
  int     bottom_1_width,
  int     bottom_1_height,
  int     dim,
  bool    has_dim,
  bool    flip,
  bool    clip,
  bmcompiler_net* p_bmcompiler_net,
  DATA_TYPE_T dtype_i = DTYPE_FP32,
  DATA_TYPE_T dtype_o = DTYPE_FP32
);

void bmcompiler_net_inf_priorbox_cpu_v2(
  const vector<int>& bottom0_shape_,
  const vector<int>& bottom1_shape_,
  const vector<int>& top_shape_,
  const string& bottom0_name_,
  const string& bottom1_name_,
  const string& top_name_,
  const void* top_data,
  float*   min_sizes,
  int     real_min_size,
  float*   max_sizes,
  int     real_max_size,
  float*   aspect_ratios,
  int     real_spect_size,
  float*   variance,
  int     real_variance_size,
  int     num_priors,
  int     img_w,
  int     img_h,
  float   step_w,
  float   step_h,
  float   offset,
  float   thTop,
  int     bottom_0_width,
  int     bottom_0_height,
  int     bottom_1_width,
  int     bottom_1_height,
  int     dim,
  bool    has_dim,
  bool    flip,
  bool    clip,
  bmcompiler_net* p_bmcompiler_net,
  DATA_TYPE_T dtype_i = DTYPE_FP32,
  DATA_TYPE_T dtype_o = DTYPE_FP32
);

template<typename T>
void bmcompiler_net_inf_common(
        LAYER_TYPE_T ltype,
        std::initializer_list<T> inputs,
        std::initializer_list<T> outputs,
        bmcompiler_net* net,
        LAYER_PARAM_T *lparam = nullptr
);
template<typename T>
void bmcompiler_net_inf_common(
  LAYER_TYPE_T ltype,
  const vector<T>& inputs,
  const vector<T>& outputs,
  bmcompiler_net* net,
  LAYER_PARAM_T *lparam = nullptr
);

void bmcompiler_net_inf_sizeslice(
  const vector<string>& inputs,
  const vector<string>& outputs,
  u32 slice_mask,
  bmcompiler_net* net
);

void bmcompiler_net_inf_yolo(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int n,
    int classes,
    int coords,
    int background,
    int softmax,
    bmcompiler_net* p_bmcompiler_net
  );

void bmcompiler_net_inf_ssd_detect_out(
    const vector<int>& bottom0_shape_,
    const string& bottom0_name_,
    const vector<int>& bottom1_shape_,
    const string& bottom1_name_,
    const vector<int>& bottom2_shape_,
    const string& bottom2_name_,
    const vector<int>& top_shape_,
    const string& top_name_,
    int num_classes,
    bool share_location,
    int background_label_id,
    int code_type,
    bool variance_encoded_in_target,
    int keep_top_k,
    float confidence_threshold,
    float nms_threshold,
    float eta,
    int top_k,
    bmcompiler_net* p_bmcompiler_net
    );

void bmcompiler_net_inf_shape_active(
        const char* input_name,
        const char* output_name,
        int active_op,
        bmcompiler_net* p_bmcompiler_net);

void bmcompiler_net_inf_identity(
        const char* input_name,
        const char* output_name,
        bmcompiler_net* p_bmcompiler_net
        );

void bmcompiler_net_inf_bitwise(
    const vector<int>& bottom0_shape_,
    const string& bottom0_name_,
    const vector<int>& bottom1_shape_,
    const string& bottom1_name_,
    const vector<int>& top_shape_,
    const string& top_name_,
    int binary_op,
    DATA_TYPE_T dtype,
    bmcompiler_net* p_bmcompiler_net);

void bmcompiler_net_inf_shift(
    const vector<vector<int>>& bottom_shape_,
    const vector<int>& top_shape_,
    const vector<string>& bottom_name_,
    const string& top_name_,
    int   is_shift_coeff,
    const void *shift_data,
    int shift_num_,
    int shift_mode,
    int shift_is_const,
    int type,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i, DATA_TYPE_T dtype_o);

void bmcompiler_net_inf_const_bitwise(
    const vector<int>& bottom0_shape_,
    const string& bottom0_name_,
    int   b_value,
    const vector<int>& top_shape_,
    const string& top_name_,
    int bitwise_op_,
    DATA_TYPE_T dtype,
    bmcompiler_net* p_bmcompiler_net);

void bmcompiler_net_inf_yolov3_detect_out(
    const vector<int>& bottom0_shape_,
    const string& bottom0_name_,
    const vector<int>& bottom1_shape_,
    const string& bottom1_name_,
    const vector<int>& bottom2_shape_,
    const string& bottom2_name_,
    const vector<int>& top_shape_,
    const string& top_name_,
    int num_classes,
    int num_boxes,
    int mask_group_size,
    int keep_top_k,
    float confidence_threshold,
    float nms_threshold,
    float bias[18],
    float anchor_scale[3],
    float mask[9],
    bmcompiler_net* p_bmcompiler_net);

void bmcompiler_net_inf_sort(
    const vector<int>& bottom_shape_,
    const vector< vector<int> >& top_shape_,
    const string& bottom_name_, const vector<string>& top_name_,
    int dim, bool stable, bool descending, bool is_argsort,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i, DATA_TYPE_T dtype_o);

void bmcompiler_net_inf_index_select(
    const vector<vector<int>>& bottom_shape_, const vector<int> & top_shape_,
    const vector<string>& bottom_name_, bool index_is_coeff, void* index_data, const string& top_name_,
    int dim, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i, DATA_TYPE_T dtype_o);

void bmcompiler_net_inf_nms(
    const vector<vector<int>>& bottom_shape_, const vector<int> & top_shape_,
    const vector<string>& bottom_name_, const string& top_name_,
    float iou_threshold, bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i, DATA_TYPE_T dtype_o,
    float score_threshold = -1);

void bmcompiler_net_inf_yolov3_detect_out_v2(
    int                 input_num,
    const char* const*  input_names,         /* input1 name, input2 name...*/
    const int* const*   input_shapes,        /* input1 shape, input2 shape */
    const int*          input_dims,          /* input1 dim, input2 dim,... */
    const char*         output_name,
    const int*          output_shape,
    const int           output_dim,
    int num_classes,
    int num_boxes,
    int mask_group_size,
    int keep_top_k,
    float confidence_threshold,
    float nms_threshold,
    float *bias,
    float *anchor_scale,
    float *mask,
    bmcompiler_net* p_bmcompiler_net);

//add_layer: layer common interface
void bmcompiler_net_inf_lut(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const vector<int>& table_shape_,
    const string& bottom_name_, const string& top_name_,
    const string& table_name_,
    const float* table,
    bool table_is_const,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_UINT8, DATA_TYPE_T dtype_o = DTYPE_FP32
  );

void bmcompiler_net_inf_conv3d(
    const vector<int>& bottom_shape_,
    const vector<int>& top_shape_,
    const string& bottom_name_,
    const string& top_name_,
    const string& layer_name_,
    const float* weight,
    const float* bias,
    int kt_, int kh_, int kw_, int groups,
    int pad_t_, int pad_t_after_,
    int pad_h_, int pad_h_after_,
    int pad_w_, int pad_w_after_,
    int stride_t, int stride_h, int stride_w, int dt, int dh, int dw, bool have_bias,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype_i = DTYPE_FP32, DATA_TYPE_T dtype_o = DTYPE_FP32,
    DATA_TYPE_T dtype_weight = DTYPE_FP32, DATA_TYPE_T dtype_bias = DTYPE_FP32,
    int rshift_num = 0,
    bool use_winograd = false
  );

void bmcompiler_net_inf_coeff2neuron(
    const vector<int>& shape,
    const string& bottom_name,
    const string& top_name,
    const void* coeff_data,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32
  );

void bmcompiler_net_inf_unfold(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int axis, int size, int step,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_gru(
        const std::vector<int> &bottom_shape_0,
        const std::string &bottom_name_0,
        const std::vector<int> &bottom_shape_1,
        const std::string &bottom_name_1,
        const std::vector<int> &top_shape_0,
        const std::string &top_name_0,
        const std::vector<int> &top_shape_1,
        const std::string &top_name_1,
        const float *weight,
        const float *bias,
        bool bidirection,
        bool batch_first,
        int num_layers,
        const std::string &layer_name,
        bmcompiler_net *p_bmcompiler_net,
        DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_pytorch_lstm(
        const std::vector<int> &bottom_shape_0,
        const std::string &bottom_name_0,
        const std::vector<int> &bottom_shape_1,
        const std::string &bottom_name_1,
        const std::vector<int> &bottom_shape_2,
        const std::string &bottom_name_2,
        const std::vector<int> &top_shape_0,
        const std::string &top_name_0,
        const std::vector<int> &top_shape_1,
        const std::string &top_name_1,
        const std::vector<int> &top_shape_2,
        const std::string &top_name_2,
        const float *weight,
        const float *bias,
        bool bidirection,
        bool batch_first,
        int num_layers,
        const std::string &layer_name,
        bmcompiler_net *p_bmcompiler_net,
        DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_matrix_band_part(
    const vector<int>& bottom_shape_, vector<int>& top_shape_,
    const string& bottom_name_, const string& top_name_,
    int lower, int upper,
    bmcompiler_net* p_bmcompiler_net,
    DATA_TYPE_T dtype = DTYPE_FP32);

void bmcompiler_net_inf_binary_shift(
    const vector<int>& a_shape,
    const vector<int>& b_shape,
    const vector<int>& top_shape,
    const string& a_name,
    const string& b_name,
    const string& top_name_,
    int           a_is_coeff,
    int           b_is_coeff,
    void*         a_data,
    void*         b_data,
    int           binary_op,
    int           rshift_num,
    int           b_is_const,
    int           b_const_val,
    int           inversed,
    bmcompiler_net* p_bmcompiler_net,
    const vector<DATA_TYPE_T>& dtype_i,
    DATA_TYPE_T dtype_o);

void bmcompiler_net_inf_tpu(
    int                 input_num,
    const char* const*  input_names,         /* input1 name, input2 name...*/
    const int* const*   input_shapes,        /* input1 shape, input2 shape */
    const int*          input_dims,    /* input1 dim, input2 dim,... */
    const int*          input_dtypes,
    int                 output_num,
    const char* const*  output_names,
    const int* const*   output_shapes,
    const int*          output_dims,
    const int*          output_dtypes,
    int                 op_type,
    const void*         layer_param,        /* bmnetc --> cpu.so, not parse in compiler */
    int                 param_size,
    bmcompiler_net* p_bmcompiler_net
);
}  // namespace bmcompiler

#if defined(DUMP_COEFF_REFS)
extern string COEFF_DIR;
#endif

#endif
