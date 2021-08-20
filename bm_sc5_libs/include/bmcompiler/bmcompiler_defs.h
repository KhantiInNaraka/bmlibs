#ifndef __BMCOMPILER_DEFS_H__
#define __BMCOMPILER_DEFS_H__

#define ARCH_BM1682 "BM1682"
#define ARCH_BM1684 "BM1684"

typedef unsigned long long u64;

#ifdef __cplusplus
namespace bmcompiler {
#endif

typedef enum {
  DTYPE_FP32 = 0,
  DTYPE_FP16 = 1,
  DTYPE_INT8 = 2,
  DTYPE_UINT8 = 3,
  DTYPE_INT16 = 4,
  DTYPE_UINT16 = 5,
  DTYPE_INT32 = 6,
  DTYPE_UINT32 = 7,
  DTYPE_UNKNOWN = -1,
} bm_data_type_t;
typedef bm_data_type_t DATA_TYPE_T;

//using fp32 as bool
#define DTYPE_BOOL DTYPE_FP32

typedef enum tensor_type {
  BMNET_NEURON = 0,
  BMNET_COEFF  = 1,
  BMNET_COEFF_NEURON = 2,
  BMNET_COEFF_FC = 3,
  BMNET_COEFF_WINOGRAD = 4,
  BMNET_NEURON_FC = 5,
  BMNET_NEURON_CONST = 6,
  BMNET_NEURON_SHAPE = 7,
  BMNET_NEURON_CPU = 8,
  BMNET_NEURON_ARRAY = 9,
  BMNET_NEURON_FLOW = 10,
  BMNET_NEURON_3IC = 11,
  TENSOR_TYPE_NUM,
  TENSOR_UNKNOWN = -1,
} TENSOR_TYPE_T;

#define NORMAL_TENSOR BMNET_NEURON
#define CONST_TENSOR  BMNET_COEFF_NEURON
#define SHAPE_TENSOR  BMNET_NEURON_SHAPE
#define FLOW_TENSOR   BMNET_NEURON_FLOW
#define OTHER_TENSOR  TENSOR_UNKNOWN
typedef TENSOR_TYPE_T bm_tensor_type_t;

typedef struct {
    bm_tensor_type_t ttype;
    bm_data_type_t dtype;
    const int *shape;
    int dims;
} bm_user_tensor_t;

typedef struct {
    int id;
    int type;
    int input_num;
    char** input_names;
    int output_num;
    char** output_names;
} bm_subnet_info_t;

typedef struct {
  int size;
  float* data;
} BmFloatArray;

typedef struct {
  int size;
  int* data;
} BmIntArray;

typedef struct {
  int scale_num;
  BmFloatArray* scale;
  int zero_point_num;
  BmIntArray* zero_point;
} BmQuantizeInfo;


static inline int get_type_len(DATA_TYPE_T dtype){
    int t=0;
    if(dtype==DTYPE_FP32){
        t = 4;
    } else if(dtype==DTYPE_INT8 || dtype == DTYPE_UINT8){
        t = 1;
    } else if(dtype == DTYPE_FP16 || dtype == DTYPE_INT16 || dtype == DTYPE_UINT16){
        t = 2;
    } else if(dtype == DTYPE_INT32 || dtype == DTYPE_UINT32){
        t = 4;
    }
    return t;
}


#ifdef __cplusplus
} // namespace bmcompiler
#endif

#endif
