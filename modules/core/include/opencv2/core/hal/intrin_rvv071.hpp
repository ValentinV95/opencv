// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Copyright (C) 2015, PingTouGe Semiconductor Co., Ltd., all rights reserved.

#ifndef OPENCV_HAL_INTRIN_RISCVV_HPP
#define OPENCV_HAL_INTRIN_RISCVV_HPP

#include <float.h>
#include <algorithm>
#include "opencv2/core/utility.hpp"

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define CV_SIMD128 1
#define CV_SIMD128_64F 0

#define CV_SIMD512 1
#define CV_SIMD512_64F 0

#ifndef __thead_c906
#define CV_SIMD_ELEM64 1
#else
#define CV_SIMD_ELEM64 0
#endif

//////////// Types ////////////
struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };

    v_uint8x16() {}
    explicit v_uint8x16(vuint8m1_t v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        uchar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = (vuint8m1_t)vle_v_u8m1((unsigned char*)v, 16);
    }
    uchar get0() const
    {
        return vmv_x_s_u8m1_u8(val, 16);
    }

    vuint8m1_t val;
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() {}
    explicit v_int8x16(vint8m1_t v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
               schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        schar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = (vint8m1_t)vle_v_i8m1((schar*)v, 16);
    }
    schar get0() const
    {
        return vmv_x_s_i8m1_i8(val, 16);
    }

    vint8m1_t val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() {}
    explicit v_uint16x8(vuint16m1_t v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        ushort v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = (vuint16m1_t)vle_v_u16m1((unsigned short*)v, 8);
    }
    ushort get0() const
    {
        return vmv_x_s_u16m1_u16(val, 8);
    }

    vuint16m1_t val;
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() {}
    explicit v_int16x8(vint16m1_t v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        short v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = (vint16m1_t)vle_v_i16m1((signed short*)v, 8);
    }
    short get0() const
    {
        return vmv_x_s_i16m1_i16(val, 8);
    }

    vint16m1_t val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4() {}
    explicit v_uint32x4(vuint32m1_t v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        unsigned v[] = {v0, v1, v2, v3};
        val = (vuint32m1_t)vle_v_u32m1((unsigned int*)v, 4);
    }
    unsigned get0() const
    {
        return vmv_x_s_u32m1_u32(val, 4);
    }

    vuint32m1_t val;
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() {}
    explicit v_int32x4(vint32m1_t v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        int v[] = {v0, v1, v2, v3};
        val = (vint32m1_t)vle_v_i32m1((signed int*)v, 4);
    }
    int get0() const
    {
        return vmv_x_s_i32m1_i32(val, 4);
    }
    vint32m1_t val;
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4() {}
    explicit v_float32x4(vfloat32m1_t v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        float v[] = {v0, v1, v2, v3};
        val = (vfloat32m1_t)vle_v_f32m1((float*)v, 4);
    }
    float get0() const
    {
        return vfmv_f_s_f32m1_f32(val, 4);
    }
    vfloat32m1_t val;
};

#if CV_SIMD_ELEM64
struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() {}
    explicit v_uint64x2(vuint64m1_t v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        uint64 v[] = {v0, v1};
        val = (vuint64m1_t)vle_v_u64m1((unsigned long*)v, 2);
    }
    uint64 get0() const
    {
        return vmv_x_s_u64m1_u64(val, 2);
    }
    vuint64m1_t val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() {}
    explicit v_int64x2(vint64m1_t v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        int64 v[] = {v0, v1};
        val = (vint64m1_t)vle_v_i64m1((long*)v, 2);
    }
    int64 get0() const
    {
        return vmv_x_s_i64m1_i64(val, 2);
    }
    vint64m1_t val;
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() {}
    explicit v_float64x2(vfloat64m1_t v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        double v[] = {v0, v1};
        val = (vfloat64m1_t)vle_v_f64m1((double*)v, 2);
    }
    double get0() const
    {
        return vfmv_f_s_f64m1_f64(val, 2);
    }
    vfloat64m1_t val;
};
#else
struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() {}
    //    explicit v_uint64x2(vuint64m1_t v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        val[0] = v0;
        val[1] = v1;
    }
    uint64 get0() const
    {
        //        return vmv_x_s_u64m1_u64(val, 2);
        return val[0];
    }
    uint64 val[2] = { 0 };
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() {}
    v_int64x2(int64 v0, int64 v1)
    {
        val[0] = v0;
        val[1] = v1;
    }
    int64 get0() const
    {
        return val[0];
    }
    int64 val[2] = { 0 };
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() {}
    v_float64x2(double v0, double v1)
    {
        val[0] = v0;
        val[1] = v1;
    }
    double get0() const
    {
        return val[0];
    }
    double val[2] = {0};
};
#endif

//////////// Types 512B ////////////
struct v_uint8x64
{
    typedef uchar lane_type;
    enum { nlanes = 64 };

    v_uint8x64() {}
    explicit v_uint8x64(vuint8m4_t v) : val(v) {}
    v_uint8x64(uchar v0, uchar v1, uchar v2, uchar v3,
        uchar v4, uchar v5, uchar v6, uchar v7,
        uchar v8, uchar v9, uchar v10, uchar v11,
        uchar v12, uchar v13, uchar v14, uchar v15,
        uchar v16, uchar v17, uchar v18, uchar v19,
        uchar v20, uchar v21, uchar v22, uchar v23,
        uchar v24, uchar v25, uchar v26, uchar v27,
        uchar v28, uchar v29, uchar v30, uchar v31,
        uchar v32, uchar v33, uchar v34, uchar v35,
        uchar v36, uchar v37, uchar v38, uchar v39,
        uchar v40, uchar v41, uchar v42, uchar v43,
        uchar v44, uchar v45, uchar v46, uchar v47,
        uchar v48, uchar v49, uchar v50, uchar v51,
        uchar v52, uchar v53, uchar v54, uchar v55,
        uchar v56, uchar v57, uchar v58, uchar v59,
        uchar v60, uchar v61, uchar v62, uchar v63)
    {
        uchar v[] = { v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
         v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31,
         v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47,
         v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63 };
        val = (vuint8m4_t)vle_v_u8m4((unsigned char*)v, 64);
    } 
    uchar get0() const
    {
        return vmv_x_s_u8m4_u8(val, 64);
    }

    vuint8m4_t val;
};

struct v_int8x64
{
    typedef schar lane_type;
    enum { nlanes = 64 };

    v_int8x64() {}
    explicit v_int8x64(vint8m4_t v) : val(v) {}
    v_int8x64(schar v0, schar v1, schar v2, schar v3,
        schar v4, schar v5, schar v6, schar v7,
        schar v8, schar v9, schar v10, schar v11,
        schar v12, schar v13, schar v14, schar v15,
        schar v16, schar v17, schar v18, schar v19,
        schar v20, schar v21, schar v22, schar v23,
        schar v24, schar v25, schar v26, schar v27,
        schar v28, schar v29, schar v30, schar v31,
        schar v32, schar v33, schar v34, schar v35,
        schar v36, schar v37, schar v38, schar v39,
        schar v40, schar v41, schar v42, schar v43,
        schar v44, schar v45, schar v46, schar v47,
        schar v48, schar v49, schar v50, schar v51,
        schar v52, schar v53, schar v54, schar v55,
        schar v56, schar v57, schar v58, schar v59,
        schar v60, schar v61, schar v62, schar v63)
    {
        schar v[] = { v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
         v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31,
         v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47,
         v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63 };
        val = (vint8m4_t)vle_v_i8m4((schar*)v, 64);
    }
    schar get0() const
    {
        return vmv_x_s_i8m4_i8(val, 64);
    }

    vint8m4_t val;
};

struct v_uint16x32
{
    typedef ushort lane_type;
    enum { nlanes = 32 };

    v_uint16x32() {}
    explicit v_uint16x32(vuint16m4_t v) : val(v) {}
    v_uint16x32(ushort v0, ushort v1, ushort v2, ushort v3,
        ushort v4, ushort v5, ushort v6, ushort v7,
        ushort v8, ushort v9, ushort v10, ushort v11,
        ushort v12, ushort v13, ushort v14, ushort v15,
        ushort v16, ushort v17, ushort v18, ushort v19,
        ushort v20, ushort v21, ushort v22, ushort v23,
        ushort v24, ushort v25, ushort v26, ushort v27,
        ushort v28, ushort v29, ushort v30, ushort v31)
    {
        ushort v[] = { v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
         v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31 };
        val = (vuint16m4_t)vle_v_u16m4((unsigned short*)v, 32);
    }
    ushort get0() const
    {
        return vmv_x_s_u16m4_u16(val, 32);
    }

    vuint16m4_t val;
};

struct v_int16x32
{
    typedef short lane_type;
    enum { nlanes = 32 };

    v_int16x32() {}
    explicit v_int16x32(vint16m4_t v) : val(v) {}
    v_int16x32(short v0, short v1, short v2, short v3,
        short v4, short v5, short v6, short v7,
        short v8, short v9, short v10, short v11,
        short v12, short v13, short v14, short v15,
        short v16, short v17, short v18, short v19,
        short v20, short v21, short v22, short v23,
        short v24, short v25, short v26, short v27,
        short v28, short v29, short v30, short v31)
    {
        short v[] = {  v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
         v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31 };
        val = (vint16m4_t)vle_v_i16m4((signed short*)v, 32);
    }
    short get0() const
    {
        return vmv_x_s_i16m4_i16(val, 32);
    }

    vint16m4_t val;
};

struct v_uint32x16
{
    typedef unsigned lane_type;
    enum { nlanes = 16 };

    v_uint32x16() {}
    explicit v_uint32x16(vuint32m4_t v) : val(v) {}
    v_uint32x16(unsigned v0, unsigned v1, unsigned v2, unsigned v3, unsigned v4, unsigned v5, unsigned v6, unsigned v7,
        unsigned v8, unsigned v9, unsigned v10, unsigned v11, unsigned v12, unsigned v13, unsigned v14, unsigned v15)
    {
        unsigned v[] = { v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 };
        val = (vuint32m4_t)vle_v_u32m4((unsigned int*)v, 16);
    }
    unsigned get0() const
    {
        return vmv_x_s_u32m4_u32(val, 16);
    }

    vuint32m4_t val;
};

struct v_int32x16
{
    typedef int lane_type;
    enum { nlanes = 16 };

    v_int32x16() {}
    explicit v_int32x16(vint32m4_t v) : val(v) {}
    v_int32x16(int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7,
        int v8, int v9, int v10, int v11, int v12, int v13, int v14, int v15)
    {
        int v[] = { v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 };
        val = (vint32m4_t)vle_v_i32m4((signed int*)v, 16);
    }
    int get0() const
    {
        return vmv_x_s_i32m4_i32(val, 16);
    }
    vint32m4_t val;
};

struct v_float32x16
{
    typedef float lane_type;
    enum { nlanes = 16 };

    v_float32x16() {}
    explicit v_float32x16(vfloat32m4_t v) : val(v) {}
    v_float32x16(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
        float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15)
    {
        float v[] = { v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 };
        val = (vfloat32m4_t)vle_v_f32m4((float*)v, 16);
    }
    float get0() const
    {
        return vfmv_f_s_f32m4_f32(val, 16);
    }
    vfloat32m4_t val;
};

#if CV_SIMD_ELEM64
struct v_uint64x8
{
    typedef uint64 lane_type;
    enum { nlanes = 8 };

    v_uint64x8() {}
    explicit v_uint64x8(vuint64m4_t v) : val(v) {}
    v_uint64x8(uint64 v0, uint64 v1, uint64 v2, uint64 v3, uint64 v4, uint64 v5, uint64 v6, uint64 v7)
    {
        uint64 v[] = { v0, v1, v2, v3, v4, v5, v6, v7 };
        val = (vuint64m4_t)vle_v_u64m4((unsigned long*)v, 8);
    }
    uint64 get0() const
    {
        return vmv_x_s_u64m4_u64(val, 8);
    }
    vuint64m4_t val;
};

struct v_int64x8
{
    typedef int64 lane_type;
    enum { nlanes = 8 };

    v_int64x8() {}
    explicit v_int64x8(vint64m4_t v) : val(v) {}
    v_int64x8(int64 v0, int64 v1, int64 v2, int64 v3, int64 v4, int64 v5, int64 v6, int64 v7)
    {
        int64 v[] = { v0, v1, v2, v3, v4, v5, v6, v7 };
        val = (vint64m4_t)vle_v_i64m4((long*)v, 8);
    }
    int64 get0() const
    {
        return vmv_x_s_i64m4_i64(val, 8);
    }
    vint64m4_t val;
};

struct v_float64x8
{
    typedef double lane_type;
    enum { nlanes = 8 };

    v_float64x8() {}
    explicit v_float64x8(vfloat64m4_t v) : val(v) {}
    v_float64x8(double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7)
    {
        double v[] = { v0, v1, v2, v3, v4, v5, v6, v7 };
        val = (vfloat64m4_t)vle_v_f64m4((double*)v, 8);
    }
    double get0() const
    {
        return vfmv_f_s_f64m4_f64(val, 8);
    }
    vfloat64m4_t val;
};
#else
struct v_uint64x8
{
    typedef uint64 lane_type;
    enum { nlanes = 8 };

    v_uint64x8() {}
    v_uint64x8(uint64 v0, uint64 v1, uint64 v2, uint64 v3,
               uint64 v4, uint64 v5, uint64 v6, uint64 v7)
    {
        uint64 v[] = { v0, v1, v2, v3, v4, v5, v6, v7 };
        for(int i =0; i <8; i++) val[i] = v[i];
    }
    uint64 get0() const
    {
        return val[0];
    }
    uint64 val[8];
};

struct v_int64x8
{
    typedef int64 lane_type;
    enum { nlanes = 8 };

    v_int64x8() {}
//    explicit v_int64x8(vint64m4_t v) : val(v) {}
    v_int64x8(int64 v0, int64 v1, int64 v2, int64 v3,
              int64 v4, int64 v5, int64 v6, int64 v7)
    {
        int64 v[] = { v0, v1, v2, v3, v4, v5, v6, v7 };
        for (int i = 0; i < 8; i++) val[i] = v[i];
    }
    int64 get0() const
    {
        return val[0];
    }
    int64 val[8];
};

struct v_float64x8
{
    typedef double lane_type;
    enum { nlanes = 8 };

    v_float64x8() {}
    v_float64x8(double v0, double v1, double v2, double v3, double v4,
                double v5, double v6, double v7)
    {
        double v[] = { v0, v1, v2, v3, v4, v5, v6, v7 };
        for (int i = 0; i < 8; i++) val[i] = v[i];
    }
    double get0() const
    {
        return val[0];
    }
    double val[8];
};
#endif

#define OPENCV_HAL_IMPL_RISCVV_GETSET(_Tp, _T) \
inline _Tp##m2_t vget_##_T##m4_##_T##m2(_Tp##m4_t v, const int32_t index) \
{ \
    _Tp##m2_t res = vundefined_##_T##m2(); \
    res = vset_##_T##m2(res, 0, vget_##_T##m4_##_T##m1(v, 2*index)); \
    res = vset_##_T##m2(res, 1, vget_##_T##m4_##_T##m1(v, 2*index + 1)); \
    return res; \
} \
inline _Tp##m4_t vget_##_T##m8_##_T##m4(_Tp##m8_t v, const int32_t index) \
{ \
    _Tp##m4_t res = vundefined_##_T##m4(); \
    res = vset_##_T##m4(res, 0, vget_##_T##m8_##_T##m1(v, 4*index)); \
    res = vset_##_T##m4(res, 1, vget_##_T##m8_##_T##m1(v, 4*index + 1)); \
    res = vset_##_T##m4(res, 2, vget_##_T##m8_##_T##m1(v, 4*index + 2)); \
    res = vset_##_T##m4(res, 3, vget_##_T##m8_##_T##m1(v, 4*index + 3)); \
    return res; \
} \
inline _Tp##m4_t vset_##_T##m4_##_T##m2(_Tp##m4_t v, const int32_t index, _Tp##m2_t val) \
{ \
    v = vset_##_T##m4(v, 2*index, vget_##_T##m2_##_T##m1(val, 0)); \
    v = vset_##_T##m4(v, 2*index + 1, vget_##_T##m2_##_T##m1(val, 1)); \
    return v; \
} \
inline _Tp##m4_t vset_##_T##m4_##_T##m1(_Tp##m4_t v, const int32_t index, _Tp##m1_t val) \
{ \
    return vset_##_T##m4(v, index, val); \
} \
inline _Tp##m8_t vset_##_T##m8_##_T##m4(_Tp##m8_t v, const int32_t index, _Tp##m4_t val) \
{ \
    v = vset_##_T##m8(v, 4*index,     vget_##_T##m4_##_T##m1(val, 0)); \
    v = vset_##_T##m8(v, 4*index + 1, vget_##_T##m4_##_T##m1(val, 1)); \
    v = vset_##_T##m8(v, 4*index + 2, vget_##_T##m4_##_T##m1(val, 2)); \
    v = vset_##_T##m8(v, 4*index + 3, vget_##_T##m4_##_T##m1(val, 3)); \
    return v; \
} \
inline _Tp##m2_t vset_##_T##m2_##_T##m1(_Tp##m2_t v, const int32_t index, _Tp##m1_t val) \
{ \
    return vset_##_T##m2(v, index, val); \
}

OPENCV_HAL_IMPL_RISCVV_GETSET(vuint8,  u8)
OPENCV_HAL_IMPL_RISCVV_GETSET(vint8,   i8)
OPENCV_HAL_IMPL_RISCVV_GETSET(vuint16,  u16)
OPENCV_HAL_IMPL_RISCVV_GETSET(vint16,   i16)
OPENCV_HAL_IMPL_RISCVV_GETSET(vuint32,  u32)
OPENCV_HAL_IMPL_RISCVV_GETSET(vint32,   i32)
OPENCV_HAL_IMPL_RISCVV_GETSET(vuint64,  u64)
OPENCV_HAL_IMPL_RISCVV_GETSET(vint64,   i64)
OPENCV_HAL_IMPL_RISCVV_GETSET(vfloat32, f32)
OPENCV_HAL_IMPL_RISCVV_GETSET(vfloat64, f64)


#define OPENCV_HAL_IMPL_RISCVV_INIT(_Tpv, _Tp, suffix) \
inline _Tp##m1_t vreinterpretq_##suffix##_##suffix(_Tp##m1_t v) { return v; } \
inline v_uint8x16 v_reinterpret_as_u8(const v_##_Tpv& v) { return v_uint8x16((vuint8m1_t)(v.val)); } \
inline v_int8x16 v_reinterpret_as_s8(const v_##_Tpv& v) { return v_int8x16((vint8m1_t)(v.val)); } \
inline v_uint16x8 v_reinterpret_as_u16(const v_##_Tpv& v) { return v_uint16x8((vuint16m1_t)(v.val)); } \
inline v_int16x8 v_reinterpret_as_s16(const v_##_Tpv& v) { return v_int16x8((vint16m1_t)(v.val)); } \
inline v_uint32x4 v_reinterpret_as_u32(const v_##_Tpv& v) { return v_uint32x4((vuint32m1_t)(v.val)); } \
inline v_int32x4 v_reinterpret_as_s32(const v_##_Tpv& v) { return v_int32x4((vint32m1_t)(v.val)); } \
inline v_uint64x2 v_reinterpret_as_u64(const v_##_Tpv& v) { return v_uint64x2((vuint64m1_t)(v.val)); } \
inline v_int64x2 v_reinterpret_as_s64(const v_##_Tpv& v) { return v_int64x2((vint64m1_t)(v.val)); } \
inline v_float32x4 v_reinterpret_as_f32(const v_##_Tpv& v) { return v_float32x4((vfloat32m1_t)(v.val)); }\
inline v_float64x2 v_reinterpret_as_f64(const v_##_Tpv& v) { return v_float64x2((vfloat64m1_t)(v.val)); }

#define OPENCV_HAL_IMPL_RISCVV_INIT_512(_Tpv, _Tp, suffix) \
inline _Tp##m4_t vreinterpretq_##suffix##_##suffix(_Tp##m4_t v) { return v; } \
inline v_uint8x64 v_reinterpret_as_u8(const v_##_Tpv& v) { return v_uint8x64((vuint8m4_t)(v.val)); } \
inline v_int8x64 v_reinterpret_as_s8(const v_##_Tpv& v) { return v_int8x64((vint8m4_t)(v.val)); } \
inline v_uint16x32 v_reinterpret_as_u16(const v_##_Tpv& v) { return v_uint16x32((vuint16m4_t)(v.val)); } \
inline v_int16x32 v_reinterpret_as_s16(const v_##_Tpv& v) { return v_int16x32((vint16m4_t)(v.val)); } \
inline v_uint32x16 v_reinterpret_as_u32(const v_##_Tpv& v) { return v_uint32x16((vuint32m4_t)(v.val)); } \
inline v_int32x16 v_reinterpret_as_s32(const v_##_Tpv& v) { return v_int32x16((vint32m4_t)(v.val)); } \
inline v_uint64x8 v_reinterpret_as_u64(const v_##_Tpv& v) { return v_uint64x8((vuint64m4_t)(v.val)); } \
inline v_int64x8 v_reinterpret_as_s64(const v_##_Tpv& v) { return v_int64x8((vint64m4_t)(v.val)); } \
inline v_float32x16 v_reinterpret_as_f32(const v_##_Tpv& v) { return v_float32x16((vfloat32m4_t)(v.val)); }\
inline v_float64x8 v_reinterpret_as_f64(const v_##_Tpv& v) { return v_float64x8((vfloat64m4_t)(v.val)); }


OPENCV_HAL_IMPL_RISCVV_INIT(uint8x16, vuint8, u8)
OPENCV_HAL_IMPL_RISCVV_INIT(int8x16, vint8, s8)
OPENCV_HAL_IMPL_RISCVV_INIT(uint16x8, vuint16, u16)
OPENCV_HAL_IMPL_RISCVV_INIT(int16x8, vint16, s16)
OPENCV_HAL_IMPL_RISCVV_INIT(uint32x4, vuint32, u32)
OPENCV_HAL_IMPL_RISCVV_INIT(int32x4, vint32, s32)
OPENCV_HAL_IMPL_RISCVV_INIT(uint64x2, vuint64, u64)
OPENCV_HAL_IMPL_RISCVV_INIT(int64x2, vint64, s64)
OPENCV_HAL_IMPL_RISCVV_INIT(float64x2, vfloat64, f64)
OPENCV_HAL_IMPL_RISCVV_INIT(float32x4, vfloat32, f32)

OPENCV_HAL_IMPL_RISCVV_INIT_512(uint8x64, vuint8, u8)
OPENCV_HAL_IMPL_RISCVV_INIT_512(int8x64, vint8, s8)
OPENCV_HAL_IMPL_RISCVV_INIT_512(uint16x32, vuint16, u16)
OPENCV_HAL_IMPL_RISCVV_INIT_512(int16x32, vint16, s16)
OPENCV_HAL_IMPL_RISCVV_INIT_512(uint32x16, vuint32, u32)
OPENCV_HAL_IMPL_RISCVV_INIT_512(int32x16, vint32, s32)
OPENCV_HAL_IMPL_RISCVV_INIT_512(uint64x8, vuint64, u64)
OPENCV_HAL_IMPL_RISCVV_INIT_512(int64x8, vint64, s64)
OPENCV_HAL_IMPL_RISCVV_INIT_512(float64x8, vfloat64, f64)
OPENCV_HAL_IMPL_RISCVV_INIT_512(float32x16, vfloat32, f32)

#define OPENCV_HAL_IMPL_RISCVV_INIT_SET(__Tp, _Tp, suffix, len, num) \
inline v_##_Tp##x##num v_setzero_##suffix() { return v_##_Tp##x##num((v##_Tp##m1_t){0}); }     \
inline v_##_Tp##x##num v_setall_##suffix(__Tp v) { return v_##_Tp##x##num(vmv_v_x_##len##m1(v, num)); }

OPENCV_HAL_IMPL_RISCVV_INIT_SET(uchar, uint8, u8, u8, 16)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(char, int8, s8, i8, 16)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(ushort, uint16, u16, u16, 8)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(short, int16, s16, i16, 8)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(unsigned int, uint32, u32, u32, 4)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(int, int32, s32, i32, 4)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(unsigned long, uint64, u64, u64, 2)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(long, int64, s64, i64, 2)
inline v_float32x4 v_setzero_f32() { return v_float32x4((vfloat32m1_t){0}); }
inline v_float32x4 v_setall_f32(float v) { return v_float32x4(vfmv_v_f_f32m1(v, 4)); }

inline v_float64x2 v_setzero_f64() { return v_float64x2(vfmv_v_f_f64m1(0, 2)); }
inline v_float64x2 v_setall_f64(double v) { return v_float64x2(vfmv_v_f_f64m1(v, 2)); }

#define OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(__Tp, _Tp, suffix, len, num) \
inline v_##_Tp##x##num v512_setzero_##suffix() { return v_##_Tp##x##num((v##_Tp##m4_t){0}); }     \
inline v_##_Tp##x##num v512_setall_##suffix(__Tp v) { return v_##_Tp##x##num(vmv_v_x_##len##m4(v, num)); }

OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(uchar, uint8, u8, u8, 64)
OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(char, int8, s8, i8, 64)
OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(ushort, uint16, u16, u16, 32)
OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(short, int16, s16, i16, 32)
OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(unsigned int, uint32, u32, u32, 16)
OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(int, int32, s32, i32, 16)
OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(unsigned long, uint64, u64, u64, 8)
OPENCV_HAL_IMPL_RISCVV_INIT_SET_512(long, int64, s64, i64, 8)

inline v_float32x16 v512_setall_f32(float v) { return v_float32x16(vfmv_v_f_f32m4(v, 16)); }
inline v_float32x16 v512_setzero_f32() { return v_float32x16((vfloat32m4_t){0}); }

inline v_float64x8 v512_setzero_f64() { return v_float64x8(vfmv_v_f_f64m4(0, 8)); }
inline v_float64x8 v512_setall_f64(double v) { return v_float64x8(vfmv_v_f_f64m4(v, 8)); }


#define OPENCV_HAL_IMPL_RISCVV_BIN_OP(bin_op, _Tpvec, intrin) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
} \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
{ \
    a.val = intrin(a.val, b.val); \
    return a; \
}

#define OPENCV_HAL_IMPL_RISCVV_BIN_OPN(bin_op, _Tpvec, intrin, num) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val, num)); \
} \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
{ \
    a.val = intrin(a.val, b.val, num); \
    return a; \
}

OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint8x16, vsaddu_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint8x16, vssubu_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int8x16, vsadd_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int8x16, vssub_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint16x8, vsaddu_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint16x8, vssubu_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int16x8, vsadd_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int16x8, vssub_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int32x4, vadd_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int32x4, vsub_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_int32x4, vmul_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint32x4, vadd_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint32x4, vsub_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_uint32x4, vmul_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int64x2, vadd_vv_i64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int64x2, vsub_vv_i64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint64x2, vadd_vv_u64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint64x2, vsub_vv_u64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_float32x4, vfadd_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_float32x4, vfsub_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_float32x4, vfmul_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(/, v_float32x4, vfdiv_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_float64x2, vfadd_vv_f64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_float64x2, vfsub_vv_f64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_float64x2, vfmul_vv_f64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(/, v_float64x2, vfdiv_vv_f64m1, 2)

//512
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint8x64, vsaddu_vv_u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint8x64, vssubu_vv_u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int8x64, vsadd_vv_i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int8x64, vssub_vv_i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint16x32, vsaddu_vv_u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint16x32, vssubu_vv_u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int16x32, vsadd_vv_i16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int16x32, vssub_vv_i16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int32x16, vadd_vv_i32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int32x16, vsub_vv_i32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_int32x16, vmul_vv_i32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint32x16, vadd_vv_u32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint32x16, vsub_vv_u32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_uint32x16, vmul_vv_u32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int64x8, vadd_vv_i64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int64x8, vsub_vv_i64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint64x8, vadd_vv_u64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint64x8, vsub_vv_u64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_float32x16, vfadd_vv_f32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_float32x16, vfsub_vv_f32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_float32x16, vfmul_vv_f32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(/, v_float32x16, vfdiv_vv_f32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_float64x8, vfadd_vv_f64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_float64x8, vfsub_vv_f64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_float64x8, vfmul_vv_f64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(/, v_float64x8, vfdiv_vv_f64m4, 8)

// TODO: exp, log, sin, cos

#define OPENCV_HAL_IMPL_RISCVV_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}

#define OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(_Tpvec, func, intrin, num) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val, num)); \
}

OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_min, vminu_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_max, vmaxu_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_min, vmin_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_max, vmax_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_min, vminu_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_max, vmaxu_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_min, vmin_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_max, vmax_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint32x4, v_min, vminu_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint32x4, v_max, vmaxu_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int32x4, v_min, vmin_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int32x4, v_max, vmax_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float32x4, v_min, vfmin_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float32x4, v_max, vfmax_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float64x2, v_min, vfmin_vv_f64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float64x2, v_max, vfmax_vv_f64m1, 2)


// 512
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x64, v_min, vminu_vv_u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x64, v_max, vmaxu_vv_u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x64, v_min, vmin_vv_i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x64, v_max, vmax_vv_i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x32, v_min, vminu_vv_u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x32, v_max, vmaxu_vv_u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x32, v_min, vmin_vv_i16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x32, v_max, vmax_vv_i16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint32x16, v_min, vminu_vv_u32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint32x16, v_max, vmaxu_vv_u32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int32x16, v_min, vmin_vv_i32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int32x16, v_max, vmax_vv_i32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float32x16, v_min, vfmin_vv_f32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float32x16, v_max, vfmax_vv_f32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float64x8, v_min, vfmin_vv_f64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float64x8, v_max, vfmax_vv_f64m4, 8)

#define OPENCV_HAL_IMPL_RISCVV_FMA_FUNC(_Tpvec, num, suffix, preffix) \
inline _Tpvec v_fma(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
{ \
    return _Tpvec(v##preffix##cc_vv_##suffix(c.val, a.val, b.val, num)); \
} \
inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
{     return v_fma(a, b, c); }

OPENCV_HAL_IMPL_RISCVV_FMA_FUNC(v_float32x4, 4, f32m1, fma)
OPENCV_HAL_IMPL_RISCVV_FMA_FUNC(v_int32x4, 4, i32m1, ma)
OPENCV_HAL_IMPL_RISCVV_FMA_FUNC(v_float64x2, 2, f64m1, fma)

//512
OPENCV_HAL_IMPL_RISCVV_FMA_FUNC(v_float32x16, 16, f32m4, fma)
OPENCV_HAL_IMPL_RISCVV_FMA_FUNC(v_int32x16, 16, i32m4, ma)
OPENCV_HAL_IMPL_RISCVV_FMA_FUNC(v_float64x8, 8, f64m4, fma)


#define OPENCV_HAL_IMPL_RISCVV_SPEC_FUNC(_Tpvec, num, suffix) \
inline _Tpvec v_sqrt(const _Tpvec& x) \
{   return _Tpvec(vfsqrt_v_##suffix(x.val, num)); } \
inline _Tpvec v_invsqrt(const _Tpvec& x) \
{    return _Tpvec(vfrdiv_vf_##suffix(vfsqrt_v_##suffix(x.val, num), 1, num)); } \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b) \
{    return v_fma(a, a, b * b); } \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b) \
{    return v_sqrt(v_fma(a, a, b * b));} \
    


OPENCV_HAL_IMPL_RISCVV_SPEC_FUNC(v_float32x4, 4, f32m1)
OPENCV_HAL_IMPL_RISCVV_SPEC_FUNC(v_float64x2, 2, f64m1)

//512
OPENCV_HAL_IMPL_RISCVV_SPEC_FUNC(v_float32x16, 16, f32m4)
OPENCV_HAL_IMPL_RISCVV_SPEC_FUNC(v_float64x8, 8, f64m4)

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    vfloat32m1_t res = vfmul_vf_f32m1(m0.val, v.val[0], 4);//vmuli_f32(m0.val, v.val, 0);
    res = vfmacc_vf_f32m1(res, v.val[1], m1.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m1(res, v.val[2], m2.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m1(res, v.val[3], m3.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    return v_float32x4(res);
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    vfloat32m1_t res = vfmul_vf_f32m1(m0.val, v.val[0], 4);//vmuli_f32(m0.val, v.val, 0);
    res = vfmacc_vf_f32m1(res, v.val[1], m1.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m1(res, v.val[2], m2.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfadd_vv_f32m1(res, a.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    return v_float32x4(res);
}

//512
inline v_float32x16 v_matmul(const v_float32x16& v, const v_float32x16& m0,
                            const v_float32x16& m1, const v_float32x16& m2,
                            const v_float32x16& m3)
{
    vfloat32m4_t res = vfmul_vf_f32m4(m0.val, v.val[0], 16);//vmuli_f32(m0.val, v.val, 0);
    res = vfmacc_vf_f32m4(res, v.val[1], m1.val, 16);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m4(res, v.val[2], m2.val, 16);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m4(res, v.val[3], m3.val, 16);//vmulai_f32(res, m1.val, v.val, 1);
    return v_float32x16(res);
}

inline v_float32x16 v_matmuladd(const v_float32x16& v, const v_float32x16& m0,
                               const v_float32x16& m1, const v_float32x16& m2,
                               const v_float32x16& a)
{
    vfloat32m4_t res = vfmul_vf_f32m4(m0.val, v.val[0], 16);//vmuli_f32(m0.val, v.val, 0);
    res = vfmacc_vf_f32m4(res, v.val[1], m1.val, 16);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m4(res, v.val[2], m2.val, 16);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfadd_vv_f32m4(res, a.val, 16);//vmulai_f32(res, m1.val, v.val, 1);
    return v_float32x16(res);
}


#define OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(_Tpvec, suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(&, _Tpvec, vand_vv_##suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(|, _Tpvec, vor_vv_##suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(^, _Tpvec, vxor_vv_##suffix, num) \
    inline _Tpvec operator ~ (const _Tpvec & a) \
    { \
        return _Tpvec(vnot_v_##suffix(a.val, num)); \
    }

OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint8x16, u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint16x8, u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint32x4, u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint64x2, u64m1, 2)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int8x16,  i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int16x8,  i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int32x4,  i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int64x2,  i64m1, 2)

//512
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint8x64, u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint16x32, u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint32x16, u32m4, 16)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint64x8, u64m4, 8)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int8x64,  i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int16x32,  i16m4, 32)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int32x16,  i32m4, 16)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int64x8,  i64m4, 8)

#define OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(bin_op, _Tpvec, intrin, suffix, num) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(vfloat##suffix##_t(intrin(vint##suffix##_t(a.val), vint##suffix##_t(b.val), num))); \
} \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
{ \
    a.val = vfloat##suffix##_t(intrin(vint##suffix##_t(a.val), vint##suffix##_t(b.val), num)); \
    return a; \
}

OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(&, v_float32x4, vand_vv_i32m1, 32m1, 4)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(|, v_float32x4, vor_vv_i32m1, 32m1, 4)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(^, v_float32x4, vxor_vv_i32m1, 32m1, 4)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(&, v_float64x2, vand_vv_i64m1, 64m1, 2)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(|, v_float64x2, vor_vv_i64m1, 64m1, 2)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(^, v_float64x2, vxor_vv_i64m1, 64m1, 2)

//512
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(&, v_float32x16, vand_vv_i32m4, 32m4, 16)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(|, v_float32x16, vor_vv_i32m4, 32m4, 16)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(^, v_float32x16, vxor_vv_i32m4, 32m4, 16)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(&, v_float64x8, vand_vv_i64m4, 64m4, 8)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(|, v_float64x8, vor_vv_i64m4, 64m4, 8)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(^, v_float64x8, vxor_vv_i64m4, 64m4, 8)

#define OPENCV_HAL_IMPL_RISCVV_FLT_NOT_OP(_Tpvec, suffix, num) \
inline _Tpvec operator ~ (const _Tpvec& a) \
{   return _Tpvec(vfloat##suffix##_t(vnot_v_i##suffix(vint##suffix##_t(a.val), num))); }

OPENCV_HAL_IMPL_RISCVV_FLT_NOT_OP(v_float32x4, 32m1, 4)
OPENCV_HAL_IMPL_RISCVV_FLT_NOT_OP(v_float64x2, 64m1, 2)

//512
OPENCV_HAL_IMPL_RISCVV_FLT_NOT_OP(v_float32x16, 32m4, 16)
OPENCV_HAL_IMPL_RISCVV_FLT_NOT_OP(v_float64x8, 64m4, 8)

inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{
    return v_int16x8(vmulh_vv_i16m1(a.val, b.val, 8));
}
inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(vmulhu_vv_u16m1(a.val, b.val, 8));
}

//512
inline v_int16x32 v_mul_hi(const v_int16x32& a, const v_int16x32& b)
{
    return v_int16x32(vmulh_vv_i16m4(a.val, b.val, 32));
}
inline v_uint16x32 v_mul_hi(const v_uint16x32& a, const v_uint16x32& b)
{
    return v_uint16x32(vmulhu_vv_u16m4(a.val, b.val, 32));
}

#define OPENCV_HAL_IMPL_RISCVV_INT_ABS_ABSDIFF(_Tpuvec, _Tpsvec, _Tpv, bsuffix, num) \
inline _Tpuvec v_abs(const _Tpsvec& x) \
{ \
    vbool##bsuffix##_t mask=vmslt_vx_i##_Tpv##_b##bsuffix(x.val, 0, num); \
    return _Tpuvec((vuint##_Tpv##_t)vrsub_vx_i##_Tpv##_m(mask, x.val, x.val, 0, num)); \
} \
inline _Tpuvec v_absdiff(const _Tpuvec& a, const _Tpuvec& b){    \
    vuint##_Tpv##_t vmax = vmaxu_vv_u##_Tpv(a.val, b.val, num);    \
    vuint##_Tpv##_t vmin = vminu_vv_u##_Tpv(a.val, b.val, num);    \
    return _Tpuvec(vsub_vv_u##_Tpv(vmax, vmin, num));\
} \
inline _Tpsvec v_absdiffs(const _Tpsvec& a, const _Tpsvec& b){ \
    vint##_Tpv##_t vmax = vmax_vv_i##_Tpv(a.val, b.val, num); \
    vint##_Tpv##_t vmin = vmin_vv_i##_Tpv(a.val, b.val, num); \
    return _Tpsvec(vssub_vv_i##_Tpv(vmax, vmin, num)); \
} \
inline _Tpuvec v_absdiff(const _Tpsvec& a, const _Tpsvec& b){    \
     vint##_Tpv##_t max = vmax_vv_i##_Tpv(a.val, b.val, num);\
     vint##_Tpv##_t min = vmin_vv_i##_Tpv(a.val, b.val, num);\
    return _Tpuvec((vuint##_Tpv##_t)vsub_vv_i##_Tpv(max, min, num));    \
}

OPENCV_HAL_IMPL_RISCVV_INT_ABS_ABSDIFF(v_uint32x4, v_int32x4, 32m1, 32, 4)
OPENCV_HAL_IMPL_RISCVV_INT_ABS_ABSDIFF(v_uint16x8, v_int16x8, 16m1, 16, 8)
OPENCV_HAL_IMPL_RISCVV_INT_ABS_ABSDIFF(v_uint8x16, v_int8x16, 8m1, 8, 16)

//512
OPENCV_HAL_IMPL_RISCVV_INT_ABS_ABSDIFF(v_uint32x16, v_int32x16, 32m4, 8, 16)
OPENCV_HAL_IMPL_RISCVV_INT_ABS_ABSDIFF(v_uint16x32, v_int16x32, 16m4, 4, 32)
OPENCV_HAL_IMPL_RISCVV_INT_ABS_ABSDIFF(v_uint8x64, v_int8x64, 8m4, 2, 64)


#define OPENCV_HAL_IMPL_RISCVV_FLT_ABS_ABSDIFF(_Tpvec, _Tpv, num) \
inline _Tpvec v_abs(const _Tpvec& x) \
{ \
    return (_Tpvec)vfsgnjx_vv_f##_Tpv(x.val, x.val, num); \
} \
inline _Tpvec v_absdiff(const _Tpvec& a, const _Tpvec& b) \
{   return v_abs(a - b); } 

OPENCV_HAL_IMPL_RISCVV_FLT_ABS_ABSDIFF(v_float32x4, 32m1, 4)
OPENCV_HAL_IMPL_RISCVV_FLT_ABS_ABSDIFF(v_float64x2, 64m1, 2)

//512
OPENCV_HAL_IMPL_RISCVV_FLT_ABS_ABSDIFF(v_float32x16, 32m4, 16)
OPENCV_HAL_IMPL_RISCVV_FLT_ABS_ABSDIFF(v_float64x8, 64m4, 8)




//  Multiply and expand
#define OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND(_Tpvec, _Tpwvec, _Tpwvr, _Tpv, _Tpwv, num, op) \
inline void v_mul_expand(const _Tpvec& a, const _Tpvec& b, \
                         _Tpwvec& c, _Tpwvec& d) \
{ \
    _Tpwvr res = vundefined_##_Tpwv(); \
    res = vw##op##_vv_##_Tpwv(a.val, b.val, num); \
    c.val = vget_##_Tpwv##_##_Tpv(res, 0); \
    d.val = vget_##_Tpwv##_##_Tpv(res, 1); \
}

OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND(v_int8x16, v_int16x8, vint16m2_t, i16m1, i16m2, 16, mul)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND(v_uint8x16, v_uint16x8, vuint16m2_t, u16m1, u16m2, 16, mulu)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND(v_int16x8, v_int32x4, vint32m2_t, i32m1, i32m2, 8, mul)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND(v_uint16x8, v_uint32x4, vuint32m2_t, u32m1, u32m2, 8, mulu)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND(v_int32x4, v_int64x2, vint64m2_t, i64m1, i64m2, 4, mul)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND(v_uint32x4, v_uint64x2, vuint64m2_t, u64m1, u64m2, 4, mulu)

//512
#define OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND_512(_Tpvec, _Tpwvec, _Tpwvr, _Tpv, _Tphv, num, op) \
inline void v_mul_expand(const _Tpvec& a, const _Tpvec& b, \
                         _Tpwvec& c, _Tpwvec& d) \
{ \
    c.val = vw##op##_vv_##_Tpv##m4(vget_##_Tphv##m4_##_Tphv##m2(a.val, 0), vget_##_Tphv##m4_##_Tphv##m2(b.val, 0), num / 2); \
    d.val = vw##op##_vv_##_Tpv##m4(vget_##_Tphv##m4_##_Tphv##m2(a.val, 1), vget_##_Tphv##m4_##_Tphv##m2(b.val, 1), num / 2); \
}
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND_512(v_int8x64, v_int16x32, vint16m4_t, i16, i8, 64, mul)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND_512(v_uint8x64, v_uint16x32, vuint16m4_t, u16, u8, 64, mulu)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND_512(v_int16x32, v_int32x16, vint32m4_t, i32, i16, 32, mul)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND_512(v_uint16x32, v_uint32x16, vuint32m4_t, u32, u16, 32, mulu)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND_512(v_int32x16, v_int64x8, vint64m4_t, i64, i32, 16, mul)
OPENCV_HAL_IMPL_RISCVV_MUL_EXPAND_512(v_uint32x16, v_uint64x8, vuint64m4_t, u64, u32, 16, mulu)


OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_add_wrap, vadd_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_add_wrap, vadd_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_add_wrap, vadd_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_add_wrap, vadd_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_sub_wrap, vsub_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_sub_wrap, vsub_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_sub_wrap, vsub_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_sub_wrap, vsub_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_mul_wrap, vmul_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_mul_wrap, vmul_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_mul_wrap, vmul_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_mul_wrap, vmul_vv_i16m1, 8)

//512
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x64, v_add_wrap, vadd_vv_u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x64, v_add_wrap, vadd_vv_i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x32, v_add_wrap, vadd_vv_u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x32, v_add_wrap, vadd_vv_i16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x64, v_sub_wrap, vsub_vv_u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x64, v_sub_wrap, vsub_vv_i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x32, v_sub_wrap, vsub_vv_u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x32, v_sub_wrap, vsub_vv_i16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x64, v_mul_wrap, vmul_vv_u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x64, v_mul_wrap, vmul_vv_i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x32, v_mul_wrap, vmul_vv_u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x32, v_mul_wrap, vmul_vv_i16m4, 32)


//////// Dot Product ////////
#define OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(func, _Tpvec, _Tpwvec) \
inline _Tpwvec v_##func(const _Tpvec& a, const _Tpvec& b, const _Tpwvec& c) \
{  return v_##func(a, b) + c; } \

// 16 >> 32
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t res = vundefined_i32m2();
    res = vwmul_vv_i32m2(a.val, b.val, 8);
    res = vrgather_vv_i32m2(res, (vuint32m2_t){0, 2, 4, 6, 1, 3, 5, 7}, 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(res, 0), vget_i32m2_i32m1(res, 1), 4));
}
inline v_int32x16 v_dotprod(const v_int16x32& a, const v_int16x32& b)
{
    //vint32m4_t res = vundefined_i32m4();
    //vint32m4_t res1 = vundefined_i32m4();
    //vint32m4_t res2 = vundefined_i32m4();
    //res = vwmul_vv_i32m4(vget_i16m4_i16m2(a.val, 0), vget_i16m4_i16m2(b.val, 0), 16);
    //res = vrgather_vv_i32m4(res, (vuint32m4_t){0, 2, 4, 6, 8, 10, 12, 14, 
    //                                           1, 3, 5, 7, 9, 11, 13, 15}, 16);
    //vset_i32m4_i32m2(res1, 0, vget_i32m4_i32m2(res, 0));
    //vset_i32m4_i32m2(res2, 0, vget_i32m4_i32m2(res, 1));
    //res = vwmul_vv_i32m4(vget_i16m4_i16m2(a.val, 1), vget_i16m4_i16m2(b.val, 1), 16);
    //res = vrgather_vv_i32m4(res, (vuint32m4_t){0, 2, 4, 6, 8, 10, 12, 14, 
    //                                           1, 3, 5, 7, 9, 11, 13, 15}, 16);
    //vset_i32m4_i32m2(res1, 1, vget_i32m4_i32m2(res, 0));
    //vset_i32m4_i32m2(res2, 1, vget_i32m4_i32m2(res, 1));
    //return v_int32x16(vadd_vv_i32m4(res1, res2, 16));
    vint32m8_t res = vundefined_i32m8();
    res = vwmul_vv_i32m8(a.val, b.val, 16);
    res = vrgather_vv_i32m8(res, (vuint32m8_t){0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
                                               1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}, 32);
    return v_int32x16(vadd_vv_i32m4(vget_i32m8_i32m4(res, 0), vget_i32m8_i32m4(res, 1), 32));
    
}

OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod, v_int16x8, v_int32x4)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod, v_int16x32, v_int32x16)

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
    vint64m2_t res = vundefined_i64m2();
    res = vwmul_vv_i64m2(a.val, b.val, 4);
    res = vrgather_vv_i64m2(res, (vuint64m2_t){0, 2, 1, 3}, 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(res, 0), vget_i64m2_i64m1(res, 1), 2));
}
inline v_int64x8 v_dotprod(const v_int32x16& a, const v_int32x16& b)
{
    //vint64m4_t res = vundefined_i64m4();
    //vint64m4_t res1 = vundefined_i64m4();
    //vint64m4_t res2 = vundefined_i64m4();
    //res = vwmul_vv_i64m4(vget_i32m4_i32m2(a.val, 0), vget_i32m4_i32m2(b.val, 0), 8);
    //res = vrgather_vv_i64m4(res, (vuint64m4_t){0, 2, 4, 6,
    //                                           1, 3, 5, 7 }, 8);
    //vset_i64m4_i64m2(res1, 0, vget_i64m4_i64m2(res, 0));
    //vset_i64m4_i64m2(res2, 0, vget_i64m4_i64m2(res, 1));
    //res = vwmul_vv_i64m4(vget_i32m4_i32m2(a.val, 1), vget_i32m4_i32m2(b.val, 1), 8);
    //res = vrgather_vv_i64m4(res, (vuint64m4_t){0, 2, 4, 6,
    //                                           1, 3, 5, 7 }, 8);
    //vset_i64m4_i64m2(res1, 1, vget_i64m4_i64m2(res, 0));
    //vset_i64m4_i64m2(res2, 1, vget_i64m4_i64m2(res, 1));
    //
    //return v_int64x8(vadd_vv_i64m4(res1, res2, 8));
    vint64m8_t res = vundefined_i64m8();
    res = vwmul_vv_i64m8(a.val, b.val, 16);
    res = vrgather_vv_i64m8(res, (vuint64m8_t){0, 2, 4, 6, 8, 10, 12, 14,
                                               1, 3, 5, 7, 9, 11, 13, 15 }, 16);
    return v_int64x8(vadd_vv_i64m4(vget_i64m8_i64m4(res, 0), vget_i64m8_i64m4(res, 1), 8));
}

OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod, v_int32x4, v_int64x2)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod, v_int32x16, v_int64x8)

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    vuint16m2_t v1 = vundefined_u16m2();
    vuint32m2_t v2 = vundefined_u32m2();
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v1 = vrgather_vv_u16m2(v1, (vuint16m2_t){0, 4, 8, 12, 
                                             1, 5, 9, 13, 
                                             2, 6, 10, 14, 
                                             3, 7, 11, 15}, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    vint16m2_t v1 = vundefined_i16m2();
    vint32m2_t v2 = vundefined_i32m2();
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v1 = vrgather_vv_i16m2(v1, (vuint16m2_t){0, 4, 8, 12, 
                                             1, 5, 9, 13, 
                                             2, 6, 10, 14, 
                                             3, 7, 11, 15}, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4));
}

inline v_uint32x16 v_dotprod_expand(const v_uint8x64& a, const v_uint8x64& b)
{
    //vuint16m4_t v1 = vundefined_u16m4();
    //vuint32m4_t res1 = vundefined_u32m4();
    //vuint32m4_t res2 = vundefined_u32m4();
    //v1 = vwmulu_vv_u16m4(vget_u8m4_u8m2(a.val, 0), vget_u8m4_u8m2(b.val, 0), 32);
    //v1 = vrgather_vv_u16m4(v1, (vuint16m4_t){0, 4, 8, 12, 16, 20, 24, 28, 
    //                                         1, 5, 9, 13, 17, 21, 25, 29, 
    //                                         2, 6, 10, 14, 18, 22, 26, 30,  
    //                                         3, 7, 11, 15, 19, 23, 27, 31}, 32);
    //vuint32m4_t v2 = vwaddu_vv_u32m4(vget_u16m4_u16m2(v1, 0), vget_u16m4_u16m2(v1, 1), 16);
    //vset_u32m4_u32m2(res1, 0, vget_u32m4_u32m2(v2, 0));
    //vset_u32m4_u32m2(res2, 0, vget_u32m4_u32m2(v2, 1));
    //
    //v1 = vwmulu_vv_u16m4(vget_u8m4_u8m2(a.val, 1), vget_u8m4_u8m2(b.val, 1), 32);
    //v1 = vrgather_vv_u16m4(v1, (vuint16m4_t){0, 4, 8, 12, 16, 20, 24, 28, 
    //                                         1, 5, 9, 13, 17, 21, 25, 29, 
    //                                         2, 6, 10, 14, 18, 22, 26, 30,  
    //                                         3, 7, 11, 15, 19, 23, 27, 31}, 32);
    //v2 = vwaddu_vv_u32m4(vget_u16m4_u16m2(v1, 0), vget_u16m4_u16m2(v1, 1), 16);
    //vset_u32m4_u32m2(res1, 1, vget_u32m4_u32m2(v2, 0));
    //vset_u32m4_u32m2(res2, 1, vget_u32m4_u32m2(v2, 1));
    //
    //return v_uint32x16(vadd_vv_u32m4(res1, res2, 16));

    vuint16m8_t v1 = vwmulu_vv_u16m8(a.val, b.val, 64);
    v1 = vrgather_vv_u16m8(v1, (vuint16m8_t){0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                                             1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
                                             2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,  
                                             3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 }, 64);
    vuint32m8_t v2 = vwaddu_vv_u32m8(vget_u16m8_u16m4(v1, 0), vget_u16m8_u16m4(v1, 1), 32);
    return v_uint32x16(vadd_vv_u32m4(vget_u32m8_u32m4(v2, 0), vget_u32m8_u32m4(v2, 1), 16));
}

inline v_int32x16 v_dotprod_expand(const v_int8x64& a, const v_int8x64& b)
{    
    //vint16m4_t v1 = vundefined_i16m4();
    //vint32m4_t res1 = vundefined_i32m4();
    //vint32m4_t res2 = vundefined_i32m4();
    //v1 = vwmul_vv_i16m4(vget_i8m4_i8m2(a.val, 0), vget_i8m4_i8m2(b.val, 0), 32);
    //v1 = vrgather_vv_i16m4(v1, (vuint16m4_t){0, 4, 8, 12, 16, 20, 24, 28, 
    //                                         1, 5, 9, 13, 17, 21, 25, 29, 
    //                                         2, 6, 10, 14, 18, 22, 26, 30,  
    //                                         3, 7, 11, 15, 19, 23, 27, 31}, 32);
    //vint32m4_t v2 = vwadd_vv_i32m4(vget_i16m4_i16m2(v1, 0), vget_i16m4_i16m2(v1, 1), 16);
    //vset_i32m4_i32m2(res1, 0, vget_i32m4_i32m2(v2, 0));
    //vset_i32m4_i32m2(res2, 0, vget_i32m4_i32m2(v2, 1));
    //
    //v1 = vwmul_vv_i16m4(vget_i8m4_i8m2(a.val, 1), vget_i8m4_i8m2(b.val, 1), 32);
    //v1 = vrgather_vv_i16m4(v1, (vuint16m4_t){0, 4, 8, 12, 16, 20, 24, 28, 
    //                                         1, 5, 9, 13, 17, 21, 25, 29, 
    //                                         2, 6, 10, 14, 18, 22, 26, 30,  
    //                                         3, 7, 11, 15, 19, 23, 27, 31}, 32);
    //v2 = vwadd_vv_i32m4(vget_i16m4_i16m2(v1, 0), vget_i16m4_i16m2(v1, 1), 16);
    //vset_i32m4_i32m2(res1, 1, vget_i32m4_i32m2(v2, 0));
    //vset_i32m4_i32m2(res2, 1, vget_i32m4_i32m2(v2, 1));
    //
    //return v_int32x16(vadd_vv_i32m4(res1, res2, 16));
    vint16m8_t v1 = vwmul_vv_i16m8(a.val, b.val, 64);
    v1 = vrgather_vv_i16m8(v1, (vuint16m8_t){0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                                             1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
                                             2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,  
                                             3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 }, 64);
    vint32m8_t v2 = vwadd_vv_i32m8(vget_i16m8_i16m4(v1, 0), vget_i16m8_i16m4(v1, 1), 8);
    return v_int32x16(vadd_vv_i32m4(vget_i32m8_i32m4(v2, 0), vget_i32m8_i32m4(v2, 1), 4));
}

OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand, v_uint8x16, v_uint32x4)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand, v_int8x16, v_int32x4)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand, v_uint8x64, v_uint32x16)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand, v_int8x64, v_int32x16)

inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    vuint32m2_t v1 = vundefined_u32m2();
    vuint64m2_t v2 = vundefined_u64m2();
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v1 = vrgather_vv_u32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2));
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1 = vundefined_i32m2();
    vint64m2_t v2 = vundefined_i64m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v1 = vrgather_vv_i32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2));
}

inline v_uint64x8 v_dotprod_expand(const v_uint16x32& a, const v_uint16x32& b)
{   
    //vuint32m4_t v1 = vundefined_u32m4();
    //vuint64m4_t res1 = vundefined_u64m4();
    //vuint64m4_t res2 = vundefined_u64m4();
    //v1 = vwmulu_vv_u32m4(vget_u16m4_u16m2(a.val, 0), vget_u16m4_u16m2(b.val, 0), 16);
    //v1 = vrgather_vv_u32m4(v1, (vuint32m4_t){0, 4, 8, 12,  
    //                                         1, 5, 9, 13,  
    //                                         2, 6, 10, 14,  
    //                                         3, 7, 11, 15}, 16);
    //vuint64m4_t v2 = vwaddu_vv_u64m4(vget_u32m4_u32m2(v1, 0), vget_u32m4_u32m2(v1, 1), 8);
    //vset_u64m4_u64m2(res1, 0, vget_u64m4_u64m2(v2, 0));
    //vset_u64m4_u64m2(res2, 0, vget_u64m4_u64m2(v2, 1));
    //
    //v1 = vwmulu_vv_u32m4(vget_u16m4_u16m2(a.val, 1), vget_u16m4_u16m2(b.val, 1), 16);
    //v1 = vrgather_vv_u32m4(v1, (vuint32m4_t){0, 4, 8, 12,  
    //                                         1, 5, 9, 13,  
    //                                         2, 6, 10, 14,  
    //                                         3, 7, 11, 15}, 16);
    //v2 = vwaddu_vv_u64m4(vget_u32m4_u32m2(v1, 0), vget_u32m4_u32m2(v1, 1), 8);
    //vset_u64m4_u64m2(res1, 1, vget_u64m4_u64m2(v2, 0));
    //vset_u64m4_u64m2(res2, 1, vget_u64m4_u64m2(v2, 1));
    //
    //return v_uint64x8(vadd_vv_u64m4(res1, res2, 8));
    
    vuint32m8_t v1 = vwmulu_vv_u32m8(a.val, b.val, 32);
    v1 = vrgather_vv_u32m8(v1, (vuint32m8_t){0, 4, 8, 12, 16, 20, 24, 28,
                                             1, 5, 9, 13, 17, 21, 25, 29,
                                             2, 6, 10, 14, 18, 22, 26, 30,  
                                             3, 7, 11, 15, 19, 23, 27, 31}, 32);
    vuint64m8_t v2 = vwaddu_vv_u64m8(vget_u32m8_u32m4(v1, 0), vget_u32m8_u32m4(v1, 1), 4);
    return v_uint64x8(vadd_vv_u64m4(vget_u64m8_u64m4(v2, 0), vget_u64m8_u64m4(v2, 1), 2));
}

inline v_int64x8 v_dotprod_expand(const v_int16x32& a, const v_int16x32& b)
{  
    //vint32m4_t v1 = vundefined_i32m4();
    //vint64m4_t res1 = vundefined_i64m4();
    //vint64m4_t res2 = vundefined_i64m4();
    //v1 = vwmul_vv_i32m4(vget_i16m4_i16m2(a.val, 0), vget_i16m4_i16m2(b.val, 0), 16);
    //v1 = vrgather_vv_i32m4(v1, (vuint32m4_t){0, 4, 8, 12,  
    //                                         1, 5, 9, 13,  
    //                                         2, 6, 10, 14,  
    //                                         3, 7, 11, 15}, 16);
    //vint64m4_t v2 = vwadd_vv_i64m4(vget_i32m4_i32m2(v1, 0), vget_i32m4_i32m2(v1, 1), 8);
    //vset_i64m4_i64m2(res1, 0, vget_i64m4_i64m2(v2, 0));
    //vset_i64m4_i64m2(res2, 0, vget_i64m4_i64m2(v2, 1));
    //
    //v1 = vwmul_vv_i32m4(vget_i16m4_i16m2(a.val, 1), vget_i16m4_i16m2(b.val, 1), 16);
    //v1 = vrgather_vv_i32m4(v1, (vuint32m4_t){0, 4, 8, 12,  
    //                                         1, 5, 9, 13,  
    //                                         2, 6, 10, 14,  
    //                                         3, 7, 11, 15}, 16);
    //v2 = vwadd_vv_i64m4(vget_i32m4_i32m2(v1, 0), vget_i32m4_i32m2(v1, 1), 8);
    //vset_i64m4_i64m2(res1, 1, vget_i64m4_i64m2(v2, 0));
    //vset_i64m4_i64m2(res2, 1, vget_i64m4_i64m2(v2, 1));
    //
    //return v_int64x8(vadd_vv_i64m4(res1, res2, 8));
    
    vint32m8_t v1 = vwmul_vv_i32m8(a.val, b.val, 32);
    v1 = vrgather_vv_i32m8(v1, (vuint32m8_t){0, 4, 8, 12, 16, 20, 24, 28,
                                             1, 5, 9, 13, 17, 21, 25, 29,
                                             2, 6, 10, 14, 18, 22, 26, 30,  
                                             3, 7, 11, 15, 19, 23, 27, 31}, 32);
    vint64m8_t v2 = vwadd_vv_i64m8(vget_i32m8_i32m4(v1, 0), vget_i32m8_i32m4(v1, 1), 4);
    return v_int64x8(vadd_vv_i64m4(vget_i64m8_i64m4(v2, 0), vget_i64m8_i64m4(v2, 1), 2));
}

OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand, v_uint16x8, v_uint64x2)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand, v_int16x8, v_int64x2)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand, v_uint16x32, v_uint64x8)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand, v_int16x32, v_int64x8)

//////// Fast Dot Product ////////
// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1 = vundefined_i32m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4));
}

inline v_int32x16 v_dotprod_fast(const v_int16x32& a, const v_int16x32& b)
{
    //vint32m4_t v1 = vundefined_i32m4();
    //vint32m4_t v2 = vundefined_i32m4();
    //
    //v1 = vwmul_vv_i32m4(vget_i16m4_i16m2(a.val, 0), vget_i16m4_i16m2(b.val, 0), 16);
    //v2 = vwmul_vv_i32m4(vget_i16m4_i16m2(a.val, 1), vget_i16m4_i16m2(b.val, 1), 16);
    //return v_int32x16(vadd_vv_i32m4(v1, v2, 16));

    vint32m8_t v1 = vundefined_i32m8();
    v1 = vwmul_vv_i32m8(a.val, b.val, 32);
    return v_int32x16(vadd_vv_i32m4(vget_i32m8_i32m4(v1, 0), vget_i32m8_i32m4(v1, 1), 16));
}

OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_fast, v_int16x8, v_int32x4)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_fast, v_int16x32, v_int32x16)


// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{
    vint64m2_t v1 = vundefined_i64m2();
    v1 = vwmul_vv_i64m2(a.val, b.val, 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v1, 0), vget_i64m2_i64m1(v1, 1), 2));
}
inline v_int64x8 v_dotprod_fast(const v_int32x16& a, const v_int32x16& b)
{
    //vint64m4_t v1 = vundefined_i64m4();
    //vint64m4_t v2 = vundefined_i64m4();
    //
    //v1 = vwmul_vv_i64m4(vget_i32m4_i32m2(a.val, 0), vget_i32m4_i32m2(b.val, 0), 8);
    //v2 = vwmul_vv_i64m4(vget_i32m4_i32m2(a.val, 1), vget_i32m4_i32m2(b.val, 1), 8);
    //return v_int64x8(vadd_vv_i64m4(v1, v2, 8));
    vint64m8_t v1 = vundefined_i64m8();
    v1 = vwmul_vv_i64m8(a.val, b.val, 16);
    return v_int64x8(vadd_vv_i64m4(vget_i64m8_i64m4(v1, 0), vget_i64m8_i64m4(v1, 1), 8));
}

OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_fast, v_int32x4, v_int64x2)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_fast, v_int32x16, v_int64x8)

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{
    vuint16m2_t v1 = vundefined_u16m2();
    vuint32m2_t v2 = vundefined_u32m2();
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4));
}

inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{
    vint16m2_t v1 = vundefined_i16m2();
    vint32m2_t v2 = vundefined_i32m2();
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4));
}

inline v_uint32x16 v_dotprod_expand_fast(const v_uint8x64& a, const v_uint8x64& b)
{
    //vuint16m4_t v1 = vundefined_u16m4();
    //vuint16m4_t v2 = vundefined_u16m4();
    //
    //v1 = vwmulu_vv_u16m4(vget_u8m4_u8m2(a.val, 0), vget_u8m4_u8m2(b.val, 0), 32);
    //v2 = vwmulu_vv_u16m4(vget_u8m4_u8m2(a.val, 1), vget_u8m4_u8m2(b.val, 1), 32);
    //vuint32m4_t rv1 = vwaddu_vv_u32m4(vget_u16m4_u16m2(v1, 0), vget_u16m4_u16m2(v1, 1), 16);
    //vuint32m4_t rv2 = vwaddu_vv_u32m4(vget_u16m4_u16m2(v2, 0), vget_u16m4_u16m2(v2, 1), 16);
    //return v_uint32x16(vadd_vv_u32m4(rv1, rv2, 16));
    vuint16m8_t v1 = vwmulu_vv_u16m8(a.val, b.val, 64);
    vuint32m8_t v2 = vwaddu_vv_u32m8(vget_u16m8_u16m4(v1, 0), vget_u16m8_u16m4(v1, 1), 32);
    return v_uint32x16(vadd_vv_u32m4(vget_u32m8_u32m4(v2, 0), vget_u32m8_u32m4(v2, 1), 16));
}

inline v_int32x16 v_dotprod_expand_fast(const v_int8x64& a, const v_int8x64& b)
{
    //vint16m4_t v1 = vundefined_i16m4();
    //vint16m4_t v2 = vundefined_i16m4();
    //
    //v1 = vwmul_vv_i16m4(vget_i8m4_i8m2(a.val, 0), vget_i8m4_i8m2(b.val, 0), 32);
    //v2 = vwmul_vv_i16m4(vget_i8m4_i8m2(a.val, 1), vget_i8m4_i8m2(b.val, 1), 32);
    //vint32m4_t rv1 = vwadd_vv_i32m4(vget_i16m4_i16m2(v1, 0), vget_i16m4_i16m2(v1, 1), 16);
    //vint32m4_t rv2 = vwadd_vv_i32m4(vget_i16m4_i16m2(v2, 0), vget_i16m4_i16m2(v2, 1), 16);
    //return v_int32x16(vadd_vv_i32m4(rv1, rv2, 16));
    vint16m8_t v1 = vwmul_vv_i16m8(a.val, b.val, 64);
    vint32m8_t v2 = vwadd_vv_i32m8(vget_i16m8_i16m4(v1, 0), vget_i16m8_i16m4(v1, 1), 32);
    return v_int32x16(vadd_vv_i32m4(vget_i32m8_i32m4(v2, 0), vget_i32m8_i32m4(v2, 1), 16));
}

OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand_fast, v_uint8x16, v_uint32x4)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand_fast, v_int8x16, v_int32x4)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand_fast, v_uint8x64, v_uint32x16)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand_fast, v_int8x64, v_int32x16)

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{
    vuint32m2_t v1 = vundefined_u32m2();
    vuint64m2_t v2 = vundefined_u64m2();
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2));
}

inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1 = vundefined_i32m2();
    vint64m2_t v2 = vundefined_i64m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2));
}

inline v_uint64x8 v_dotprod_expand_fast(const v_uint16x32& a, const v_uint16x32& b)
{
    //vuint32m4_t v1 = vundefined_u32m4();
    //vuint32m4_t v2 = vundefined_u32m4();
    //
    //v1 = vwmulu_vv_u32m4(vget_u16m4_u16m2(a.val, 0), vget_u16m4_u16m2(b.val, 0), 16);
    //v2 = vwmulu_vv_u32m4(vget_u16m4_u16m2(a.val, 1), vget_u16m4_u16m2(b.val, 1), 16);
    //vuint64m4_t rv1 = vwaddu_vv_u64m4(vget_u32m4_u32m2(v1, 0), vget_u32m4_u32m2(v1, 1), 8);
    //vuint64m4_t rv2 = vwaddu_vv_u64m4(vget_u32m4_u32m2(v2, 0), vget_u32m4_u32m2(v2, 1), 8);
    //return v_uint64x8(vadd_vv_u64m4(rv1, rv2, 8));
    vuint32m8_t v1 = vwmulu_vv_u32m8(a.val, b.val, 32);
    vuint64m8_t v2 = vwaddu_vv_u64m8(vget_u32m8_u32m4(v1, 0), vget_u32m8_u32m4(v1, 1), 16);
    return v_uint64x8(vadd_vv_u64m4(vget_u64m8_u64m4(v2, 0), vget_u64m8_u64m4(v2, 1), 8));
}

inline v_int64x8 v_dotprod_expand_fast(const v_int16x32& a, const v_int16x32& b)
{
    //vint32m4_t v1 = vundefined_i32m4();
    //vint32m4_t v2 = vundefined_i32m4();
    //
    //v1 = vwmul_vv_i32m4(vget_i16m4_i16m2(a.val, 0), vget_i16m4_i16m2(b.val, 0), 16);
    //v2 = vwmul_vv_i32m4(vget_i16m4_i16m2(a.val, 1), vget_i16m4_i16m2(b.val, 1), 16);
    //vint64m4_t rv1 = vwadd_vv_i64m4(vget_i32m4_i32m2(v1, 0), vget_i32m4_i32m2(v1, 1), 8);
    //vint64m4_t rv2 = vwadd_vv_i64m4(vget_i32m4_i32m2(v2, 0), vget_i32m4_i32m2(v2, 1), 8);
    //return v_int64x8(vadd_vv_i64m4(rv1, rv2, 8));
    vint32m8_t v1 = vwmul_vv_i32m8(a.val, b.val, 32);
    vint64m8_t v2 = vwadd_vv_i64m8(vget_i32m8_i32m4(v1, 0), vget_i32m8_i32m4(v1, 1), 16);
    return  v_int64x8(vadd_vv_i64m4(vget_i64m8_i64m4(v2, 0), vget_i64m8_i64m4(v2, 1), 8));
}

OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand_fast, v_uint16x8, v_uint64x2)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand_fast, v_int16x8, v_int64x2)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand_fast, v_uint16x32, v_uint64x8)
OPENCV_HAL_IMPL_RISCVV_DOT_PRODUCT3(dotprod_expand_fast, v_int16x32, v_int64x8)


#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(_Tpvec, _Tpvec2, len, scalartype, func, intrin, num) \
inline scalartype v_reduce_##func(const v_##_Tpvec##x##num& a) \
{\
    v##_Tpvec2##m1_t val = vmv_v_x_##len##m1(0, num); \
    val = intrin(val, a.val, val, num);    \
    return vmv_x_s_##len##m1_##len(val, num);    \
}

OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int8, int16, i16, int, sum, vwredsum_vs_i8m1_i16m1, 16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int16, int32, i32, int, sum, vwredsum_vs_i16m1_i32m1, 8)
#if CV_SIMD_ELEM64
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int32, int64, i64, int, sum, vwredsum_vs_i32m1_i64m1, 4)
#else
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int32, int32, i32, int, sum, vredsum_vs_i32m1_i32m1, 4)
#endif
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint8, uint16, u16, unsigned, sum, vwredsumu_vs_u8m1_u16m1, 16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint16, uint32, u32, unsigned, sum, vwredsumu_vs_u16m1_u32m1, 8)
#if CV_SIMD_ELEM64
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint32, uint64m1, u64m1, 1, unsigned, sum, vwredsumu_vs_u32m1_u64m1, 4)
#else
inline unsigned v_reduce_sum(const v_uint32x4& a) \
{ return vext_x_v_u32m1_u32((vuint32m1_t)a.val, 0, 4)+vext_x_v_u32m1_u32((vuint32m1_t)a.val, 1, 4) + \
    vext_x_v_u32m1_u32((vuint32m1_t)a.val, 2, 4)+vext_x_v_u32m1_u32((vuint32m1_t)a.val, 3, 4); \
}
#endif
inline float v_reduce_sum(const v_float32x4& a) \
{\
    vfloat32m1_t val = vfmv_v_f_f32m1(0.0, 4); \
    val = vfredsum_vs_f32m1_f32m1(val, a.val, val, 4);    \
    return vfmv_f_s_f32m1_f32(val, 4);    \
}
inline double v_reduce_sum(const v_float64x2& a) \
{\
    vfloat64m1_t val = vfmv_v_f_f64m1(0.0, 2); \
    val = vfredsum_vs_f64m1_f64m1(val, a.val, val, 2);    \
    return vfmv_f_s_f64m1_f64(val, 2);    \
}
inline uint64 v_reduce_sum(const v_uint64x2& a)
{ return vext_x_v_u64m1_u64((vuint64m1_t)a.val, 0, 2)+vext_x_v_u64m1_u64((vuint64m1_t)a.val, 1, 2); }

inline int64 v_reduce_sum(const v_int64x2& a)
{ return vext_x_v_i64m1_i64((vint64m1_t)a.val, 0, 2)+vext_x_v_i64m1_i64((vint64m1_t)a.val, 1, 2); }

//512
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int8, int16, i16, int, sum, vwredsum_vs_i8m4_i16m1, 64)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int16, int32, i32, int, sum, vwredsum_vs_i16m4_i32m1, 32)
#if CV_SIMD_ELEM64
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int32, int64, i64, int, sum, vwredsum_vs_i32m4_i64m1, 16)
#else
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int32, int32, i32, int, sum, vredsum_vs_i32m4_i32m1, 16)
#endif
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint8, uint16, u16, unsigned, sum, vwredsumu_vs_u8m4_u16m1, 64)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint16, uint32, u32, unsigned, sum, vwredsumu_vs_u16m4_u32m1, 32)
#if CV_SIMD_ELEM64
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint32, uint64, u64, unsigned, sum, vwredsumu_vs_u32m4_u64m1, 16)
#else
inline unsigned v_reduce_sum(const v_uint32x16& a)
{ 
    vuint32m2_t v = vadd_vv_u32m2(vget_u32m4_u32m2(a.val, 0), vget_u32m4_u32m2(a.val, 1), 16);
    vuint32m1_t v1 = vadd_vv_u32m1(vget_u32m2_u32m1(v, 0), vget_u32m2_u32m1(v, 1), 8);
    return vext_x_v_u32m1_u32(v1, 0, 4)+vext_x_v_u32m1_u32(v1, 1, 4) +
           vext_x_v_u32m1_u32(v1, 2, 4)+vext_x_v_u32m1_u32(v1, 3, 4);
}
#endif
inline float v_reduce_sum(const v_float32x16& a) 
{
    vfloat32m1_t val = vfmv_v_f_f32m1(0.0, 4); 
    val = vfredsum_vs_f32m4_f32m1(val, a.val, val, 16);    
    return vfmv_f_s_f32m1_f32(val, 4);    
}
inline double v_reduce_sum(const v_float64x8& a) 
{
    vfloat64m1_t val = vfmv_v_f_f64m1(0.0, 2); 
    val = vfredsum_vs_f64m4_f64m1(val, a.val, val, 8);    
    return vfmv_f_s_f64m1_f64(val, 2);    
}
inline uint64 v_reduce_sum(const v_uint64x8& a)
{  
    vuint64m2_t v = vadd_vv_u64m2(vget_u64m4_u64m2(a.val, 0), vget_u64m4_u64m2(a.val, 1), 8);
    vuint64m1_t v1 = vadd_vv_u64m1(vget_u64m2_u64m1(v, 0), vget_u64m2_u64m1(v, 1), 4);
    return vext_x_v_u64m1_u64(v1, 0, 2)+vext_x_v_u64m1_u64(v1, 1, 2); 
}
inline int64 v_reduce_sum(const v_int64x8& a)
{  
    vint64m2_t v = vadd_vv_i64m2(vget_i64m4_i64m2(a.val, 0), vget_i64m4_i64m2(a.val, 1), 8);
    vint64m1_t v1 = vadd_vv_i64m1(vget_i64m2_i64m1(v, 0), vget_i64m2_i64m1(v, 1), 4);
    return vext_x_v_i64m1_i64(v1, 0, 2)+vext_x_v_i64m1_i64(v1, 1, 2); 
}


#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(_Tpvec, _Tpvec2, regnum, scalartype, func, funcu, num) \
inline scalartype v_reduce_##func(const v_##_Tpvec##x##num& a) \
{\
    v##_Tpvec##m1_t val = (v##_Tpvec##m1_t)vmv_v_x_i8m1(0, 16); \
    val = v##funcu##_vs_##_Tpvec2##m##regnum##_##_Tpvec2##m1(val, a.val, val, num);    \
    return val[0];    \
}

#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(func)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int8,  i8, 1, int, func, red##func, 16)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int16, i16, 1, int, func, red##func, 8)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int32, i32, 1, int, func, red##func, 4)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int64, i64, 1, int, func, red##func, 2)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint8,  u8, 1, unsigned, func, red##func##u, 16)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint16, u16, 1, unsigned, func, red##func##u, 8)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint32, u32, 1, unsigned, func, red##func##u, 4)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(float32, f32, 1, float, func, fred##func, 4) \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int8,  i8, 4, int, func, red##func, 64)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int16, i16, 4, int, func, red##func, 32)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int32, i32, 4, int, func, red##func, 16)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int64, i64, 4, int, func, red##func, 8)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint8,  u8, 4, unsigned, func, red##func##u, 64)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint16, u16, 4, unsigned, func, red##func##u, 32)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint32, u32, 4, unsigned, func, red##func##u, 16)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(float32, f32, 4, float, func, fred##func, 16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(max)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(min)

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    vfloat32m1_t a0 = vfmv_v_f_f32m1(0.0, 4);
    vfloat32m1_t b0 = vfmv_v_f_f32m1(0.0, 4);
    vfloat32m1_t c0 = vfmv_v_f_f32m1(0.0, 4);
    vfloat32m1_t d0 = vfmv_v_f_f32m1(0.0, 4);
    a0 = vfredsum_vs_f32m1_f32m1(a0, a.val, a0, 4);
    b0 = vfredsum_vs_f32m1_f32m1(b0, b.val, b0, 4);
    c0 = vfredsum_vs_f32m1_f32m1(c0, c.val, c0, 4);
    d0 = vfredsum_vs_f32m1_f32m1(d0, d.val, d0, 4);
    return v_float32x4(a0[0], b0[0], c0[0], d0[0]);
}

//512
inline v_float32x16 v_reduce_sum4(const v_float32x16& a, const v_float32x16& b,
                                  const v_float32x16& c, const v_float32x16& d)
{
    float res0, res1, res2, res3;
    vfloat32m1_t f0 = vfmv_v_f_f32m1(0.0, 4);
    f0 = vfredsum_vs_f32m4_f32m1(f0, a.val, f0, 4);
    res0 = f0[0];
    f0 = vfmv_v_f_f32m1(0.0, 4);
    f0 = vfredsum_vs_f32m4_f32m1(f0, b.val, f0, 4);
    res1 = f0[0];
    f0 = vfmv_v_f_f32m1(0.0, 4);
    f0 = vfredsum_vs_f32m4_f32m1(f0, c.val, f0, 4);
    res2 = f0[0];
    f0 = vfmv_v_f_f32m1(0.0, 4);
    f0 = vfredsum_vs_f32m4_f32m1(f0, d.val, f0, 4);
    res3 = f0[0];
    return v_float32x16(res0, res1, res2, res3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    vfloat32m1_t a0 = vfmv_v_f_f32m1(0.0, 4);
    v_float32x4 val;
    val = v_absdiff(a, b);
    a0 = vfredsum_vs_f32m1_f32m1(a0, val.val, a0, 4);
    return a0[0];
}

inline float v_reduce_sad(const v_float32x16& a, const v_float32x16& b)
{
    vfloat32m1_t a0 = vfmv_v_f_f32m1(0.0, 4);
    v_float32x16 val;
    val = v_absdiff(a, b);
    a0 = vfredsum_vs_f32m4_f32m1(a0, val.val, a0, 16);
    return a0[0];
}

#define OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(_Tpvec, _Tpvec2) \
inline unsigned v_reduce_sad(const _Tpvec& a, const _Tpvec&b){    \
    _Tpvec2 x = v_absdiff(a, b);    \
    return v_reduce_sum(x);    \
}

OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int8x16, v_uint8x16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint8x16, v_uint8x16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int16x8, v_uint16x8)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint16x8, v_uint16x8)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int32x4, v_uint32x4)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint32x4, v_uint32x4)

//512
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int8x64, v_uint8x64)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint8x64, v_uint8x64)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int16x32, v_uint16x32)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint16x32, v_uint16x32)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int32x16, v_uint32x16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint32x16, v_uint32x16)

#define OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(_Tpvec, _Tp, _T, num, uv) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmseq_vv_##_Tp##_b##_T(a.val, b.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmsne_vv_##_Tp##_b##_T(a.val, b.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmslt##uv##_Tp##_b##_T(a.val, b.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmslt##uv##_Tp##_b##_T(b.val, a.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmsle##uv##_Tp##_b##_T(a.val, b.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmsle##uv##_Tp##_b##_T(b.val, a.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \

OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int8x16, i8m1,  8, 16, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int16x8, i16m1, 16, 8, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int32x4, i32m1, 32, 4, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int64x2, i64m1, 64, 2, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint8x16, u8m1, 8, 16, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint16x8, u16m1, 16, 8, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint32x4, u32m1, 32, 4, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint64x2, u64m1, 64, 2, u_vv_)

//512
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int8x64, i8m4,  2, 64, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int16x32, i16m4, 4, 32, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int32x16, i32m4, 8, 16, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int64x8, i64m4, 16, 8, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint8x64, u8m4, 2, 64, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint16x32, u16m4, 4, 32, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint32x16, u32m4, 8, 16, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint64x8, u64m4, 16, 8, u_vv_)

#define OPENCV_HAL_IMPL_RISCVV_FLT_CMP_OP(_Tpvec, _Tp, _T, num) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmfeq_vv_f##_Tp##_b##_T(a.val, b.val, num); \
    vint##_Tp##_t res = vmerge_vxm_i##_Tp(mask, vmv_v_x_i##_Tp(0.0, num), -1, num); \
    return _Tpvec((vfloat##_Tp##_t)res); \
} \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmfne_vv_f##_Tp##_b##_T(a.val, b.val, num); \
    vint##_Tp##_t res = vmerge_vxm_i##_Tp(mask, vmv_v_x_i##_Tp(0.0, num), -1, num); \
    return _Tpvec((vfloat##_Tp##_t)res); \
} \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmflt_vv_f##_Tp##_b##_T(a.val, b.val, num); \
    vint##_Tp##_t res = vmerge_vxm_i##_Tp(mask, vmv_v_x_i##_Tp(0.0, num), -1, num); \
    return _Tpvec((vfloat##_Tp##_t)res); \
} \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmfle_vv_f##_Tp##_b##_T(a.val, b.val, num); \
    vint##_Tp##_t res = vmerge_vxm_i##_Tp(mask, vmv_v_x_i##_Tp(0.0, num), -1, num); \
    return _Tpvec((vfloat##_Tp##_t)res); \
} \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmfgt_vv_f##_Tp##_b##_T(a.val, b.val, num); \
    vint##_Tp##_t res = vmerge_vxm_i##_Tp(mask, vmv_v_x_i##_Tp(0.0, num), -1, num); \
    return _Tpvec((vfloat##_Tp##_t)res); \
} \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmfge_vv_f##_Tp##_b##_T(a.val, b.val, num); \
    vint##_Tp##_t res = vmerge_vxm_i##_Tp(mask, vmv_v_x_i##_Tp(0.0, num), -1, num); \
    return _Tpvec((vfloat##_Tp##_t)res); \
} \
inline _Tpvec v_not_nan(const _Tpvec& a) \
{ \
    vbool##_T##_t mask = vmford_vv_f##_Tp##_b##_T(a.val, a.val, num); \
    vint##_Tp##_t res = vmerge_vxm_i##_Tp(mask, vmv_v_x_i##_Tp(0.0, num), -1, num); \
    return _Tpvec((vfloat##_Tp##_t)res); \
}

OPENCV_HAL_IMPL_RISCVV_FLT_CMP_OP(v_float32x4, 32m1, 32, 4)
OPENCV_HAL_IMPL_RISCVV_FLT_CMP_OP(v_float64x2, 64m1, 64, 2)


OPENCV_HAL_IMPL_RISCVV_FLT_CMP_OP(v_float32x16, 32m4, 8, 16)
OPENCV_HAL_IMPL_RISCVV_FLT_CMP_OP(v_float64x8, 64m4, 16, 8)

#define OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(_Tp, _T) \
inline void v_transpose4x4(const v_##_Tp##32x4& a0, const v_##_Tp##32x4& a1, \
                         const v_##_Tp##32x4& a2, const v_##_Tp##32x4& a3, \
                         v_##_Tp##32x4& b0, v_##_Tp##32x4& b1, \
                         v_##_Tp##32x4& b2, v_##_Tp##32x4& b3) \
{ \
    v##_Tp##32m4_t val = vundefined_##_T##m4();    \
    val = vset_##_T##m4(val, 0, a0.val);    \
    val = vset_##_T##m4(val, 1, a1.val);    \
    val = vset_##_T##m4(val, 2, a2.val);    \
    val = vset_##_T##m4(val, 3, a3.val);   \
    val = vrgather_vv_##_T##m4(val, (vuint32m4_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);    \
    b0.val = vget_##_T##m4_##_T##m1(val, 0);   \
    b1.val = vget_##_T##m4_##_T##m1(val, 1);   \
    b2.val = vget_##_T##m4_##_T##m1(val, 2);   \
    b3.val = vget_##_T##m4_##_T##m1(val, 3);   \
}
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(uint, u32)
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(int, i32)
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(float, f32)

//512
#define OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4_512(_Tp, _T) \
inline void v_transpose4x4(const v_##_Tp##32x16& a0, const v_##_Tp##32x16& a1, \
                         const v_##_Tp##32x16& a2, const v_##_Tp##32x16& a3, \
                         v_##_Tp##32x16& b0, v_##_Tp##32x16& b1, \
                         v_##_Tp##32x16& b2, v_##_Tp##32x16& b3) \
{ \
    for (int i = 0; i < 4; i++) \
    { \
        v##_Tp##32m4_t val = vundefined_##_T##m4();    \
        val = vset_##_T##m4(val, 0, vget_##_T##m4_##_T##m1(a0.val, i));    \
        val = vset_##_T##m4(val, 1, vget_##_T##m4_##_T##m1(a1.val, i));    \
        val = vset_##_T##m4(val, 2, vget_##_T##m4_##_T##m1(a2.val, i));    \
        val = vset_##_T##m4(val, 3, vget_##_T##m4_##_T##m1(a3.val, i));    \
        val = vrgather_vv_##_T##m4(val, (vuint32m4_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);    \
        b0.val = vset_##_T##m4(b0.val, i, vget_##_T##m4_##_T##m1(val, 0));   \
        b1.val = vset_##_T##m4(b1.val, i, vget_##_T##m4_##_T##m1(val, 1));   \
        b2.val = vset_##_T##m4(b2.val, i, vget_##_T##m4_##_T##m1(val, 2));   \
        b3.val = vset_##_T##m4(b3.val, i, vget_##_T##m4_##_T##m1(val, 3));   \
    } \
}
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4_512(uint, u32)
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4_512(int, i32)
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4_512(float, f32)


#define OPENCV_HAL_IMPL_RISCVV_SHIFT_LEFT(_Tpvec, suffix, _T, regnum, num) \
inline _Tpvec operator << (const _Tpvec& a, int n) \
{ return _Tpvec((vsll_vx_##_T##m##regnum(a.val, n, num))); } \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return _Tpvec((vsll_vx_##_T##m##regnum(a.val, n, num))); }

#define OPENCV_HAL_IMPL_RISCVV_SHIFT_RIGHT(_Tpvec, suffix, _T, regnum, num, intric) \
inline _Tpvec operator >> (const _Tpvec& a, int n) \
{ return _Tpvec((v##intric##_vx_##_T##m##regnum(a.val, n, num))); } \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return _Tpvec((v##intric##_vx_##_T##m##regnum(a.val, n, num))); }\
template<int n> inline _Tpvec v_rshr(const _Tpvec& a) \
{ return _Tpvec((v##intric##_vx_##_T##m##regnum(vadd_vx_##_T##m##regnum(a.val, 1<<(n-1), num), n, num))); }

// trade efficiency for convenience
#define OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(suffix, _T, regnum, num, intrin) \
OPENCV_HAL_IMPL_RISCVV_SHIFT_LEFT(v_##suffix##x##num, suffix, _T, regnum, num) \
OPENCV_HAL_IMPL_RISCVV_SHIFT_RIGHT(v_##suffix##x##num, suffix, _T, regnum, num, intrin)

OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint8,  u8,  1, 16, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint16, u16, 1, 8, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint32, u32, 1, 4, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint64, u64, 1, 2, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int8,   i8,  1, 16, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int16,  i16, 1, 8, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int32,  i32, 1, 4, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int64,  i64, 1, 2, sra)

//512
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint8,  u8,  4, 64, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint16, u16, 4, 32, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint32, u32, 4, 16, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint64, u64, 4, 8, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int8,   i8,  4, 64, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int16,  i16, 4, 32, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int32,  i32, 4, 16, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int64,  i64, 4, 8, sra)

#if 0
#define VUP4(n) {0, 1, 2, 3}
#define VUP8(n) {0, 1, 2, 3, 4, 5, 6, 7}
#define VUP16(n) {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
#define VUP2(n) {0, 1}
#endif
#define OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(_Tpvec, suffix, _T, num, num2, regnum, regnum2, vmv, len) \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{    \
    suffix##m##regnum##_t tmp = vmv##_##_T##m##regnum(0, num);\
        tmp = vslideup_vx_##_T##m##regnum##_m(vmset_m_##len(num), tmp, a.val, n, num);\
        return _Tpvec(tmp);\
} \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{     \
        return _Tpvec(vslidedown_vx_##_T##m##regnum(a.val, n, num));\
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    suffix##m##regnum2##_t tmp = vundefined_##_T##m##regnum2();    \
    tmp = vset_##_T##m##regnum2##_##_T##m##regnum(tmp, 0, a.val);          \
    tmp = vset_##_T##m##regnum2##_##_T##m##regnum(tmp, 1, b.val);          \
        tmp = vslidedown_vx_##_T##m##regnum2(tmp, n, num2);\
        return _Tpvec(vget_##_T##m##regnum2##_##_T##m##regnum(tmp, 0));\
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    suffix##m##regnum2##_t tmp = vundefined_##_T##m##regnum2();    \
    tmp = vset_##_T##m##regnum2##_##_T##m##regnum(tmp, 0, b.val);    \
    tmp = vset_##_T##m##regnum2##_##_T##m##regnum(tmp, 1, a.val);    \
        tmp = vslideup_vx_##_T##m##regnum2(tmp, n, num2);\
        return _Tpvec(vget_##_T##m##regnum2##_##_T##m##regnum(tmp, 1));\
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ \
    CV_UNUSED(b); return a; \
}

OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint8x16, vuint8, u8,   16, 32, 1, 2, vmv_v_x, b8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int8x16, vint8, i8,     16, 32, 1, 2, vmv_v_x, b8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint16x8, vuint16, u16,  8, 16, 1, 2, vmv_v_x, b16)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int16x8, vint16, i16,    8, 16, 1, 2, vmv_v_x, b16)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint32x4, vuint32, u32,  4,  8, 1, 2, vmv_v_x, b32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int32x4, vint32, i32,    4,  8, 1, 2, vmv_v_x, b32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint64x2, vuint64, u64,  2,  4, 1, 2, vmv_v_x, b64)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int64x2, vint64, i64,    2,  4, 1, 2, vmv_v_x, b64)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_float32x4, vfloat32, f32, 4, 8, 1, 2, vfmv_v_f, b32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_float64x2, vfloat64, f64, 2, 4, 1, 2, vfmv_v_f, b64)

//512
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint8x64, vuint8, u8,     64, 128, 4, 8, vmv_v_x, b2)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int8x64, vint8, i8,       64, 128, 4, 8, vmv_v_x, b2)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint16x32, vuint16, u16,  32,  64, 4, 8, vmv_v_x, b4)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int16x32, vint16, i16,    32,  64, 4, 8, vmv_v_x, b4)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint32x16, vuint32, u32,  16,  32, 4, 8, vmv_v_x, b8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int32x16, vint32, i32,    16,  32, 4, 8, vmv_v_x, b8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint64x8, vuint64, u64,    8,  16, 4, 8, vmv_v_x, b16)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int64x8, vint64, i64,      8,  16, 4, 8, vmv_v_x, b16)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_float32x16, vfloat32, f32, 16, 32, 4, 8, vfmv_v_f, b8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_float64x8, vfloat64, f64,  8,  16, 4, 8, vfmv_v_f, b16)

#define OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(_Tpvec, _Tp, _Tp2, len, hnum, num, prefix) \
inline _Tpvec prefix##_load_low(const _Tp* ptr) \
{ return _Tpvec(vle_v_##len(ptr, hnum)); }\
inline _Tpvec prefix##_load_aligned(const _Tp* ptr) \
{ return _Tpvec(vle_v_##len(ptr, num)); } \
inline _Tpvec prefix##_load(const _Tp* ptr) \
{ return _Tpvec((_Tp2##_t)vle_v_##len((const _Tp *)ptr, num)); } \
inline _Tpvec prefix##_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
/*  typedef uint64 CV_DECL_ALIGNED(1) unaligned_uint64; \
  vuint64m1_t tmp = {*(unaligned_uint64*)ptr0, *(unaligned_uint64*)ptr1};\
    return _Tpvec(_Tp2##_t(tmp)); } */\
return v_rotate_left<hnum>(prefix##_load_low(ptr1), v_rotate_left<hnum>(prefix##_load_low(ptr0)));}\
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ vse_v_##len(ptr, a.val, hnum);}\
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
  _Tp2##_t a0 = vslidedown_vx_##len(a.val, hnum, num);    \
  vse_v_##len(ptr, a0, hnum);}\
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ vse_v_##len(ptr, a.val, num); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ vse_v_##len(ptr, a.val, num); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ vse_v_##len(ptr, a.val, num); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ vse_v_##len(ptr, a.val, num); }

OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint8x16,  uchar,          vuint8m1,  u8m1, 8, 16, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int8x16,   schar,           vint8m1,  i8m1, 8, 16, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint16x8,  ushort,        vuint16m1, u16m1, 4,  8, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int16x8,   short,          vint16m1, i16m1, 4,  8, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint32x4,  unsigned,      vuint32m1, u32m1, 2,  4, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int32x4,   int,            vint32m1, i32m1, 2,  4, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint64x2,  unsigned long, vuint64m1, u64m1, 1,  2, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int64x2,   long,           vint64m1, i64m1, 1,  2, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_float32x4, float,        vfloat32m1, f32m1, 2,  4, v)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_float64x2, double,       vfloat64m1, f64m1, 1,  2, v)

//512
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint8x64,   uchar,          vuint8m4,  u8m4, 32, 64, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int8x64,    schar,           vint8m4,  i8m4, 32, 64, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint16x32,  ushort,        vuint16m4, u16m4, 16, 32, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int16x32,   short,          vint16m4, i16m4, 16, 32, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint32x16,  unsigned,      vuint32m4, u32m4,  8, 16, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int32x16,   int,            vint32m4, i32m4,  8, 16, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint64x8,   unsigned long, vuint64m4, u64m4,  4,  8, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int64x8,    long,           vint64m4, i64m4,  4,  8, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_float32x16, float,        vfloat32m4, f32m4,  8, 16, v512)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_float64x8,  double,       vfloat64m4, f64m4,  4,  8, v512)

////////////// Lookup table access ////////////////////

inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
#if 1
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]],
        tab[idx[ 1]],
        tab[idx[ 2]],
        tab[idx[ 3]],
        tab[idx[ 4]],
        tab[idx[ 5]],
        tab[idx[ 6]],
        tab[idx[ 7]],
        tab[idx[ 8]],
        tab[idx[ 9]],
        tab[idx[10]],
        tab[idx[11]],
        tab[idx[12]],
        tab[idx[13]],
        tab[idx[14]],
        tab[idx[15]]
    };
    return v_int8x16(vle_v_i8m1(elems, 16));
#else
    int32xm4_t index32 = vlev_int32xm4(idx, 16);
    vint16m2_t index16 = vnsra_vx_i16m2_int32xm4(index32, 0, 16);
    vint8m1_t index = vnsra_vx_i8m1_i16m2(index16, 0, 16);
    return v_int8x16(vlxbv_i8m1(tab, index, 16));
#endif
}

inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx){
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[3]],
        tab[idx[3] + 1],
        tab[idx[4]],
        tab[idx[4] + 1],
        tab[idx[5]],
        tab[idx[5] + 1],
        tab[idx[6]],
        tab[idx[6] + 1],
        tab[idx[7]],
        tab[idx[7] + 1]
    };
    return v_int8x16(vle_v_i8m1(elems, 16));
}
inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[0] + 2],
        tab[idx[0] + 3],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[1] + 2],
        tab[idx[1] + 3],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[2] + 2],
        tab[idx[2] + 3],
        tab[idx[3]],
        tab[idx[3] + 1],
        tab[idx[3] + 2],
        tab[idx[3] + 3]
    };
    return v_int8x16(vle_v_i8m1(elems, 16));
}

inline v_uint8x16 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((schar*)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((schar*)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((schar*)tab, idx)); }

inline v_int16x8 v_lut(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]],
        tab[idx[4]],
        tab[idx[5]],
        tab[idx[6]],
        tab[idx[7]]
    };
    return v_int16x8(vle_v_i16m1(elems, 8));
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[3]],
        tab[idx[3] + 1]
    };
    return v_int16x8(vle_v_i16m1(elems, 8));
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[0] + 2],
        tab[idx[0] + 3],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[1] + 2],
        tab[idx[1] + 3]
    };
    return v_int16x8(vle_v_i16m1(elems, 8));
}
inline v_uint16x8 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((short*)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((short*)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((short*)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]]
    };
    return v_int32x4(vle_v_i32m1(elems, 4));
}
inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1]
    };
    return v_int32x4(vle_v_i32m1(elems, 4));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(vle_v_i32m1(tab+idx[0], 4));
}
inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((int*)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((int*)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((int*)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
{
    vint64m1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_int64x2(res);
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(vle_v_i64m1(tab+idx[0], 2));
}

inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx)
{
    vuint64m1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_uint64x2(res);
}
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx)
{
    return v_uint64x2(vle_v_u64m1(tab+idx[0], 2));
}

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]]
    };
    return v_float32x4(vle_v_f32m1(elems, 4));
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[0]+1],
        tab[idx[1]],
        tab[idx[1]+1]
    };
    return v_float32x4(vle_v_f32m1(elems, 4));
}
inline v_float32x4 v_lut_quads(const float* tab, const int* idx)
{
    return v_float32x4(vle_v_f32m1(tab + idx[0], 4));
}
inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    vfloat64m1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_float64x2(res);
}
inline v_float64x2 v_lut_pairs(const double* tab, const int* idx)
{
    return v_float64x2(vle_v_f64m1(tab+idx[0], 2));
}

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idxvec.val[0]],
        tab[idxvec.val[1]],
        tab[idxvec.val[2]],
        tab[idxvec.val[3]]
    };
    return v_int32x4(vle_v_i32m1(elems, 4));
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    unsigned CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idxvec.val[0]],
        tab[idxvec.val[1]],
        tab[idxvec.val[2]],
        tab[idxvec.val[3]]
    };
    return v_uint32x4(vle_v_u32m1(elems, 4));
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idxvec.val[0]],
        tab[idxvec.val[1]],
        tab[idxvec.val[2]],
        tab[idxvec.val[3]]
    };
    return v_float32x4(vle_v_f32m1(elems, 4));
}
inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    vfloat64m1_t res = {tab[idxvec.val[0]], tab[idxvec.val[1]]};
    return v_float64x2(res);
}
inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    vint32m1_t index_x = vmul_vx_i32m1(idxvec.val, 4, 4);
    vint32m1_t index_y = vadd_vx_i32m1(index_x, 4, 4);

    x.val = vlxe_v_f32m1(tab, index_x, 4);
    y.val = vlxe_v_f32m1(tab, index_y, 4);
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    x = v_float64x2(tab[idx[0]], tab[idx[1]]);
    y = v_float64x2(tab[idx[0]+1], tab[idx[1]+1]);
}

//512
inline v_int8x64 v512_lut(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[64] =
    {
        tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]],
        tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
        tab[idx[ 8]], tab[idx[ 9]], tab[idx[10]], tab[idx[11]],
        tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]],
        tab[idx[16]], tab[idx[17]], tab[idx[18]], tab[idx[19]],
        tab[idx[20]], tab[idx[21]], tab[idx[22]], tab[idx[23]],
        tab[idx[24]], tab[idx[25]], tab[idx[26]], tab[idx[27]],
        tab[idx[28]], tab[idx[29]], tab[idx[30]], tab[idx[31]],
        tab[idx[32]], tab[idx[33]], tab[idx[34]], tab[idx[35]],
        tab[idx[36]], tab[idx[37]], tab[idx[38]], tab[idx[39]],
        tab[idx[40]], tab[idx[41]], tab[idx[42]], tab[idx[43]],
        tab[idx[44]], tab[idx[45]], tab[idx[46]], tab[idx[47]],
        tab[idx[48]], tab[idx[49]], tab[idx[50]], tab[idx[51]],
        tab[idx[52]], tab[idx[53]], tab[idx[54]], tab[idx[55]],
        tab[idx[56]], tab[idx[57]], tab[idx[58]], tab[idx[59]],
        tab[idx[60]], tab[idx[61]], tab[idx[62]], tab[idx[63]]
    };
    return v_int8x64(vle_v_i8m4(elems, 64));
}

inline v_int8x64 v512_lut_pairs(const schar* tab, const int* idx){
    schar CV_DECL_ALIGNED(32) elems[64] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 1]], tab[idx[ 1] + 1],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 3]], tab[idx[ 3] + 1],
        tab[idx[ 4]], tab[idx[ 4] + 1], tab[idx[ 5]], tab[idx[ 5] + 1],
        tab[idx[ 6]], tab[idx[ 6] + 1], tab[idx[ 7]], tab[idx[ 7] + 1],
        tab[idx[ 8]], tab[idx[ 8] + 1], tab[idx[ 9]], tab[idx[ 9] + 1],
        tab[idx[10]], tab[idx[10] + 1], tab[idx[11]], tab[idx[11] + 1],
        tab[idx[12]], tab[idx[12] + 1], tab[idx[13]], tab[idx[13] + 1],
        tab[idx[14]], tab[idx[14] + 1], tab[idx[15]], tab[idx[15] + 1],
        tab[idx[16]], tab[idx[16] + 1], tab[idx[17]], tab[idx[17] + 1],
        tab[idx[18]], tab[idx[18] + 1], tab[idx[19]], tab[idx[19] + 1],
        tab[idx[20]], tab[idx[20] + 1], tab[idx[21]], tab[idx[21] + 1],
        tab[idx[22]], tab[idx[22] + 1], tab[idx[23]], tab[idx[23] + 1],
        tab[idx[24]], tab[idx[24] + 1], tab[idx[25]], tab[idx[25] + 1],
        tab[idx[26]], tab[idx[26] + 1], tab[idx[27]], tab[idx[27] + 1],
        tab[idx[28]], tab[idx[28] + 1], tab[idx[29]], tab[idx[29] + 1],
        tab[idx[30]], tab[idx[30] + 1], tab[idx[31]], tab[idx[31] + 1]
    };
    return v_int8x64(vle_v_i8m4(elems, 64));
}
inline v_int8x64 v512_lut_quads(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[64] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 0] + 2], tab[idx[ 0] + 3],
        tab[idx[ 1]], tab[idx[ 1] + 1], tab[idx[ 1] + 2], tab[idx[ 1] + 3],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 2] + 2], tab[idx[ 2] + 3],
        tab[idx[ 3]], tab[idx[ 3] + 1], tab[idx[ 3] + 2], tab[idx[ 3] + 3],
        tab[idx[ 4]], tab[idx[ 4] + 1], tab[idx[ 4] + 2], tab[idx[ 4] + 3],
        tab[idx[ 5]], tab[idx[ 5] + 1], tab[idx[ 5] + 2], tab[idx[ 5] + 3],
        tab[idx[ 6]], tab[idx[ 6] + 1], tab[idx[ 6] + 2], tab[idx[ 6] + 3],
        tab[idx[ 7]], tab[idx[ 7] + 1], tab[idx[ 7] + 2], tab[idx[ 7] + 3],
        tab[idx[ 8]], tab[idx[ 8] + 1], tab[idx[ 8] + 2], tab[idx[ 8] + 3],
        tab[idx[ 9]], tab[idx[ 9] + 1], tab[idx[ 9] + 2], tab[idx[ 9] + 3],
        tab[idx[10]], tab[idx[10] + 1], tab[idx[10] + 2], tab[idx[10] + 3],
        tab[idx[11]], tab[idx[11] + 1], tab[idx[11] + 2], tab[idx[11] + 3],
        tab[idx[12]], tab[idx[12] + 1], tab[idx[12] + 2], tab[idx[12] + 3],
        tab[idx[13]], tab[idx[13] + 1], tab[idx[13] + 2], tab[idx[13] + 3],
        tab[idx[14]], tab[idx[14] + 1], tab[idx[14] + 2], tab[idx[14] + 3],
        tab[idx[15]], tab[idx[15] + 1], tab[idx[15] + 2], tab[idx[15] + 3]
    };
    return v_int8x64(vle_v_i8m4(elems, 64));
}

inline v_uint8x64 v512_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut((schar*)tab, idx)); }
inline v_uint8x64 v512_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut_pairs((schar*)tab, idx)); }
inline v_uint8x64 v512_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut_quads((schar*)tab, idx)); }

inline v_int16x32 v512_lut(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[32] =
    {
        tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]],
        tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
        tab[idx[ 8]], tab[idx[ 9]], tab[idx[10]], tab[idx[11]],
        tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]],
        tab[idx[16]], tab[idx[17]], tab[idx[18]], tab[idx[19]],
        tab[idx[20]], tab[idx[21]], tab[idx[22]], tab[idx[23]],
        tab[idx[24]], tab[idx[25]], tab[idx[26]], tab[idx[27]],
        tab[idx[28]], tab[idx[29]], tab[idx[30]], tab[idx[31]]
    };
    return v_int16x32(vle_v_i16m4(elems, 32));
}
inline v_int16x32 v512_lut_pairs(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[32] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 1]], tab[idx[ 1] + 1],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 3]], tab[idx[ 3] + 1],
        tab[idx[ 4]], tab[idx[ 4] + 1], tab[idx[ 5]], tab[idx[ 5] + 1],
        tab[idx[ 6]], tab[idx[ 6] + 1], tab[idx[ 7]], tab[idx[ 7] + 1],
        tab[idx[ 8]], tab[idx[ 8] + 1], tab[idx[ 9]], tab[idx[ 9] + 1],
        tab[idx[10]], tab[idx[10] + 1], tab[idx[11]], tab[idx[11] + 1],
        tab[idx[12]], tab[idx[12] + 1], tab[idx[13]], tab[idx[13] + 1],
        tab[idx[14]], tab[idx[14] + 1], tab[idx[15]], tab[idx[15] + 1]
    };
    return v_int16x32(vle_v_i16m4(elems, 32));
}
inline v_int16x32 v512_lut_quads(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[32] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 0] + 2], tab[idx[ 0] + 3],
        tab[idx[ 1]], tab[idx[ 1] + 1], tab[idx[ 1] + 2], tab[idx[ 1] + 3],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 2] + 2], tab[idx[ 2] + 3],
        tab[idx[ 3]], tab[idx[ 3] + 1], tab[idx[ 3] + 2], tab[idx[ 3] + 3],
        tab[idx[ 4]], tab[idx[ 4] + 1], tab[idx[ 4] + 2], tab[idx[ 4] + 3],
        tab[idx[ 5]], tab[idx[ 5] + 1], tab[idx[ 5] + 2], tab[idx[ 5] + 3],
        tab[idx[ 6]], tab[idx[ 6] + 1], tab[idx[ 6] + 2], tab[idx[ 6] + 3],
        tab[idx[ 7]], tab[idx[ 7] + 1], tab[idx[ 7] + 2], tab[idx[ 7] + 3]
    };
    return v_int16x32(vle_v_i16m4(elems, 32));
}
inline v_uint16x32 v512_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut((short*)tab, idx)); }
inline v_uint16x32 v512_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut_pairs((short*)tab, idx)); }
inline v_uint16x32 v512_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut_quads((short*)tab, idx)); }

inline v_int32x16 v512_lut(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]],
        tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
        tab[idx[ 8]], tab[idx[ 9]], tab[idx[10]], tab[idx[11]],
        tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]]
    };
    return v_int32x16(vle_v_i32m4(elems, 16));
}
inline v_int32x16 v512_lut_pairs(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 1]], tab[idx[ 1] + 1],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 3]], tab[idx[ 3] + 1],
        tab[idx[ 4]], tab[idx[ 4] + 1], tab[idx[ 5]], tab[idx[ 5] + 1],
        tab[idx[ 6]], tab[idx[ 6] + 1], tab[idx[ 7]], tab[idx[ 7] + 1]
    };
    return v_int32x16(vle_v_i32m4(elems, 16));
}
inline v_int32x16 v512_lut_quads(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 0] + 2], tab[idx[ 0] + 3],
        tab[idx[ 1]], tab[idx[ 1] + 1], tab[idx[ 1] + 2], tab[idx[ 1] + 3],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 2] + 2], tab[idx[ 2] + 3],
        tab[idx[ 3]], tab[idx[ 3] + 1], tab[idx[ 3] + 2], tab[idx[ 3] + 3]
    };
    return v_int32x16(vle_v_i32m4(elems, 16));
}
inline v_uint32x16 v512_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut((int*)tab, idx)); }
inline v_uint32x16 v512_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut_pairs((int*)tab, idx)); }
inline v_uint32x16 v512_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut_quads((int*)tab, idx)); }

inline v_int64x8 v512_lut(const int64_t* tab, const int* idx)
{
    int64_t CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]],
        tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]]
    };
    return v_int64x8(vle_v_i64m4(elems, 8));
}
inline v_int64x8 v512_lut_pairs(const int64_t* tab, const int* idx)
{
    int64_t CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 1]], tab[idx[ 1] + 1],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 3]], tab[idx[ 3] + 1]
    };
    return v_int64x8(vle_v_i64m4(elems, 8));
}
inline v_uint64x8 v512_lut(const unsigned long* tab, const int* idx) { return v_reinterpret_as_u64(v512_lut((int64_t*)tab, idx)); }
inline v_uint64x8 v512_lut_pairs(const unsigned long* tab, const int* idx) { return v_reinterpret_as_u64(v512_lut_pairs((int64_t*)tab, idx)); }


inline v_float32x16 v512_lut(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]],
        tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
        tab[idx[ 8]], tab[idx[ 9]], tab[idx[10]], tab[idx[11]],
        tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]]
    };
    return v_float32x16(vle_v_f32m4(elems, 16));
}
inline v_float32x16 v512_lut_pairs(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 1]], tab[idx[ 1] + 1],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 3]], tab[idx[ 3] + 1],
        tab[idx[ 4]], tab[idx[ 4] + 1], tab[idx[ 5]], tab[idx[ 5] + 1],
        tab[idx[ 6]], tab[idx[ 6] + 1], tab[idx[ 7]], tab[idx[ 7] + 1]
    };
    return v_float32x16(vle_v_f32m4(elems, 16));
}
inline v_float32x16 v512_lut_quads(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 0] + 2], tab[idx[ 0] + 3],
        tab[idx[ 1]], tab[idx[ 1] + 1], tab[idx[ 1] + 2], tab[idx[ 1] + 3],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 2] + 2], tab[idx[ 2] + 3],
        tab[idx[ 3]], tab[idx[ 3] + 1], tab[idx[ 3] + 2], tab[idx[ 3] + 3]
    };
    return v_float32x16(vle_v_f32m4(elems, 16));
}
inline v_float64x8 v512_lut(const double* tab, const int* idx)
{
    double CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]],
        tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]]
    };
    return v_float64x8(vle_v_f64m4(elems, 8));
}
inline v_float64x8 v512_lut_pairs(const double* tab, const int* idx)
{
    double CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[ 0]], tab[idx[ 0] + 1], tab[idx[ 1]], tab[idx[ 1] + 1],
        tab[idx[ 2]], tab[idx[ 2] + 1], tab[idx[ 3]], tab[idx[ 3] + 1]
    };
    return v_float64x8(vle_v_f64m4(elems, 8));
}

inline v_int32x16 v_lut(const int* tab, const v_int32x16& idxvec)
{
    int CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idxvec.val[ 0]],tab[idxvec.val[ 1]],tab[idxvec.val[ 2]],tab[idxvec.val[ 3]],
        tab[idxvec.val[ 4]],tab[idxvec.val[ 5]],tab[idxvec.val[ 6]],tab[idxvec.val[ 7]],
        tab[idxvec.val[ 8]],tab[idxvec.val[ 9]],tab[idxvec.val[10]],tab[idxvec.val[11]],
        tab[idxvec.val[12]],tab[idxvec.val[13]],tab[idxvec.val[14]],tab[idxvec.val[15]]
    };
    return v_int32x16(vle_v_i32m4(elems, 16));
}

inline v_uint32x16 v_lut(const unsigned* tab, const v_int32x16& idxvec)
{
    unsigned CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idxvec.val[ 0]],tab[idxvec.val[ 1]],tab[idxvec.val[ 2]],tab[idxvec.val[ 3]],
        tab[idxvec.val[ 4]],tab[idxvec.val[ 5]],tab[idxvec.val[ 6]],tab[idxvec.val[ 7]],
        tab[idxvec.val[ 8]],tab[idxvec.val[ 9]],tab[idxvec.val[10]],tab[idxvec.val[11]],
        tab[idxvec.val[12]],tab[idxvec.val[13]],tab[idxvec.val[14]],tab[idxvec.val[15]]
    };
    return v_uint32x16(vle_v_u32m4(elems, 16));
}

inline v_float32x16 v_lut(const float* tab, const v_int32x16& idxvec)
{
    float CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idxvec.val[ 0]],tab[idxvec.val[ 1]],tab[idxvec.val[ 2]],tab[idxvec.val[ 3]],
        tab[idxvec.val[ 4]],tab[idxvec.val[ 5]],tab[idxvec.val[ 6]],tab[idxvec.val[ 7]],
        tab[idxvec.val[ 8]],tab[idxvec.val[ 9]],tab[idxvec.val[10]],tab[idxvec.val[11]],
        tab[idxvec.val[12]],tab[idxvec.val[13]],tab[idxvec.val[14]],tab[idxvec.val[15]]
    };
    return v_float32x16(vle_v_f32m4(elems, 16));
}
inline v_float64x8 v_lut(const double* tab, const v_int32x16& idxvec)
{
    vfloat64m4_t res = {tab[idxvec.val[ 0]],tab[idxvec.val[ 1]],
                        tab[idxvec.val[ 2]],tab[idxvec.val[ 3]],
                        tab[idxvec.val[ 4]],tab[idxvec.val[ 5]],
                        tab[idxvec.val[ 6]],tab[idxvec.val[ 7]]};
    return v_float64x8(res);
}
inline void v_lut_deinterleave(const float* tab, const v_int32x16& idxvec, v_float32x16& x, v_float32x16& y)
{
    vint32m4_t index_x = vmul_vx_i32m4(idxvec.val, 4, 16);
    vint32m4_t index_y = vadd_vx_i32m4(index_x, 4, 16);

    x.val = vlxe_v_f32m4(tab, index_x, 16);
    y.val = vlxe_v_f32m4(tab, index_y, 16);
}

inline void v_lut_deinterleave(const double* tab, const v_int32x16& idxvec, v_float64x8& x, v_float64x8& y)
{
    int CV_DECL_ALIGNED(32) idx[16];
    v_store_aligned(idx, idxvec);

    x = v_float64x8(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
                    tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]);
    y = v_float64x8(tab[idx[0]+1], tab[idx[1]+1],tab[idx[2]+1], tab[idx[3]+1],
                    tab[idx[4]+1], tab[idx[5]+1],tab[idx[6]+1], tab[idx[7]+1]);
}

//#define OPENCV_HAL_IMPL_RISCVV_PACKS(_Tp, _Tp2, _T2, num2, _T1, num, intrin, shr, _Type, regnum, regnum2) \
inline v_##_Tp##x##num v_pack(const v_##_Tp2##x##num2& a, const v_##_Tp2##x##num2& b) \
{ \
    v##_Tp2##m##regnum2##_t  tmp = vundefined_##_T2##m##regnum2();    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 1, b.val);    \
    return v_##_Tp##x##num(shr##_##_T1##m##regnum(tmp, 0, num)); \
}\
template<int n> inline \
v_##_Tp##x##num v_rshr_pack(const v_##_Tp2##x##num2& a, const v_##_Tp2##x##num2& b) \
{ \
    v##_Tp2##m##regnum2##_t  tmp = vundefined_##_T2##m##regnum2();    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 1, b.val);    \
    return v_##_Tp##x##num(intrin##_##_T1##m##regnum(tmp, n, num)); \
}\
inline void v_pack_store(_Type* ptr, const v_##_Tp2##x##num2& a) \
{ \
    v##_Tp2##m##regnum2##_t tmp = vundefined_##_T2##m##regnum2();    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 1, vmv_v_x_##_T2##m##regnum(0, num2));    \
    asm("" ::: "memory");                                       \
    vse_v_##_T1##m##regnum(ptr, shr##_##_T1##m##regnum(tmp, 0, num), num2); \
}\
template<int n> inline \
void v_rshr_pack_store(_Type* ptr, const v_##_Tp2##x##num2& a) \
{ \
    v##_Tp2##m##regnum2##_t tmp = vundefined_##_T2##m##regnum2();    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 1, vmv_v_x_##_T2##m##regnum(0, num2));    \
    vse_v_##_T1##m##regnum(ptr, intrin##_##_T1##m##regnum(tmp, n, num), num2); \
}

#define OPENCV_HAL_IMPL_RISCVV_PACKS(_Tp, _Tp2, _T2, num2, _T1, num, intrin, shr, _Type, regnum, regnum2) \
inline v_##_Tp##x##num v_pack(const v_##_Tp2##x##num2& a, const v_##_Tp2##x##num2& b) \
{ \
    v##_Tp2##m##regnum2##_t  tmp = vundefined_##_T2##m##regnum2();    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 1, b.val);    \
    return v_##_Tp##x##num(shr##_##_T1##m##regnum(tmp, 0, num)); \
}\
template<int n> inline \
v_##_Tp##x##num v_rshr_pack(const v_##_Tp2##x##num2& a, const v_##_Tp2##x##num2& b) \
{ \
    v##_Tp2##m##regnum2##_t  tmp = vundefined_##_T2##m##regnum2();    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 1, b.val);    \
    return v_##_Tp##x##num(intrin##_##_T1##m##regnum(tmp, n, num)); \
}\
inline void v_pack_store(_Type* ptr, const v_##_Tp2##x##num2& a) \
{ \
    v##_Tp2##m##regnum2##_t tmp = vundefined_##_T2##m##regnum2();    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 1, vmv_v_x_##_T2##m##regnum(0, num2));    \
    asm("" ::: "memory");                                       \
    vse_v_##_T1##m##regnum(ptr, shr##_##_T1##m##regnum(tmp, 0, num), num2); \
}\
template<int n> inline \
void v_rshr_pack_store(_Type* ptr, const v_##_Tp2##x##num2& a) \
{ \
    v##_Tp2##m##regnum2##_t tmp = vundefined_##_T2##m##regnum2();    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##_T2##m##regnum2##_##_T2##m##regnum(tmp, 1, vmv_v_x_##_T2##m##regnum(0, num2));    \
    vse_v_##_T1##m##regnum(ptr, intrin##_##_T1##m##regnum(tmp, n, num), num2); \
}

OPENCV_HAL_IMPL_RISCVV_PACKS(int8, int16, i16, 8, i8, 16, vnclip_vx, vnclip_vx, signed char, 1, 2)
OPENCV_HAL_IMPL_RISCVV_PACKS(int16, int32, i32, 4, i16, 8, vnclip_vx, vnclip_vx, signed short, 1, 2)
OPENCV_HAL_IMPL_RISCVV_PACKS(int32, int64, i64, 2, i32, 4, vnclip_vx, vnsra_vx, int, 1, 2)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint8, uint16, u16, 8, u8, 16, vnclipu_vx, vnclipu_vx, unsigned char, 1, 2)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint16, uint32, u32, 4, u16, 8, vnclipu_vx, vnclipu_vx, unsigned short, 1, 2)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint32, uint64, u64, 2, u32, 4, vnclipu_vx, vnsrl_vx, unsigned int, 1, 2)

//512
OPENCV_HAL_IMPL_RISCVV_PACKS(int8, int16, i16, 32, i8, 64, vnclip_vx, vnclip_vx, signed char, 4, 8)
OPENCV_HAL_IMPL_RISCVV_PACKS(int16, int32, i32, 16, i16, 32, vnclip_vx, vnclip_vx, signed short, 4, 8)
OPENCV_HAL_IMPL_RISCVV_PACKS(int32, int64, i64, 8, i32, 16, vnclip_vx, vnsra_vx, int, 4, 8)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint8, uint16, u16, 32, u8, 64, vnclipu_vx, vnclipu_vx, unsigned char, 4, 8)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint16, uint32, u32, 16, u16, 32, vnclipu_vx, vnclipu_vx, unsigned short, 4, 8)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint32, uint64, u64, 8, u32, 16, vnclipu_vx, vnsrl_vx, unsigned int, 4, 8)

// pack boolean
inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    vuint16m2_t tmp = vundefined_u16m2();    \
    tmp = vset_u16m2(tmp, 0, a.val);    \
    tmp = vset_u16m2(tmp, 1, b.val);    \
    return v_uint8x16(vnsrl_vx_u8m1(tmp, 0, 16));
}

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    vuint32m4_t vabcd = vundefined_u32m4();    \
    vuint16m2_t v16 = vundefined_u16m2();    \
    vabcd = vset_u32m4(vabcd, 0, a.val);    \
    vabcd = vset_u32m4(vabcd, 1, b.val);    \
    vabcd = vset_u32m4(vabcd, 2, c.val);    \
    vabcd = vset_u32m4(vabcd, 3, d.val);    \
    v16 = vnsrl_vx_u16m2(vabcd, 0, 16);
    return v_uint8x16(vnsrl_vx_u8m1(v16, 0, 16));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    vuint64m8_t v64 = vundefined_u64m8();    \
    vuint32m4_t v32 = vundefined_u32m4();    \
    vuint16m2_t v16 = vundefined_u16m2();    \
    v64 = vset_u64m8(v64, 0, a.val);    \
    v64 = vset_u64m8(v64, 1, b.val);    \
    v64 = vset_u64m8(v64, 2, c.val);    \
    v64 = vset_u64m8(v64, 3, d.val);    \
    v64 = vset_u64m8(v64, 4, e.val);    \
    v64 = vset_u64m8(v64, 5, f.val);    \
    v64 = vset_u64m8(v64, 6, g.val);    \
    v64 = vset_u64m8(v64, 7, h.val);    \
    v32 = vnsrl_vx_u32m4(v64, 0, 16);   \
    v16 = vnsrl_vx_u16m2(v32, 0, 16);   \
    return v_uint8x16(vnsrl_vx_u8m1(v16, 0, 16));
}

// pack boolean 512
inline v_uint8x64 v_pack_b(const v_uint16x32& a, const v_uint16x32& b)
{
    vuint16m8_t tmp = vundefined_u16m8();
    tmp = vset_u16m8_u16m4(tmp, 0, a.val);
    tmp = vset_u16m8_u16m4(tmp, 1, b.val);
    return v_uint8x64(vnsrl_vx_u8m4(tmp, 0, 64));
}

inline v_uint8x64 v_pack_b(const v_uint32x16& a, const v_uint32x16& b,
                           const v_uint32x16& c, const v_uint32x16& d)
{
    vuint8m4_t res = vundefined_u8m4();
    vuint32m8_t v32 = vundefined_u32m8();
    vuint16m4_t v16 = vundefined_u16m4();
    v32 = vset_u32m8_u32m4(v32, 0, a.val);
    v32 = vset_u32m8_u32m4(v32, 1, b.val);
    v16 = vnsrl_vx_u16m4(v32, 0, 32);
    res = vset_u8m4_u8m2(res, 0, vnsrl_vx_u8m2(v16, 0, 32));
    v32 = vset_u32m8_u32m4(v32, 0, c.val);
    v32 = vset_u32m8_u32m4(v32, 1, d.val);
    v16 = vnsrl_vx_u16m4(v32, 0, 32);
    res = vset_u8m4_u8m2(res, 1, vnsrl_vx_u8m2(v16, 0, 32));
    return v_uint8x64(res);
}

inline v_uint8x64 v_pack_b(const v_uint64x8& a, const v_uint64x8& b, const v_uint64x8& c,
                           const v_uint64x8& d, const v_uint64x8& e, const v_uint64x8& f,
                           const v_uint64x8& g, const v_uint64x8& h)
{
    vuint8m4_t res = vundefined_u8m4();
    vuint64m8_t v64 = vundefined_u64m8();
    vuint16m2_t v16 = vundefined_u16m2();
    v64 = vset_u64m8_u64m4(v64, 0, a.val);
    v64 = vset_u64m8_u64m4(v64, 1, b.val);
    v16 = vnsrl_vx_u16m2(vnsrl_vx_u32m4(v64, 0, 16), 0, 16);
    res = vset_u8m4_u8m1(res, 0, vnsrl_vx_u8m1(v16, 0, 16));
    v64 = vset_u64m8_u64m4(v64, 0, c.val);
    v64 = vset_u64m8_u64m4(v64, 1, d.val);
    v16 = vnsrl_vx_u16m2(vnsrl_vx_u32m4(v64, 0, 16), 0, 16);
    res = vset_u8m4_u8m1(res, 1, vnsrl_vx_u8m1(v16, 0, 16));
    v64 = vset_u64m8_u64m4(v64, 0, e.val);
    v64 = vset_u64m8_u64m4(v64, 1, f.val);
    v16 = vnsrl_vx_u16m2(vnsrl_vx_u32m4(v64, 0, 16), 0, 16);
    res = vset_u8m4_u8m1(res, 2, vnsrl_vx_u8m1(v16, 0, 16));
    v64 = vset_u64m8_u64m4(v64, 0, g.val);
    v64 = vset_u64m8_u64m4(v64, 1, h.val);
    v16 = vnsrl_vx_u16m2(vnsrl_vx_u32m4(v64, 0, 16), 0, 16);
    res = vset_u8m4_u8m1(res, 3, vnsrl_vx_u8m1(v16, 0, 16));
    return v_uint8x64(res);
}

//inline v_uint8x16 v_pack_u(const v_int16x8& a, const v_int16x8& b) \
//{ \
//    int16xm2_u tmp;    \
//    tmp.m1[0] = (vint16m1_t)a.val;    \
//    tmp.m1[1] = (vint16m1_t)b.val;    \
//    e8xm1_t mask = (e8xm1_t)vmsge_vx_e16xm2_i16m2(tmp.v, 0, 16);\
//    return v_uint8x16(vnclipuvi_mask_u8m1_u16m2(vmv_v_x_u8m1(0, 16), (vuint16m2_t)tmp.v, 0, mask, 16));
//}

#define OPENCV_HAL_IMPL_RISCVV_PACK_U(tp1, num1, tp2, num2, _Tp, regnum, regnum2) \
inline v_uint##tp1##x##num1 v_pack_u(const v_int##tp2##x##num2& a, const v_int##tp2##x##num2& b) \
{ \
    vint##tp2##m##regnum2##_t tmp = vundefined_##i##tp2##m##regnum2();    \
    tmp = vset_##i##tp2##m##regnum2##_##i##tp2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##i##tp2##m##regnum2##_##i##tp2##m##regnum(tmp, 1, b.val);    \
    vint##tp2##m##regnum2##_t val = vmax_vx_i##tp2##m##regnum2(tmp, 0, num1);\
    return v_uint##tp1##x##num1(vnclipu_vx_u##tp1##m##regnum((vuint##tp2##m##regnum2##_t)val, 0, num1));    \
} \
inline void v_pack_u_store(_Tp* ptr, const v_int##tp2##x##num2& a) \
{ \
    vint##tp2##m##regnum2##_t tmp = vundefined_##i##tp2##m##regnum2();    \
    tmp = vset_##i##tp2##m##regnum2##_##i##tp2##m##regnum(tmp, 0, a.val);    \
    vint##tp2##m##regnum2##_t val = vmax_vx_i##tp2##m##regnum2(tmp, 0, num1);\
    return vse_v_u##tp1##m##regnum(ptr, vnclipu_vx_u##tp1##m##regnum((vuint##tp2##m##regnum2##_t)val, 0, num1), num2);    \
} \
template<int n> inline \
v_uint##tp1##x##num1 v_rshr_pack_u(const v_int##tp2##x##num2& a, const v_int##tp2##x##num2& b) \
{ \
    vint##tp2##m##regnum2##_t tmp = vundefined_##i##tp2##m##regnum2();    \
    tmp = vset_##i##tp2##m##regnum2##_##i##tp2##m##regnum(tmp, 0, a.val);    \
    tmp = vset_##i##tp2##m##regnum2##_##i##tp2##m##regnum(tmp, 1, b.val);    \
    vint##tp2##m##regnum2##_t val = vmax_vx_i##tp2##m##regnum2(tmp, 0, num1);\
    return v_uint##tp1##x##num1(vnclipu_vx_u##tp1##m##regnum((vuint##tp2##m##regnum2##_t)val, n, num1));    \
} \
template<int n> inline \
void v_rshr_pack_u_store(_Tp* ptr, const v_int##tp2##x##num2& a) \
{ \
    vint##tp2##m##regnum2##_t tmp = vundefined_##i##tp2##m##regnum2();    \
    tmp = vset_##i##tp2##m##regnum2##_##i##tp2##m##regnum(tmp, 0, a.val);    \
    vint##tp2##m##regnum2##_t val_ = vmax_vx_i##tp2##m##regnum2(tmp, 0, num1);\
    vuint##tp1##m##regnum##_t val = vnclipu_vx_u##tp1##m##regnum((vuint##tp2##m##regnum2##_t)val_, n, num1);    \
    return vse_v_u##tp1##m##regnum(ptr, val, num2);\
}
OPENCV_HAL_IMPL_RISCVV_PACK_U(8, 16, 16, 8, unsigned char , 1, 2)
OPENCV_HAL_IMPL_RISCVV_PACK_U(16, 8, 32, 4, unsigned short, 1, 2)

//512
OPENCV_HAL_IMPL_RISCVV_PACK_U( 8, 64, 16, 32, unsigned char , 4, 8)
OPENCV_HAL_IMPL_RISCVV_PACK_U(16, 32, 32, 16, unsigned short, 4, 8)

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

// saturating multiply 8-bit, 16-bit
#define OPENCV_HAL_IMPL_RISCVV_MUL_SAT(_Tpvec, _Tpwvec)            \
    inline _Tpvec operator * (const _Tpvec& a, const _Tpvec& b)  \
    {                                                            \
        _Tpwvec c, d;                                            \
        v_mul_expand(a, b, c, d);                                \
        return v_pack(c, d);                                     \
    }                                                            \
    inline _Tpvec& operator *= (_Tpvec& a, const _Tpvec& b)      \
    { a = a * b; return a; }

OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_int8x16,  v_int16x8)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_uint8x16, v_uint16x8)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_int16x8,  v_int32x4)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_uint16x8, v_uint32x4)

//512
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_int8x64,  v_int16x32)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_uint8x64, v_uint16x32)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_int16x32,  v_int32x16)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_uint16x32, v_uint32x16)

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
static const signed char popCountTable[256] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

inline vuint8m1_t vcnt_u8(vuint8m1_t val){
    vuint8m1_t v0 = val & 1;
    return vlxe_v_u8m1((unsigned char*)popCountTable, val >> 1, 16)+v0;
}
//512
inline vuint8m4_t vcnt_u8(vuint8m4_t val){
    vuint8m4_t v0 = val & 1;
    return vlxe_v_u8m4((unsigned char*)popCountTable, val >> 1, 64)+v0;
}

inline v_uint8x16
v_popcount(const v_uint8x16& a)
{
    return v_uint8x16(vcnt_u8(a.val));
}

inline v_uint8x16
v_popcount(const v_int8x16& a)
{
    return v_uint8x16(vcnt_u8((vuint8m1_t)a.val));
}

inline v_uint16x8
v_popcount(const v_uint16x8& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0x0E0C0A0806040200, 0, 0x0F0D0B0907050301, 0};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint16m2_t res = vwaddu_vv_u16m2(vget_u8m2_u8m1(tmp, 0), vget_u8m2_u8m1(tmp, 1), 8);
    return v_uint16x8(vget_u16m2_u16m1(res, 0));
}

inline v_uint16x8
v_popcount(const v_int16x8& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0x0E0C0A0806040200, 0, 0x0F0D0B0907050301, 0};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint16m2_t res = vwaddu_vv_u16m2(vget_u8m2_u8m1(tmp, 0), vget_u8m2_u8m1(tmp, 1), 8);
    return v_uint16x8(vget_u16m2_u16m1(res, 0));
}

inline v_uint32x4
v_popcount(const v_uint32x4& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0xFFFFFFFF0C080400, 0xFFFFFFFF0D090501,
                     0xFFFFFFFF0E0A0602, 0xFFFFFFFF0F0B0703};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint16m2_t res_ = vwaddu_vv_u16m2(vget_u8m2_u8m1(tmp, 0), vget_u8m2_u8m1(tmp, 1), 16);
    vuint32m2_t res  = vwaddu_vv_u32m2(vget_u16m2_u16m1(res_, 0), vget_u16m2_u16m1(res_, 1), 8);
    return v_uint32x4(vget_u32m2_u32m1(res, 0));
}

inline v_uint32x4
v_popcount(const v_int32x4& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0xFFFFFFFF0C080400, 0xFFFFFFFF0D090501,
                     0xFFFFFFFF0E0A0602, 0xFFFFFFFF0F0B0703};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint16m2_t res_ = vwaddu_vv_u16m2(vget_u8m2_u8m1(tmp, 0), vget_u8m2_u8m1(tmp, 1), 16);
    vuint32m2_t res  = vwaddu_vv_u32m2(vget_u16m2_u16m1(res_, 0), vget_u16m2_u16m1(res_, 1), 8);
    return v_uint32x4(vget_u32m2_u32m1(res, 0));
}

inline v_uint64x2
v_popcount(const v_uint64x2& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0x0706050403020100, 0x0000000000000000,
                     0x0F0E0D0C0B0A0908, 0x0000000000000000};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint8m1_t zero = vmv_v_x_u8m1(0, 16);
    vuint8m1_t res1 = zero;
    vuint8m1_t res2 = zero;
    res1 = vredsum_vs_u8m1_u8m1(res1, vget_u8m2_u8m1(tmp, 0), zero, 8);
    res2 = vredsum_vs_u8m1_u8m1(res2, vget_u8m2_u8m1(tmp, 1), zero, 8);

    return v_uint64x2((unsigned long)vmv_x_s_u8m1_u8(res1, 8), (unsigned long)vmv_x_s_u8m1_u8(res2, 8));
}

inline v_uint64x2
v_popcount(const v_int64x2& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0x0706050403020100, 0x0000000000000000,
                     0x0F0E0D0C0B0A0908, 0x0000000000000000};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint8m1_t zero = vmv_v_x_u8m1(0, 16);
    vuint8m1_t res1 = zero;
    vuint8m1_t res2 = zero;
    res1 = vredsum_vs_u8m1_u8m1(res1, vget_u8m2_u8m1(tmp, 0), zero, 8);
    res2 = vredsum_vs_u8m1_u8m1(res2, vget_u8m2_u8m1(tmp, 1), zero, 8);

    return v_uint64x2((unsigned long)vmv_x_s_u8m1_u8(res1, 8), (unsigned long)vmv_x_s_u8m1_u8(res2, 8));
}

//512
inline v_uint8x64
v_popcount(const v_uint8x64& a)
{
    return v_uint8x64(vcnt_u8(a.val));
}

inline v_uint8x64
v_popcount(const v_int8x64& a)
{
    return v_uint8x64(vcnt_u8((vuint8m4_t)a.val));
}

inline v_uint16x32
v_popcount(const v_uint16x32& a)
{
    vuint8m8_t tmp = vundefined_u8m8();
    tmp = vset_u8m8_u8m4(tmp, 0, vcnt_u8((vuint8m4_t)a.val));
    vuint64m8_t mask = (vuint64m8_t){0x3E3C3A3836343230, 0x2E2C2A2826242220, 0x1E1C1A1816141210, 0x0E0C0A0806040200, 0,0,0,0, 
                                     0x3F3D3B3937353331, 0x2F2D2B2927252321, 0x1F1D1B1917151311, 0x0F0D0B0907050301, 0,0,0,0};
    tmp = vrgather_vv_u8m8(tmp, (vuint8m8_t)mask, 128);
    vuint16m8_t res = vwaddu_vv_u16m8(vget_u8m8_u8m4(tmp, 0), vget_u8m8_u8m4(tmp, 1), 32);
    return v_uint16x32(vget_u16m8_u16m4(res, 0));
}

inline v_uint16x32
v_popcount(const v_int16x32& a)
{
    vuint8m8_t tmp = vundefined_u8m8();
    tmp = vset_u8m8_u8m4(tmp, 0, vcnt_u8((vuint8m4_t)a.val));
    vuint64m8_t mask = (vuint64m8_t){0x3E3C3A3836343230, 0x2E2C2A2826242220, 0x1E1C1A1816141210, 0x0E0C0A0806040200, 0,0,0,0, 
                                     0x3F3D3B3937353331, 0x2F2D2B2927252321, 0x1F1D1B1917151311, 0x0F0D0B0907050301, 0,0,0,0};
    tmp = vrgather_vv_u8m8(tmp, (vuint8m8_t)mask, 128);
    vuint16m8_t res = vwaddu_vv_u16m8(vget_u8m8_u8m4(tmp, 0), vget_u8m8_u8m4(tmp, 1), 32);
    return v_uint16x32(vget_u16m8_u16m4(res, 0));
}

inline v_uint32x16
v_popcount(const v_uint32x16& a)
{
    vuint8m8_t tmp = vundefined_u8m8();
    tmp = vset_u8m8_u8m4(tmp, 0, vcnt_u8((vuint8m4_t)a.val));
    vuint64m8_t mask = (vuint64m8_t){0x1C1814100C080400, 0x3C3834302C282420, 0,0,
                                     0x1D1915110D090501, 0x3D3935312D292521, 0,0,
                                     0x1E1A16120E0A0602, 0x3E3A36322E2A2622, 0,0, 
                                     0x1F1B17130F0B0703, 0x3F3B37332F2B2723, 0,0};
    tmp = vrgather_vv_u8m8(tmp, (vuint8m8_t)mask, 128);
    vuint16m8_t res_ = vwaddu_vv_u16m8(vget_u8m8_u8m4(tmp, 0), vget_u8m8_u8m4(tmp, 1), 64);
    vuint32m8_t res  = vwaddu_vv_u32m8(vget_u16m8_u16m4(res_, 0), vget_u16m8_u16m4(res_, 1), 32);
    return v_uint32x16(vget_u32m8_u32m4(res, 0));
}

inline v_uint32x16
v_popcount(const v_int32x16& a)
{
    vuint8m8_t tmp = vundefined_u8m8();
    tmp = vset_u8m8_u8m4(tmp, 0, vcnt_u8((vuint8m4_t)a.val));
    vuint64m8_t mask = (vuint64m8_t){0x1C1814100C080400, 0x3C3834302C282420, 0,0,
                                     0x1D1915110D090501, 0x3D3935312D292521, 0,0,
                                     0x1E1A16120E0A0602, 0x3E3A36322E2A2622, 0,0, 
                                     0x1F1B17130F0B0703, 0x3F3B37332F2B2723, 0,0};
    tmp = vrgather_vv_u8m8(tmp, (vuint8m8_t)mask, 128);
    vuint16m8_t res_ = vwaddu_vv_u16m8(vget_u8m8_u8m4(tmp, 0), vget_u8m8_u8m4(tmp, 1), 64);
    vuint32m8_t res  = vwaddu_vv_u32m8(vget_u16m8_u16m4(res_, 0), vget_u16m8_u16m4(res_, 1), 32);
    return v_uint32x16(vget_u32m8_u32m4(res, 0));
}

inline v_uint64x8
v_popcount(const v_uint64x8& a)
{
    vuint8m8_t tmp = vundefined_u8m8();
    tmp = vset_u8m8_u8m4(tmp, 0, vcnt_u8((vuint8m4_t)a.val));
    vuint64m8_t mask = (vuint64m8_t){0x3F372F271F170F07,0, 0x3E362E261E160E06,0, 
                                     0x3D352D251D150D05,0, 0x3C342C241C140C04,0,
                                     0x3B332B231B130B03,0, 0x3A322A221A120A02,0, 
                                     0x3931292119110901,0, 0x3830282018100800,0
                                     };
    tmp = vrgather_vv_u8m8(tmp, (vuint8m8_t)mask, 128); 
    vuint16m8_t res_ = vwaddu_vv_u16m8(vget_u8m8_u8m4(tmp, 0), vget_u8m8_u8m4(tmp, 1), 64);
    vuint32m8_t res32  = vwaddu_vv_u32m8(vget_u16m8_u16m4(res_, 0), vget_u16m8_u16m4(res_, 1), 32);
    vuint64m8_t res64  = vwaddu_vv_u64m8(vget_u32m8_u32m4(res32, 0), vget_u32m8_u32m4(res32, 1), 16);
    return v_uint64x8(vget_u64m8_u64m4(res64, 0));
}

inline v_uint64x8
v_popcount(const v_int64x8& a)
{
    vuint8m8_t tmp = vundefined_u8m8();
    tmp = vset_u8m8_u8m4(tmp, 0, vcnt_u8((vuint8m4_t)a.val));
    vuint64m8_t mask = (vuint64m8_t){0x3F372F271F170F07,0, 0x3E362E261E160E06,0, 
                                     0x3D352D251D150D05,0, 0x3C342C241C140C04,0,
                                     0x3B332B231B130B03,0, 0x3A322A221A120A02,0, 
                                     0x3931292119110901,0, 0x3830282018100800,0
                                     };
    tmp = vrgather_vv_u8m8(tmp, (vuint8m8_t)mask, 128); 
    vuint16m8_t res_ = vwaddu_vv_u16m8(vget_u8m8_u8m4(tmp, 0), vget_u8m8_u8m4(tmp, 1), 64);
    vuint32m8_t res32  = vwaddu_vv_u32m8(vget_u16m8_u16m4(res_, 0), vget_u16m8_u16m4(res_, 1), 32);
    vuint64m8_t res64  = vwaddu_vv_u64m8(vget_u32m8_u32m4(res32, 0), vget_u32m8_u32m4(res32, 1), 16);
    return v_uint64x8(vget_u64m8_u64m4(res64, 0));
}

//128
#define SMASK 1, 2, 4, 8, 16, 32, 64, 128
inline int v_signmask(const v_uint8x16& a)
{
    vuint8m1_t t0  = vsrl_vx_u8m1(a.val, 7, 16);
    vuint8m1_t m1  = (vuint8m1_t){SMASK, SMASK};
    vuint16m2_t t1 = vwmulu_vv_u16m2(t0, m1, 16);
    vuint32m1_t res = vmv_v_x_u32m1(0, 4);
    vuint32m2_t t2 = vwmulu_vx_u32m2(vget_u16m2_u16m1(t1, 1), 256, 8);
    res = vredsum_vs_u32m2_u32m1(res, t2, res, 8);
    res = vwredsumu_vs_u16m1_u32m1(res, vget_u16m2_u16m1(t1, 0), res, 8);
    return vmv_x_s_u32m1_u32(res, 8);
}
inline int v_signmask(const v_int8x16& a)
{
    vuint8m1_t t0 = vsrl_vx_u8m1((vuint8m1_t)a.val, 7, 16);
    vuint8m1_t m1 = (vuint8m1_t){SMASK, SMASK};
    vint16m2_t t1 = (vint16m2_t)vwmulu_vv_u16m2(t0, m1, 16);
    vint32m1_t res = vmv_v_x_i32m1(0, 4);
    vint32m2_t t2 = vwmul_vx_i32m2(vget_i16m2_i16m1(t1, 1), 256, 8);
    res = vredsum_vs_i32m2_i32m1(res, t2, res, 8);
    res = vwredsum_vs_i16m1_i32m1(res, vget_i16m2_i16m1(t1, 0), res, 8);
    return vmv_x_s_i32m1_i32(res, 8);
}

inline int v_signmask(const v_int16x8& a)
{
    vint16m1_t t0 = (vint16m1_t)vsrl_vx_u16m1((vuint16m1_t)a.val, 15, 8);
    vint16m1_t m1 = (vint16m1_t){SMASK};
    vint16m1_t t1 = vmul_vv_i16m1(t0, m1, 8);
    vint16m1_t res = vmv_v_x_i16m1(0, 8);
    res = vredsum_vs_i16m1_i16m1(res, t1, res, 8);
    return vmv_x_s_i16m1_i16(res, 8);
}
inline int v_signmask(const v_uint16x8& a)
{
    vint16m1_t t0 = (vint16m1_t)vsrl_vx_u16m1((vuint16m1_t)a.val, 15, 8);
    vint16m1_t m1 = (vint16m1_t){SMASK};
    vint16m1_t t1 = vmul_vv_i16m1(t0, m1, 8);
    vint16m1_t res = vmv_v_x_i16m1(0, 8);
    res = vredsum_vs_i16m1_i16m1(res, t1, res, 8);
    return vmv_x_s_i16m1_i16(res, 8);
}
inline int v_signmask(const v_int32x4& a)
{
    vint32m1_t t0 = (vint32m1_t)vsrl_vx_u32m1((vuint32m1_t)a.val, 31, 4);
    vint32m1_t m1 = (vint32m1_t){1, 2, 4, 8};
    vint32m1_t res = vmv_v_x_i32m1(0, 4);
    vint32m1_t t1 = vmul_vv_i32m1(t0, m1, 4);
    res = vredsum_vs_i32m1_i32m1(res, t1, res, 4);
    return vmv_x_s_i32m1_i32(res, 4);
}
inline int v_signmask(const v_uint32x4& a)
{
    vint32m1_t t0 = (vint32m1_t)vsrl_vx_u32m1(a.val, 31, 4);
    vint32m1_t m1 = (vint32m1_t){1, 2, 4, 8};
    vint32m1_t res = vmv_v_x_i32m1(0, 4);
    vint32m1_t t1 = vmul_vv_i32m1(t0, m1, 4);
    res = vredsum_vs_i32m1_i32m1(res, t1, res, 4);
    return vmv_x_s_i32m1_i32(res, 4);
}
inline int v_signmask(const v_uint64x2& a)
{
    vuint64m1_t v0 = vsrl_vx_u64m1(a.val, 63, 2);
    int res = (int)vext_x_v_u64m1_u64(v0, 0, 2) + ((int)vext_x_v_u64m1_u64(v0, 1, 2) << 1);
    return res;
}
inline int v_signmask(const v_int64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float32x4& a)
{
    vint32m1_t t0 = (vint32m1_t)vsrl_vx_u32m1((vuint32m1_t)a.val, 31, 4);
    vint32m1_t m1 = (vint32m1_t){1, 2, 4, 8};
    vint32m1_t res = vmv_v_x_i32m1(0, 4);
    vint32m1_t t1 = vmul_vv_i32m1(t0, m1, 4);
    res = vredsum_vs_i32m1_i32m1(res, t1, res, 4);
    return vmv_x_s_i32m1_i32(res, 4);
}

//512
inline int64 v_signmask(const v_uint8x64& a)
{
    int64 res = v_signmask(v_uint8x16(vget_u8m4_u8m1(a.val, 0)));
    res |= int64(v_signmask(v_uint8x16(vget_u8m4_u8m1(a.val, 1))))<<16;
    res |= int64(v_signmask(v_uint8x16(vget_u8m4_u8m1(a.val, 2))))<<32;
    res |= int64(v_signmask(v_uint8x16(vget_u8m4_u8m1(a.val, 3))))<<48;
    return res;
}
inline int64 v_signmask(const v_int8x64& a)
{
    int64 res = v_signmask(v_int8x16(vget_i8m4_i8m1(a.val, 0)));
    res |= int64(v_signmask(v_int8x16(vget_i8m4_i8m1(a.val, 1))))<<16;
    res |= int64(v_signmask(v_int8x16(vget_i8m4_i8m1(a.val, 2))))<<32;
    res |= int64(v_signmask(v_int8x16(vget_i8m4_i8m1(a.val, 3))))<<48;
    return res;
}

inline int v_signmask(const v_int16x32& a)
{
    int res =  v_signmask(v_int16x8(vget_i16m4_i16m1(a.val, 0)));
    res |= int(v_signmask(v_int16x8(vget_i16m4_i16m1(a.val, 1))))<<8;
    res |= int(v_signmask(v_int16x8(vget_i16m4_i16m1(a.val, 2))))<<16;
    res |= int(v_signmask(v_int16x8(vget_i16m4_i16m1(a.val, 3))))<<24;
    return res;
}
inline int v_signmask(const v_uint16x32& a)
{
    int res =  v_signmask(v_uint16x8(vget_u16m4_u16m1(a.val, 0)));
    res |= int(v_signmask(v_uint16x8(vget_u16m4_u16m1(a.val, 1))))<<8;
    res |= int(v_signmask(v_uint16x8(vget_u16m4_u16m1(a.val, 2))))<<16;
    res |= int(v_signmask(v_uint16x8(vget_u16m4_u16m1(a.val, 3))))<<24;
    return res;
}
inline int v_signmask(const v_int32x16& a)
{
    int res =  v_signmask(v_int32x4(vget_i32m4_i32m1(a.val, 0)));
    res |= int(v_signmask(v_int32x4(vget_i32m4_i32m1(a.val, 1))))<<8;
    res |= int(v_signmask(v_int32x4(vget_i32m4_i32m1(a.val, 2))))<<16;
    res |= int(v_signmask(v_int32x4(vget_i32m4_i32m1(a.val, 3))))<<24;
    return res;
}
inline int v_signmask(const v_uint32x16& a)
{
    int res =  v_signmask(v_uint32x4(vget_u32m4_u32m1(a.val, 0)));
    res |= int(v_signmask(v_uint32x4(vget_u32m4_u32m1(a.val, 1))))<<8;
    res |= int(v_signmask(v_uint32x4(vget_u32m4_u32m1(a.val, 2))))<<16;
    res |= int(v_signmask(v_uint32x4(vget_u32m4_u32m1(a.val, 3))))<<24;
    return res;
}
inline int v_signmask(const v_uint64x8& a)
{
    int res =  v_signmask(v_uint64x2(vget_u64m4_u64m1(a.val, 0)));
    res |= int(v_signmask(v_uint64x2(vget_u64m4_u64m1(a.val, 1))))<<8;
    res |= int(v_signmask(v_uint64x2(vget_u64m4_u64m1(a.val, 2))))<<16;
    res |= int(v_signmask(v_uint64x2(vget_u64m4_u64m1(a.val, 3))))<<24;
    return res;
}
inline int v_signmask(const v_int64x8& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float64x8& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float32x16& a)
{ return v_signmask(v_reinterpret_as_u32(a)); }

#define OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD(_Tpvec) \
inline int v_scan_forward(const _Tpvec& a) { \
int val = v_signmask(a); \
if(val==0) return 0; \
else return trailingZeros32(val); }

OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_int8x16)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_uint8x16)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_int16x8)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_uint16x8)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_int32x4)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_uint32x4)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_float32x4)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_int64x2)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_uint64x2)

//512
inline int v_scan_forward(const v_int8x64& a) {
    int64 mask = v_signmask(a);
    int mask32 = (int)mask;
    return mask != 0 ? mask32 != 0 ? 
           trailingZeros32(mask32) : 32 + trailingZeros32((int)(mask >> 32)) : 0;
}
inline int v_scan_forward(const v_uint8x64& a) {
    int64 mask = v_signmask(a);
    int mask32 = (int)mask;
    return mask != 0 ? mask32 != 0 ? 
           trailingZeros32(mask32) : 32 + trailingZeros32((int)(mask >> 32)) : 0;
}
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_int16x32)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_uint16x32)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_int32x16)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_uint32x16)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_float32x16)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_int64x8)
OPENCV_HAL_IMPL_RISCVV_SCAN_FORWARD (v_uint64x8)


#define OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(_Tpvec, suffix, _T, shift, num) \
inline bool v_check_all(const v_##_Tpvec& a) \
{ \
    suffix##m1_t v0 = vsrl_vx_##_T(vnot_v_##_T(a.val, num), shift, num); \
    vuint32m1_t v1 = vuint32m1_t(v0); \
    return (v1[0] | v1[1] | v1[2] | v1[3]) == 0; \
} \
inline bool v_check_any(const v_##_Tpvec& a) \
{ \
    suffix##m1_t v0 = vsrl_vx_##_T(a.val, shift, num); \
    vuint32m1_t v1 = vuint32m1_t(v0); \
    return (v1[0] | v1[1] | v1[2] | v1[3]) != 0; \
}

OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint8x16, vuint8,  u8m1, 7, 16)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint16x8, vuint16, u16m1, 15, 8)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint32x4, vuint32, u32m1, 31, 4)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint64x2, vuint64, u64m1, 63, 2)


#define OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_512(_Tpvec, suffix, _T, shift, num) \
inline bool v_check_all(const v_##_Tpvec& a) \
{ \
    suffix##m4_t v0 = vsrl_vx_##_T(vnot_v_##_T(a.val, num), shift, num); \
    vuint32m4_t v1 = vuint32m4_t(v0); \
    return (v1[0] | v1[1] | v1[2] | v1[3] | v1[4] | v1[5] | v1[6] | v1[7] | \
       v1[8] | v1[9] | v1[10] | v1[11] | v1[12] | v1[13] | v1[14] | v1[15]) == 0; \
} \
inline bool v_check_any(const v_##_Tpvec& a) \
{ \
    suffix##m4_t v0 = vsrl_vx_##_T(a.val, shift, num); \
    vuint32m4_t v1 = vuint32m4_t(v0); \
    return (v1[0] | v1[1] | v1[2] | v1[3] | v1[4] | v1[5] | v1[6] | v1[7] | \
       v1[8] | v1[9] | v1[10] | v1[11] | v1[12] | v1[13] | v1[14] | v1[15]) != 0; \
}

OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_512(uint8x64, vuint8,  u8m4, 7, 64)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_512(uint16x32, vuint16, u16m4, 15, 32)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_512(uint32x16, vuint32, u32m4, 31, 16)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_512(uint64x8, vuint64, u64m4, 63, 8)

#define OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(_Tpvec1, _T) \
inline bool v_check_all(const _Tpvec1& a) \
{ return v_check_all(v_reinterpret_as_##_T(a)); } \
inline bool v_check_any(const _Tpvec1& a) \
{ return v_check_any(v_reinterpret_as_##_T(a)); }

OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_int8x16, u8)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_int16x8, u16)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_int32x4, u32)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_int64x2, u64)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_float32x4, u32)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_float64x2, u64)

OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_int8x64, u8)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_int16x32, u16)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_int32x16, u32)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_int64x8, u64)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_float32x16, u32)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY_REINTER(v_float64x8, u64)


#define OPENCV_HAL_IMPL_RISCVV_SELECT(_Tpvec, suffix, _Tpvec2, num) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(vmerge_vvm_##suffix(_Tpvec2(mask.val), b.val, a.val, num)); \
}

#define OPENCV_HAL_IMPL_RISCVV_SELECT_512(_Tpvec, suffix, _Tpvec2, num, data) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    v##data##m4_t res = vundefined_##suffix##m4(); \
    res = vset_##suffix##m4(res, 0, vmerge_vvm_##suffix##m1(\
                                _Tpvec2(vget_##suffix##m4_##suffix##m1(mask.val, 0)), \
                                vget_##suffix##m4_##suffix##m1(b.val, 0), \
                                vget_##suffix##m4_##suffix##m1(a.val, 0), num/4)); \
    res = vset_##suffix##m4(res, 1, vmerge_vvm_##suffix##m1(\
                                _Tpvec2(vget_##suffix##m4_##suffix##m1(mask.val, 1)), \
                                vget_##suffix##m4_##suffix##m1(b.val, 1), \
                                vget_##suffix##m4_##suffix##m1(a.val, 1), num/4)); \
    res = vset_##suffix##m4(res, 2, vmerge_vvm_##suffix##m1(\
                                _Tpvec2(vget_##suffix##m4_##suffix##m1(mask.val, 2)), \
                                vget_##suffix##m4_##suffix##m1(b.val, 2), \
                                vget_##suffix##m4_##suffix##m1(a.val, 2), num/4)); \
    res = vset_##suffix##m4(res, 3, vmerge_vvm_##suffix##m1(\
                                _Tpvec2(vget_##suffix##m4_##suffix##m1(mask.val, 3)), \
                                vget_##suffix##m4_##suffix##m1(b.val, 3), \
                                vget_##suffix##m4_##suffix##m1(a.val, 3), num/4)); \
    return _Tpvec(res); \
}

OPENCV_HAL_IMPL_RISCVV_SELECT(v_int8x16,  i8m1, vbool8_t, 16)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_int16x8,  i16m1, vbool16_t, 8)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_int32x4,  i32m1, vbool32_t, 4)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint8x16, u8m1, vbool8_t, 16)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint16x8, u16m1, vbool16_t, 8)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint32x4, u32m1, vbool32_t, 4)


OPENCV_HAL_IMPL_RISCVV_SELECT_512(v_int8x64,  i8, vbool8_t, 64, int8)
OPENCV_HAL_IMPL_RISCVV_SELECT_512(v_int16x32,  i16, vbool16_t, 32, int16)
OPENCV_HAL_IMPL_RISCVV_SELECT_512(v_int32x16,  i32, vbool32_t, 16, int32)
OPENCV_HAL_IMPL_RISCVV_SELECT_512(v_uint8x64, u8, vbool8_t, 64, uint8)
OPENCV_HAL_IMPL_RISCVV_SELECT_512(v_uint16x32, u16, vbool16_t, 32, uint16)
OPENCV_HAL_IMPL_RISCVV_SELECT_512(v_uint32x16, u32, vbool32_t, 16, uint32)
OPENCV_HAL_IMPL_RISCVV_SELECT_512(v_uint64x8, u64, vbool64_t, 8, uint64)


inline v_float32x4 v_select(const v_float32x4& mask, const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4((vfloat32m1_t)vmerge_vvm_u32m1((vbool32_t)mask.val, (vuint32m1_t)b.val, (vuint32m1_t)a.val, 4));
}
inline v_float64x2 v_select(const v_float64x2& mask, const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2((vfloat64m1_t)vmerge_vvm_u64m1((vbool64_t)mask.val, (vuint64m1_t)b.val, (vuint64m1_t)a.val, 2));
}


inline v_float32x16 v_select(const v_float32x16& mask, const v_float32x16& a, const v_float32x16& b)
{
    return v_reinterpret_as_f32(v_select(v_reinterpret_as_u32(mask),v_reinterpret_as_u32(a),v_reinterpret_as_u32(b)));;
}
inline v_float64x8 v_select(const v_float64x8& mask, const v_float64x8& a, const v_float64x8& b)
{
    return v_reinterpret_as_f64(v_select(v_reinterpret_as_u64(mask),v_reinterpret_as_u64(a),v_reinterpret_as_u64(b)));;
}

#define OPENCV_HAL_IMPL_RISCVV_EXPAND(add, _Tpvec, _Tpwvec, _Tp, _Tp1, num1, _Tp2, num2, _T1, _T2, regnum1, regnum2, prefix) \
inline void v_expand(const _Tpvec& a, v_##_Tpwvec& b0, v_##_Tpwvec& b1) \
{ \
    _T1##_t b = vw##add##_vv_##_Tp2##m##regnum2(a.val, vmv_v_x_##_Tp1(0, num1), num1);    \
    b0.val = vget_##_Tp2##m##regnum2##_##_Tp2##m##regnum1(b, 0);  \
    b1.val = vget_##_Tp2##m##regnum2##_##_Tp2##m##regnum1(b, 1);  \
} \
inline v_##_Tpwvec v_expand_low(const _Tpvec& a) \
{ \
    _T1##_t b = vw##add##_vv_##_Tp2##m##regnum2(a.val, vmv_v_x_##_Tp1(0, num2), num2);    \
    return v_##_Tpwvec(vget_##_Tp2##m##regnum2##_##_Tp2##m##regnum1(b, 0)); \
} \
inline v_##_Tpwvec v_expand_high(const _Tpvec& a) \
{ \
    _T1##_t b = vw##add##_vv_##_Tp2##m##regnum2(a.val, vmv_v_x_##_Tp1(0, num1), num1);    \
    return v_##_Tpwvec(vget_##_Tp2##m##regnum2##_##_Tp2##m##regnum1(b, 1)); \
} \
inline v_##_Tpwvec prefix##_load_expand(const _Tp* ptr) \
{ \
    _T2##_t val = vle##_v_##_Tp1(ptr, num2);    \
    _T1##_t b = vw##add##_vv_##_Tp2##m##regnum2(val, vmv_v_x_##_Tp1(0, num2), num2);    \
    return v_##_Tpwvec(vget_##_Tp2##m##regnum2##_##_Tp2##m##regnum1(b, 0)); \
}

OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint8x16, uint16x8, uchar, u8m1, 16, u16, 8, vuint16m2, vuint8m1, 1, 2, v)
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint16x8, uint32x4, ushort,  u16m1, 8, u32, 4, vuint32m2, vuint16m1, 1, 2, v)
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint32x4, uint64x2, uint,  u32m1, 4, u64, 2, vuint64m2, vuint32m1, 1, 2, v)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int8x16, int16x8, schar,  i8m1, 16, i16, 8, vint16m2, vint8m1, 1, 2, v)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int16x8, int32x4, short,  i16m1, 8, i32, 4, vint32m2, vint16m1, 1, 2, v)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int32x4, int64x2, int,  i32m1, 4, i64, 2, vint64m2, vint32m1, 1, 2, v)
// 512
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint8x64, uint16x32, uchar, u8m4, 64, u16, 32, vuint16m8, vuint8m4, 4, 8, v512)
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint16x32, uint32x16, ushort,  u16m4, 32, u32, 16, vuint32m8, vuint16m4, 4, 8, v512)
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint32x16, uint64x8, uint,  u32m4, 16, u64, 8, vuint64m8, vuint32m4, 4, 8, v512)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int8x64, int16x32, schar,  i8m4, 64, i16, 32, vint16m8, vint8m4, 4, 8, v512)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int16x32, int32x16, short,  i16m4, 32, i32, 16, vint32m8, vint16m4, 4, 8, v512)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int32x16, int64x8, int,  i32m4, 16, i64, 8, vint64m8, vint32m4, 4, 8, v512)


inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    vuint16m2_t b = vundefined_u16m2();
    vuint32m2_t c = vundefined_u32m2();
    vuint8m1_t val = vle_v_u8m1(ptr, 4);    \
    b = vwaddu_vv_u16m2(val, vmv_v_x_u8m1(0, 4), 4);    \
    c = vwaddu_vv_u32m2(vget_u16m2_u16m1(b, 0), vmv_v_x_u16m1(0, 4), 4);    \
    return v_uint32x4(vget_u32m2_u32m1(c, 0));
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    vint16m2_t b = vundefined_i16m2();
    vint32m2_t c = vundefined_i32m2();
    vint8m1_t val = vle_v_i8m1(ptr, 4);    \
    b = vwadd_vv_i16m2(val, vmv_v_x_i8m1(0, 4), 4);    \
    c = vwadd_vv_i32m2(vget_i16m2_i16m1(b, 0), vmv_v_x_i16m1(0, 4), 4);    \
    return v_int32x4(vget_i32m2_i32m1(c, 0));
}

inline v_uint32x16 v512_load_expand_q(const uchar* ptr)
{
    vuint16m8_t b = vundefined_u16m8();
    vuint32m8_t c = vundefined_u32m8();
    vuint8m4_t val = vle_v_u8m4(ptr, 16); 
    b = vwaddu_vv_u16m8(val, vmv_v_x_u8m4(0, 16), 16);
    c = vwaddu_vv_u32m8(vget_u16m8_u16m4(b, 0), vmv_v_x_u16m4(0, 16), 16);
    return v_uint32x16(vget_u32m8_u32m4(c, 0));
}

inline v_int32x16 v512_load_expand_q(const schar* ptr)
{
    vint16m8_t b = vundefined_i16m8();
    vint32m8_t c = vundefined_i32m8();
    vint8m4_t val = vle_v_i8m4(ptr, 16);
    b = vwadd_vv_i16m8(val, vmv_v_x_i8m4(0, 16), 16);
    c = vwadd_vv_i32m8(vget_i16m8_i16m4(b, 0), vmv_v_x_i16m4(0, 16), 16);
    return v_int32x16(vget_i32m8_i32m4(c, 0));
}

#define VITL2_16 (vuint32m2_t){0x11011000, 0x13031202, 0x15051404, 0x17071606, 0x19091808, 0x1B0B1A0A, 0x1D0D1C0C, 0x1F0F1E0E}
#define VITL2_8 (vuint32m2_t){0x00080000, 0x00090001, 0x000A0002, 0x000B0003, 0x000C0004, 0x000D0005, 0x000E0006, 0x000F0007}
#define VITL2_4 (vuint32m2_t){0x00000000, 0x00000004, 0x00000001, 0x00000005, 0x00000002, 0x00000006, 0x00000003, 0x00000007}
#define VITL2_2 (vuint32m2_t){0, 0, 2, 0, 1, 0, 3, 0}

#define VITL8_64 (vuint32m8_t){0x41014000, 0x43034202, 0x45054404, 0x47074606, 0x49094808, 0x4B0B4A0A, 0x4D0D4C0C, 0x4F0F4E0E, \
                               0x51115010, 0x53135212, 0x55155414, 0x57175616, 0x59195818, 0x5B1B5A1A, 0x5D1D5C1C, 0x5F1F5E1E, \
                               0x61216020, 0x63236222, 0x65256424, 0x67276626, 0x69296828, 0x6B2B6A2A, 0x6D2D6C2C, 0x6F2F6E2E, \
                               0x71317030, 0x73337232, 0x75357434, 0x77377636, 0x79397838, 0x7B3B7A3A, 0x7D3D7C3C, 0x7F3F7E3E}
#define VITL8_32 (vuint32m8_t){0x00200000, 0x00210001, 0x00220002, 0x00230003, 0x00240004, 0x00250005, 0x00260006, 0x00270007, \
                              0x00280008, 0x00290009, 0x002A000A, 0x002B000B, 0x002C000C, 0x002D000D, 0x002E000E, 0x002F000F, \
                              0x00300010, 0x00310011, 0x00320012, 0x00330013, 0x00340014, 0x00350015, 0x00360016, 0x00370017, \
                              0x00380018, 0x00390019, 0x003A001A, 0x003B001B, 0x003C001C, 0x003D001D, 0x003E001E, 0x003F001F}
#define VITL8_16 (vuint32m8_t){0x00000000, 0x00000010, 0x00000001, 0x00000011, 0x00000002, 0x00000012, 0x00000003, 0x00000013, \
                              0x00000004, 0x00000014, 0x00000005, 0x00000015, 0x00000006, 0x00000016, 0x00000007, 0x00000017, \
                              0x00000008, 0x00000018, 0x00000009, 0x00000019, 0x0000000A, 0x0000001A, 0x0000000B, 0x0000001B, \
                              0x0000000C, 0x0000001C, 0x0000000D, 0x0000001D, 0x0000000E, 0x0000001E, 0x0000000F, 0x0000001F}
#define VITL8_8 (vuint32m8_t){0, 0, 8, 0, 1, 0, 9, 0, 2, 0, 10, 0, 3, 0, 11, 0, 4, 0, 12, 0, 5, 0, 13, 0, 6, 0, 14, 0, 7, 0, 15, 0}


#define OPENCV_HAL_IMPL_RISCVV_UNPACKS(_Tpvec, _Tp, _T, _UTp, _UT, num, num2, len, numh, regnum1, regnum2) \
inline void v_zip(const v_##_Tpvec& a0, const v_##_Tpvec& a1, v_##_Tpvec& b0, v_##_Tpvec& b1) \
{ \
    v##_Tp##m##regnum2##_t tmp = vundefined_##_T##m##regnum2();\
    tmp = vset_##_T##m##regnum2##_##_T##m##regnum1(tmp, 0, a0.val); \
    tmp = vset_##_T##m##regnum2##_##_T##m##regnum1(tmp, 1, a1.val); \
    vuint32m##regnum2##_t mask = VITL##regnum2##_##num;    \
    tmp = (v##_Tp##m##regnum2##_t)vrgather_vv_##_T##m##regnum2((v##_Tp##m##regnum2##_t)tmp, (v##_UTp##m##regnum2##_t)mask, num2);    \
    b0.val = vget_##_T##m##regnum2##_##_T##m##regnum1(tmp, 0); \
    b1.val = vget_##_T##m##regnum2##_##_T##m##regnum1(tmp, 1); \
} \
inline v_##_Tpvec v_combine_low(const v_##_Tpvec& a, const v_##_Tpvec& b) \
{ \
    v##_Tp##m##regnum1##_t b0 = vslideup_vx_##_T##m##regnum1##_m(vmset_m_##len(num), a.val, b.val, numh, num);    \
    return v_##_Tpvec(b0);\
} \
inline v_##_Tpvec v_combine_high(const v_##_Tpvec& a, const v_##_Tpvec& b) \
{ \
    v##_Tp##m##regnum1##_t b0 = vslidedown_vx_##_T##m##regnum1(b.val, numh, num);    \
    v##_Tp##m##regnum1##_t a0 = vslidedown_vx_##_T##m##regnum1(a.val, numh, num);    \
    v##_Tp##m##regnum1##_t b1 = vslideup_vx_##_T##m##regnum1##_m(vmset_m_##len(num), a0, b0, numh, num);    \
    return v_##_Tpvec(b1);\
} \
inline void v_recombine(const v_##_Tpvec& a, const v_##_Tpvec& b, v_##_Tpvec& c, v_##_Tpvec& d) \
{ \
    c.val = vslideup_vx_##_T##m##regnum1##_m(vmset_m_##len(num), a.val, b.val, numh, num);    \
    v##_Tp##m##regnum1##_t b0 = vslidedown_vx_##_T##m##regnum1(b.val, numh, num);    \
    v##_Tp##m##regnum1##_t a0 = vslidedown_vx_##_T##m##regnum1(a.val, numh, num);    \
    d.val = vslideup_vx_##_T##m##regnum1##_m(vmset_m_##len(num), a0, b0, numh, num);    \
}

OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint8x16, uint8, u8, uint8, u8, 16, 32, b8, 8, 1, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int8x16, int8, i8, uint8, u8, 16, 32, b8, 8, 1, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint16x8, uint16, u16, uint16, u16, 8, 16, b16, 4, 1, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int16x8, int16, i16, uint16, u16, 8, 16, b16, 4, 1, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint32x4, uint32, u32, uint32, u32, 4, 8, b32, 2, 1, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int32x4, int32, i32, uint32, u32, 4, 8, b32, 2, 1, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(float32x4, float32, f32, uint32, u32, 4, 8, b32, 2, 1, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(float64x2, float64, f64, uint64, u64, 2, 4, b64, 1, 1, 2)

//512
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint8x64, uint8, u8, uint8, u8, 64, 128, b2, 32, 4, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int8x64, int8, i8, uint8, u8, 64, 128, b2, 32, 4, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint16x32, uint16, u16, uint16, u16, 32, 64, b4, 16, 4, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int16x32, int16, i16, uint16, u16, 32, 64, b4, 16, 4, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint32x16, uint32, u32, uint32, u32, 16, 32, b8, 8, 4, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int32x16, int32, i32, uint32, u32, 16, 32, b8, 8, 4, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(float32x16, float32, f32, uint32, u32, 16, 32, b8, 8, 4, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(float64x8, float64, f64, uint64, u64, 8, 16, b16, 4, 4, 8)

inline v_uint8x16 v_reverse(const v_uint8x16 &a)
{
    vuint64m1_t mask = (vuint64m1_t){0x08090A0B0C0D0E0F, 0x0001020304050607};
    return v_uint8x16(vrgather_vv_u8m1(a.val, (vuint8m1_t)mask, 16));
}
inline v_int8x16 v_reverse(const v_int8x16 &a)
{
    vint64m1_t mask = (vint64m1_t){0x08090A0B0C0D0E0F, 0x0001020304050607};
    return v_int8x16(vrgather_vv_i8m1(a.val, (vuint8m1_t)mask, 16));
}

inline v_uint16x8 v_reverse(const v_uint16x8 &a)
{
    vuint64m1_t mask = (vuint64m1_t){0x0004000500060007, 0x000000100020003};
    return v_uint16x8(vrgather_vv_u16m1(a.val, (vuint16m1_t)mask, 8));
}

inline v_int16x8 v_reverse(const v_int16x8 &a)
{
    vint64m1_t mask = (vint64m1_t){0x0004000500060007, 0x000000100020003};
    return v_int16x8(vrgather_vv_i16m1(a.val, (vuint16m1_t)mask, 8));
}
inline v_uint32x4 v_reverse(const v_uint32x4 &a)
{
    return v_uint32x4(vrgather_vv_u32m1(a.val, (vuint32m1_t){3, 2, 1, 0}, 4));
}

inline v_int32x4 v_reverse(const v_int32x4 &a)
{
    return v_int32x4(vrgather_vv_i32m1(a.val, (vuint32m1_t){3, 2, 1, 0}, 4));
}

inline v_float32x4 v_reverse(const v_float32x4 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x2 v_reverse(const v_uint64x2 &a)
{
    return v_uint64x2(a.val[1], a.val[0]);
}

inline v_int64x2 v_reverse(const v_int64x2 &a)
{
    return v_int64x2(a.val[1], a.val[0]);
}

inline v_float64x2 v_reverse(const v_float64x2 &a)
{
    return v_float64x2(a.val[1], a.val[0]);
}

//512
inline v_uint8x64 v_reverse(const v_uint8x64 &a)
{
    vuint64m4_t mask = (vuint64m4_t){0x38393A3B3C3D3E3F, 0x3031323334353637,
                                     0x28292A2B2C2D2E2F, 0x2021222324252627,
                                     0x18191A1B1C1D1E1F, 0x1011121314151617,
                                     0x08090A0B0C0D0E0F, 0x0001020304050607};
    return v_uint8x64(vrgather_vv_u8m4(a.val, (vuint8m4_t)mask, 64));
}
inline v_int8x64 v_reverse(const v_int8x64 &a)
{ return v_reinterpret_as_s8(v_reverse(v_reinterpret_as_u8(a))); }

inline v_uint16x32 v_reverse(const v_uint16x32 &a)
{
    vuint64m4_t mask = (vuint64m4_t){0x001C001D001E001F, 0x00180019001A001B, 
                                     0x0014001500160017, 0x0010001100120013,
                                     0x000C000D000E000F, 0x00080009000A000B, 
                                     0x0004000500060007, 0x0000000100020003};
    return v_uint16x32(vrgather_vv_u16m4(a.val, (vuint16m4_t)mask, 32));
}
inline v_int16x32 v_reverse(const v_int16x32 &a)
{ return v_reinterpret_as_s16(v_reverse(v_reinterpret_as_u16(a))); }

inline v_uint32x16 v_reverse(const v_uint32x16 &a)
{
    vuint64m4_t mask = (vuint64m4_t){0x0000000E0000000F, 0x0000000C0000000D,
                                     0x0000000A0000000B, 0x0000000800000009,
                                     0x0000000600000007, 0x0000000400000005,
                                     0x0000000200000003, 0x0000000000000001};
    return v_uint32x16(vrgather_vv_u32m4(a.val, (vuint32m4_t)mask, 16));
}
inline v_int32x16 v_reverse(const v_int32x16 &a)
{ return v_reinterpret_as_s32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_float32x16 v_reverse(const v_float32x16 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x8 v_reverse(const v_uint64x8 &a)
{
    vuint64m4_t mask = (vuint64m4_t){7, 6, 5, 4, 3, 2, 1, 0};
    return v_uint64x8(vrgather_vv_u64m4(a.val, (vuint64m4_t)mask, 8));
}
inline v_int64x8 v_reverse(const v_int64x8 &a)
{ return v_reinterpret_as_s64(v_reverse(v_reinterpret_as_u64(a))); }

inline v_float64x8 v_reverse(const v_float64x8 &a)
{ return v_reinterpret_as_f64(v_reverse(v_reinterpret_as_u64(a))); }

#define OPENCV_HAL_IMPL_RISCVV_EXTRACT(_Tpvec) \
template <int n> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b) \
{ return v_rotate_right<n>(a, b);}
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint8x16)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int8x16)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint16x8)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int16x8)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint32x4)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int32x4)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint64x2)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int64x2)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_float32x4)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_float64x2)

//512
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint8x64)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int8x64)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint16x32)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int16x32)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint32x16)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int32x16)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint64x8)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int64x8)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_float32x16)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_float64x8)


#define OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(_Tpvec, _Tp) \
template<int i> inline _Tp v_extract_n(_Tpvec v) { return v.val[i]; }

OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint8x16, uchar)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int8x16, schar)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint16x8, ushort)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int16x8, short)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint32x4, uint)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int32x4, int)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint64x2, uint64)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int64x2, int64)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_float32x4, float)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_float64x2, double)

//512
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint8x64, uchar)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int8x64, schar)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint16x32, ushort)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int16x32, short)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint32x16, uint)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int32x16, int)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint64x8, uint64)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int64x8, int64)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_float32x16, float)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_float64x8, double)

#define OPENCV_HAL_IMPL_RISCVV_BROADCAST(_Tpvec, _Tp, num) \
template<int i> inline _Tpvec v_broadcast_element(_Tpvec v) { return _Tpvec(vrgather_vx_##_Tp(v.val, i, num)); }

OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint8x16, u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int8x16, i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint16x8, u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int16x8, i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint32x4, u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int32x4, i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint64x2, u64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int64x2, i64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_float32x4, f32m1, 4)

//512
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint8x64, u8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int8x64, i8m4, 64)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint16x32, u16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int16x32, i16m4, 32)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint32x16, u32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int32x16, i32m4, 16)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint64x8, u64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int64x8, i64m4, 8)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_float32x16, f32m4, 16)

inline v_int32x4 v_round(const v_float32x4& a)
{
    __builtin_riscv_fsrm(0);
    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)a.val, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), a.val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}
inline v_int32x4 v_floor(const v_float32x4& a)
{
    __builtin_riscv_fsrm(2);
    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)a.val, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), a.val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    __builtin_riscv_fsrm(3);
    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)a.val, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), a.val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{
    __builtin_riscv_fsrm(1);
    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)a.val, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), a.val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

//512
inline v_int32x16 v_round(const v_float32x16& a)
{
    __builtin_riscv_fsrm(0);
    vint32m4_t nan = vand_vx_i32m4((vint32m4_t)a.val, 0x7f800000, 16);
    vbool8_t mask = vmsne_vx_i32m4_b8(nan, 0x7f800000, 16);
    vint32m4_t val = vfcvt_x_f_v_i32m4_m(mask, vmv_v_x_i32m4(0, 16), a.val, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}
inline v_int32x16 v_floor(const v_float32x16& a)
{
    __builtin_riscv_fsrm(2);
    vint32m4_t nan = vand_vx_i32m4((vint32m4_t)a.val, 0x7f800000, 16);
    vbool8_t mask = vmsne_vx_i32m4_b8(nan, 0x7f800000, 16);
    vint32m4_t val = vfcvt_x_f_v_i32m4_m(mask, vmv_v_x_i32m4(0, 16), a.val, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}

inline v_int32x16 v_ceil(const v_float32x16& a)
{
    __builtin_riscv_fsrm(3);
    vint32m4_t nan = vand_vx_i32m4((vint32m4_t)a.val, 0x7f800000, 16);
    vbool8_t mask = vmsne_vx_i32m4_b8(nan, 0x7f800000, 16);
    vint32m4_t val = vfcvt_x_f_v_i32m4_m(mask, vmv_v_x_i32m4(0, 16), a.val, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}

inline v_int32x16 v_trunc(const v_float32x16& a)
{
    __builtin_riscv_fsrm(1);
    vint32m4_t nan = vand_vx_i32m4((vint32m4_t)a.val, 0x7f800000, 16);
    vbool8_t mask = vmsne_vx_i32m4_b8(nan, 0x7f800000, 16);
    vint32m4_t val = vfcvt_x_f_v_i32m4_m(mask, vmv_v_x_i32m4(0, 16), a.val, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}

//128
inline v_int32x4 v_round(const v_float64x2& a)
{
    __builtin_riscv_fsrm(0);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    //_val = vset_f64m2(_val, 1, a.val);
    _val = vset_f64m2(_val, 1, vfmv_v_f_f64m1(0, 2));
    vint32m1_t val = vfncvt_x_f_v_i32m1(_val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}
inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{
    __builtin_riscv_fsrm(0);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    _val = vset_f64m2(_val, 1, b.val);
    vint32m1_t val = vfncvt_x_f_v_i32m1(_val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}
inline v_int32x4 v_floor(const v_float64x2& a)
{
    __builtin_riscv_fsrm(2);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 2);

    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)aval, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), aval, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_ceil(const v_float64x2& a)
{
    __builtin_riscv_fsrm(3);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 2);

    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)aval, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), aval, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_trunc(const v_float64x2& a)
{
    __builtin_riscv_fsrm(1);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 2);

    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)aval, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), aval, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}


inline v_int32x16 v_round(const v_float64x8& a)
{
    __builtin_riscv_fsrm(0);
    vfloat64m8_t _val = vundefined_f64m8();
    _val = vset_f64m8_f64m4(_val, 0, a.val);
    _val = vset_f64m8_f64m4(_val, 1, vfmv_v_f_f64m4(0, 4));
    vint32m4_t val = vfncvt_x_f_v_i32m4(_val, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}
inline v_int32x16 v_round(const v_float64x8& a, const v_float64x8& b)
{
    __builtin_riscv_fsrm(0);
    vfloat64m8_t _val = vundefined_f64m8();
    _val = vset_f64m8_f64m4(_val, 0, a.val);
    _val = vset_f64m8_f64m4(_val, 1, b.val);
    vint32m4_t val = vfncvt_x_f_v_i32m4(_val, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}
inline v_int32x16 v_floor(const v_float64x8& a)
{
    __builtin_riscv_fsrm(2);
    vfloat64m8_t _val = vundefined_f64m8();
    _val = vset_f64m8_f64m4(_val, 0, a.val);
    vfloat32m4_t aval = vfncvt_f_f_v_f32m4(_val, 8);

    vint32m4_t nan = vand_vx_i32m4((vint32m4_t)aval, 0x7f800000, 16);
    vbool8_t mask = vmsne_vx_i32m4_b8(nan, 0x7f800000, 16);
    vint32m4_t val = vfcvt_x_f_v_i32m4_m(mask, vmv_v_x_i32m4(0, 16), aval, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}

inline v_int32x16 v_ceil(const v_float64x8& a)
{
    __builtin_riscv_fsrm(3);
    vfloat64m8_t _val = vundefined_f64m8();
    _val = vset_f64m8_f64m4(_val, 0, a.val);
    vfloat32m4_t aval = vfncvt_f_f_v_f32m4(_val, 8);

    vint32m4_t nan = vand_vx_i32m4((vint32m4_t)aval, 0x7f800000, 16);
    vbool8_t mask = vmsne_vx_i32m4_b8(nan, 0x7f800000, 16);
    vint32m4_t val = vfcvt_x_f_v_i32m4_m(mask, vmv_v_x_i32m4(0, 16), aval, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}

inline v_int32x16 v_trunc(const v_float64x8& a)
{
    __builtin_riscv_fsrm(1);
    vfloat64m8_t _val = vundefined_f64m8();
    _val = vset_f64m8_f64m4(_val, 0, a.val);
    vfloat32m4_t aval = vfncvt_f_f_v_f32m4(_val, 8);

    vint32m4_t nan = vand_vx_i32m4((vint32m4_t)aval, 0x7f800000, 16);
    vbool8_t mask = vmsne_vx_i32m4_b8(nan, 0x7f800000, 16);
    vint32m4_t val = vfcvt_x_f_v_i32m4_m(mask, vmv_v_x_i32m4(0, 16), aval, 16);
    __builtin_riscv_fsrm(0);
    return v_int32x16(val);
}

#define OPENCV_HAL_IMPL_RISCVV_LOAD_DEINTERLEAVED(intrin, _Tpvec, num, _Tp, _T)    \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b) \
{ \
    v##_Tpvec##m1x2_t ret = intrin##2e_v_##_T##m1x2(ptr, num);\
    a.val = vget_##_T##m1x2_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x2_##_T##m1(ret, 1);  \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, v_##_Tpvec##x##num& c) \
{ \
    v##_Tpvec##m1x3_t ret = intrin##3e_v_##_T##m1x3(ptr, num);\
    a.val = vget_##_T##m1x3_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x3_##_T##m1(ret, 1);  \
    c.val = vget_##_T##m1x3_##_T##m1(ret, 2);  \
}\
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, \
                                v_##_Tpvec##x##num& c, v_##_Tpvec##x##num& d) \
{ \
    v##_Tpvec##m1x4_t ret = intrin##4e_v_##_T##m1x4(ptr, num);\
    a.val = vget_##_T##m1x4_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x4_##_T##m1(ret, 1);  \
    c.val = vget_##_T##m1x4_##_T##m1(ret, 2);  \
    d.val = vget_##_T##m1x4_##_T##m1(ret, 3);  \
} \

#define OPENCV_HAL_IMPL_RISCVV_STORE_INTERLEAVED(intrin, _Tpvec, num, _Tp, _T)    \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m1x2_t ret = vundefined_##_T##m1x2();      \
    ret = vset_##_T##m1x2(ret, 0, a.val);  \
    ret = vset_##_T##m1x2(ret, 1, b.val);  \
    intrin##2e_v_##_T##m1x2(ptr, ret, num); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m1x3_t ret = vundefined_##_T##m1x3();       \
    ret = vset_##_T##m1x3(ret, 0, a.val);  \
    ret = vset_##_T##m1x3(ret, 1, b.val);  \
    ret = vset_##_T##m1x3(ret, 2, c.val);  \
    intrin##3e_v_##_T##m1x3(ptr, ret, num); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, const v_##_Tpvec##x##num& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    v##_Tpvec##m1x4_t ret = vundefined_##_T##m1x4();             \
    ret = vset_##_T##m1x4(ret, 0, a.val);  \
    ret = vset_##_T##m1x4(ret, 1, b.val);  \
    ret = vset_##_T##m1x4(ret, 2, c.val);  \
    ret = vset_##_T##m1x4(ret, 3, d.val);  \
    intrin##4e_v_##_T##m1x4(ptr, ret, num); \
}

#define OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(_Tpvec, _Tp, num, ld, st, _T) \
OPENCV_HAL_IMPL_RISCVV_LOAD_DEINTERLEAVED(ld, _Tpvec, num, _Tp, _T)    \
OPENCV_HAL_IMPL_RISCVV_STORE_INTERLEAVED(st, _Tpvec, num, _Tp, _T)

//OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(uint8, uchar, )
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(int8, schar, 16, vlseg, vsseg, i8)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(int16, short, 8, vlseg, vsseg, i16)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(int32, int, 4, vlseg, vsseg, i32)

OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(uint8, unsigned char, 16, vlseg, vsseg, u8)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(uint16, unsigned short, 8, vlseg, vsseg, u16)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(uint32, unsigned int, 4, vlseg, vsseg, u32)

//512
#define OPENCV_HAL_IMPL_RISCVV_LOAD_DEINTERLEAVED_512(intrin, _Tpvec, num, _Tp, _T)    \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b) \
{ \
    v##_Tpvec##m4x2_t ret = intrin##2e_v_##_T##m4x2(ptr, num);\
    a.val = vget_##_T##m4x2_##_T##m4(ret, 0);  \
    b.val = vget_##_T##m4x2_##_T##m4(ret, 1);  \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, v_##_Tpvec##x##num& c) \
{ \
    a.val = (v##_Tpvec##m4_t){0}; b.val = (v##_Tpvec##m4_t){0}; c.val = (v##_Tpvec##m4_t){0}; \
    v##_Tpvec##m2x3_t ret = intrin##3e_v_##_T##m2x3(ptr, num);\
    a.val = vset_##_T##m4_##_T##m2(a.val, 0, vget_##_T##m2x3_##_T##m2(ret, 0));  \
    b.val = vset_##_T##m4_##_T##m2(b.val, 0, vget_##_T##m2x3_##_T##m2(ret, 1));  \
    c.val = vset_##_T##m4_##_T##m2(c.val, 0, vget_##_T##m2x3_##_T##m2(ret, 2));  \
    ret = intrin##3e_v_##_T##m2x3(ptr + 3*num, num);\
    a.val = vset_##_T##m4_##_T##m2(a.val, 1, vget_##_T##m2x3_##_T##m2(ret, 0));  \
    b.val = vset_##_T##m4_##_T##m2(b.val, 1, vget_##_T##m2x3_##_T##m2(ret, 1));  \
    c.val = vset_##_T##m4_##_T##m2(c.val, 1, vget_##_T##m2x3_##_T##m2(ret, 2));  \
}\
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, \
                                v_##_Tpvec##x##num& c, v_##_Tpvec##x##num& d) \
{ \
    a.val = (v##_Tpvec##m4_t){0}; b.val = (v##_Tpvec##m4_t){0}; \
    c.val = (v##_Tpvec##m4_t){0}; d.val = (v##_Tpvec##m4_t){0}; \
    v##_Tpvec##m1x4_t ret = intrin##4e_v_##_T##m1x4(ptr, num);\
    a.val = vset_##_T##m4_##_T##m1(a.val, 0, vget_##_T##m1x4_##_T##m1(ret, 0));  \
    b.val = vset_##_T##m4_##_T##m1(b.val, 0, vget_##_T##m1x4_##_T##m1(ret, 1));  \
    c.val = vset_##_T##m4_##_T##m1(c.val, 0, vget_##_T##m1x4_##_T##m1(ret, 2));  \
    d.val = vset_##_T##m4_##_T##m1(d.val, 0, vget_##_T##m1x4_##_T##m1(ret, 3));  \
    ret = intrin##4e_v_##_T##m1x4(ptr + 4*num, num);\
    a.val = vset_##_T##m4_##_T##m1(a.val, 1, vget_##_T##m1x4_##_T##m1(ret, 0));  \
    b.val = vset_##_T##m4_##_T##m1(b.val, 1, vget_##_T##m1x4_##_T##m1(ret, 1));  \
    c.val = vset_##_T##m4_##_T##m1(c.val, 1, vget_##_T##m1x4_##_T##m1(ret, 2));  \
    d.val = vset_##_T##m4_##_T##m1(d.val, 1, vget_##_T##m1x4_##_T##m1(ret, 3));  \
    ret = intrin##4e_v_##_T##m1x4(ptr + 8*num, num);\
    a.val = vset_##_T##m4_##_T##m1(a.val, 2, vget_##_T##m1x4_##_T##m1(ret, 0));  \
    b.val = vset_##_T##m4_##_T##m1(b.val, 2, vget_##_T##m1x4_##_T##m1(ret, 1));  \
    c.val = vset_##_T##m4_##_T##m1(c.val, 2, vget_##_T##m1x4_##_T##m1(ret, 2));  \
    d.val = vset_##_T##m4_##_T##m1(d.val, 2, vget_##_T##m1x4_##_T##m1(ret, 3));  \
    ret = intrin##4e_v_##_T##m1x4(ptr + 12*num, num);\
    a.val = vset_##_T##m4_##_T##m1(a.val, 3, vget_##_T##m1x4_##_T##m1(ret, 0));  \
    b.val = vset_##_T##m4_##_T##m1(b.val, 3, vget_##_T##m1x4_##_T##m1(ret, 1));  \
    c.val = vset_##_T##m4_##_T##m1(c.val, 3, vget_##_T##m1x4_##_T##m1(ret, 2));  \
    d.val = vset_##_T##m4_##_T##m1(d.val, 3, vget_##_T##m1x4_##_T##m1(ret, 3));  \
} \

#define OPENCV_HAL_IMPL_RISCVV_STORE_INTERLEAVED_512(intrin, _Tpvec, num, _Tp, _T)    \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m4x2_t ret = vundefined_##_T##m4x2();      \
    ret = vset_##_T##m4x2(ret, 0, a.val);  \
    ret = vset_##_T##m4x2(ret, 1, b.val);  \
    intrin##2e_v_##_T##m4x2(ptr, ret, num); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m2x3_t ret = vundefined_##_T##m2x3();       \
    ret = vset_##_T##m2x3(ret, 0, vget_##_T##m4_##_T##m2(a.val, 0));  \
    ret = vset_##_T##m2x3(ret, 1, vget_##_T##m4_##_T##m2(b.val, 0));  \
    ret = vset_##_T##m2x3(ret, 2, vget_##_T##m4_##_T##m2(c.val, 0));  \
    intrin##3e_v_##_T##m2x3(ptr, ret, num); \
    ret = vset_##_T##m2x3(ret, 0, vget_##_T##m4_##_T##m2(a.val, 1));  \
    ret = vset_##_T##m2x3(ret, 1, vget_##_T##m4_##_T##m2(b.val, 1));  \
    ret = vset_##_T##m2x3(ret, 2, vget_##_T##m4_##_T##m2(c.val, 1));  \
    intrin##3e_v_##_T##m2x3(ptr + 3*num, ret, num); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, const v_##_Tpvec##x##num& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    v##_Tpvec##m1x4_t ret = vundefined_##_T##m1x4();             \
    ret = vset_##_T##m1x4(ret, 0, vget_##_T##m4_##_T##m1(a.val, 0));  \
    ret = vset_##_T##m1x4(ret, 1, vget_##_T##m4_##_T##m1(b.val, 0));  \
    ret = vset_##_T##m1x4(ret, 2, vget_##_T##m4_##_T##m1(c.val, 0));  \
    ret = vset_##_T##m1x4(ret, 3, vget_##_T##m4_##_T##m1(d.val, 0));  \
    intrin##4e_v_##_T##m1x4(ptr, ret, num); \
    ret = vset_##_T##m1x4(ret, 0, vget_##_T##m4_##_T##m1(a.val, 1));  \
    ret = vset_##_T##m1x4(ret, 1, vget_##_T##m4_##_T##m1(b.val, 1));  \
    ret = vset_##_T##m1x4(ret, 2, vget_##_T##m4_##_T##m1(c.val, 1));  \
    ret = vset_##_T##m1x4(ret, 3, vget_##_T##m4_##_T##m1(d.val, 1));  \
    intrin##4e_v_##_T##m1x4(ptr + 4*num, ret, num); \
    ret = vset_##_T##m1x4(ret, 0, vget_##_T##m4_##_T##m1(a.val, 2));  \
    ret = vset_##_T##m1x4(ret, 1, vget_##_T##m4_##_T##m1(b.val, 2));  \
    ret = vset_##_T##m1x4(ret, 2, vget_##_T##m4_##_T##m1(c.val, 2));  \
    ret = vset_##_T##m1x4(ret, 3, vget_##_T##m4_##_T##m1(d.val, 2));  \
    intrin##4e_v_##_T##m1x4(ptr + 8*num, ret, num); \
    ret = vset_##_T##m1x4(ret, 0, vget_##_T##m4_##_T##m1(a.val, 3));  \
    ret = vset_##_T##m1x4(ret, 1, vget_##_T##m4_##_T##m1(b.val, 3));  \
    ret = vset_##_T##m1x4(ret, 2, vget_##_T##m4_##_T##m1(c.val, 3));  \
    ret = vset_##_T##m1x4(ret, 3, vget_##_T##m4_##_T##m1(d.val, 3));  \
    intrin##4e_v_##_T##m1x4(ptr + 12*num, ret, num); \
}

#define OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(_Tpvec, _Tp, num, ld, st, _T) \
OPENCV_HAL_IMPL_RISCVV_LOAD_DEINTERLEAVED_512(ld, _Tpvec, num, _Tp, _T)    \
OPENCV_HAL_IMPL_RISCVV_STORE_INTERLEAVED_512(st, _Tpvec, num, _Tp, _T)

OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(int8, schar, 64, vlseg, vsseg, i8)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(int16, short, 32, vlseg, vsseg, i16)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(int32, int, 16, vlseg, vsseg, i32)

OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(uint8, unsigned char, 64, vlseg, vsseg, u8)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(uint16, unsigned short, 32, vlseg, vsseg, u16)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(uint32, unsigned int, 16, vlseg, vsseg, u32)


#define OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(_Tpvec, _Tp, num, _T) \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b) \
{ \
    v##_Tpvec##m1x2_t ret = vlseg2e_v_##_T##m1x2(ptr, num); \
    a.val = vget_##_T##m1x2_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x2_##_T##m1(ret, 1);  \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, v_##_Tpvec##x##num& c) \
{ \
    v##_Tpvec##m1x3_t ret = vlseg3e_v_##_T##m1x3(ptr, num);    \
    a.val = vget_##_T##m1x3_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x3_##_T##m1(ret, 1);  \
    c.val = vget_##_T##m1x3_##_T##m1(ret, 2);  \
}\
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, \
                                v_##_Tpvec##x##num& c, v_##_Tpvec##x##num& d) \
{ \
    v##_Tpvec##m1x4_t ret = vlseg4e_v_##_T##m1x4(ptr, num);    \
    a.val = vget_##_T##m1x4_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x4_##_T##m1(ret, 1);  \
    c.val = vget_##_T##m1x4_##_T##m1(ret, 2);  \
    d.val = vget_##_T##m1x4_##_T##m1(ret, 3);  \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m1x2_t ret = vundefined_##_T##m1x2();    \
    ret = vset_##_T##m1x2(ret, 0, a.val);  \
    ret = vset_##_T##m1x2(ret, 1, b.val);  \
    vsseg2e_v_##_T##m1x2(ptr, ret, num);    \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m1x3_t ret = vundefined_##_T##m1x3();    \
    ret = vset_##_T##m1x3(ret, 0, a.val);  \
    ret = vset_##_T##m1x3(ret, 1, b.val);  \
    ret = vset_##_T##m1x3(ret, 2, c.val);  \
    vsseg3e_v_##_T##m1x3(ptr, ret, num);    \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, const v_##_Tpvec##x##num& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    v##_Tpvec##m1x4_t ret = vundefined_##_T##m1x4();    \
    ret = vset_##_T##m1x4(ret, 0, a.val);  \
    ret = vset_##_T##m1x4(ret, 1, b.val);  \
    ret = vset_##_T##m1x4(ret, 2, c.val);  \
    ret = vset_##_T##m1x4(ret, 3, d.val);  \
    vsseg4e_v_##_T##m1x4(ptr, ret, num);    \
}
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(float32, float, 4, f32)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(float64, double, 2, f64)

OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(uint64, unsigned long, 2, u64)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(int64, long, 2, i64)

//512
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(float32, float, 16, vlseg, vsseg, f32)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(float64, double, 8, vlseg, vsseg, f64)

OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(uint64, unsigned long, 8, vlseg, vsseg, u64)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_512(int64, long, 8, vlseg, vsseg, i64)

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(vfcvt_f_x_v_f32m1(a.val, 4));
}

#if CV_SIMD128_64F
inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 2);
    return v_float32x4(aval);
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    _val = vset_f64m2(_val, 1, b.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 4);
    return v_float32x4(aval);
}

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{
    vfloat32m1_t val = vfcvt_f_x_v_f32m1(a.val, 4);
    vfloat64m2_t _val = vfwcvt_f_f_v_f64m2(val, 4);
    return v_float64x2(vget_f64m2_f64m1(_val, 0));
}

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{
    vfloat32m1_t val = vfcvt_f_x_v_f32m1(a.val, 4);
    vfloat64m2_t _val = vfwcvt_f_f_v_f64m2(val, 4);
    return v_float64x2(vget_f64m2_f64m1(_val, 1));
}

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{
    vfloat64m2_t _val  = vfwcvt_f_f_v_f64m2(a.val, 4);
    return v_float64x2(vget_f64m2_f64m1(_val, 0));
}

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{
    vfloat64m2_t _val  = vfwcvt_f_f_v_f64m2(a.val, 4);
    return v_float64x2(vget_f64m2_f64m1(_val, 1));
}

inline v_float64x2 v_cvt_f64(const v_int64x2& a)
{
    return v_float64x2(vfcvt_f_x_v_f64m1(a.val, 2));
}

#endif

//512
inline v_float32x16 v_cvt_f32(const v_int32x16& a)
{
    return v_float32x16(vfcvt_f_x_v_f32m4(a.val, 16));
}

#if CV_SIMD512_64F
inline v_float32x16 v_cvt_f32(const v_float64x8& a)
{
    vfloat64m8_t _val = vundefined_f64m8();
    _val = vset_f64m8_f64m4(_val, 0, a.val);
    vfloat32m4_t aval = vfncvt_f_f_v_f32m4(_val, 8);
    return v_float32x16(aval);
}

inline v_float32x16 v_cvt_f32(const v_float64x8& a, const v_float64x8& b)
{
    vfloat64m8_t _val = vundefined_f64m8();
    _val = vset_f64m8_f64m4(_val, 0, a.val);
    _val = vset_f64m8_f64m4(_val, 1, b.val);
    vfloat32m4_t aval = vfncvt_f_f_v_f32m4(_val, 16);
    return v_float32x16(aval);
}

inline v_float64x8 v_cvt_f64(const v_int32x16& a)
{
    vfloat32m4_t val = vfcvt_f_x_v_f32m4(a.val, 16);
    vfloat64m8_t _val = vfwcvt_f_f_v_f64m8(val, 16);
    return v_float64x8(vget_f64m8_f64m4(_val, 0));
}

inline v_float64x8 v_cvt_f64_high(const v_int32x16& a)
{
    vfloat32m4_t val = vfcvt_f_x_v_f32m4(a.val, 16);
    vfloat64m8_t _val = vfwcvt_f_f_v_f64m8(val, 16);
    return v_float64x8(vget_f64m8_f64m4(_val, 1));
}

inline v_float64x8 v_cvt_f64(const v_float32x16& a)
{
    vfloat64m8_t _val  = vfwcvt_f_f_v_f64m8(a.val, 16);
    return v_float64x8(vget_f64m8_f64m4(_val, 0));
}

inline v_float64x8 v_cvt_f64_high(const v_float32x16& a)
{
    vfloat64m8_t _val  = vfwcvt_f_f_v_f64m8(a.val, 16);
    return v_float64x8(vget_f64m8_f64m4(_val, 1));
}

inline v_float64x8 v_cvt_f64(const v_int64x8& a)
{
    return v_float64x8(vfcvt_f_x_v_f64m4(a.val, 8));
}

#endif

inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
{
    vuint64m1_t m0 = {0x0705060403010200, 0x0F0D0E0C0B090A08};
    return v_int8x16(vrgather_vv_i8m1(vec.val, (vuint8m1_t)m0, 16));
}
inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec)
{
    return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec)));
}

inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
{
    vuint64m1_t m0 = {0x0703060205010400, 0x0F0B0E0A0D090C08};
    return v_int8x16(vrgather_vv_i8m1(vec.val, (vuint8m1_t)m0, 16));
}
inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec)
{
    return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec)));
}

inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
{
    vuint64m1_t m0 = {0x0706030205040100, 0x0F0E0B0A0D0C0908};
    return v_int16x8((vint16m1_t)vrgather_vv_u8m1((vuint8m1_t)vec.val, (vuint8m1_t)m0, 16));
}
inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
{
    vuint64m1_t m0 = {0x0B0A030209080100, 0x0F0E07060D0C0504};
    return v_int16x8((vint16m1_t)vrgather_vv_u8m1((vuint8m1_t)(vec.val), (vuint8m1_t)m0, 16));
}
inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
{
    vuint64m1_t m0 = {0x0B0A090803020100, 0x0F0E0D0C07060504};
    return v_int32x4((vint32m1_t)vrgather_vv_u8m1((vuint8m1_t)(vec.val), (vuint8m1_t)m0, 16));
}
inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec) { return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x4 v_interleave_pairs(const v_float32x4& vec) { return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
{
    vuint64m1_t m0 = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
    return v_int8x16((vint8m1_t)vrgather_vv_u8m1((vuint8m1_t)(vec.val), (vuint8m1_t)m0, 16));
}
inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
{
    vuint64m1_t m0 = {0x0908050403020100, 0xFFFFFFFF0D0C0B0A};
    return v_int16x8((vint16m1_t)vrgather_vv_u8m1((vuint8m1_t)(vec.val), (vuint8m1_t)m0, 16));
}
inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_pack_triplets(const v_int32x4& vec) { return vec; }
inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec) { return vec; }
inline v_float32x4 v_pack_triplets(const v_float32x4& vec) { return vec; }

//512
inline v_int8x64 v_interleave_pairs(const v_int8x64& vec)
{
    vuint64m4_t m0 = {0x0705060403010200, 0x0F0D0E0C0B090A08,
                      0x1715161413111210, 0x1F1D1E1C1B191A18,
                      0x2725262423212220, 0x2F2D2E2C2B292A28,
                      0x3735363433313230, 0x3F3D3E3C3B393A38};
    return v_int8x64(vrgather_vv_i8m4(vec.val, (vuint8m4_t)m0, 64));
}
inline v_uint8x64 v_interleave_pairs(const v_uint8x64& vec)
{ return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }

inline v_int8x64 v_interleave_quads(const v_int8x64& vec)
{
    vuint64m4_t m0 = {0x0703060205010400, 0x0F0B0E0A0D090C08,
                      0x1713161215111410, 0x1F1B1E1A1D191C18,
                      0x2723262225212420, 0x2F2B2E2A2D292C28,
                      0x3733363235313430, 0x3F3B3E3A3D393C38};
    return v_int8x64(vrgather_vv_i8m4(vec.val, (vuint8m4_t)m0, 64));
}
inline v_uint8x64 v_interleave_quads(const v_uint8x64& vec)
{ return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x32 v_interleave_pairs(const v_int16x32& vec)
{
    vuint64m4_t m0 = {0x0706030205040100, 0x0F0E0B0A0D0C0908,
                      0x1716131215141110, 0x1F1E1B1A1D1C1918,
                      0x2726232225242120, 0x2F2E2B2A2D2C2928,
                      0x3736333235343130, 0x3F3E3B3A3D3C3938};
    return v_int16x32((vint16m4_t)vrgather_vv_u8m4((vuint8m4_t)vec.val, (vuint8m4_t)m0, 64));
}
inline v_uint16x32 v_interleave_pairs(const v_uint16x32& vec) 
{ return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x32 v_interleave_quads(const v_int16x32& vec)
{
    vuint64m4_t m0 = {0x0B0A030209080100, 0x0F0E07060D0C0504,
                      0x1B1A131219181110, 0x1F1E17161D1C1514,
                      0x2B2A232229282120, 0x2F2E27262D2C2524,
                      0x3B3A333239383130, 0x3F3E37363D3C3534};
    return v_int16x32((vint16m4_t)vrgather_vv_u8m4((vuint8m4_t)(vec.val), (vuint8m4_t)m0, 64));
}
inline v_uint16x32 v_interleave_quads(const v_uint16x32& vec) 
{ return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x16 v_interleave_pairs(const v_int32x16& vec)
{
    vuint64m4_t m0 = {0x0B0A090803020100, 0x0F0E0D0C07060504,
                      0x1B1A191813121110, 0x1F1E1D1C17161514,
                      0x2B2A292823222120, 0x2F2E2D2C27262524,
                      0x3B3A393833323130, 0x3F3E3D3C37363534};
    return v_int32x16((vint32m4_t)vrgather_vv_u8m4((vuint8m4_t)(vec.val), (vuint8m4_t)m0, 64));
}
inline v_uint32x16 v_interleave_pairs(const v_uint32x16& vec) 
{ return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x16 v_interleave_pairs(const v_float32x16& vec) 
{ return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_int8x64 v_pack_triplets(const v_int8x64& vec)
{
    vuint64m4_t m0 = {0x0908060504020100, 0x141211100E0D0C0A,
                      0x1E1D1C1A19181615, 0x2928262524222120,
                      0x343231302E2D2C2A, 0x3E3D3C3A39383635,
                      0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    return v_int8x64((vint8m4_t)vrgather_vv_u8m4((vuint8m4_t)(vec.val), (vuint8m4_t)m0, 64));
}
inline v_uint8x64 v_pack_triplets(const v_uint8x64& vec) 
{ return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x32 v_pack_triplets(const v_int16x32& vec)
{
    vuint64m4_t m0 = {0x0908050403020100, 0x131211100D0C0B0A,
                      0x1D1C1B1A19181514, 0x2928252423222120, 
                      0x333231302D2C2B2A, 0x3D3C3B3A39383534,
                      0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    return v_int16x32((vint16m4_t)vrgather_vv_u8m4((vuint8m4_t)(vec.val), (vuint8m4_t)m0, 64));
}
inline v_uint16x32 v_pack_triplets(const v_uint16x32& vec) 
{ return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x16 v_pack_triplets(const v_int32x16& vec)
{
    vuint64m4_t m0 = {0x0706050403020100, 0x131211100B0A0908,
                      0x1B1A191817161514, 0x2726252423222120,
                      0x333231302B2A2928, 0x3B3A393837363534,
                      0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    return v_int32x16((vint32m4_t)vrgather_vv_u8m4((vuint8m4_t)(vec.val), (vuint8m4_t)m0, 64));
}
inline v_uint32x16 v_pack_triplets(const v_uint32x16& vec)
{ return v_reinterpret_as_u32(v_pack_triplets(v_reinterpret_as_s32(vec))); }
inline v_float32x16 v_pack_triplets(const v_float32x16& vec)
{ return v_reinterpret_as_f32(v_pack_triplets(v_reinterpret_as_s32(vec))); }


#if CV_SIMD128_64F
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x2 v_dotprod_expand(const v_int32x4& a,   const v_int32x4& b,
                                    const v_float64x2& c)
{ return v_dotprod_expand(a, b) + c; }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{
    vint64m2_t v1 = vwmul_vv_i64m2(a.val, b.val, 4);
    vfloat64m1_t res = vfcvt_f_x_v_f64m1(vadd_vv_i64m1(vget_i64m2_i64m1(v1, 0), vget_i64m2_i64m1(v1, 1), 2), 2);
    return v_float64x2(res);
}
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ v_float64x2 res = v_dotprod_expand_fast(a, b);
  return res + c; }
#endif

#if CV_SIMD512_64F
inline v_float64x8 v_dotprod_expand(const v_int32x16& a, const v_int32x16& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x8 v_dotprod_expand(const v_int32x16& a,   const v_int32x16& b,
                                    const v_float64x8& c)
{ return v_dotprod_expand(a, b) + c; }
inline v_float64x8 v_dotprod_expand_fast(const v_int32x16& a, const v_int32x16& b)
{
    vint64m8_t v1 = vwmul_vv_i64m8(a.val, b.val, 16);
    vfloat64m4_t res = vfcvt_f_x_v_f64m4(vadd_vv_i64m4(vget_i64m8_i64m4(v1, 0), vget_i64m8_i64m4(v1, 1), 8), 8);
    return v_float64x8(res);
}
inline v_float64x8 v_dotprod_expand_fast(const v_int32x16& a, const v_int32x16& b, const v_float64x8& c)
{ v_float64x8 res = v_dotprod_expand_fast(a, b);
  return res + c; }
#endif

////// FP16 support ///////
inline v_float32x4 v_load_expand(const float16_t* ptr)
{
    vfloat16m1_t v = vle_v_f16m1((__fp16*)ptr, 4);
    vfloat32m2_t v32 = vfwcvt_f_f_v_f32m2(v, 4);
    return v_float32x4(vget_f32m2_f32m1(v32, 0));
}

inline void v_pack_store(float16_t* ptr, const v_float32x4& v)
{
    vfloat32m2_t v32 = vundefined_f32m2();
    v32 = vset_f32m2(v32, 0, v.val);
    vfloat16m1_t hv = vfncvt_f_f_v_f16m1(v32, 4);
    vse_v_f16m1((__fp16*)ptr, hv, 4);
}

//512
inline v_float32x16 v512_load_expand(const float16_t* ptr)
{
    vfloat16m4_t v = vle_v_f16m4((__fp16*)ptr, 16);
    vfloat32m8_t v32 = vfwcvt_f_f_v_f32m8(v, 16);
    return v_float32x16(vget_f32m8_f32m4(v32, 0));
}

inline void v_pack_store(float16_t* ptr, const v_float32x16& v)
{
    vfloat32m8_t v32 = vundefined_f32m8();
    v32 = vset_f32m8_f32m4(v32, 0, v.val);
    vfloat16m4_t hv = vfncvt_f_f_v_f16m4(v32, 16);
    vse_v_f16m4((__fp16*)ptr, hv, 16);
}


inline void v_cleanup() {}
inline void v512_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}
#endif
