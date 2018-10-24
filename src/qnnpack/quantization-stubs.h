/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <qnnpack.h>
#include <qnnpack/params.h>

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

enum qnnp_status qnnp_quantize(size_t n, const float* input, uint8_t* output, uint8_t zero_point, float scale, pthreadpool_t threadpool)

typedef enum qnnp_status (*quantization_function)(
    size_t n,
    const float* input,
    uint8_t* output,
    uint8_t zero_point,
    float scale,
    pthreadpool_t threadpool);

#define DECLARE_QUANTIZATION_FUNCTION(fn_name) \
    enum qnnp_status fn_name( \
        size_t n, \
        const float* input, \
        uint8_t* output, \
        uint8_t zero_point, \
        float scale, \
        pthreadpool_t threadpool);

DECLARE_QUANTIZATION_FUNCTION(qnnp_quantize_fp32_uint8__scalar)
DECLARE_QUANTIZATION_FUNCTION(qnnp_quantize_fp32_uint8__neon)
DECLARE_QUANTIZATION_FUNCTION(qnnp_quantize_fp32_uint8__sse)


#ifdef __cplusplus
} /* extern "C" */
#endif
