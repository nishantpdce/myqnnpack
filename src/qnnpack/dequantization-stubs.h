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

enum qnnp_status qnnp_dequantize(size_t n, const uint8_t* input, float* output, uint8_t zero_point, float scale, pthreadpool_t threadpool)

typedef enum qnnp_status (*dequantization_function)(
    size_t n,
    const uint8_t* input,
    float* output,
    uint8_t zero_point,
    float scale,
    pthreadpool_t threadpool);

#define DECLARE_DEQUANTIZATION_FUNCTION(fn_name) \
    enum qnnp_status fn_name( \
        size_t n, \
        const uint8_t* input, \
        float* output, \
        uint8_t zero_point, \
        float scale, \
        pthreadpool_t threadpool);

DECLARE_DEQUANTIZATION_FUNCTION(qnnp_dequantize_uint8_fp32__scalar)
DECLARE_DEQUANTIZATION_FUNCTION(qnnp_dequantize_uint8_fp32__neon)
DECLARE_DEQUANTIZATION_FUNCTION(qnnp_dequantize_uint8_fp32__sse)


#ifdef __cplusplus
} /* extern "C" */
#endif
