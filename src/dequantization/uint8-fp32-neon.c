/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <arm_neon.h>

#include <qnnpack/dequantization-stubs.h>

void qnnp_dequantize_uint8_fp32__neon(
    size_t n,
    const uint8_t* input,
    float* output,
    uint8_t zero_point,
    float scale)
{
  assert(n % 16 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const float32x4_t vscale = vdupq_n_f32(scale);
  const uint32x4_t vzero_point = vdupq_n_u32((uint32_t) zero_point);

  for (; n != 0; n -= 16) {
    uint8x16_t vinput8x16 = vld1q_u8(input);
    uint8x8_t vinput8x8l = vget_low_u8(vinput8x16);
    uint8x8_t vinput8x8h = vget_high_u8(vinput8x16);


    uint16x8_t vinput16x8l = vmovl_u8(vinput8x8l);
    uint16x8_t vinput16x8h = vmovl_u8(vinput8x8h);
    input += 16;

    uint16x4_t vinput16x4la = vget_low_u16(vinput16x8l);
    uint16x4_t vinput16x4ha = vget_high_u16(vinput16x8l);
    uint16x4_t vinput16x4lb = vget_low_u16(vinput16x8h);
    uint16x4_t vinput16x4hb = vget_high_u16(vinput16x8h);

    uint32x4_t vinput32x4a = vmovl_u16(vinput16x4la);
    uint32x4_t vinput32x4b = vmovl_u16(vinput16x4ha);
    uint32x4_t vinput32x4c = vmovl_u16(vinput16x4lb);
    uint32x4_t vinput32x4d = vmovl_u16(vinput16x4hb);

    const float32x4_t resulta = vmulq_f32(vcvtq_f32_u32(vsubq_u32(vinput32x4a, vzero_point)), vscale);
    vst1q_f32(output, resulta);
    output += 4;
    const float32x4_t resultb = vmulq_f32(vcvtq_f32_u32(vsubq_u32(vinput32x4b, vzero_point)), vscale);
    vst1q_f32(output, resultb);
    output += 4;
    const float32x4_t resultc = vmulq_f32(vcvtq_f32_u32(vsubq_u32(vinput32x4c, vzero_point)), vscale);
    vst1q_f32(output, resultc);
    output += 4;
    const float32x4_t resultd = vmulq_f32(vcvtq_f32_u32(vsubq_u32(vinput32x4d, vzero_point)), vscale);
    vst1q_f32(output, resultd);
    output += 4;
  }
}
