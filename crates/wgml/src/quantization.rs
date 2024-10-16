//! Quantization and unquantization structures.
//!
//! This is inspired heavily from [ggml-common.h](https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-common.h#L144).

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
/// A single `f16` value.
pub struct BlockF16 {
    pub data: u16,
}

impl BlockF16 {
    pub fn dequantize(self) -> f32 {
        decode_f16(self.data)
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43-L46
pub struct BlockQ8_0 {
    pub scale: u16, // f16
    pub data: [i8; 32],
}

impl BlockQ8_0 {
    pub const ELEMENTS_PER_BLOCK: usize = 32;

    // See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1609
    pub fn dequantize(self) -> [f32; 32] {
        let scale = decode_f16(self.scale);
        self.data.map(|v| v as f32 * scale)
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11-L14
pub struct BlockQ4_0 {
    pub d: u16, // f16
    pub qs: [u8; 32 / 2],
}

impl BlockQ4_0 {
    pub const ELEMENTS_PER_BLOCK: usize = 32;
    // See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1515
    pub fn dequantize(self) -> [f32; 32] {
        let mut result = [0.0; 32];
        let d = decode_f16(self.d);

        for j in 0..Self::ELEMENTS_PER_BLOCK / 2 {
            let x0 = (self.qs[j] & 0x0F) as i32 - 8;
            let x1 = (self.qs[j] >> 4) as i32 - 8;

            result[j] = x0 as f32 * d;
            result[j + Self::ELEMENTS_PER_BLOCK / 2] = x1 as f32 * d;
        }

        result
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L18-L22
pub struct BlockQ4_1 {
    pub d: u16, // f16
    pub m: u16,
    pub qs: [u8; 32 / 2],
}

impl BlockQ4_1 {
    pub const ELEMENTS_PER_BLOCK: usize = 32;
    // See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1535
    pub fn dequantize(self) -> [f32; 32] {
        let mut result = [0.0; 32];
        let d = decode_f16(self.d);
        let m = decode_f16(self.m);

        for j in 0..Self::ELEMENTS_PER_BLOCK / 2 {
            let x0 = self.qs[j] & 0x0F;
            let x1 = self.qs[j] >> 4;

            result[j] = x0 as f32 * d + m;
            result[j + Self::ELEMENTS_PER_BLOCK / 2] = x1 as f32 * d + m;
        }

        result
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L26-L30
pub struct BlockQ5_0 {
    pub d: u16, // f16
    pub qh: [u8; 4],
    pub qs: [u8; 32 / 2],
}

impl BlockQ5_0 {
    pub const ELEMENTS_PER_BLOCK: usize = 32;
    // See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1556
    pub fn dequantize(self) -> [f32; 32] {
        let mut result = [0.0; 32];
        let d = decode_f16(self.d);
        let qh: u32 = bytemuck::cast(self.qh);

        for j in 0..Self::ELEMENTS_PER_BLOCK / 2 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((self.qs[j] as u32 & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((self.qs[j] as u32 >> 4) | xh_1) as i32 - 16;

            result[j] = x0 as f32 * d;
            result[j + Self::ELEMENTS_PER_BLOCK / 2] = x1 as f32 * d;
        }

        result
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L34-L39
pub struct BlockQ5_1 {
    pub d: u16,           // delta
    pub m: u16,           // min
    pub qh: [u8; 4],      // 5-th bit of quants
    pub qs: [u8; 32 / 2], // nibbles / quants
}

impl BlockQ5_1 {
    pub const ELEMENTS_PER_BLOCK: usize = 32;
    // See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1582
    pub fn dequantize(self) -> [f32; 32] {
        let mut result = [0.0; 32];
        let d = decode_f16(self.d);
        let m = decode_f16(self.m);
        let qh: u32 = bytemuck::cast(self.qh);

        for j in 0..Self::ELEMENTS_PER_BLOCK / 2 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = (self.qs[j] as u32 & 0x0F) | xh_0;
            let x1 = (self.qs[j] as u32 >> 4) | xh_1;

            result[j] = x0 as f32 * d + m;
            result[j + Self::ELEMENTS_PER_BLOCK / 2] = x1 as f32 * d + m;
        }

        result
    }
}

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L161-L165
pub struct BlockQ8_K {
    pub d: f32,                  // delta
    pub qs: [i8; QK_K],          // quants
    pub bsums: [i16; QK_K / 16], // sum of quants in groups of 16
}

impl BlockQ8_K {
    pub const ELEMENTS_PER_BLOCK: usize = QK_K;
    pub fn dequantize(self) -> [f32; QK_K] {
        let mut result = [0.0; QK_K];
        for j in 0..QK_K {
            result[j] = self.d * self.qs[j] as f32;
        }
        result
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152-L157
pub struct BlockQ6_K {
    pub ql: [u8; QK_K / 2],      // quants, lower 4 bits
    pub qh: [u8; QK_K / 4],      // quants, upper 2 bits
    pub scales: [i8; QK_K / 16], // scales, quantized with 8 bits
    pub d: u16,                  // super-block scale
}

impl BlockQ6_K {
    pub const ELEMENTS_PER_BLOCK: usize = QK_K;
    // https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L2970
    pub fn dequantize(self) -> [f32; QK_K] {
        let mut result = [0.0; QK_K];

        let d = decode_f16(self.d);
        let mut i = 0;

        for _ in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;

                let q1 = ((self.ql[i * 64 + l + 0] & 0xF) | (((self.qh[i * 32 + l] >> 0) & 3) << 4))
                    as i8
                    - 32;
                let q2 = ((self.ql[i * 64 + l + 32] & 0xF)
                    | (((self.qh[i * 32 + l] >> 2) & 3) << 4)) as i8
                    - 32;
                let q3 = ((self.ql[i * 64 + l + 0] >> 4) | (((self.qh[i * 32 + l] >> 4) & 3) << 4))
                    as i8
                    - 32;
                let q4 = ((self.ql[i * 64 + l + 32] >> 4) | (((self.qh[i * 32 + l] >> 6) & 3) << 4))
                    as i8
                    - 32;

                result[i * 128 + l + 0] = d * self.scales[i * 8 + is + 0] as f32 * q1 as f32;
                result[i * 128 + l + 32] = d * self.scales[i * 8 + is + 2] as f32 * q2 as f32;
                result[i * 128 + l + 64] = d * self.scales[i * 8 + is + 4] as f32 * q3 as f32;
                result[i * 128 + l + 96] = d * self.scales[i * 8 + is + 6] as f32 * q4 as f32;
            }

            i += 1;
        }

        result
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L130-L135
pub struct BlockQ5_K {
    pub d: u16,                     // super-block scale
    pub dmin: u16,                  // super-block scale for quantized mins
    pub scales: [u8; K_SCALE_SIZE], // scales and mins, quantized with 6 bits
    pub qh: [u8; QK_K / 8],         // quants, high bit
    pub qs: [u8; QK_K / 2],         // quants, low 4 bits
}

impl BlockQ5_K {
    pub const ELEMENTS_PER_BLOCK: usize = QK_K;
    pub fn dequantize(self) -> [f32; QK_K] {
        let mut result = [0.0; QK_K];
        let mut iq = 0;
        let mut is = 0;

        let d = decode_f16(self.d);
        let min = decode_f16(self.dmin);

        let mut sc = 0;
        let mut m = 0;
        let mut u1 = 1;
        let mut u2 = 2;

        for j in (0..QK_K).step_by(64) {
            get_scale_min_k4(is, &self.scales, &mut sc, &mut m);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            get_scale_min_k4(is + 1, &self.scales, &mut sc, &mut m);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;

            for l in 0..32 {
                result[j + l] = d1
                    * ((self.qs[iq + l] & 0xF) + (if self.qh[l] & u1 != 0 { 16 } else { 0 }))
                        as f32
                    - m1;
            }

            for l in 0..32 {
                result[j + l + 32] = d2
                    * ((self.qs[iq + l] >> 4) + (if self.qh[l] & u2 != 0 { 16 } else { 0 })) as f32
                    - m2;
            }

            iq += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }

        result
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L109-L113
pub struct BlockQ4_K {
    pub d: u16,                     // super-block scales for quantized scales
    pub dmin: u16,                  // super-block scale for quantized mins
    pub scales: [u8; K_SCALE_SIZE], // scales and mins, quantized with 6 bits
    pub qs: [u8; QK_K / 2],         // 4-bit quants
}

impl BlockQ4_K {
    pub const ELEMENTS_PER_BLOCK: usize = QK_K;
    // See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L2548
    pub fn dequantize(self) -> [f32; QK_K] {
        let mut result = [0.0; QK_K];
        let d = decode_f16(self.d);
        let min = decode_f16(self.dmin);

        let mut is = 0;
        let mut sc = 0u8;
        let mut m = 0u8;
        let mut iq = 0;

        for j in (0..QK_K).step_by(64) {
            get_scale_min_k4(is, &self.scales, &mut sc, &mut m);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            get_scale_min_k4(is + 1, &self.scales, &mut sc, &mut m);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;

            for l in 0..32 {
                result[j + l] = d1 * (self.qs[iq + l] & 0xF) as f32 - m1;
            }

            for l in 0..32 {
                result[j + l + 32] = d2 * (self.qs[iq + l] >> 4) as f32 - m2;
            }

            iq += 32;
            is += 2;
        }

        result
    }
}

fn get_scale_min_k4(j: usize, q: &[u8], d: &mut u8, m: &mut u8) {
    if j < 4 {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// From https://stackoverflow.com/questions/36008434/how-can-i-decode-f16-to-f32-using-only-the-stable-standard-library
pub fn decode_f16(half: u16) -> f32 {
    let exp: u16 = half >> 10 & 0x1f;
    let mant: u16 = half & 0x3ff;
    let val: f32 = if exp == 0 {
        (mant as f32) * (2.0f32).powi(-24)
    } else if exp != 31 {
        (mant as f32 + 1024f32) * (2.0f32).powi(exp as i32 - 25)
    } else if mant == 0 {
        ::std::f32::INFINITY
    } else {
        ::std::f32::NAN
    };
    if half & 0x8000 != 0 {
        -val
    } else {
        val
    }
}
