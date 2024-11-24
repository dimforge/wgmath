//! Loading gguf files.

use crate::quantization::{
    BlockF16, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0, BlockQ5_1, BlockQ5_K, BlockQ6_K,
    BlockQ8_0, BlockQ8_K,
};
use bytemuck::{Pod, PodCastError};
use std::collections::HashMap;

const GGUF_FILE_MAGIC_LE: u32 = 0x46554747; // "fugg" (little-endian)
const GGUF_FILE_MAGIC_BE: u32 = 0x47475546; // "gguf"
                                            // const GGUF_QNT_VERSION_FACTOR: u32 = 1000; // do not change this

#[derive(thiserror::Error, Debug, Copy, Clone)]
pub enum GgufParseError {
    #[error(
        "the input file isn’t a ggml binary file. Got a magic number of {0:#x} instead of 0x67676d6c"
    )]
    IncorrectMagicNumber(u32),
    #[error("metadata value of type {0} is not part of the supported gguf version")]
    UnsupportedMetadataValueType(u32),
    #[error("tensor of type {0} is not part of the supported gguf version")]
    UnsupportedTensorType(u32),
    #[error("invalid vocabulary size (expected {expected}, found {found})")]
    VocabSizeMismatch { expected: u32, found: u32 },
    #[error(transparent)]
    Cast(#[from] PodCastError),
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum GgufTensorData {
    F32(Vec<f32>),
    F16(Vec<BlockF16>),
    Q4_0(Vec<BlockQ4_0>),
    Q4_1(Vec<BlockQ4_1>),
    Q5_0(Vec<BlockQ5_0>),
    Q5_1(Vec<BlockQ5_1>),
    Q8_0(Vec<BlockQ8_0>),
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K(Vec<BlockQ4_K>),
    Q5_K(Vec<BlockQ5_K>),
    Q6_K(Vec<BlockQ6_K>),
    Q8_K(Vec<BlockQ8_K>),
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    IQ1_M,
}

impl GgufTensorData {
    fn from_u32(tag: u32) -> Result<Self, GgufParseError> {
        match tag {
            0 => Ok(GgufTensorData::F32(Vec::new())),
            1 => Ok(GgufTensorData::F16(Vec::new())),
            2 => Ok(GgufTensorData::Q4_0(Vec::new())),
            3 => Ok(GgufTensorData::Q4_1(Vec::new())),
            6 => Ok(GgufTensorData::Q5_0(Vec::new())),
            7 => Ok(GgufTensorData::Q5_1(Vec::new())),
            8 => Ok(GgufTensorData::Q8_0(Vec::new())),
            9 => Ok(GgufTensorData::Q8_1),
            10 => Ok(GgufTensorData::Q2_K),
            11 => Ok(GgufTensorData::Q3_K),
            12 => Ok(GgufTensorData::Q4_K(Vec::new())),
            13 => Ok(GgufTensorData::Q5_K(Vec::new())),
            14 => Ok(GgufTensorData::Q6_K(Vec::new())),
            15 => Ok(GgufTensorData::Q8_K(Vec::new())),
            16 => Ok(GgufTensorData::IQ2_XXS),
            17 => Ok(GgufTensorData::IQ2_XS),
            18 => Ok(GgufTensorData::IQ3_XXS),
            19 => Ok(GgufTensorData::IQ1_S),
            20 => Ok(GgufTensorData::IQ4_NL),
            21 => Ok(GgufTensorData::IQ3_S),
            22 => Ok(GgufTensorData::IQ2_S),
            23 => Ok(GgufTensorData::IQ4_XS),
            24 => Ok(GgufTensorData::I8(Vec::new())),
            25 => Ok(GgufTensorData::I16(Vec::new())),
            26 => Ok(GgufTensorData::I32(Vec::new())),
            27 => Ok(GgufTensorData::I64(Vec::new())),
            28 => Ok(GgufTensorData::F64(Vec::new())),
            29 => Ok(GgufTensorData::IQ1_M),
            _ => Err(GgufParseError::UnsupportedTensorType(tag)),
        }
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(vals) => Some(&vals[..]),
            _ => None,
        }
    }

    pub fn dequantize(&self) -> Option<Vec<f32>> {
        match self {
            Self::F32(v) => Some(v.clone()),
            Self::F16(v) => Some(v.iter().map(|v| v.dequantize()).collect()),
            Self::Q8_0(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            Self::Q5_0(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            Self::Q5_1(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            Self::Q4_0(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            Self::Q4_1(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            Self::Q8_K(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            Self::Q6_K(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            Self::Q5_K(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            Self::Q4_K(v) => Some(v.iter().flat_map(|v| v.dequantize().into_iter()).collect()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum GgufMetadataValueArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Array(Vec<GgufMetadataValueArray>),
}

impl GgufMetadataValueArray {
    /// Is this array empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of elements in this array.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Array(v) => v.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(GgufMetadataValueArray),
}

impl GgufMetadataValue {
    pub fn unwrap_u32(&self) -> u32 {
        if let Self::U32(val) = self {
            *val
        } else {
            panic!("unwrap: nexpected GGUF attribute type.")
        }
    }

    pub fn unwrap_array_len(&self) -> usize {
        if let Self::Array(val) = self {
            val.len()
        } else {
            panic!("unwrap: nexpected GGUF attribute type.")
        }
    }

    pub fn as_string_array(&self) -> &[String] {
        if let Self::Array(GgufMetadataValueArray::String(val)) = self {
            val
        } else {
            panic!("unwrap: nexpected GGUF attribute type.")
        }
    }

    pub fn as_f32_array(&self) -> &[f32] {
        if let Self::Array(GgufMetadataValueArray::F32(val)) = self {
            val
        } else {
            panic!("unwrap: nexpected GGUF attribute type.")
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    dimensions: [u64; 4], // Currently at most 4, but this may change in the future
    // Offset of the tensor data, relative to the start of the file.
    offset: u64,
    // The tensor data. The vec might be empty if the user requested
    // not to read the data in memory right away.
    data: GgufTensorData,
}

impl GgufTensor {
    pub fn dimensions(&self) -> [u64; 4] {
        self.dimensions
    }

    pub fn data(&self) -> &GgufTensorData {
        &self.data
    }
}

#[derive(Debug, Clone)]
pub struct Gguf {
    pub version: u32,
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: HashMap<String, GgufTensor>,
}

#[derive(Debug)]
struct Header {
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
    alignment: u32,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum MetadataValueType {
    // The value is a 8-bit unsigned integer.
    U8 = 0,
    // The value is a 8-bit signed integer.
    I8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    U16 = 2,
    // The value is a 16-bit signed little-endian integer.
    I16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    U32 = 4,
    // The value is a 32-bit signed little-endian integer.
    I32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    F32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String = 8,
    // The value is an array of other values, with the length and type prepended.
    //
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array = 9,
    // The value is a 64-bit unsigned little-endian integer.
    U64 = 10,
    // The value is a 64-bit signed little-endian integer.
    I64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    F64 = 12,
}

impl MetadataValueType {
    fn from_u32(val: u32) -> Result<Self, GgufParseError> {
        match val {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(GgufParseError::UnsupportedMetadataValueType(val)),
        }
    }

    fn read_value(
        self,
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<GgufMetadataValue, GgufParseError> {
        Ok(match self {
            Self::U8 => GgufMetadataValue::U8(read_pod(bytes, offset)?),
            Self::I8 => GgufMetadataValue::I8(read_pod(bytes, offset)?),
            Self::U16 => GgufMetadataValue::U16(read_pod(bytes, offset)?),
            Self::I16 => GgufMetadataValue::I16(read_pod(bytes, offset)?),
            Self::U32 => GgufMetadataValue::U32(read_pod(bytes, offset)?),
            Self::I32 => GgufMetadataValue::I32(read_pod(bytes, offset)?),
            Self::F32 => GgufMetadataValue::F32(read_pod(bytes, offset)?),
            Self::Bool => GgufMetadataValue::Bool(read_pod::<u8>(bytes, offset) != Ok(0)),
            Self::U64 => GgufMetadataValue::U64(read_pod(bytes, offset)?),
            Self::I64 => GgufMetadataValue::I64(read_pod(bytes, offset)?),
            Self::F64 => GgufMetadataValue::F64(read_pod(bytes, offset)?),
            Self::String => GgufMetadataValue::String(read_string(bytes, offset)?),
            Self::Array => {
                let ty = MetadataValueType::from_u32(read_pod(bytes, offset)?)?;
                GgufMetadataValue::Array(ty.read_array(bytes, offset)?)
            }
        })
    }

    fn read_array(
        self,
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<GgufMetadataValueArray, GgufParseError> {
        let len: u64 = read_pod(bytes, offset)?;
        let len = len as usize;

        Ok(match self {
            Self::U8 => GgufMetadataValueArray::U8(read_array_unaligned(len, bytes, offset)?),
            Self::I8 => GgufMetadataValueArray::I8(read_array_unaligned(len, bytes, offset)?),
            Self::U16 => GgufMetadataValueArray::U16(read_array_unaligned(len, bytes, offset)?),
            Self::I16 => GgufMetadataValueArray::I16(read_array_unaligned(len, bytes, offset)?),
            Self::U32 => GgufMetadataValueArray::U32(read_array_unaligned(len, bytes, offset)?),
            Self::I32 => GgufMetadataValueArray::I32(read_array_unaligned(len, bytes, offset)?),
            Self::F32 => GgufMetadataValueArray::F32(read_array_unaligned(len, bytes, offset)?),
            Self::Bool => {
                GgufMetadataValueArray::Bool(read_bool_array_unaligned(len, bytes, offset)?)
            }
            Self::U64 => GgufMetadataValueArray::U64(read_array_unaligned(len, bytes, offset)?),
            Self::I64 => GgufMetadataValueArray::I64(read_array_unaligned(len, bytes, offset)?),
            Self::F64 => GgufMetadataValueArray::F64(read_array_unaligned(len, bytes, offset)?),
            Self::String => GgufMetadataValueArray::String(
                (0..len)
                    .map(|_| read_string(bytes, offset))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Self::Array => {
                let arrays: Result<_, GgufParseError> = (0..len)
                    .map(|_| {
                        let ty = MetadataValueType::from_u32(read_pod(bytes, offset)?)?;
                        ty.read_array(bytes, offset)
                    })
                    .collect();
                GgufMetadataValueArray::Array(arrays?)
            }
        })
    }
}

impl Gguf {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GgufParseError> {
        let mut offset = 0;
        let mut header = Self::load_header(bytes, &mut offset)?;
        println!("Found header: {:?}", header);
        let metadata = Self::load_metadata(&header, bytes, &mut offset)?;
        header.alignment = metadata
            .get("general.alignment")
            .and_then(|val| {
                let GgufMetadataValue::U32(align) = val else {
                    return None;
                };
                Some(*align)
            })
            .unwrap_or(32);
        let tensors = Self::load_tensors(&header, bytes, &mut offset)?;

        Ok(Self {
            version: header.version,
            metadata,
            tensors,
        })
    }

    fn load_header(bytes: &[u8], offset: &mut usize) -> Result<Header, GgufParseError> {
        let magic: u32 = read_pod(bytes, offset)?;

        if magic != GGUF_FILE_MAGIC_LE && magic != GGUF_FILE_MAGIC_BE {
            return Err(GgufParseError::IncorrectMagicNumber(magic));
        }

        let version: u32 = read_pod(bytes, offset)?;
        let tensor_count: u64 = read_pod(bytes, offset)?;
        let metadata_count: u64 = read_pod(bytes, offset)?;
        Ok(Header {
            version,
            tensor_count,
            metadata_count,
            alignment: 32, // Will be read/set after reading metadatas.
        })
    }

    fn load_metadata(
        header: &Header,
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<HashMap<String, GgufMetadataValue>, GgufParseError> {
        let mut metadata = HashMap::new();
        for _ in 0..header.metadata_count {
            let key = read_string(bytes, offset)?;
            let value_type = MetadataValueType::from_u32(read_pod(bytes, offset)?)?;
            let value = value_type.read_value(bytes, offset)?;
            metadata.insert(key, value);
        }
        Ok(metadata)
    }

    fn load_tensors(
        header: &Header,
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<HashMap<String, GgufTensor>, GgufParseError> {
        let mut result = HashMap::new();
        for _ in 0..header.tensor_count {
            let name = read_string(bytes, offset)?;
            let ndims: u32 = read_pod(bytes, offset)?;
            let mut dimensions = [1u64; 4];

            assert!(
                ndims <= 4,
                "Tensors of dimensions larger than 4 are not supported yet."
            );

            for i_dim in 0..ndims {
                dimensions[i_dim as usize] = read_pod(bytes, offset)?;
            }

            let data = GgufTensorData::from_u32(read_pod(bytes, offset)?)?;
            let offset: u64 = read_pod(bytes, offset)?;
            result.insert(
                name,
                GgufTensor {
                    dimensions,
                    offset,
                    data,
                },
            );
        }

        // Populate tensors (TODO: this should be optional.)
        let tensor_data_start_offset = align_offset(*offset as u64, header.alignment as u64);
        for tensor in result.values_mut() {
            tensor.offset += tensor_data_start_offset;
            let mut tensor_offset = tensor.offset as usize;
            let len = tensor.dimensions[0]
                * tensor.dimensions[1]
                * tensor.dimensions[2]
                * tensor.dimensions[3];
            match &mut tensor.data {
                GgufTensorData::F32(ref mut data) => {
                    *data = read_array_unaligned(len as usize, bytes, &mut tensor_offset)?;
                }
                GgufTensorData::F16(ref mut data) => {
                    *data = read_array_unaligned(len as usize, bytes, &mut tensor_offset)?;
                }
                GgufTensorData::Q8_0(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ8_0::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                GgufTensorData::Q5_0(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ5_0::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                GgufTensorData::Q5_1(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ5_1::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                GgufTensorData::Q4_0(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ4_0::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                GgufTensorData::Q4_1(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ4_1::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                GgufTensorData::Q8_K(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ8_K::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                GgufTensorData::Q6_K(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ6_K::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                GgufTensorData::Q5_K(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ5_K::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                GgufTensorData::Q4_K(ref mut data) => {
                    *data = read_array_unaligned(
                        len as usize / BlockQ4_K::ELEMENTS_PER_BLOCK,
                        bytes,
                        &mut tensor_offset,
                    )?;
                }
                _ => { /* Unsupported. */ }
            }
        }

        Ok(result)
    }

    pub fn print_metadata(&self) {
        for (key, val) in &self.metadata {
            if let GgufMetadataValue::Array(arr) = val {
                println!("{} = array[{}]", key, arr.len());
            } else {
                println!("{} = {:?}", key, val);
            }
        }
    }

    pub fn print_tensors(&self) {
        let mut keys: Vec<_> = self.tensors.keys().collect();
        keys.sort();
        for key in keys {
            let tensor = &self.tensors[key];
            println!("{} -> {:?}", key, tensor.dimensions());
        }
    }
}

fn read_string(bytes: &[u8], offset: &mut usize) -> Result<String, PodCastError> {
    let key_len: u64 = read_pod(bytes, offset)?;
    let key = String::from_utf8_lossy(read_array(key_len as usize, bytes, offset)?);
    Ok(key.to_string())
}

fn read_pod<T: Pod>(bytes: &[u8], offset: &mut usize) -> Result<T, PodCastError> {
    let sz = std::mem::size_of::<T>();
    let bytes = &bytes[*offset..(*offset + sz)];
    *offset += sz;
    bytemuck::try_pod_read_unaligned(bytes)
}

fn read_array_unaligned<T: Pod>(
    len: usize,
    bytes: &[u8],
    offset: &mut usize,
) -> Result<Vec<T>, PodCastError> {
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        result.push(read_pod(bytes, offset)?);
    }
    Ok(result)
}

// TODO: this could actually always be considered aligned.
fn read_bool_array_unaligned(
    len: usize,
    bytes: &[u8],
    offset: &mut usize,
) -> Result<Vec<bool>, PodCastError> {
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        result.push(read_pod::<u8>(bytes, offset)? != 0);
    }
    Ok(result)
}

fn read_array<'a, T: Pod>(
    len: usize,
    bytes: &'a [u8],
    offset: &mut usize,
) -> Result<&'a [T], PodCastError> {
    let sz = std::mem::size_of::<T>() * len;
    let bytes = &bytes[*offset..(*offset + len)];
    *offset += sz;
    bytemuck::try_cast_slice(bytes)
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset + (alignment - (offset % alignment)) % alignment
}

#[cfg(test)]
mod test {
    use crate::gguf::Gguf;

    #[test]
    fn load_dummy_gguf() {
        // NOTE: the dummy.gguf test file was generated with llama.ccp:
        //       `llama.cpp/build/bin/llama-gguf dummy.gguf w`
        let bytes = std::fs::read("../../assets/gguf/dummy.gguf").unwrap();
        let gguf = Gguf::from_bytes(&bytes).unwrap();
        println!("gguf: {:#?}", gguf);
    }
}
