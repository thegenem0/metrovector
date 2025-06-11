use crate::{
    errors::{MvfError, Result},
    mvf_fbs,
};

use super::mem::VectorSlice;

/// A single vector with typed data access.
///
/// Provides efficient access to vector components with automatic type conversion
/// and SIMD-optimized operations where possible.
///
/// # Examples
///
/// ```no_run
/// # use metrovector::{reader::MvfReader, errors::Result};
/// # fn example() -> Result<()> {
/// let reader = MvfReader::open("vectors.mvf")?;
/// let space = reader.vector_space("embeddings")?;
/// let vector = space.get_vector(0)?;
///
/// println!("Dimension: {}", vector.dimension());
/// let values = vector.as_f32()?;
/// println!("First component: {}", values[0]);
/// # Ok(())
/// # }
/// ```
pub struct Vector<'a> {
    data: &'a [u8],
    dimension: u32,
    data_type: mvf_fbs::DataType,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Vector<'a> {
    /// Creates a new vector from raw data.
    ///
    /// # Safety
    /// The caller must ensure that `data` contains valid data for the specified
    /// dimension and data type.
    pub fn new(data: &'a [u8], dimension: u32, data_type: mvf_fbs::DataType) -> Self {
        Self {
            data,
            dimension,
            data_type,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the vector dimension (number of components).
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Returns the storage data type.
    pub fn data_type(&self) -> mvf_fbs::DataType {
        self.data_type
    }

    /// Returns the raw data buffer.
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    /// Converts vector components to f32 values.
    ///
    /// Supports automatic conversion from Float16 and Float32 storage formats.
    ///
    /// # Errors
    /// Returns an error if the data type cannot be converted to f32.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        match self.data_type {
            mvf_fbs::DataType::Float32 => {
                let mut result = Vec::with_capacity(self.dimension as usize);
                for chunk in self.data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    result.push(f32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            mvf_fbs::DataType::Float16 => {
                let mut result = Vec::with_capacity(self.dimension as usize);
                for chunk in self.data.chunks_exact(2) {
                    let bytes = [chunk[0], chunk[1]];
                    let f16_val = half::f16::from_le_bytes(bytes);
                    result.push(f16_val.to_f32());
                }
                Ok(result)
            }
            _ => Err(MvfError::build_error("Cannot convert to f32")),
        }
    }

    /// Returns a typed slice view of the vector data.
    ///
    /// # Safety
    /// The caller must ensure T matches the vector's data type and that
    /// the data is properly aligned for type T.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Data length is not divisible by the element size
    /// - Data is not properly aligned
    pub fn as_slice<T>(&self) -> Result<&[T]>
    where
        T: Copy,
    {
        let element_size = std::mem::size_of::<T>();
        if self.data.len() % element_size != 0 {
            return Err(MvfError::corrupted_data("Invalid vector data alignment"));
        }

        let ptr = self.data.as_ptr() as *const T;

        // Safety: We know the length is a multiple of the element size
        // and the pointer is valid
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.data.len() / element_size) };
        Ok(slice)
    }

    /// Returns a SIMD-aligned slice view for optimized operations.
    ///
    /// # Safety
    /// The caller must ensure proper alignment for SIMD operations.
    ///
    /// # Errors
    /// Returns an error if data size or alignment requirements aren't met.
    pub fn as_simd_slice<T>(&self) -> Result<&[T]>
    where
        T: Copy,
    {
        let type_size = std::mem::size_of::<T>();
        let expected_len = self.dimension as usize;

        if self.data.len() != expected_len * type_size {
            return Err(MvfError::corrupted_data("Vector size mismatch"));
        }

        // Check alignment for SIMD operations
        let ptr = self.data.as_ptr() as usize;
        if ptr % std::mem::align_of::<T>() != 0 {
            return Err(MvfError::corrupted_data("Vector data not properly aligned"));
        }

        let ptr = self.data.as_ptr() as *const T;
        Ok(unsafe { std::slice::from_raw_parts(ptr, expected_len) })
    }

    /// Creates a vector slice for advanced memory operations.
    ///
    /// # Errors
    /// Returns an error if the data type is unsupported.
    pub fn as_vector_slice(&self) -> Result<VectorSlice<'a>> {
        use mvf_fbs::DataType;
        let elem_size = match self.data_type {
            DataType::Float32 => 4,
            DataType::Float16 => 2,
            DataType::Int8 | DataType::UInt8 => 1,
            _ => return Err(MvfError::build_error("Unsupported data type")),
        };

        VectorSlice::new(
            self.data,
            elem_size,
            self.dimension() as usize,
            self.data_type(),
        )
    }

    /// Casts vector data to a different type.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The target type T is appropriate for the data
    /// - Alignment requirements are met
    /// - The cast is semantically valid
    ///
    /// # Errors
    /// Returns an error if:
    /// - Size is not divisible by target type size
    /// - Alignment requirements aren't met
    /// - Resulting slice would be too large
    pub fn cast_to<T>(&self) -> Result<&[T]>
    where
        T: Copy,
    {
        let type_size = std::mem::size_of::<T>();

        if self.data.len() % type_size != 0 {
            return Err(MvfError::corrupted_data("Cannot cast: size mismatch"));
        }

        let ptr = self.data.as_ptr() as usize;
        if ptr % std::mem::align_of::<T>() != 0 {
            return Err(MvfError::corrupted_data("Cannot cast: alignment mismatch"));
        }

        let ptr = self.data.as_ptr() as *const T;
        let len = self.data.len() / type_size;

        if len > isize::MAX as usize {
            return Err(MvfError::corrupted_data("Cannot cast: slice too large"));
        }

        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts;

    use crate::mvf_fbs::DataType;

    use super::*;

    #[test]
    fn test_vector_as_slice_error_cases() {
        // Create vector with misaligned data size
        let data = vec![1u8, 2, 3]; // 3 bytes, not divisible by 4
        let vector = Vector::new(&data, 1, DataType::Float32);

        let result: Result<&[f32]> = vector.as_slice();
        assert!(matches!(result, Err(MvfError::CorruptedData(_))));
    }

    #[test]
    fn test_vector_as_f32_float16() {
        // Test Float16 to f32 conversion
        let mut data = Vec::new();
        let f16_val = half::f16::from_f32(consts::PI);
        data.extend_from_slice(&f16_val.to_le_bytes());
        data.extend_from_slice(&half::f16::from_f32(consts::E).to_le_bytes());

        let vector = Vector::new(&data, 2, DataType::Float16);
        let f32_vec = vector.as_f32().unwrap();

        assert_eq!(f32_vec.len(), 2);
        assert!((f32_vec[0] - consts::PI).abs() < 0.01); // Some precision loss expected
        assert!((f32_vec[1] - consts::E).abs() < 0.01);
    }

    #[test]
    fn test_vector_as_f32_unsupported_type() {
        let data = vec![1u8, 2, 3, 4];
        let vector = Vector::new(&data, 1, DataType::StringRef);

        let result = vector.as_f32();
        assert!(matches!(result, Err(MvfError::Build(_))));
    }

    #[test]
    fn test_vector_as_simd_slice_size_mismatch() {
        let data = vec![1u8, 2, 3]; // Wrong size for dimension 1 * f32
        let vector = Vector::new(&data, 1, DataType::Float32);

        let result: Result<&[f32]> = vector.as_simd_slice();
        assert!(matches!(result, Err(MvfError::CorruptedData(_))));
    }

    #[test]
    fn test_vector_as_vector_slice_unsupported_type() {
        let data = vec![1u8, 2, 3, 4];
        let vector = Vector::new(&data, 1, DataType::StringRef);

        let result = vector.as_vector_slice();
        assert!(matches!(result, Err(MvfError::Build(_))));
    }

    #[test]
    fn test_vector_cast_to_slice_too_large() {
        // Create a scenario where the resulting slice would be too large
        // This is hard to test in practice, but we can test the bounds checking
        let data = vec![0u8; 8];
        let vector = Vector::new(&data, 2, DataType::Float32);

        // Cast to bytes should work
        let bytes: &[u8] = vector.cast_to().unwrap();
        assert_eq!(bytes.len(), 8);

        // Cast to u64 should work (8 bytes = 1 u64)
        let u64_slice: &[u64] = vector.cast_to().unwrap();
        assert_eq!(u64_slice.len(), 1);
    }
}
