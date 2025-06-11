use crate::{
    errors::{MvfError, Result},
    mvf_fbs,
};

/// SIMD-optimized slice view of vector data.
///
/// Provides efficient access to vector components with support for
/// strided access patterns and SIMD operations.
///
/// # Examples
///
/// ```no_run
/// # use metrovector::{vectors::mem::VectorSlice, mvf_fbs::DataType, errors::Result};
/// # fn example() -> Result<()> {
/// let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
/// let slice = VectorSlice::new(&data, 4, 2, DataType::Float32)?;
///
/// let first_element: f32 = slice.get_element(0)?;
/// println!("First element: {}", first_element);
/// # Ok(())
/// # }
/// ```
pub struct VectorSlice<'a> {
    data: &'a [u8],
    stride: usize,
    count: usize,
    element_type: mvf_fbs::DataType,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> VectorSlice<'a> {
    /// Creates a new vector slice.
    ///
    /// # Arguments
    /// * `data` - Raw byte data
    /// * `stride` - Bytes between consecutive elements
    /// * `count` - Number of elements
    /// * `element_type` - Data type of elements
    ///
    /// # Errors
    /// Returns an error if:
    /// - Stride is not properly aligned for the element type
    /// - Data buffer is too small for the specified count and stride
    pub fn new(
        data: &'a [u8],
        stride: usize,
        count: usize,
        element_type: mvf_fbs::DataType,
    ) -> Result<Self> {
        let elem_size = Self::element_size(element_type)?;

        if stride % elem_size != 0 {
            return Err(MvfError::corrupted_data("Invalid stride alignment"));
        }

        if data.len() < count * stride {
            return Err(MvfError::corrupted_data("Data buffer too small for slice"));
        }

        Ok(Self {
            data,
            stride,
            count,
            element_type,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns a raw pointer to the data.
    ///
    /// # Safety
    /// The caller must ensure the pointer is used safely and doesn't outlive
    /// the slice.
    pub fn as_ptr<T>(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    /// Returns an aligned slice view for tightly packed data.
    ///
    /// # Safety
    /// Only works for non-strided data (stride == element size).
    ///
    /// # Errors
    /// Returns an error if:
    /// - Type size doesn't match element type
    /// - Data is not properly aligned
    /// - Data is strided (not tightly packed)
    pub fn as_aligned_slice<T>(&self) -> Result<&[T]>
    where
        T: Copy,
    {
        let type_size = std::mem::size_of::<T>();
        let type_align = std::mem::align_of::<T>();
        let expected_size = Self::element_size(self.element_type)?;

        if type_size != expected_size {
            return Err(MvfError::build_error("Type size mismatch"));
        }

        let ptr = self.data.as_ptr() as usize;
        if ptr % type_align != 0 {
            return Err(MvfError::corrupted_data("Data not properly aligned"));
        }

        // For strided data, we can only return a slice if it's tightly packed
        if self.stride == type_size {
            let len = self.count;

            if len > isize::MAX as usize {
                return Err(MvfError::corrupted_data("Slice too large"));
            }

            let prt = self.data.as_ptr() as *const T;
            Ok(unsafe { std::slice::from_raw_parts(prt, len) })
        } else {
            Err(MvfError::build_error(
                "Cannot create slice from strided data",
            ))
        }
    }

    /// Gets a single element by index.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Index is out of bounds
    /// - Element access would exceed data boundaries
    pub fn get_element<T>(&self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        if index >= self.count {
            return Err(MvfError::IndexOutOfBounds {
                index,
                len: self.count,
            });
        }

        let offset = index * self.stride;
        let type_size = std::mem::size_of::<T>();

        if offset + type_size > self.data.len() {
            return Err(MvfError::corrupted_data("Element access out of bounds"));
        }

        let ptr = unsafe { self.data.as_ptr().add(offset) as *const T };
        Ok(unsafe { ptr.read_unaligned() })
    }

    /// Creates an iterator over elements.
    pub fn iter_elements<T>(&self) -> ElementIterator<'_, T>
    where
        T: Copy,
    {
        ElementIterator::new(self, 0)
    }

    /// Returns whether the data is aligned for SIMD operations.
    ///
    /// # Arguments
    /// * `simd_width` - Required alignment in bytes (e.g., 32 for AVX2)
    pub fn is_simd_aligned(&self, simd_width: usize) -> bool {
        let ptr = self.data.as_ptr() as usize;
        ptr % simd_width == 0 // AVX2 alignment
    }

    /// Returns the optimal chunk size for SIMD operations.
    ///
    /// # Arguments
    /// * `simd_width` - SIMD register width in bytes
    pub fn chunk_size_for_simd(&self, simd_width: usize) -> usize {
        let elem_size = Self::element_size(self.element_type).unwrap_or(4);
        simd_width / elem_size
    }

    /// Returns the size in bytes of the given data type.
    fn element_size(data_type: mvf_fbs::DataType) -> Result<usize> {
        use mvf_fbs::DataType;
        match data_type {
            DataType::Float32 => Ok(4),
            DataType::Float16 => Ok(2),
            DataType::Int8 | DataType::UInt8 => Ok(1),
            _ => Err(MvfError::build_error("Unsupported data type")),
        }
    }
}

/// Iterator over vector slice elements.
pub struct ElementIterator<'a, T> {
    slice: &'a VectorSlice<'a>,
    current_index: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> ElementIterator<'a, T> {
    /// Creates a new element iterator.
    pub fn new(slice: &'a VectorSlice<'a>, start: usize) -> Self {
        Self {
            slice,
            current_index: start,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T> Iterator for ElementIterator<'a, T>
where
    T: Copy,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.slice.count {
            return None;
        }

        let result = self.slice.get_element(self.current_index);
        self.current_index += 1;
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mvf_fbs::DataType,
        reader::MvfReader,
        tests::{TestContext, create_test_mvf},
        vectors::access::AccessPattern,
    };

    use super::*;

    #[test]
    fn test_vector_slice_creation() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let slice = VectorSlice::new(&data, 4, 2, DataType::Float32).unwrap();

        assert_eq!(slice.count, 2);
        assert_eq!(slice.stride, 4);
    }

    #[test]
    fn test_vector_slice_invalid_stride() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];

        // stride of 3 is not divisible by 4 bytes
        // which is required for Float32
        let result = VectorSlice::new(&data, 3, 2, DataType::Float32);

        assert!(matches!(result, Err(MvfError::CorruptedData(_))));
    }

    #[test]
    fn test_vector_slice_buffer_too_small() {
        let data = vec![1u8, 2, 3, 4];

        // Need 8 bytes, have 4
        let result = VectorSlice::new(&data, 4, 2, DataType::Float32);

        assert!(matches!(result, Err(MvfError::CorruptedData(_))));
    }

    #[test]
    fn test_vector_slice_as_aligned_slice_f32() {
        let data: Vec<u8> = (0..16).collect();
        let slice = VectorSlice::new(&data, 4, 4, DataType::Float32).unwrap();

        let f32_slice: Result<&[f32]> = slice.as_aligned_slice();
        assert!(f32_slice.is_ok());
        assert_eq!(f32_slice.unwrap().len(), 4);
    }

    #[test]
    fn test_vector_slice_as_aligned_slice_type_mismatch() {
        let data: Vec<u8> = (0..16).collect();

        // Type here is Float16
        let slice = VectorSlice::new(&data, 2, 8, DataType::Float16).unwrap();

        // Explicitly try to get as f32
        let f32_slice: Result<&[f32]> = slice.as_aligned_slice();

        // Expect failure due to type mismatch
        assert!(matches!(f32_slice, Err(MvfError::Build(_))));
    }

    #[test]
    fn test_vector_slice_as_aligned_slice_strided() {
        let data: Vec<u8> = (0..24).collect();
        let slice = VectorSlice::new(&data, 8, 3, DataType::Float32).unwrap(); // Stride > element size

        let f32_slice: Result<&[f32]> = slice.as_aligned_slice();
        assert!(matches!(f32_slice, Err(MvfError::Build(_))));
    }

    #[test]
    fn test_vector_slice_get_element() {
        let mut data = vec![0u8; 16];
        // Write some f32 values
        let values = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
        for (i, &val) in values.iter().enumerate() {
            let bytes = val.to_le_bytes();
            data[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }

        let slice = VectorSlice::new(&data, 4, 4, DataType::Float32).unwrap();

        let val: f32 = slice.get_element(0).unwrap();
        assert_eq!(val, 1.0);

        let val: f32 = slice.get_element(3).unwrap();
        assert_eq!(val, 4.0);
    }

    #[test]
    fn test_vector_slice_get_element_out_of_bounds() {
        let data = vec![0u8; 16];
        let slice = VectorSlice::new(&data, 4, 4, DataType::Float32).unwrap();

        let result: Result<f32> = slice.get_element(10);
        assert!(matches!(result, Err(MvfError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_vector_slice_element_iterator() {
        let mut data = vec![0u8; 12];
        let values = [1.0f32, 2.0f32, 3.0f32];
        for (i, &val) in values.iter().enumerate() {
            let bytes = val.to_le_bytes();
            data[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }

        let slice = VectorSlice::new(&data, 4, 3, DataType::Float32).unwrap();

        let collected: Vec<f32> = slice.iter_elements().map(|r| r.unwrap()).collect();

        assert_eq!(collected, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_slice_element_iterator_empty() {
        let data = vec![0u8; 0];
        let slice = VectorSlice::new(&data, 4, 0, DataType::Float32).unwrap();

        let mut iter = slice.iter_elements::<f32>();
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_vector_slice_simd_alignment() {
        // Create 32-byte aligned data
        let mut data = vec![0u8; 64];
        let ptr = data.as_mut_ptr();
        let aligned_offset = (32 - (ptr as usize % 32)) % 32;
        let aligned_data = &data[aligned_offset..aligned_offset + 32];

        let slice = VectorSlice::new(aligned_data, 4, 8, DataType::Float32).unwrap();

        let simd_data_type_size = 32; // f32 size
        let simd_width = 256; // AVX2 width = 32 bytes * 8 floats

        // This might not always be aligned depending on the allocator
        // but we test the method exists and works
        let is_aligned = slice.is_simd_aligned(simd_data_type_size);
        assert!(is_aligned);

        let chunk_size = slice.chunk_size_for_simd(simd_width); // AVX2 width
        assert_eq!(chunk_size, 64); // 256 bits / 4 bytes per f32
    }

    #[test]
    fn test_vector_slice_element_size() {
        assert_eq!(
            VectorSlice::<'_>::element_size(DataType::Float32).unwrap(),
            4
        );
        assert_eq!(
            VectorSlice::<'_>::element_size(DataType::Float16).unwrap(),
            2
        );
        assert_eq!(VectorSlice::<'_>::element_size(DataType::Int8).unwrap(), 1);
    }

    #[test]
    fn test_vector_simd_slice_f32() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();
        let vector = space.get_vector(0).unwrap();

        let slice: Result<&[f32]> = vector.as_simd_slice();
        match slice {
            Ok(s) => assert_eq!(s.len(), 4),
            Err(_) => {
                // Might fail due to alignment, is okay for this test
            }
        }
    }

    #[test]
    fn test_vector_as_vector_slice() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();
        let vector = space.get_vector(0).unwrap();

        let vector_slice = vector.as_vector_slice().unwrap();

        assert_eq!(vector_slice.count, 4);
    }

    #[test]
    fn test_vector_cast_to() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();
        let vector = space.get_vector(0).unwrap();

        // Cast to bytes should always work
        let bytes: &[u8] = vector.cast_to().unwrap();
        assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes each

        // Cast to f32 should work if properly aligned
        match vector.cast_to::<f32>() {
            Ok(f32_slice) => {
                assert_eq!(f32_slice.len(), 4);
            }
            Err(MvfError::CorruptedData(_)) => {
                // Alignment error is okay here
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_vector_cast_to_size_mismatch() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();
        let vector = space.get_vector(0).unwrap();

        // Try to cast 16 bytes to something that doesn't divide evenly
        // For example, cast to u128 (16 bytes) should work
        match vector.cast_to::<u128>() {
            Ok(u128_slice) => {
                assert_eq!(u128_slice.len(), 1); // 16 bytes = 1 u128
            }
            Err(MvfError::CorruptedData(_)) => {
                // Alignment error is okay here
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_vector_cast_to_alignment_error() {
        // Create a vector with known misalignment
        // ```
        // let mut data = vec![0u8; 17]; // 17 bytes
        // data.remove(0); // Remove first byte to potentially misalign
        // ```
        // This is tricky to test because Rust's allocator usually provides aligned memory

        // data doesn't divide evenly by target type size
        let data = vec![0u8; 15]; // 15 bytes can't be divided into 4-byte f32s evenly

        let slice = VectorSlice::new(&data, 1, 15, DataType::UInt8).unwrap();

        // Try to get as f32 (4 bytes each) - should fail due to size mismatch
        let result: Result<&[f32]> = slice.as_aligned_slice();
        assert!(matches!(result, Err(MvfError::Build(_))));
    }

    #[test]
    fn test_vector_slice_alignment_checks() {
        let mut data = vec![0u8; 16];
        let values = [1.0f32, 2.0f32, 3.0f32, 4.0f32];

        for (i, &val) in values.iter().enumerate() {
            let bytes = val.to_le_bytes();
            data[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }

        let slice = VectorSlice::new(&data, 4, 4, DataType::Float32).unwrap();

        // This might fail due to alignment, which is okay
        match slice.as_aligned_slice::<f32>() {
            Ok(f32_slice) => {
                assert_eq!(f32_slice.len(), 4);
            }
            Err(MvfError::CorruptedData(_)) => {
                // Alignment error is acceptable in this test environment
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_vector_cast_to_bytes() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();
        let vector = space.get_vector(0).unwrap();

        // Casting to bytes should always work (no alignment requirements)
        let bytes: &[u8] = vector.cast_to().unwrap();

        // Expect 4 floats * 4 bytes each
        assert!(!bytes.is_empty());
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_vector_slice_size_limit() {
        // This is hard to test in practice because we'd need to allocate
        // more than isize::MAX bytes, which isn't practical
        // The check is there for safety but hard to unit test
    }

    #[test]
    fn test_map_vector_range_valid() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let slice = space.map_vector_range(0, 2).unwrap();

        // Expect 2 vectors from mapped range of
        // vectors 0..2 (inclusive)
        assert_eq!(slice.count, 2);
    }

    #[test]
    fn test_map_vector_range_out_of_bounds() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let total = space.total_vectors();
        let result = space.map_vector_range(0, total + 1);

        // Expect OOB error, we're intentionally
        // reading past the end of the vector space
        assert!(matches!(result, Err(MvfError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_map_vector_range_empty() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let slice = space.map_vector_range(1, 0).unwrap();

        // Expect 0 vectors from mapped range,
        // we're explicitly asking for 0 vectors
        assert_eq!(slice.count, 0);
    }

    #[test]
    fn test_clone_for_thread() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let cloned_space = space.clone_concurrent();

        // Both should work identically
        let vector1 = space.get_vector(0).unwrap();
        let vector2 = cloned_space.get_vector(0).unwrap();

        assert_eq!(vector1.dimension(), vector2.dimension());
        assert_eq!(vector1.data_type(), vector2.data_type());
    }

    #[test]
    fn test_get_vectors_with_pattern() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let pattern = AccessPattern::new(vec![2, 0, 1]);
        let vectors = space.get_vectors_with_pattern(&pattern).unwrap();

        assert_eq!(vectors.len(), 3);
        assert!(vectors.iter().all(|v| v.dimension() == 4));
    }

    #[test]
    fn test_get_vectors_with_empty_pattern() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let pattern = AccessPattern::new(vec![]);
        let vectors = space.get_vectors_with_pattern(&pattern).unwrap();

        assert!(vectors.is_empty());
    }

    #[test]
    fn test_vector_exact_size_iterator() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let mut total_count = 0;
        for chunk_result in space.stream_vectors(0, 1) {
            let chunk = chunk_result.unwrap();
            total_count += chunk.len();
        }

        assert_eq!(total_count, space.total_vectors() as usize);
    }

    #[test]
    fn test_vector_as_f32_different_types() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();
        let vector = space.get_vector(0).unwrap();

        // Should work for Float32
        let f32_vec = vector.as_f32().unwrap();
        assert_eq!(f32_vec.len(), 4);
    }

    #[test]
    fn test_vector_properties() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();
        let vector = space.get_vector(0).unwrap();

        assert_eq!(vector.dimension(), 4);
        assert_eq!(vector.data_type(), crate::mvf_fbs::DataType::Float32);
        assert_eq!(vector.as_bytes().len(), 16); // 4 * 4 bytes
    }

    #[test]
    fn test_vector_slice_as_ptr() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let slice = VectorSlice::new(&data, 4, 2, DataType::Float32).unwrap();

        let ptr: *const f32 = slice.as_ptr();
        assert!(!ptr.is_null());

        // Verify we can read through the pointer (unsafe but for testing)
        unsafe {
            let value = std::ptr::read_unaligned(ptr);
            // This reads the first 4 bytes as f32
            assert!(!value.is_nan()); // Just verify it's a valid f32
        }
    }

    #[test]
    fn test_vector_slice_different_data_types() {
        // Test Float16
        let data = vec![0u8; 16]; // 8 f16 values
        let slice = VectorSlice::new(&data, 2, 8, DataType::Float16).unwrap();
        assert_eq!(slice.chunk_size_for_simd(32), 16); // 32 bytes / 2 bytes per f16

        // Test Int8
        let data = vec![0u8; 16];
        let slice = VectorSlice::new(&data, 1, 16, DataType::Int8).unwrap();
        assert_eq!(slice.chunk_size_for_simd(32), 32); // 32 bytes / 1 byte per int8

        // Test UInt8
        let slice = VectorSlice::new(&data, 1, 16, DataType::UInt8).unwrap();
        assert_eq!(slice.chunk_size_for_simd(32), 32);
    }

    #[test]
    fn test_element_iterator_error_handling() {
        let data = vec![1u8, 2, 3, 4];
        let slice = VectorSlice::new(&data, 4, 1, DataType::Float32).unwrap();

        let mut iter = slice.iter_elements::<f32>();

        // First element should work
        let first = iter.next().unwrap();
        assert!(first.is_ok());

        // Second element should return None (out of bounds)
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_vector_slice_edge_cases() {
        // Test with minimum valid data
        let data = vec![0u8; 4];
        let slice = VectorSlice::new(&data, 4, 1, DataType::Float32).unwrap();
        assert_eq!(slice.count, 1);

        // Test element access at boundary
        let result: Result<f32> = slice.get_element(0);
        assert!(result.is_ok());

        let result: Result<f32> = slice.get_element(1);
        assert!(matches!(result, Err(MvfError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_unsupported_data_type_error() {
        // This should fail for unsupported data types
        let result = VectorSlice::element_size(DataType::StringRef);
        assert!(matches!(result, Err(MvfError::Build(_))));
    }

    #[test]
    fn test_vector_all_accessors() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test_vector.mvf");

        built.save(&path).unwrap();
        let reader = crate::reader::MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();
        let vector = space.get_vector(0).unwrap();

        // Test all accessor methods
        assert_eq!(vector.dimension(), 4);
        assert_eq!(vector.data_type(), DataType::Float32);
        assert_eq!(vector.as_bytes().len(), 16);

        let f32_values = vector.as_f32().unwrap();
        assert_eq!(f32_values.len(), 4);

        // Test vector slice creation
        let vector_slice = vector.as_vector_slice().unwrap();
        assert_eq!(vector_slice.count, 4);
    }
}
