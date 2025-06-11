use crate::{
    METRO_MAGIC,
    errors::{MvfError, Result},
    mvf_fbs::{self, DataBlock},
};

use super::{
    access::AccessPattern, dimension::DimensionSlice, iterator::VectorChunkIterator,
    mem::VectorSlice, vector::Vector,
};

/// A collection of vectors with shared properties and indexing.
///
/// Represents a named vector space within an MVF file, providing efficient
/// access to individual vectors and batch operations.
///
/// # Examples
///
/// ```no_run
/// # use metrovector::{reader::MvfReader, errors::Result};
/// # fn example() -> Result<()> {
/// let reader = MvfReader::open("vectors.mvf")?;
/// let space = reader.vector_space("embeddings")?;
///
/// println!("Space: {}", space.name());
/// println!("Dimension: {}", space.dimension());
/// println!("Total vectors: {}", space.total_vectors());
///
/// let vector = space.get_vector(0)?;
/// let values = vector.as_f32()?;
/// # Ok(())
/// # }
/// ```
pub struct VectorSpace<'a> {
    space: mvf_fbs::VectorSpace<'a>,
    data: &'a [u8],
    blocks: flatbuffers::Vector<'a, DataBlock>,
    index: usize,
}

impl<'a> VectorSpace<'a> {
    /// Creates a new vector space view.
    ///
    /// # Safety
    /// Internal constructor - assumes all parameters are valid and consistent
    /// with the file format.
    pub(crate) fn new(
        space: mvf_fbs::VectorSpace<'a>,
        file_data: &'a [u8],
        blocks: flatbuffers::Vector<'a, DataBlock>,
        index: usize,
    ) -> Self {
        Self {
            space,
            data: file_data,
            blocks,
            index,
        }
    }

    /// Returns the vector space name.
    pub fn name(&self) -> &str {
        self.space.name()
    }

    /// Returns the vector space dimension (number of components).
    pub fn dimension(&self) -> u32 {
        self.space.dimension()
    }

    /// Returns the total number of vectors in the space.
    pub fn total_vectors(&self) -> u64 {
        self.space.total_vectors()
    }

    /// Returns the vector type (Dense, Sparse, etc.).
    pub fn vector_type(&self) -> mvf_fbs::VectorType {
        self.space.vector_type()
    }

    /// Returns the distance metric used for similarity calculations.
    pub fn distance_metric(&self) -> mvf_fbs::DistanceMetric {
        self.space.distance_metric()
    }

    /// Returns the storage data type for vector components.
    pub fn data_type(&self) -> mvf_fbs::DataType {
        self.space.data_type()
    }

    /// Gets a vector by index.
    ///
    /// # Arguments
    /// * `index` - Zero-based vector index
    ///
    /// # Errors
    /// Returns an error if:
    /// - Index is out of bounds
    /// - Data block is corrupted or invalid
    /// - Vector data cannot be accessed
    pub fn get_vector(&self, index: u64) -> Result<Vector<'a>> {
        if index >= self.total_vectors() {
            return Err(MvfError::IndexOutOfBounds {
                index: index as usize,
                len: self.total_vectors() as usize,
            });
        }

        let block_index = self.space.vectors_block_index() as usize;
        if block_index >= self.blocks.len() {
            return Err(MvfError::corrupted_data("Invalid vector block index"));
        }

        let block = self.blocks.get(block_index);

        // Push offset over by METRO_MAGIC.len() to account for the magic bytes
        // as this is a relative offset calculation
        let data_start_offset = METRO_MAGIC.len();
        let start_offset = data_start_offset + block.offset() as usize;
        let block_data = &self.data[start_offset..start_offset + block.size() as usize];

        let element_size = match self.data_type() {
            mvf_fbs::DataType::Float32 => 4,
            mvf_fbs::DataType::Float16 => 2,
            mvf_fbs::DataType::Int8 | mvf_fbs::DataType::UInt8 => 1,
            _ => return Err(MvfError::build_error("Unsupported vector data type")),
        };

        let vector_size = self.dimension() as usize * element_size;
        let vector_offset = index as usize * vector_size;

        if vector_offset + vector_size > block_data.len() {
            return Err(MvfError::IndexOutOfBounds {
                index: index as usize,
                len: block_data.len() / vector_size,
            });
        }

        let vector_data = &block_data[vector_offset..vector_offset + vector_size];

        Ok(Vector::new(vector_data, self.dimension(), self.data_type()))
    }

    /// Maps a range of vectors for batch processing.
    ///
    /// Creates a memory-mapped view of multiple consecutive vectors for
    /// efficient batch operations.
    ///
    /// # Arguments
    /// * `start` - Starting vector index
    /// * `count` - Number of vectors to map
    ///
    /// # Errors
    /// Returns an error if the range extends beyond available vectors.
    pub fn map_vector_range(&self, start: u64, count: u64) -> Result<VectorSlice<'a>> {
        if start + count > self.total_vectors() {
            return Err(MvfError::IndexOutOfBounds {
                index: (start + count) as usize,
                len: self.total_vectors() as usize,
            });
        }

        let block_index = self.space.vectors_block_index() as usize;
        let block = self.blocks.get(block_index);

        let data_start_offset = METRO_MAGIC.len();
        let start_offset = data_start_offset + block.offset() as usize;
        let block_data = &self.data[start_offset..start_offset + block.size() as usize];

        let elem_size = match self.data_type() {
            mvf_fbs::DataType::Float32 => 4,
            mvf_fbs::DataType::Float16 => 2,
            mvf_fbs::DataType::Int8 | mvf_fbs::DataType::UInt8 => 1,
            _ => return Err(MvfError::build_error("Unsupported vector data type")),
        };

        let vector_size = self.dimension() as usize * elem_size;
        let range_offset = start as usize * vector_size;
        let range_size = count as usize * vector_size;

        if range_offset + range_size > block_data.len() {
            return Err(MvfError::corrupted_data("Vector range out of bounds"));
        }

        let range_data = &block_data[range_offset..range_offset + range_size];

        VectorSlice::new(range_data, vector_size, count as usize, self.data_type())
    }

    /// Creates a clone for concurrent access.
    ///
    /// The clone shares the same memory-mapped data, making it safe and
    /// efficient for concurrent read operations across threads.
    pub fn clone_concurrent(&self) -> VectorSpace<'a> {
        VectorSpace {
            space: self.space,
            data: self.data,
            blocks: self.blocks,
            index: self.index,
        }
    }

    /// Gets multiple vectors efficiently using a pre-computed access pattern.
    ///
    /// # Arguments
    /// * `pattern` - Optimized access pattern from `prepare_access_pattern`
    ///
    /// # Errors
    /// Returns an error if any vector access fails.
    pub fn get_vectors_with_pattern(&self, pattern: &AccessPattern) -> Result<Vec<Vector<'a>>> {
        let mut vectors = Vec::with_capacity(pattern.indices().len());

        for &(start_idx, end_idx) in pattern.block_ranges() {
            let indices = &pattern.indices()[start_idx..end_idx];

            for &index in indices {
                vectors.push(self.get_vector(index)?);
            }
        }
        Ok(vectors)
    }

    /// Gets multiple vectors by index with automatic optimization.
    ///
    /// # Arguments
    /// * `indices` - Vector indices to retrieve
    ///
    /// # Errors
    /// Returns an error if any vector access fails.
    pub fn get_vectors_batch(&self, indices: &[u64]) -> Result<Vec<Vector<'a>>> {
        let mut vectors = Vec::with_capacity(indices.len());

        // Pre-sort indices for better cache locality
        let pattern = AccessPattern::new(indices.to_vec());

        for &index in pattern.indices() {
            vectors.push(self.get_vector(index)?);
        }

        Ok(vectors)
    }

    /// Creates a streaming iterator over vectors.
    ///
    /// Provides memory-efficient access to large vector collections by
    /// processing them in chunks.
    ///
    /// # Arguments
    /// * `start` - Starting vector index
    /// * `chunk_size` - Number of vectors per chunk
    pub fn stream_vectors(&self, start: u64, chunk_size: usize) -> VectorChunkIterator {
        VectorChunkIterator::new(self, start, chunk_size, self.total_vectors())
    }

    /// Prepares an optimized access pattern for batch operations.
    ///
    /// Pre-processes indices to minimize cache misses and optimize memory
    /// access patterns for subsequent batch operations.
    ///
    /// # Arguments
    /// * `indices` - Vector indices that will be accessed
    pub fn prepare_access_pattern(&self, indices: &[u64]) -> AccessPattern {
        AccessPattern::new(indices.to_vec())
    }

    /// Gets a slice view of a specific dimension across multiple vectors.
    ///
    /// Efficient for operations that only need one component from many vectors.
    ///
    /// # Arguments
    /// * `dim` - Dimension index (0-based)
    /// * `start` - Starting vector index
    /// * `count` - Number of vectors
    ///
    /// # Errors
    /// Returns an error if:
    /// - Dimension index is out of bounds
    /// - Vector range is invalid
    pub fn get_dimension_slice(&self, dim: u32, start: u64, count: u64) -> Result<DimensionSlice> {
        if dim >= self.dimension() {
            return Err(MvfError::IndexOutOfBounds {
                index: dim as usize,
                len: self.dimension() as usize,
            });
        }

        if start + count > self.total_vectors() {
            return Err(MvfError::IndexOutOfBounds {
                index: (start + count) as usize,
                len: self.total_vectors() as usize,
            });
        }

        let block_idx = self.space.vectors_block_index() as usize;
        let block = self.blocks.get(block_idx);

        let data_start_offset = METRO_MAGIC.len();
        let start_offset = data_start_offset + block.offset() as usize;
        let block_data = &self.data[start_offset..start_offset + block.size() as usize];

        let elem_size = match self.data_type() {
            mvf_fbs::DataType::Float32 => 4,
            mvf_fbs::DataType::Float16 => 2,
            mvf_fbs::DataType::Int8 | mvf_fbs::DataType::UInt8 => 1,
            _ => return Err(MvfError::build_error("Unsupported vector data type")),
        };

        Ok(DimensionSlice::new(
            block_data,
            dim,
            start,
            count,
            self.dimension(),
            elem_size,
            self.data_type(),
        ))
    }
}

/// Iterator over vectors in a vector space.
pub struct VectorIterator<'a> {
    space: &'a VectorSpace<'a>,
    current: u64,
    total: u64,
}

impl<'a> Iterator for VectorIterator<'a> {
    type Item = Result<Vector<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }

        let result = self.space.get_vector(self.current);
        self.current += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.total - self.current) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for VectorIterator<'a> {}

#[cfg(test)]
mod tests {
    use crate::{
        errors::MvfError,
        reader::MvfReader,
        tests::{TestContext, create_test_mvf},
    };

    #[test]
    fn test_batch_vector_access_empty() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let vectors = space.get_vectors_batch(&[]).unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn test_batch_vector_access_single() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let vectors = space.get_vectors_batch(&[1]).unwrap();
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].dimension(), 4);
    }

    #[test]
    fn test_batch_vector_access_out_of_bounds() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let result = space.get_vectors_batch(&[0, 1, 999]);
        assert!(matches!(result, Err(MvfError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_batch_vector_access_duplicates() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let indices = vec![0, 2, 1, 2, 0];
        let vectors = space.get_vectors_batch(&indices).unwrap();

        // Should return 3 unique vectors (deduplicated)
        assert_eq!(vectors.len(), 3);
    }

    #[test]
    fn test_vector_streaming_empty_range() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let total_vectors = space.total_vectors();
        let mut stream = space.stream_vectors(total_vectors, 10);

        assert!(stream.next().is_none());
    }

    #[test]
    fn test_vector_streaming_single_chunk() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let mut chunks = 0;
        let mut total_vectors = 0;

        for chunk_result in space.stream_vectors(0, 100) {
            let chunk = chunk_result.unwrap();
            chunks += 1;
            total_vectors += chunk.len();
        }

        assert_eq!(chunks, 1);
        assert_eq!(total_vectors, space.total_vectors() as usize);
    }

    #[test]
    fn test_vector_streaming_multiple_chunks() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let mut chunks = 0;
        let mut total_vectors = 0;

        for chunk_result in space.stream_vectors(0, 1) {
            let chunk = chunk_result.unwrap();
            chunks += 1;
            total_vectors += chunk.len();
        }

        assert_eq!(chunks, space.total_vectors() as usize);
        assert_eq!(total_vectors, space.total_vectors() as usize);
    }

    #[test]
    fn test_vector_streaming_partial_range() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let mut total_vectors = 0;
        for chunk_result in space.stream_vectors(1, 1) {
            let chunk = chunk_result.unwrap();
            total_vectors += chunk.len();
        }

        assert_eq!(total_vectors, (space.total_vectors() - 1) as usize);
    }

    #[test]
    fn test_dimension_slice_valid() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let slice = space.get_dimension_slice(0, 0, 2).unwrap();

        let value0 = slice.get_value(0).unwrap();
        let value1 = slice.get_value(1).unwrap();

        assert!(!value0.is_nan());
        assert!(!value1.is_nan());
    }

    #[test]
    fn test_dimension_slice_out_of_bounds_dimension() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let result = space.get_dimension_slice(10, 0, 1);
        assert!(matches!(result, Err(MvfError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_dimension_slice_out_of_bounds_vectors() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let total = space.total_vectors();
        let result = space.get_dimension_slice(0, 0, total + 1);
        assert!(matches!(result, Err(MvfError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_dimension_slice_iterator() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let slice = space.get_dimension_slice(1, 0, 3).unwrap();
        let values: Vec<_> = slice.iter_values().collect();

        assert_eq!(values.len(), 3);
        assert!(values.iter().all(|v| v.is_ok()));
    }

    #[test]
    fn test_dimension_slice_iterator_empty() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let slice = space.get_dimension_slice(0, 0, 0).unwrap();
        let mut iter = slice.iter_values();

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dimension_value_out_of_bounds() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        let slice = space.get_dimension_slice(0, 0, 2).unwrap();
        let result = slice.get_value(5);
        assert!(matches!(result, Err(MvfError::IndexOutOfBounds { .. })));
    }
}
