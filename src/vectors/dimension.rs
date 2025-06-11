use crate::{
    errors::{MvfError, Result},
    mvf_fbs,
};

/// Efficient access to a specific dimension across multiple vectors.
///
/// Provides optimized access patterns for operations that need only one
/// component from many vectors, such as statistical analysis or filtering.
///
/// # Examples
///
/// ```no_run
/// # use metrovector::{reader::MvfReader, errors::Result};
/// # fn example() -> Result<()> {
/// let reader = MvfReader::open("vectors.mvf")?;
/// let space = reader.vector_space("embeddings")?;
///
/// // Get the first dimension of vectors 0-99
/// let dim_slice = space.get_dimension_slice(0, 0, 100)?;
///
/// // Access individual values
/// let first_value = dim_slice.get_value(0)?;
/// println!("First vector's first dimension: {}", first_value);
///
/// // Or iterate over all values
/// for value in dim_slice.iter_values() {
///     println!("Value: {}", value?);
/// }
/// # Ok(())
/// # }
/// ```
pub struct DimensionSlice<'a> {
    data: &'a [u8],
    dimension: u32,
    start_vector: u64,
    vector_count: u64,
    vector_dimension: u32,
    element_size: usize,
    data_type: mvf_fbs::DataType,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> DimensionSlice<'a> {
    /// Creates a new dimension slice.
    ///
    /// # Safety
    /// Internal constructor - assumes all parameters are valid and consistent.
    pub(super) fn new(
        data: &'a [u8],
        dimension: u32,
        start_vector: u64,
        vector_count: u64,
        vector_dimension: u32,
        element_size: usize,
        data_type: mvf_fbs::DataType,
    ) -> Self {
        Self {
            data,
            dimension,
            start_vector,
            vector_count,
            vector_dimension,
            element_size,
            data_type,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a> DimensionSlice<'a> {
    /// Gets the value at the specified vector index.
    ///
    /// # Arguments
    /// * `vector_index` - Index of the vector (relative to the slice start)
    ///
    /// # Errors
    /// Returns an error if:
    /// - Vector index is out of bounds
    /// - Data access exceeds slice boundaries
    /// - Data type is unsupported
    pub fn get_value(&self, vector_index: u64) -> Result<f32> {
        if vector_index >= self.vector_count {
            return Err(MvfError::IndexOutOfBounds {
                index: vector_index as usize,
                len: self.vector_count as usize,
            });
        }

        let vector_size = self.vector_dimension as usize * self.element_size;
        let vector_offset = (self.start_vector + vector_index) as usize * vector_size;
        let dim_offset = vector_offset + self.dimension as usize * self.element_size;

        if dim_offset + self.element_size > self.data.len() {
            return Err(MvfError::corrupted_data("Dimension slice out of bounds"));
        }

        let value = match self.data_type {
            mvf_fbs::DataType::Float32 => {
                let bytes = [
                    self.data[dim_offset],
                    self.data[dim_offset + 1],
                    self.data[dim_offset + 2],
                    self.data[dim_offset + 3],
                ];
                f32::from_le_bytes(bytes)
            }
            mvf_fbs::DataType::Float16 => {
                let bytes = [self.data[dim_offset], self.data[dim_offset + 1]];
                half::f16::from_le_bytes(bytes).to_f32()
            }
            _ => {
                return Err(MvfError::build_error(
                    "Unsupported data type for dimension slice",
                ));
            }
        };

        Ok(value)
    }

    /// Creates an iterator over dimension values.
    pub fn iter_values(&self) -> DimensionValueIterator<'_> {
        DimensionValueIterator::new(self)
    }
}

/// Iterator over dimension values.
pub struct DimensionValueIterator<'a> {
    slice: &'a DimensionSlice<'a>,
    current_vector: u64,
}

impl<'a> DimensionValueIterator<'a> {
    /// Creates a new dimension value iterator.
    pub fn new(slice: &'a DimensionSlice<'a>) -> Self {
        Self {
            slice,
            current_vector: 0,
        }
    }
}

impl<'a> Iterator for DimensionValueIterator<'a> {
    type Item = Result<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_vector >= self.slice.vector_count {
            return None;
        }

        let result = self.slice.get_value(self.current_vector);
        self.current_vector += 1;
        Some(result)
    }
}
