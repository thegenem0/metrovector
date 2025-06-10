use crate::{
    errors::{MvfError, Result},
    mvf_fbs::{self, DataBlock},
};

pub struct VectorSpace<'a> {
    space: mvf_fbs::VectorSpace<'a>,
    data: &'a [u8],
    blocks: flatbuffers::Vector<'a, DataBlock>,
    index: usize,
}

impl<'a> VectorSpace<'a> {
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

    pub fn name(&self) -> &str {
        self.space.name()
    }

    pub fn dimension(&self) -> u32 {
        self.space.dimension()
    }

    pub fn total_vectors(&self) -> u64 {
        self.space.total_vectors()
    }

    pub fn vector_type(&self) -> mvf_fbs::VectorType {
        self.space.vector_type()
    }

    pub fn distance_metric(&self) -> mvf_fbs::DistanceMetric {
        self.space.distance_metric()
    }

    pub fn data_type(&self) -> mvf_fbs::DataType {
        self.space.data_type()
    }

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
        let start_offset = block.offset() as usize;
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

        Ok(Vector {
            data: vector_data,
            dimension: self.dimension(),
            data_type: self.data_type(),
            _phantom: std::marker::PhantomData,
        })
    }
}

pub struct Vector<'a> {
    data: &'a [u8],
    dimension: u32,
    data_type: mvf_fbs::DataType,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Vector<'a> {
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    pub fn data_type(&self) -> mvf_fbs::DataType {
        self.data_type
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

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
}

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
