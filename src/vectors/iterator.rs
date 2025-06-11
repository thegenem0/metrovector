use crate::errors::Result;

use super::{vector::Vector, vector_space::VectorSpace};

/// Iterator that yields chunks of vectors for memory-efficient processing.
///
/// Useful for processing large vector collections without loading everything
/// into memory at once.
///
/// # Examples
///
/// ```no_run
/// # use metrovector::{reader::MvfReader, errors::Result};
/// # fn example() -> Result<()> {
/// let reader = MvfReader::open("vectors.mvf")?;
/// let space = reader.vector_space("embeddings")?;
///
/// // Process vectors in chunks of 1000
/// for chunk_result in space.stream_vectors(0, 1000) {
///     let chunk = chunk_result?;
///     println!("Processing {} vectors", chunk.len());
///     
///     for vector in chunk {
///         // Process individual vector
///         let values = vector.as_f32()?;
///         // ... do something with values
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct VectorChunkIterator<'a> {
    space: &'a VectorSpace<'a>,
    current_index: u64,
    chunk_size: usize,
    end_index: u64,
}

impl<'a> VectorChunkIterator<'a> {
    /// Creates a new vector chunk iterator.
    ///
    /// # Arguments
    /// * `space` - Vector space to iterate over
    /// * `start_index` - Starting vector index
    /// * `chunk_size` - Number of vectors per chunk
    /// * `end_index` - Ending vector index (exclusive)
    pub fn new(
        space: &'a VectorSpace<'a>,
        start_index: u64,
        chunk_size: usize,
        end_index: u64,
    ) -> Self {
        Self {
            space,
            current_index: start_index,
            chunk_size,
            end_index,
        }
    }
}

impl<'a> Iterator for VectorChunkIterator<'a> {
    type Item = Result<Vec<Vector<'a>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.end_index {
            return None;
        }

        let chunk_end = (self.current_index + self.chunk_size as u64).min(self.end_index);
        let mut chunk = Vec::with_capacity((chunk_end - self.current_index) as usize);

        for i in self.current_index..chunk_end {
            match self.space.get_vector(i) {
                Ok(vector) => chunk.push(vector),
                Err(err) => return Some(Err(err)),
            }
        }
        self.current_index = chunk_end;
        Some(Ok(chunk))
    }
}
