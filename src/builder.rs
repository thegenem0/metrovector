use std::{collections::HashMap, path::Path};

use flatbuffers::FlatBufferBuilder;

use crate::{
    METRO_MAGIC,
    errors::{MvfError, Result},
    io::MvfWriter,
    mvf_fbs::{
        CompressionAlgorithm, DataBlock, DataType, DistanceMetric, FileFooter, FileFooterArgs,
        FlatIndex, FlatIndexArgs, HNSWIndex, HNSWIndexArgs, IVFIndex, IVFIndexArgs, Index,
        MetadataColumn, MetadataColumnArgs, VectorSpace, VectorSpaceArgs, VectorType,
    },
};

/// Builder for constructing MVF files.
///
/// Provides a fluent API for creating multi-dimensional vector files with
/// support for multiple vector spaces, different data types, and indexing strategies.
///
/// # Examples
///
/// ```no_run
/// use metrovector::{builder::MvfBuilder, mvf_fbs::{VectorType, DistanceMetric, DataType},
/// errors::MvfError};
///
/// let mut builder = MvfBuilder::new();
///
/// builder.add_vector_space(
///     "embeddings",
///     512,
///     VectorType::Dense,
///     DistanceMetric::Cosine,
///     DataType::Float32,
/// );
///
/// let vectors = vec![vec![1.0; 512], vec![2.0; 512]];
/// builder.add_vectors("embeddings", &vectors)?;
///
/// let built = builder.build();
/// built.save("embeddings.mvf")?;
/// # Ok::<(), MvfError>(())
/// ```
pub struct MvfBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    vector_spaces: Vec<VectorSpaceBuilder>,
    metadata_columns: Vec<MetadataColumnBuilder>,
    data_blocks: Vec<Vec<u8>>,
    string_heap: Vec<u8>,
    string_offsets: HashMap<String, u32>,
}

/// Internal builder for vector spaces.
struct VectorSpaceBuilder {
    name: String,
    dimension: u32,
    vector_type: VectorType,
    distance_metric: DistanceMetric,
    data_type: DataType,
    vectors: Vec<u8>,
    vector_ids: Option<Vec<u64>>,
    index_config: Option<IndexConfig>,
    tombstones: Option<Vec<u64>>,
}

/// Internal builder for metadata columns.
struct MetadataColumnBuilder {
    name: String,
    data_type: DataType,
    data: Vec<u8>,
    null_count: u64,
    min_value: Option<Vec<u8>>,
    max_value: Option<Vec<u8>>,
}

/// Index configuration options.
#[derive(Debug)]
enum IndexConfig {
    /// Flat/brute-force index.
    Flat,
    /// Inverted File index.
    Ivf { num_lists: u32, centroids: Vec<u8> },
    /// Hierarchical Navigable Small World index.
    Hnsw {
        entry_point: u64,
        max_connections: u32,
        graph: Vec<u8>,
    },
}

impl<'a> MvfBuilder<'a> {
    /// Creates a new MVF builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the MVF format version this builder creates.
    pub fn version(&self) -> u16 {
        1
    }

    /// Adds a new vector space to the file.
    ///
    /// # Arguments
    /// * `name` - Unique name for the vector space
    /// * `dimension` - Number of dimensions per vector
    /// * `vector_type` - Type of vectors (Dense, Sparse, etc.)
    /// * `distance_metric` - Distance metric for similarity calculations
    /// * `data_type` - Storage format for vector components
    ///
    /// # Returns
    /// The index of the created vector space.
    pub fn add_vector_space(
        &mut self,
        name: &str,
        dimension: u32,
        vector_type: VectorType,
        distance_metric: DistanceMetric,
        data_type: DataType,
    ) -> usize {
        let space = VectorSpaceBuilder {
            name: name.to_string(),
            dimension,
            vector_type,
            distance_metric,
            data_type,
            vectors: Vec::new(),
            vector_ids: None,
            index_config: None,
            tombstones: None,
        };

        self.vector_spaces.push(space);
        self.vector_spaces.len() - 1
    }

    /// Adds vectors to a vector space.
    ///
    /// # Type Parameters
    /// * `T` - Numeric type that can be converted to f32
    ///
    /// # Arguments
    /// * `space_name` - Name of the target vector space
    /// * `vectors` - Slice of vectors to add
    ///
    /// # Errors
    /// Returns [`MvfError`]
    /// - If the vector space doesn't exist
    /// - If vector dimensions don't match the space's dimension
    /// - If data type conversion fails
    pub fn add_vectors<T>(&mut self, space_name: &str, vectors: &[Vec<T>]) -> Result<()>
    where
        T: Copy + Into<f32>,
    {
        let space = self
            .vector_spaces
            .iter_mut()
            .find(|s| s.name == space_name)
            .ok_or_else(|| crate::errors::MvfError::VectorSpaceNotFound(space_name.to_string()))?;

        if vectors.is_empty() {
            return Ok(());
        }

        let dimension = vectors[0].len() as u32;
        if space.dimension == 0 {
            space.dimension = dimension;
        } else if space.dimension != dimension {
            return Err(crate::errors::MvfError::DimensionMismatch {
                expected: space.dimension as usize,
                actual: dimension as usize,
            });
        }

        match space.data_type {
            DataType::Float32 => {
                for v in vectors {
                    for &value in v {
                        let value: f32 = value.into();
                        space.vectors.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
            DataType::Float16 => {
                for v in vectors {
                    for &value in v {
                        let f16_bytes = half::f16::from_f32(value.into()).to_le_bytes();
                        space.vectors.extend_from_slice(&f16_bytes);
                    }
                }
            }
            _ => return Err(MvfError::build_error("Unsupported data type for vectors")),
        }

        Ok(())
    }

    /// Adds a metadata column.
    ///
    /// # Type Parameters
    /// * `T` - Type that can be converted to Vec<u8>
    ///
    /// # Arguments
    /// * `name` - Column name
    /// * `data_type` - Data type of the column
    /// * `values` - Column values
    ///
    /// # Errors
    /// Returns [`MvfError`]
    /// - If data conversion fails.
    pub fn add_metadata_column<T>(
        &mut self,
        name: &str,
        data_type: DataType,
        values: &[T],
    ) -> Result<()>
    where
        T: Copy + Into<Vec<u8>>,
    {
        let mut column = MetadataColumnBuilder {
            name: name.to_string(),
            data_type,
            data: Vec::new(),
            null_count: 0,
            min_value: None,
            max_value: None,
        };

        for value in values {
            let bytes: Vec<u8> = (*value).into();
            column.data.extend_from_slice(&bytes);
        }

        self.metadata_columns.push(column);
        Ok(())
    }

    /// Builds the final MVF structure.
    ///
    /// Consumes the builder and returns a `BuiltMvf` ready for serialization.
    pub fn build(self) -> BuiltMvf {
        let mut data_blocks = Vec::new();
        let mut current_offset = 0u64;

        for space in &self.vector_spaces {
            let vector_block = DataBlock::new(
                current_offset,
                space.vectors.len() as u64,
                CompressionAlgorithm::None,
                0,
                crc32fast::hash(&space.vectors),
            );

            data_blocks.push((vector_block, space.vectors.clone()));
            current_offset += space.vectors.len() as u64;

            if let Some(ref ids) = space.vector_ids {
                let ids_bytes = ids
                    .iter()
                    .flat_map(|&id| id.to_le_bytes())
                    .collect::<Vec<u8>>();

                let ids_block = DataBlock::new(
                    current_offset,
                    ids_bytes.len() as u64,
                    CompressionAlgorithm::None,
                    0,
                    crc32fast::hash(&ids_bytes),
                );
                data_blocks.push((ids_block, ids_bytes.clone()));
                current_offset += ids_bytes.len() as u64;
            }
        }

        for column in &self.metadata_columns {
            let column_block = DataBlock::new(
                current_offset,
                column.data.len() as u64,
                CompressionAlgorithm::None,
                0,
                crc32fast::hash(&column.data),
            );
            data_blocks.push((column_block, column.data.clone()));
            current_offset += column.data.len() as u64;
        }

        let string_heap_block_index = if !self.string_heap.is_empty() {
            let heap_block = DataBlock::new(
                current_offset,
                self.string_heap.len() as u64,
                CompressionAlgorithm::None,
                0,
                crc32fast::hash(&self.string_heap),
            );

            data_blocks.push((heap_block, self.string_heap.clone()));
            Some(data_blocks.len() - 1)
        } else {
            None
        };

        BuiltMvf {
            data_blocks,
            vector_spaces: self.vector_spaces,
            metadata_columns: self.metadata_columns,
            string_heap_block_index,
        }
    }

    /// Gets a mutable reference to a vector space.
    fn get_vector_space_mut(&mut self, index: usize) -> Option<&mut VectorSpaceBuilder> {
        self.vector_spaces.get_mut(index)
    }

    /// Adds a string to the string heap and returns its offset.
    fn add_string(&mut self, s: &str) -> u32 {
        if let Some(offset) = self.string_offsets.get(s) {
            return *offset;
        }

        let offset = self.string_heap.len() as u32;
        self.string_heap.extend_from_slice(s.as_bytes());
        self.string_heap.push(0); // Push null terminator
        self.string_offsets.insert(s.to_string(), offset);
        offset
    }
}

/// Reference to a vector space being built.
///
/// Provides a fluent API for configuring vector space properties.
pub struct VectorSpaceBuilderRef<'a, 'b> {
    builder: &'a mut MvfBuilder<'b>,
    index: usize,
}

impl<'a, 'b> VectorSpaceBuilderRef<'a, 'b> {
    /// Sets the vector dimension.
    pub fn dimension(self, dim: u32) -> Self {
        self.builder.vector_spaces[self.index].dimension = dim;
        self
    }

    /// Sets the vector type.
    pub fn vector_type(self, vt: VectorType) -> Self {
        self.builder.vector_spaces[self.index].vector_type = vt;
        self
    }

    /// Sets the distance metric.
    pub fn distance_metric(self, dm: DistanceMetric) -> Self {
        self.builder.vector_spaces[self.index].distance_metric = dm;
        self
    }

    /// Sets the data type.
    pub fn data_type(self, dt: DataType) -> Self {
        self.builder.vector_spaces[self.index].data_type = dt;
        self
    }

    /// Configures a flat index for this vector space.
    pub fn with_flat_index(self) -> Self {
        self.builder.vector_spaces[self.index].index_config = Some(IndexConfig::Flat);
        self
    }

    /// Configures an IVF index for this vector space.
    pub fn with_ivf_index(self, num_lists: u32, centroids: Vec<u8>) -> Self {
        self.builder.vector_spaces[self.index]
            .index_config
            .replace(IndexConfig::Ivf {
                num_lists,
                centroids,
            });
        self
    }

    /// Configures an HNSW index for this vector space.
    pub fn with_hnsw_index(self, entry_point: u64, max_connections: u32, graph: Vec<u8>) -> Self {
        self.builder.vector_spaces[self.index]
            .index_config
            .replace(IndexConfig::Hnsw {
                entry_point,
                max_connections,
                graph,
            });
        self
    }
}

/// A built MVF file ready to be written to disk.
///
/// Contains all the data and metadata needed to create a valid MVF file.
pub struct BuiltMvf {
    data_blocks: Vec<(DataBlock, Vec<u8>)>, // data block, string heap
    vector_spaces: Vec<VectorSpaceBuilder>,
    metadata_columns: Vec<MetadataColumnBuilder>,
    string_heap_block_index: Option<usize>,
}

impl BuiltMvf {
    /// Saves the MVF to a file.
    ///
    /// # Errors
    /// Returns [`MvfError`]
    /// - If file creation or writing fails.
    pub fn save<P: AsRef<Path>>(self, path: P) -> Result<()> {
        let writer = MvfWriter::create(path)?;
        writer.write(self)
    }

    /// Serializes the MVF to bytes.
    ///
    /// # Errors
    /// Returns [`MvfError`]
    pub fn to_bytes(self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        // First 4 bytes are `MVF1`
        bytes.extend_from_slice(&METRO_MAGIC);

        for (_, block_data) in &self.data_blocks {
            bytes.extend_from_slice(block_data);
        }

        let mut builder = FlatBufferBuilder::with_capacity(1024);

        let mut fb_vector_spaces = Vec::new();
        for (i, space) in self.vector_spaces.iter().enumerate() {
            let name = builder.create_string(&space.name);

            let (index_type, index_type_type) = match &space.index_config {
                Some(IndexConfig::Ivf {
                    num_lists,
                    centroids,
                }) => {
                    let centroids_offset = builder.create_vector(centroids);
                    let ivf_index = IVFIndex::create(
                        &mut builder,
                        &IVFIndexArgs {
                            num_lists: *num_lists,
                            centroids_block_index: centroids_offset.value(),
                            lists_block_index: 0,
                        },
                    );
                    (Some(ivf_index.as_union_value()), Index::IVFIndex)
                }
                Some(IndexConfig::Hnsw {
                    entry_point,
                    max_connections,
                    ..
                }) => {
                    let hnsw_index = HNSWIndex::create(
                        &mut builder,
                        &HNSWIndexArgs {
                            entry_point: *entry_point,
                            max_connections: *max_connections,
                            graph_block_index: 0, // TODO(thegeneme0): Implement this
                        },
                    );
                    (Some(hnsw_index.as_union_value()), Index::HNSWIndex)
                }

                _ => {
                    let flat_index = FlatIndex::create(&mut builder, &FlatIndexArgs {});
                    (Some(flat_index.as_union_value()), Index::FlatIndex)
                }
            };

            let vector_space = VectorSpace::create(
                &mut builder,
                &VectorSpaceArgs {
                    name: Some(name),
                    dimension: space.dimension,
                    total_vectors: (space.vectors.len() / (space.dimension as usize * 4)) as u64,
                    vector_type: space.vector_type,
                    distance_metric: space.distance_metric,
                    data_type: space.data_type,
                    vectors_block_index: i as u32,
                    index_type,
                    index_type_type,
                    vector_ids_block_index: 0,
                    sparse_metadata: None,
                    tombstones: None,
                },
            );

            fb_vector_spaces.push(vector_space);
        }

        let vector_spaces = builder.create_vector(&fb_vector_spaces);

        let mut fb_data_blocks = Vec::new();
        for (block_data, _) in &self.data_blocks {
            fb_data_blocks.push(*block_data);
        }

        let data_blocks = builder.create_vector(&fb_data_blocks);

        let mut fb_metadata_columns = Vec::new();
        for (i, column) in self.metadata_columns.iter().enumerate() {
            let name = builder.create_string(&column.name);
            let min_val = column.min_value.as_ref().map(|v| builder.create_vector(v));
            let max_val = column.max_value.as_ref().map(|v| builder.create_vector(v));

            let metadata_column = MetadataColumn::create(
                &mut builder,
                &MetadataColumnArgs {
                    name: Some(name),
                    data_type: column.data_type,
                    data_block_index: (self.vector_spaces.len() + i) as u32,
                    null_count: column.null_count,
                    min_value: min_val,
                    max_value: max_val,
                },
            );

            fb_metadata_columns.push(metadata_column);
        }

        let metadata_columns = if !fb_metadata_columns.is_empty() {
            Some(builder.create_vector(&fb_metadata_columns))
        } else {
            None
        };

        let file_footer = FileFooter::create(
            &mut builder,
            &FileFooterArgs {
                format_version: 1,
                vector_spaces: Some(vector_spaces),
                block_manifest: Some(data_blocks),
                metadata_columns,
                string_heap_block_index: self
                    .string_heap_block_index
                    .map(|i| i as u32)
                    .unwrap_or(0),
                extensions: None,
                compatibility_version: 1,
                deprecated_fields: None,
            },
        );

        builder.finish_minimal(file_footer);

        let footer_data = builder.finished_data();
        bytes.extend_from_slice(footer_data);

        // Last N-8..N-4 bytes are the footer size
        let footer_len = footer_data.len() as u32;
        bytes.extend_from_slice(&footer_len.to_le_bytes());

        // Last 4 bytes are `MVF1`
        bytes.extend_from_slice(&METRO_MAGIC);

        Ok(bytes)
    }
}

impl Default for MvfBuilder<'_> {
    fn default() -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(1024),
            vector_spaces: Vec::new(),
            metadata_columns: Vec::new(),
            data_blocks: Vec::new(),
            string_heap: Vec::new(),
            string_offsets: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{I32Bytes, StringBytes, TestContext, create_test_mvf, create_test_vectors};

    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = MvfBuilder::new();
        assert_eq!(builder.version(), 1);
        assert_eq!(builder.vector_spaces.len(), 0);
        assert_eq!(builder.metadata_columns.len(), 0);
    }

    #[test]
    fn test_add_vector_space() {
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "test_space",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        assert_eq!(idx, 0);
        assert_eq!(builder.vector_spaces.len(), 1);
        assert_eq!(builder.vector_spaces[0].name, "test_space");
    }

    #[test]
    fn test_add_vectors_success() {
        let mut builder = MvfBuilder::new();
        let _idx = builder.add_vector_space(
            "test_space",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        let vectors = create_test_vectors();
        let result = builder.add_vectors("test_space", &vectors);

        assert!(result.is_ok());
        assert_eq!(builder.vector_spaces[0].dimension, 4);
        assert!(!builder.vector_spaces[0].vectors.is_empty());
    }

    #[test]
    fn test_add_vectors_nonexistent_space() {
        let mut builder = MvfBuilder::new();
        let vectors = create_test_vectors();

        let result = builder.add_vectors("nonexistent", &vectors);

        assert!(matches!(result, Err(MvfError::VectorSpaceNotFound(_))));
    }

    #[test]
    fn test_add_vectors_dimension_mismatch() {
        let mut builder = MvfBuilder::new();
        let _idx = builder.add_vector_space(
            "test_space",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        // Add vectors with dimension 4
        let vectors1 = vec![vec![1.0, 2.0, 3.0, 4.0]];
        builder.add_vectors("test_space", &vectors1).unwrap();

        // Try to add vectors with dimension 3
        let vectors2 = vec![vec![1.0, 2.0, 3.0]];
        let result = builder.add_vectors("test_space", &vectors2);

        assert!(matches!(result, Err(MvfError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_add_vectors_different_data_types() {
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "test_space",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        // Test Float32
        builder.vector_spaces[idx].data_type = DataType::Float32;
        let vectors = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        assert!(builder.add_vectors("test_space", &vectors).is_ok());

        // Test Float16
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "test_space",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );
        builder.vector_spaces[idx].data_type = DataType::Float16;
        let vectors = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        assert!(builder.add_vectors("test_space", &vectors).is_ok());
    }

    #[test]
    fn test_add_metadata_column() {
        let mut builder = MvfBuilder::new();
        let values: Vec<i32> = vec![1, 2, 3, 4, 5];
        let byte_values: Vec<I32Bytes> = values.iter().map(|&v| I32Bytes(v)).collect();

        let result = builder.add_metadata_column("test_column", DataType::UInt32, &byte_values);

        assert!(result.is_ok());
        assert_eq!(builder.metadata_columns.len(), 1);
        assert_eq!(builder.metadata_columns[0].name, "test_column");
    }

    #[test]
    fn test_build_mvf() {
        let mut builder = MvfBuilder::new();
        let _idx = builder.add_vector_space(
            "test_space",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        let vectors = create_test_vectors();
        builder.add_vectors("test_space", &vectors).unwrap();

        let built = builder.build();
        assert!(!built.data_blocks.is_empty());
        assert_eq!(built.vector_spaces.len(), 1);
    }

    #[test]
    fn test_built_mvf_to_bytes() {
        let built = create_test_mvf();

        let result = built.to_bytes();
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());

        assert!(bytes.len() > 100);
    }

    #[test]
    fn test_built_mvf_save() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        let result = built.save(&path);
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_vector_space_builder_ref() {
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "test_space",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        // Test method chaining
        let space = &mut builder.vector_spaces[idx];
        space.dimension = 128;
        space.vector_type = VectorType::Dense;
        space.distance_metric = DistanceMetric::Cosine;
        space.data_type = DataType::Float32;

        assert_eq!(space.dimension, 128);
        assert_eq!(space.vector_type, VectorType::Dense);
        assert_eq!(space.distance_metric, DistanceMetric::Cosine);
        assert_eq!(space.data_type, DataType::Float32);
    }

    #[test]
    fn test_add_metadata_column_integers() {
        let mut builder = MvfBuilder::new();

        // Test with integer metadata
        let values: Vec<i32> = vec![1, 2, 3, 4, 5];
        let byte_values: Vec<I32Bytes> = values.iter().map(|&v| I32Bytes(v)).collect();

        let result = builder.add_metadata_column("ids", DataType::UInt32, &byte_values);
        assert!(result.is_ok());
        assert_eq!(builder.metadata_columns.len(), 1);
        assert_eq!(builder.metadata_columns[0].name, "ids");
        assert_eq!(builder.metadata_columns[0].data_type, DataType::UInt32);
    }

    #[test]
    fn test_add_metadata_column_strings() {
        let mut builder = MvfBuilder::new();

        // Test with string metadata
        let strings = ["hello", "world", "test"];
        let byte_values: Vec<StringBytes> = strings.iter().map(|s| StringBytes(s)).collect();

        let result = builder.add_metadata_column("labels", DataType::StringRef, &byte_values);
        assert!(result.is_ok());
        assert_eq!(builder.metadata_columns.len(), 1);
    }

    #[test]
    fn test_add_metadata_column_empty() {
        let mut builder = MvfBuilder::new();
        let empty_values: Vec<StringBytes> = vec![];

        let result = builder.add_metadata_column("empty", DataType::StringRef, &empty_values);
        assert!(result.is_ok());
        assert_eq!(builder.metadata_columns[0].data.len(), 0);
    }

    #[test]
    fn test_vector_space_builder_ref_methods() {
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "test_space",
            0, // Will be modified
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        // Test all VectorSpaceBuilderRef methods
        let space = &mut builder.vector_spaces[idx];

        // Test dimension setting
        space.dimension = 128;
        assert_eq!(space.dimension, 128);

        // Test vector type setting
        space.vector_type = VectorType::Sparse;
        assert_eq!(space.vector_type, VectorType::Sparse);

        // Test distance metric setting
        space.distance_metric = DistanceMetric::Cosine;
        assert_eq!(space.distance_metric, DistanceMetric::Cosine);

        // Test data type setting
        space.data_type = DataType::Float16;
        assert_eq!(space.data_type, DataType::Float16);
    }

    #[test]
    fn test_vector_space_index_configurations() {
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "test_space",
            128,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        let space = &mut builder.vector_spaces[idx];

        // Test flat index
        space.index_config = Some(IndexConfig::Flat);
        assert!(matches!(space.index_config, Some(IndexConfig::Flat)));

        // Test IVF index
        let centroids = vec![1u8, 2, 3, 4];
        space.index_config = Some(IndexConfig::Ivf {
            num_lists: 100,
            centroids: centroids.clone(),
        });
        if let Some(IndexConfig::Ivf {
            num_lists,
            centroids: c,
        }) = &space.index_config
        {
            assert_eq!(*num_lists, 100);
            assert_eq!(*c, centroids);
        } else {
            panic!("Expected IVF index config");
        }

        // Test HNSW index
        let graph = vec![5u8, 6, 7, 8];
        space.index_config = Some(IndexConfig::Hnsw {
            entry_point: 42,
            max_connections: 16,
            graph: graph.clone(),
        });
        if let Some(IndexConfig::Hnsw {
            entry_point,
            max_connections,
            graph: g,
        }) = &space.index_config
        {
            assert_eq!(*entry_point, 42);
            assert_eq!(*max_connections, 16);
            assert_eq!(*g, graph);
        } else {
            panic!("Expected HNSW index config");
        }
    }

    #[test]
    fn test_string_heap_management() {
        let mut builder = MvfBuilder::new();

        // Test adding strings to heap
        let offset1 = builder.add_string("hello");
        let offset2 = builder.add_string("world");
        let offset3 = builder.add_string("hello"); // Duplicate

        assert_eq!(offset1, 0);
        assert_eq!(offset2, 6); // "hello\0" = 6 bytes
        assert_eq!(offset3, 0); // Should reuse existing offset

        // Verify string heap content
        let expected = b"hello\0world\0";
        assert_eq!(builder.string_heap, expected);

        // Verify string offsets map
        assert_eq!(builder.string_offsets.len(), 2);
        assert_eq!(builder.string_offsets["hello"], 0);
        assert_eq!(builder.string_offsets["world"], 6);
    }

    #[test]
    fn test_get_vector_space_mut() {
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "test",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        // Test valid index
        let space = builder.get_vector_space_mut(idx);
        assert!(space.is_some());
        assert_eq!(space.unwrap().name, "test");

        // Test invalid index
        let invalid_space = builder.get_vector_space_mut(999);
        assert!(invalid_space.is_none());
    }

    #[test]
    fn test_build_with_metadata_and_string_heap() {
        let mut builder = MvfBuilder::new();

        // Add vector space
        builder.add_vector_space(
            "embeddings",
            4,
            VectorType::Dense,
            DistanceMetric::Cosine,
            DataType::Float32,
        );

        // Add vectors
        let vectors = create_test_vectors();
        builder.add_vectors("embeddings", &vectors).unwrap();

        // Add metadata
        let ids: Vec<i32> = vec![1, 2, 3];
        let id_bytes: Vec<I32Bytes> = ids.iter().map(|&v| I32Bytes(v)).collect();
        builder
            .add_metadata_column("ids", DataType::UInt32, &id_bytes)
            .unwrap();

        let labels = ["cat", "dog", "bird"];
        let label_bytes: Vec<StringBytes> = labels.iter().map(|s| StringBytes(s)).collect();
        builder
            .add_metadata_column("labels", DataType::StringRef, &label_bytes)
            .unwrap();

        // Add some strings to heap
        builder.add_string("test_string_1");
        builder.add_string("test_string_2");

        let built = builder.build();

        // Verify structure
        assert_eq!(built.vector_spaces.len(), 1);
        assert_eq!(built.metadata_columns.len(), 2);
        assert!(built.string_heap_block_index.is_some());
        assert!(built.data_blocks.len() > 2); // Vector data + metadata columns + string heap
    }

    #[test]
    fn test_add_vectors_empty_space_auto_dimension() {
        let mut builder = MvfBuilder::new();
        let _idx = builder.add_vector_space(
            "auto_dim",
            0, // Auto-dimension
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        let vectors = vec![vec![1.0, 2.0, 3.0]]; // 3D vector
        let result = builder.add_vectors("auto_dim", &vectors);

        assert!(result.is_ok());
        assert_eq!(builder.vector_spaces[0].dimension, 3);
    }

    #[test]
    fn test_add_vectors_unsupported_data_type() {
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "test",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );

        // Force unsupported data type
        builder.vector_spaces[idx].data_type = DataType::StringRef;

        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let result = builder.add_vectors("test", &vectors);

        assert!(matches!(result, Err(MvfError::Build(_))));
    }

    #[test]
    fn test_built_mvf_with_complex_indexes() {
        let mut builder = MvfBuilder::new();
        let idx = builder.add_vector_space(
            "indexed_space",
            128,
            VectorType::Dense,
            DistanceMetric::Cosine,
            DataType::Float32,
        );

        let centroids = vec![0u8; 512]; // 4 centroids * 128 dimensions * 1 byte
        builder.vector_spaces[idx].index_config = Some(IndexConfig::Ivf {
            num_lists: 4,
            centroids,
        });

        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..128).map(|j| (i * 128 + j) as f32).collect())
            .collect();

        builder.add_vectors("indexed_space", &vectors).unwrap();

        let built = builder.build();
        let bytes = built.to_bytes().unwrap();

        assert!(!bytes.is_empty());
        assert!(bytes.len() > 1000); // Should be substantial with 128D vectors
    }
}
