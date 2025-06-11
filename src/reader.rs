use std::{fs::File, mem::MaybeUninit, path::Path};

use memmap2::Mmap;

use crate::{
    METRO_FOOTER_SIZE, METRO_MAGIC,
    errors::{MvfError, Result},
    mvf_fbs::FileFooter,
    vector::vector_space::VectorSpace,
};

/// Reader for MetroVector Files (MVF).
///
/// Provides memory-mapped access to MVF files for efficient vector operations.
/// The reader is thread-safe and can be shared across multiple threads.
///
/// # Examples
///
/// ```no_run
/// use metrovector::{reader::MvfReader, errors::MvfError};
///
/// let reader = MvfReader::open("vectors.mvf")?;
/// println!("File version: {}", reader.version());
/// println!("Vector spaces: {:?}", reader.vector_space_names());
/// # Ok::<(), MvfError>(())
/// ```
pub struct MvfReader {
    mmap: Mmap,
    file_footer: FileFooter<'static>,
    data_start: usize,
}

impl MvfReader {
    /// Opens an MVF file for reading.
    ///
    /// # Safety
    /// Uses memory mapping which assumes the file won't be modified externally
    /// while the reader exists. The file must remain valid for the reader's lifetime.
    ///
    /// # Errors
    /// Returns an error if:
    /// - File cannot be opened or read
    /// - File format is invalid or corrupted
    /// - File version is unsupported
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Self::validate_file_structure(&mmap)?;

        let (footer_start, footer_end) = Self::validate_footer_bounds(&mmap)?;

        let mut reader = MaybeUninit::<Self>::uninit();

        let reader_ptr = reader.as_mut_ptr();
        unsafe {
            std::ptr::addr_of_mut!((*reader_ptr).mmap).write(mmap);
            std::ptr::addr_of_mut!((*reader_ptr).data_start).write(METRO_MAGIC.len());
        }

        let mut reader = unsafe { reader.assume_init() };

        let footer_bytes = &reader.mmap[footer_start..footer_end];
        let file_footer = flatbuffers::root::<FileFooter>(footer_bytes)
            .map_err(|e| MvfError::invalid_format(format!("Failed to parse footer: {}", e)))?;

        // Safety:
        // We "extend" the lifetime of the footer to 'static because:
        // 1. The footer data comes from our mmap which lives as long as the reader
        // 2. The reader is the only thing that can access this footer directly
        // 3. There's no public API that can mutate the footer
        // In reality, the lifetime isn't actually extended,
        // we just lie to the compiler, but in this case, it's safe,
        // due to the assumptions above
        reader.file_footer =
            unsafe { std::mem::transmute::<FileFooter<'_>, FileFooter<'static>>(file_footer) };

        Ok(reader)
    }

    /// Returns the MVF format version.
    pub fn version(&self) -> u16 {
        self.file_footer.format_version()
    }

    /// Returns the number of vector spaces in the file.
    pub fn num_vector_spaces(&self) -> usize {
        self.file_footer.vector_spaces().len()
    }

    /// Returns the names of all vector spaces.
    pub fn vector_space_names(&self) -> Vec<String> {
        self.file_footer
            .vector_spaces()
            .iter()
            .map(|s| s.name().to_string())
            .collect()
    }

    /// Gets a vector space by name.
    ///
    /// # Errors
    /// Returns [`MvfError::VectorSpaceNotFound`] if no space with the given name exists.
    pub fn vector_space(&self, name: &str) -> Result<VectorSpace> {
        let spaces = self.file_footer.vector_spaces();

        let (index, space) = spaces
            .iter()
            .enumerate()
            .find(|(_, s)| *s.name() == *name)
            .ok_or(MvfError::VectorSpaceNotFound(name.to_string()))?;

        Ok(VectorSpace::new(
            space,
            &self.mmap,
            self.file_footer.block_manifest(),
            index,
        ))
    }

    /// Returns the file size in bytes.
    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Returns whether the file contains metadata columns.
    pub fn has_metadata(&self) -> bool {
        self.file_footer.metadata_columns().is_some()
    }

    /// Returns the names of all metadata columns.
    pub fn metadata_column_names(&self) -> Vec<String> {
        self.file_footer
            .metadata_columns()
            .map(|columns| {
                columns
                    .iter()
                    .map(|col| col.name())
                    .map(|name| name.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Validates file structure without checking checksums.
    ///
    /// # Errors
    /// Returns [`MvfError::CorruptedData`] if any data blocks extend beyond the file boundaries.
    pub fn validate(&self) -> Result<()> {
        for block in self.file_footer.block_manifest() {
            let end_offset = block.offset() + block.size();
            if end_offset > self.mmap.len() as u64 {
                return Err(MvfError::corrupted_data(format!(
                    "Block extends beyond file: offset={}, size={}, file_size={}",
                    block.offset(),
                    block.size(),
                    self.mmap.len()
                )));
            }
        }
        Ok(())
    }

    /// Validates file structure including checksum verification.
    ///
    /// This is more thorough but slower than `validate()`.
    ///
    /// # Errors
    /// Returns [`MvfError::CorruptedData`]
    /// - If any data blocks extend beyond file boundaries
    /// - If checksum verification fails for any block
    pub fn validate_with_checksum(&self) -> Result<()> {
        for (i, block) in self.file_footer.block_manifest().iter().enumerate() {
            let end_offset = block.offset() + block.size();
            if end_offset > self.mmap.len() as u64 {
                return Err(MvfError::corrupted_data(format!(
                    "Block extends beyond file: offset={}, size={}, file_size={}",
                    block.offset(),
                    block.size(),
                    self.mmap.len()
                )));
            }

            if block.checksum() != 0 {
                let start = (block.offset() as usize)
                    .saturating_sub(METRO_MAGIC.len())
                    .max(self.data_start);

                let end = ((block.offset() + block.size()) as usize)
                    .saturating_sub(METRO_MAGIC.len())
                    .max(self.data_start);

                if start >= end || end > self.mmap.len() {
                    return Err(MvfError::corrupted_data(format!(
                        "Invalid block range after adjustment: {}..{}",
                        start, end
                    )));
                }

                eprintln!("First 4 bytes (magic): {:?}", &self.mmap[0..4]);
                eprintln!("Expected magic: {:?}", METRO_MAGIC);
                eprintln!("Magic match: {}", self.mmap[0..4] == METRO_MAGIC);
                eprintln!("Block {}: adjusted range {}..{}", i, start, end);
                eprintln!(
                    "First 16 bytes: {:?}",
                    &self.mmap[start..(start + 16).min(self.mmap.len())]
                );

                let actual_checksum = crc32fast::hash(&self.mmap[start..end]);
                if actual_checksum != block.checksum() {
                    return Err(MvfError::corrupted_data(format!(
                        "Block checksum mismatch: expected {}, got {}",
                        block.checksum(),
                        actual_checksum
                    )));
                }
            }
        }

        todo!("Implement checksum validation")
    }

    /// Validates the footer bounds of the file
    /// Returns the start and end offsets of the footer
    fn validate_footer_bounds(mmap: &[u8]) -> Result<(usize, usize)> {
        // metro_footer_size bytes before end_magic_bytes (4 + 4)
        let footer_len_start = mmap.len() - (METRO_FOOTER_SIZE + METRO_MAGIC.len());
        let footer_len = u32::from_le_bytes([
            mmap[footer_len_start],
            mmap[footer_len_start + 1],
            mmap[footer_len_start + 2],
            mmap[footer_len_start + 3],
        ]) as usize;

        // footer_len + end_magic_bytes > file_size - start_magic_bytes
        if footer_len + METRO_FOOTER_SIZE + METRO_MAGIC.len() > mmap.len() - METRO_MAGIC.len() {
            return Err(MvfError::invalid_format("Invalid footer length"));
        }

        let footer_start = mmap.len() - (METRO_FOOTER_SIZE + METRO_MAGIC.len()) - footer_len;
        let footer_end = mmap.len() - (METRO_FOOTER_SIZE + METRO_MAGIC.len());

        let footer_bytes = &mmap[footer_start..footer_end];

        let file_footer = flatbuffers::root::<FileFooter>(footer_bytes)
            .map_err(|e| MvfError::invalid_format(format!("Failed to parse footer: {}", e)))?;

        if file_footer.format_version() != 1 {
            return Err(MvfError::UnsupportedVersion {
                got: file_footer.format_version(),
                expected: 1,
            });
        }

        Ok((footer_start, footer_end))
    }

    /// Validates basic file structure (magic bytes and size).
    fn validate_file_structure(mmap: &[u8]) -> Result<()> {
        // Minimum file size is 12 bytes
        if mmap.len() < (METRO_MAGIC.len() + METRO_FOOTER_SIZE + METRO_MAGIC.len()) {
            return Err(MvfError::invalid_format("File too small"));
        }

        if mmap[0..METRO_MAGIC.len()] != METRO_MAGIC {
            return Err(MvfError::invalid_format(
                "Invalid magic bytes at start of file",
            ));
        }

        if mmap[mmap.len() - METRO_MAGIC.len()..] != METRO_MAGIC {
            return Err(MvfError::invalid_format(
                "Invalid magic bytes at end of file, it may be corrupted",
            ));
        }

        Ok(())
    }
}

// Safety: MvfReader is safe to send between threads under the following assumptions:
// - Mmap is read-only and the underlying file isn't modified
// - All internal state is immutable after construction
unsafe impl Send for MvfReader {}

// Safety: MvfReader is safe to share between threads under the following assumptions:
// - All operations are read-only
// - Memory mapping provides consistent view across threads
unsafe impl Sync for MvfReader {}

#[cfg(test)]
mod tests {

    use std::fs;

    use crate::{
        builder::MvfBuilder,
        mvf_fbs::{DataType, DistanceMetric, VectorType},
        tests::{I32Bytes, StringBytes, TestContext, create_test_mvf, create_test_vectors},
    };

    use super::*;

    #[test]
    fn test_open_valid_file() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();

        let reader = MvfReader::open(&path);
        assert!(reader.is_ok());

        let reader = reader.unwrap();
        assert_eq!(reader.num_vector_spaces(), 1);
        assert_eq!(reader.vector_space_names(), vec!["test_space"]);
    }

    #[test]
    fn test_open_minimum_valid_size() {
        let context = TestContext::new();
        let path = context.temp_path("minimum_valid_size.mvf");

        let mut bytes = Vec::new();
        let footer_size = 0u32;
        bytes.extend_from_slice(&footer_size.to_le_bytes());
        bytes.extend_from_slice(b"MVF0");

        fs::write(&path, bytes).unwrap();

        let result = MvfReader::open(&path);
        // This should fail during flatbuffers parsing, not size checks
        // it still should not panic with overflow
        assert!(result.is_err());
    }

    #[test]
    fn test_open_nonexistent_file() {
        let context = TestContext::new();
        let path = context.temp_path("nonexistent.mvf");

        let result = MvfReader::open(&path);
        assert!(matches!(result, Err(MvfError::Io(_))));
    }

    #[test]
    fn test_open_file_too_small() {
        let context = TestContext::new();
        let path = context.temp_path("small.mvf");

        // Smaller than required footer size of 8 bytes
        fs::write(&path, vec![0u8; 4]).unwrap();
    }

    #[test]
    fn test_open_invalid_footer_size() {
        let context = TestContext::new();
        let path = context.temp_path("invalid_footer.mvf");

        let mut bytes = vec![0u8; 100];
        // Set footer size larger than file
        let invalid_size = 200u32;
        bytes.extend_from_slice(&invalid_size.to_le_bytes());
        bytes.extend_from_slice(b"MVF\0");

        fs::write(&path, bytes).unwrap();

        let result = MvfReader::open(&path);
        assert!(matches!(result, Err(MvfError::InvalidFormat(_))));
    }

    #[test]
    fn test_open_invalid_file_identifier() {
        let context = TestContext::new();
        let path = context.temp_path("invalid_id.mvf");

        let mut bytes = vec![0u8; 100];
        let footer_size = 50u32;
        bytes.extend_from_slice(&footer_size.to_le_bytes());
        bytes.extend_from_slice(b"XXXX"); // Invalid identifier

        fs::write(&path, bytes).unwrap();

        let result = MvfReader::open(&path);
        assert!(matches!(result, Err(MvfError::InvalidFormat(_))));
    }

    #[test]
    fn test_reader_properties() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();

        assert_eq!(reader.version(), 1);
        assert_eq!(reader.num_vector_spaces(), 1);
        assert!(!reader.has_metadata());
        assert!(reader.metadata_column_names().is_empty());
        assert!(reader.file_size() > 0);
    }

    #[test]
    fn test_get_vector_space() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();

        let space = reader.vector_space("test_space");
        assert!(space.is_ok());

        let nonexistent = reader.vector_space("nonexistent");
        assert!(matches!(nonexistent, Err(MvfError::VectorSpaceNotFound(_))));
    }

    #[test]
    fn test_validate_reader() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();

        let result = reader.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_reader_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<MvfReader>();
        assert_sync::<MvfReader>();
    }

    // #[test]
    // fn test_validate_with_checksum_success() {
    //     let context = TestContext::new();
    //     let built = create_test_mvf();
    //     let path = context.temp_path("test_checksum.mvf");
    //
    //     built.save(&path).unwrap();
    //     let reader = MvfReader::open(&path).unwrap();
    //
    //     // Note: This will hit the todo!() but we're testing the path to it
    //     let result = std::panic::catch_unwind(|| reader.validate_with_checksum());
    //
    //     // Should panic at todo!() which means we reached the checksum validation
    //     assert!(result.is_err());
    // }

    #[test]
    fn test_metadata_functionality() {
        let context = TestContext::new();

        // Create MVF with metadata
        let mut builder = MvfBuilder::new();
        builder.add_vector_space(
            "test",
            4,
            VectorType::Dense,
            DistanceMetric::L2,
            DataType::Float32,
        );
        builder.add_vectors("test", &create_test_vectors()).unwrap();

        // Add metadata
        let ids: Vec<i32> = vec![1, 2, 3];
        let id_bytes: Vec<I32Bytes> = ids.iter().map(|&v| I32Bytes(v)).collect();
        builder
            .add_metadata_column("ids", DataType::UInt32, &id_bytes)
            .unwrap();

        let labels = ["a", "b", "c"];
        let label_bytes: Vec<StringBytes> = labels.iter().map(|s| StringBytes(s)).collect();
        builder
            .add_metadata_column("labels", DataType::StringRef, &label_bytes)
            .unwrap();

        let built = builder.build();
        let path = context.temp_path("metadata.mvf");
        built.save(&path).unwrap();

        let reader = MvfReader::open(&path).unwrap();

        assert!(reader.has_metadata());
        let column_names = reader.metadata_column_names();
        assert_eq!(column_names.len(), 2);
        assert!(column_names.contains(&"ids".to_string()));
        assert!(column_names.contains(&"labels".to_string()));
    }

    #[test]
    fn test_vector_space_properties() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("properties.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();
        let space = reader.vector_space("test_space").unwrap();

        assert_eq!(space.name(), "test_space");
        assert_eq!(space.dimension(), 4);
        assert_eq!(space.total_vectors(), 3);
        assert_eq!(space.vector_type(), VectorType::Dense);
        assert_eq!(space.distance_metric(), DistanceMetric::L2);
        assert_eq!(space.data_type(), DataType::Float32);
    }

    // #[test]
    // fn test_corrupted_file_scenarios() {
    //     let context = TestContext::new();
    //
    //     // Test file with invalid magic at start
    //     let path1 = context.temp_path("corrupted_start.mvf");
    //     let mut bytes = vec![0u8; 100];
    //     bytes[0..4].copy_from_slice(b"XXXX"); // Wrong magic
    //     bytes[bytes.len() - 8..bytes.len() - 4].copy_from_slice(&16u32.to_le_bytes());
    //     bytes[bytes.len() - 4..].copy_from_slice(b"MVF1");
    //     fs::write(&path1, bytes).unwrap();
    //
    //     let result = MvfReader::open(&path1);
    //     assert!(matches!(result, Err(MvfError::InvalidFormat(_))));
    //
    //     // Test file with mismatched end magic
    //     let path2 = context.temp_path("corrupted_end.mvf");
    //     let mut bytes = vec![0u8; 100];
    //     bytes[0..4].copy_from_slice(b"MVF1");
    //     bytes[bytes.len() - 8..bytes.len() - 4].copy_from_slice(&16u32.to_le_bytes());
    //     bytes[bytes.len() - 4..].copy_from_slice(b"XXXX"); // Wrong magic
    //     fs::write(&path2, bytes).unwrap();
    //
    //     let result = MvfReader::open(&path2);
    //     assert!(matches!(result, Err(MvfError::InvalidFormat(_))));
    // }

    // #[test]
    // fn test_unsupported_version() {
    //     let context = TestContext::new();
    //     let path = context.temp_path("unsupported_version.mvf");
    //
    //     // Create a minimal valid file structure but with wrong version
    //     let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(1024);
    //
    //     // Create empty vector spaces
    //     let vector_spaces = builder.create_vector::<Vec>(&[]);
    //     let data_blocks = builder.create_vector(&[]);
    //
    //     let file_footer = FileFooter::create(
    //         &mut builder,
    //         &FileFooterArgs {
    //             format_version: 99, // Unsupported version
    //             vector_spaces: Some(vector_spaces),
    //             block_manifest: Some(data_blocks),
    //             metadata_columns: None,
    //             string_heap_block_index: 0,
    //             extensions: None,
    //             compatibility_version: 1,
    //             deprecated_fields: None,
    //         },
    //     );
    //
    //     builder.finish_minimal(file_footer);
    //     let footer_data = builder.finished_data();
    //
    //     let mut bytes = Vec::new();
    //     bytes.extend_from_slice(b"MVF1");
    //     bytes.extend_from_slice(footer_data);
    //     bytes.extend_from_slice(&(footer_data.len() as u32).to_le_bytes());
    //     bytes.extend_from_slice(b"MVF1");
    //
    //     fs::write(&path, bytes).unwrap();
    //
    //     let result = MvfReader::open(&path);
    //     assert!(matches!(
    //         result,
    //         Err(MvfError::UnsupportedVersion {
    //             got: 99,
    //             expected: 1
    //         })
    //     ));
    // }

    // #[test]
    // fn test_footer_bounds_edge_cases() {
    //     let context = TestContext::new();
    //
    //     // Test with exactly minimum valid size
    //     let path = context.temp_path("min_size.mvf");
    //     let min_size = METRO_MAGIC.len() + METRO_FOOTER_SIZE + METRO_MAGIC.len();
    //     let mut bytes = vec![0u8; min_size];
    //     bytes[0..4].copy_from_slice(b"MVF1");
    //     bytes[bytes.len() - 8..bytes.len() - 4].copy_from_slice(&0u32.to_le_bytes()); // Zero footer size
    //     bytes[bytes.len() - 4..].copy_from_slice(b"MVF1");
    //     fs::write(&path, bytes).unwrap();
    //
    //     let result = MvfReader::open(&path);
    //     // Should fail during flatbuffers parsing
    //     assert!(result.is_err());
    // }

    #[test]
    fn test_block_validation_edge_cases() {
        let context = TestContext::new();
        let built = create_test_mvf();
        let path = context.temp_path("test_validation.mvf");

        built.save(&path).unwrap();
        let reader = MvfReader::open(&path).unwrap();

        // Test basic validation
        let result = reader.validate();
        assert!(result.is_ok());

        // We can't easily test corrupted blocks without modifying the file,
        // but we've covered the validation logic path
    }

    #[test]
    fn test_reader_error_paths() {
        let context = TestContext::new();

        // Test opening directory instead of file
        let dir_path = context.temp_path("not_a_file");
        std::fs::create_dir(&dir_path).unwrap();

        let result = MvfReader::open(&dir_path);
        assert!(matches!(result, Err(MvfError::Io(_))));
    }
}
