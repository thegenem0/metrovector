use std::{fs::File, path::Path};

use memmap2::Mmap;

use crate::{
    METRO_FOOTER_SIZE, METRO_MAGIC,
    errors::{MvfError, Result},
    mvf_fbs::{FileFooter, VectorSpace},
};

pub struct MvfReader {
    mmap: Mmap,
    file_footer: FileFooter<'static>,
    data_start: usize,
}

impl MvfReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

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

        #[allow(invalid_value)]
        // We move ownership of mmap to the reader
        // This is what we're returning to the caller

        // Safety:
        // We're zeroing the memory here,
        // but we set it to a valid value within the same scope later
        // If the reader fails, this is dropped,
        // and the zeroed memory is never used again
        let mut reader = Self {
            mmap,
            file_footer: unsafe { std::mem::zeroed() },
            data_start: METRO_MAGIC.len(), // Start after initial magic bytes
        };

        let footer_start = reader.mmap.len() - (METRO_FOOTER_SIZE + METRO_MAGIC.len()) - footer_len;
        let footer_end = reader.mmap.len() - (METRO_FOOTER_SIZE + METRO_MAGIC.len());
        let footer_bytes = &reader.mmap[footer_start..footer_end];

        let file_footer = flatbuffers::root::<FileFooter>(footer_bytes)
            .map_err(|e| MvfError::invalid_format(format!("Failed to parse footer: {}", e)))?;

        if file_footer.format_version() != 1 {
            return Err(MvfError::UnsupportedVersion {
                got: file_footer.format_version(),
                expected: 1,
            });
        }

        // Safety:
        // We're extending the lifetime of the foother to the lifetime of the reader
        // and the reader is the only thing that can mutate the foother
        reader.file_footer = unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            std::mem::transmute(file_footer)
        };

        Ok(reader)
    }

    pub fn version(&self) -> u16 {
        self.file_footer.format_version()
    }

    pub fn num_vector_spaces(&self) -> usize {
        self.file_footer.vector_spaces().len()
    }

    pub fn vector_space_names(&self) -> Vec<String> {
        self.file_footer
            .vector_spaces()
            .iter()
            .map(|s| s.name().to_string())
            .collect()
    }

    pub fn vector_space(&self, name: &str) -> Result<VectorSpace> {
        match self
            .file_footer
            .vector_spaces()
            .iter()
            .enumerate()
            .find(|(_, s)| *s.name() == *name)
        {
            Some((_, space)) => Ok(space),
            None => Err(MvfError::VectorSpaceNotFound(name.to_string())),
        }
    }

    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    pub fn has_metadata(&self) -> bool {
        self.file_footer.metadata_columns().is_some()
    }

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

            if block.checksum() != 0 {
                let start = block.offset() as usize;
                let end = (block.offset() + block.size()) as usize;
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
        Ok(())
    }
}

unsafe impl Send for MvfReader {}
unsafe impl Sync for MvfReader {}

#[cfg(test)]
mod tests {

    use std::fs;

    use crate::tests::{TestContext, create_test_mvf};

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

        assert_eq!(reader.version(), 3);
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
}
