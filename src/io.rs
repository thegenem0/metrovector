use std::{fs::File, io::Write, path::Path};

use crate::{builder::BuiltMvf, errors::Result};

/// Writer for creating MVF files.
///
/// # Examples
///
/// ```no_run
/// use metrovector::{io::MvfWriter, builder::MvfBuilder, errors::MvfError};
///
/// let mut builder = MvfBuilder::new();
/// // ... configure builder ...
/// let built = builder.build();
///
/// let writer = MvfWriter::create("output.mvf")?;
/// writer.write(built)?;
/// # Ok::<(), MvfError>(())
/// ```
pub struct MvfWriter {
    file: File,
}

impl MvfWriter {
    /// Creates a new MVF file for writing.
    ///
    /// # Errors
    /// Returns [`MvfError::Io`] if the file cannot be created or opened for writing.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self { file })
    }

    /// Writes a built MVF to the file.
    ///
    /// # Errors
    /// Returns [`MvfError::Io`]
    /// - If the MVF data cannot be serialized
    /// - If writing to the file fails
    /// - If file cannot be flushed
    pub fn write(mut self, mvf: BuiltMvf) -> Result<()> {
        let bytes = mvf.to_bytes()?;
        self.file.write_all(&bytes)?;
        self.file.flush()?;
        Ok(())
    }
}
