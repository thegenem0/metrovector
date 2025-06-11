use std::{fs::File, io::Write, path::Path};

use crate::{builder::BuiltMvf, errors::Result};

pub struct MvfWriter {
    file: File,
}

impl MvfWriter {
    /// Create a new MVF file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self { file })
    }

    /// Write a built MVF to the file
    pub fn write(mut self, mvf: BuiltMvf) -> Result<()> {
        let bytes = mvf.to_bytes()?;
        self.file.write_all(&bytes)?;
        self.file.flush()?;
        Ok(())
    }
}
