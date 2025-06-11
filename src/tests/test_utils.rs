use std::path::PathBuf;
use tempfile::TempDir;

use crate::{
    builder::MvfBuilder,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
};

pub struct TestContext {
    pub temp_dir: TempDir,
}

impl TestContext {
    pub fn new() -> Self {
        Self {
            temp_dir: TempDir::new().expect("Failed to create temp dir"),
        }
    }

    pub fn temp_path(&self, filename: &str) -> PathBuf {
        self.temp_dir.path().join(filename)
    }
}

pub fn create_test_vectors() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ]
}

pub fn create_test_mvf() -> crate::builder::BuiltMvf {
    let mut builder = MvfBuilder::new();

    let _space_idx = builder.add_vector_space(
        "test_space",
        4,
        VectorType::Dense,
        DistanceMetric::L2,
        DataType::Float32,
    );

    builder
        .add_vectors("test_space", &create_test_vectors())
        .expect("Failed to add vectors");

    builder.build()
}

// pub fn create_invalid_mvf_bytes() -> Vec<u8> {
//     vec![0x00; 16] // Too small and invalid
// }
//
// pub fn create_minimal_valid_mvf_bytes() -> Vec<u8> {
//     let mut bytes = vec![0u8; 1024]; // Data section
//
//     // Add footer size (4 bytes)
//     let footer_size = 100u32;
//     bytes.extend_from_slice(&footer_size.to_le_bytes());
//
//     // Add file identifier
//     bytes.extend_from_slice(b"MVF0");
//
//     bytes
// }
