# Metro Vector Format (MVF)

A high-performance, compact binary format for storing and querying vector embeddings, designed as a foundational building block for vector databases.

[![CI](https://github.com/thegenem0/metrovector/actions/workflows/ci.yaml/badge.svg)](https://github.com/thegenem0/metrovector/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/thegenem0/metrovector/graph/badge.svg?token=A334YLOQAA)](https://codecov.io/gh/thegenem0/metrovector)
## Overview

MVF (Metro Vector Format) is a binary file format optimized for storing large collections of high-dimensional vectors with associated metadata.
It provides memory-efficient storage, fast random access, and support for various vector data types commonly used in machine learning and AI applications.

### Key Features

- 🚀 **High-performance**: Memory-mapped file access with zero-copy reads
- 💾 **Compact Storage**: Efficient binary encoding with optional compression
- 🔢 **Multiple Data Types**: Float32, Float16, Int8, UInt8 vector support
- 📊 **Rich Metadata**: Store vector spaces with distance metrics and indexing hints
- 🛡️ **Data Integrity**: Built-in checksums and validation
- ⚡ **Random Access**: O(1) vector retrieval by index
- 🧠 **Vector DB Ready**: Designed as a storage layer for vector databases

## Architecture

```bash
┌─────────────────────────────────────────────────────────────┐
│                    MVF File Structure                       │
├─────────────────────────────────────────────────────────────┤
│ Magic Header (4B) │ Vector Data Blocks │ Footer │ Magic (4B)│
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────────┐
            │     Vector Data Block           │
            ├─────────────────────────────────┤
            │ Vector 0: [1.0, 2.0, 3.0, ...]  │
            │ Vector 1: [4.0, 5.0, 6.0, ...]  │
            │ Vector 2: [7.0, 8.0, 9.0, ...]  │
            │          ...                    │
            └─────────────────────────────────┘
```

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
metrovector-format = "0.1.0"
```

## Quick Start

### Creating an MVF File

```rust
use metrovector_format::{
    builder::MvfBuilder,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some sample vectors
    let vectors = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];

    // Build the MVF file
    let mut builder = MvfBuilder::new();
    
    builder.add_vector_space(
        "embeddings",              // Space name
        4,                         // Dimensions
        VectorType::Dense,         // Vector type
        DistanceMetric::Euclidean, // Distance metric
        DataType::Float32,         // Data type
    );

    builder.add_vectors("embeddings", &vectors)?;
    let built_mvf = builder.build();

    // Write to file
    built_mvf.save("vectors.mvf")?;
    
    Ok(())
}
```

### Reading an MVF File

```rust
use metrovector_format::MvfReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open the MVF file
    let mvf_file = MvfFile::open("vectors.mvf")?;
    
    println!("File version: {}", mvf_file.version());
    println!("Vector spaces: {}", mvf_file.vector_spaces().len());

    // Get the first vector space by name
    let space = mvf_file.vector_space(mvf_file.vector_space_names().first().unwrap())?;

    println!("Space '{}': {} vectors, {} dimensions", 
             space.name(), space.total_vectors(), space.dimension());

    // Read vectors
    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;
        println!("Vector {}: {:?}", i, data);
    }

    Ok(())
}
```

## Examples

The examples/ directory contains comprehensive examples:

```bash
# Basic usage
cargo run --example simple

# Working with different data types  
cargo run --example data_types

# Performance benchmarking with large datasets
# Note: Generating all of the data takes as much ram as the dataset size
cargo run --example large_dataset -- --size 4gb

# Similarity search
cargo run --example similarity_search
```

## Performance

Performance characteristics on modern hardware:

| Operation | Throughput (vectors/sec) | Latency |
| --------- | ------------------------ | ------- |
| Sequential Read | ~1.5M | ~0.5μs per vector |
| Random Access | ~500K | ~2μs per vector |
| File Opening | - | ~10ms (any size) |
| Memory Usage | ~0 (memory mapped) | - |

## API Reference

This is a high-level overview of the API.

### Supported Data Types

- **Float32**: Standard 32-bit floating point (most common)
- **Float16**: Half-precision for memory efficiency
- **Int8**: Signed 8-bit integers for quantized vectors
- **UInt8**: Unsigned 8-bit integers

### Distance Metrics

- **L2**: Euclidean distance
- **Cosine**: Cosine similarity
- **Dot**: Dot product


## Testing

Run tests with:

```bash
cargo nextest run
```

Get test coverage with:

```bash
cargo llvm-cov nextest --html --ingore-filename-regex 'generated'
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT license.

Built with ❤️ in Rust for the vector AI community.
