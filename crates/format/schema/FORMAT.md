# Overview

The MetroVector Format (MVF) is a binary file format designed for efficient
storage and retrieval of high-dimensional vector embeddings with associated
metadata and search indices.

The format uses FlatBuffers for schema evolution and cross-platform compatibility.

## File Structure

Start Magic | Data Blocks | Footer Length | End Magic
-----------+-------------+---------------+----------
4 | Variable | 4 | 4
0 | 4 | N | N+M

Byte Layout Details:

Offset | Size | Field | Description
--------+----------+---------------+-------------
0 | 4 | Start Magic | ASCII "MVF1"
4 | Variable | Data Blocks | Concatenated Binary Data Blocks
N | Variable | Footer | FlatBuffer-encoded file metadata
N+M | 4 | Footer Length | Little-endian u32 length of Footer
N+M+4 | 4 | End Magic | ASCII "MVF1"

## Section Details

### Start Magic

Offset: 0x00000000
┌──────┬──────┬──────┬──────┐
│ 0x4D │ 0x56 │ 0x46 │ 0x31 │
│  M   │  V   │  F   │  1   │
└──────┴──────┴──────┴──────┘

### Data Blocks (Bytes 4 to N)

Offset: 0x00000004
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│    Block 0      │    Block 1      │    Block 2      │      ...        │
│   (Vectors)     │  (Metadata)     │  (Index Data)   │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

#### Data Block Structure

Each data block is a contiguous sequence of bytes with no internal headers. 
The block layout is defined by the corresponding DataBlock entry in the footer:

```rust
DataBlock {
    offset: u64,                       // Absolute offset from start of file
    size: u64,                         // Size in bytes
    compression: CompressionAlgorithm,
    compression_level: u32,
    checksum: u32                      // CRC32 checksum of uncompressed data
}
```

#### Vector Data Layout

For `DataType::Float32` vectors with dimension D:

Vector 0: [f32; D] | Vector 1: [f32; D] | Vector 2: [f32; D] | ...
┌─────────────────┬─────────────────┬─────────────────┬─────
│ D * 4 bytes     │ D * 4 bytes     │ D * 4 bytes     │ ...
└─────────────────┴─────────────────┴─────────────────┴─────

Each `f32` value is stored in little-endian format.

#### Metadata Layout

For columnar metadata storage, each column is stored as a separate data block:

Column Data: [value0][value1][value2]...

Data type encoding:

Int32: 4 bytes little-endian
Int64: 8 bytes little-endian
Float32: 4 bytes little-endian (IEEE 754)
Float64: 8 bytes little-endian (IEEE 754)
String: Length-prefixed UTF-8 (4-byte length + data)


### FlatBuffer Footer (Bytes N to N+M)

The footer contains all file metadata encoded as a FlatBuffer. 
This includes vector space definitions, data block manifests, and metadata schema.

```cpp
table FileFooter {
    format_version: uint16;
    compatibility_version: uint16;
    vector_spaces: [VectorSpace];
    block_manifest: [DataBlock];
    metadata_columns: [MetadataColumn];
    string_heap_block_index: uint32;
    extensions: [Extension];
    deprecated_fields: [ubyte];
}

table VectorSpace {
    name: string;
    dimension: uint32;
    total_vectors: uint64;
    vector_type: VectorType;
    distance_metric: DistanceMetric;
    data_type: DataType;
    vectors_block_index: uint32;
    vector_ids_block_index: uint32;
    index: Index;
    sparse_metadata: SparseMetadata;
    tombstones: [uint64];
}

table DataBlock {
    offset: uint64;
    size: uint64;
    compression: CompressionAlgorithm;
    compression_level: uint32;
    checksum: uint32;
}
```

### Footer Length (Bytes N+M to N+M+4)

Allows reading the footer without parsing the entire file.
Stored as a little-endian u32.

### End Magic (Bytes N+M+4 to N+M+8)

Offset: 0x0000004C
┌──────┬──────┬──────┬──────┐
│ 0x4D │ 0x56 │ 0x46 │ 0x31 │
│  M   │  V   │  F   │  1   │
└──────┴──────┴──────┴──────┘


## Example File Layout

#### Small Example (1000 vectors, 128 dimensions)

File Size: 524,348 bytes

┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Start Magic   │   Vector Data   │    Footer       │  Footer Length  │   End Magic     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
0x00000000        0x00000004        0x0007D004        0x0007D0C4        0x0007D0C8
4 bytes           512,000 bytes     192 bytes         4 bytes           4 bytes

#### Large Example (multiple vector spaces + metadata)

File Size: 2,048,576 bytes

┌─────┬─────────────┬─────────────┬─────────────┬─────────┬─────────┬─────────┐
│Magic│ Vectors_A   │ Vectors_B   │ Metadata_1  │ Index   │ Footer  │ End     │
└─────┴─────────────┴─────────────┴─────────────┴─────────┴─────────┴─────────┘
0     4             1MB           1.5MB         1.8MB     2MB       ~2MB
