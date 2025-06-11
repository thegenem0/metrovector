//! Simple example showing basic vector operations with MVF files
//!
//! Run with: cargo run --example simple

use metrovector::{
    builder::MvfBuilder,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
    reader::MvfReader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MVF Simple Example ===\n");

    // Some sample vectors
    let vectors = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
        vec![2.0, 4.0, 6.0, 8.0],
        vec![1.0, 3.0, 5.0, 7.0],
    ];

    println!("Building MVF file with {} vectors", vectors.len());

    let mut builder = MvfBuilder::new();
    let _space_idx = builder.add_vector_space(
        "embeddings",
        4, // 4-dimensional vectors
        VectorType::Dense,
        DistanceMetric::L2,
        DataType::Float32,
    );

    builder.add_vectors("embeddings", &vectors)?;

    // Add some metadata using the typed helper methods
    #[derive(Copy, Clone)]
    struct I32Metadata(i32);

    impl From<I32Metadata> for Vec<u8> {
        fn from(val: I32Metadata) -> Self {
            val.0.to_le_bytes().to_vec()
        }
    }

    let ids = vec![
        I32Metadata(1),
        I32Metadata(2),
        I32Metadata(3),
        I32Metadata(4),
        I32Metadata(5),
    ];
    builder.add_metadata_column("vector_ids", DataType::UInt32, &ids)?;

    let built_mvf = builder.build();

    let temp_file = std::env::temp_dir().join("simple_example.mvf");
    built_mvf.save(&temp_file)?;
    println!("   Written to: {:?}", temp_file);

    println!("\nReading MVF file");
    let reader = MvfReader::open(&temp_file)?;

    println!("   File version: {}", reader.version());
    println!("   Vector spaces: {}", reader.num_vector_spaces());
    println!("   File size: {:.2} KB", reader.file_size() as f64 / 1024.0);

    if reader.has_metadata() {
        println!("   Metadata columns: {:?}", reader.metadata_column_names());
    }

    let space_names = reader.vector_space_names();
    let space = reader.vector_space(&space_names[0])?;

    println!("   Space name: '{}'", space.name());
    println!("   Dimensions: {}", space.dimension());
    println!("   Total vectors: {}", space.total_vectors());
    println!("   Vector type: {:?}", space.vector_type());
    println!("   Distance metric: {:?}", space.distance_metric());
    println!("   Data type: {:?}", space.data_type());

    println!("\nReading vectors:");
    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;
        println!("   Vector {}: {:?}", i, data);
    }

    println!("\nBatch vector reading:");
    let indices = vec![0, 2, 4]; // Read vectors 0, 2, and 4
    let batch_vectors = space.get_vectors_batch(&indices)?;
    for (i, vector) in batch_vectors.iter().enumerate() {
        let data = vector.as_f32()?;
        println!("   Batch vector {} (index {}): {:?}", i, indices[i], data);
    }

    println!("\nSimilarity search:");
    let query = vec![2.5, 4.5, 6.5, 8.5];
    println!("   Query vector: {:?}", query);

    let mut best_match = None;
    let mut best_distance = f32::INFINITY;

    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;

        // Euclidean distance
        let distance: f32 = query
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        println!("   Vector {} distance: {:.2}", i, distance);

        if distance < best_distance {
            best_distance = distance;
            best_match = Some((i, data));
        }
    }

    if let Some((index, vector)) = best_match {
        println!(
            "   Best match: Vector {} (distance: {:.2})",
            index, best_distance
        );
        println!("   Result: {:?}", vector);
    }

    println!("\nFile validation:");
    reader.validate()?;
    println!("   File validation passed!");

    std::fs::remove_file(&temp_file).ok();
    println!("\n   Cleaned up temporary file.");

    Ok(())
}
