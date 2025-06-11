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

    // Create some sample vectors
    let vectors = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
        vec![2.0, 4.0, 6.0, 8.0],
        vec![1.0, 3.0, 5.0, 7.0],
    ];

    println!("1. Building MVF file with {} vectors", vectors.len());

    // Build the MVF file
    let mut builder = MvfBuilder::new();
    let _space_idx = builder.add_vector_space(
        "embeddings",
        4, // 4-dimensional vectors
        VectorType::Dense,
        DistanceMetric::L2,
        DataType::Float32,
    );

    builder.add_vectors("embeddings", &vectors)?;
    let built_mvf = builder.build();

    // Write to file
    let temp_file = std::env::temp_dir().join("simple_example.mvf");
    built_mvf.save(&temp_file)?;
    println!("   Written to: {:?}", temp_file);

    // Read the file back
    println!("\n2. Reading MVF file");
    let mvf_file = MvfReader::open(&temp_file)?;

    println!("   File version: {}", mvf_file.version());
    println!("   Vector spaces: {}", mvf_file.num_vector_spaces());

    // Get the first vector space
    let space = mvf_file
        .vector_space(mvf_file.vector_space_names().first().unwrap())
        .unwrap();

    println!("   Space name: '{}'", space.name());
    println!("   Dimensions: {}", space.dimension());
    println!("   Total vectors: {}", space.total_vectors());
    println!("   Data type: {:?}", space.data_type());

    // Read individual vectors
    println!("\n3. Reading vectors:");
    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;
        println!("   Vector {}: {:?}", i, data);
    }

    // Demonstrate similarity search
    println!("\n4. Similarity search:");
    let query = vec![2.5, 4.5, 6.5, 8.5];
    println!("   Query vector: {:?}", query);

    let mut best_match = None;
    let mut best_distance = f32::INFINITY;

    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;

        // Calculate Euclidean distance
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

    // Clean up
    std::fs::remove_file(&temp_file).ok();

    Ok(())
}
