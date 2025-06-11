//! Example showing different data types and conversions
//!
//! Run with: cargo run --example data_types

use metrovector_format::{
    builder::MvfBuilder,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
    reader::MvfReader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MVF Data Types Example ===\n");

    let temp_dir = std::env::temp_dir();

    // Example 1: Float32 vectors
    println!("1. Float32 vectors");
    let float32_vectors = vec![vec![1.0, 2.5, 3.7, 4.2], vec![5.1, 6.8, 7.3, 8.9]];

    let mut builder = MvfBuilder::new();
    builder.add_vector_space(
        "float32",
        4,
        VectorType::Dense,
        DistanceMetric::Cosine,
        DataType::Float32,
    );
    builder.add_vectors("float32", &float32_vectors)?;
    let mvf = builder.build();

    let float32_file = temp_dir.join("float32_example.mvf");
    mvf.save(&float32_file)?;

    // Read and display
    let file = MvfReader::open(&float32_file)?;
    let space = file.vector_space(file.vector_space_names().first().unwrap())?;

    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;
        println!("   Float32 vector {}: {:?}", i, data);
    }

    // Example 2: Working with raw bytes
    println!("\n2. Raw byte access");
    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let bytes = vector.as_bytes();
        println!("   Vector {} raw bytes: {} bytes", i, bytes.len());
        println!("   First 8 bytes: {:?}", &bytes[..8.min(bytes.len())]);
    }

    // Example 3: Type-safe slice access
    println!("\n3. Type-safe slice access");
    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let slice: &[f32] = vector.as_slice()?;
        println!("   Vector {} as slice: {:?}", i, slice);

        // Calculate statistics
        let sum: f32 = slice.iter().sum();
        let mean = sum / slice.len() as f32;
        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();

        println!(
            "     Sum: {:.2}, Mean: {:.2}, L2 Norm: {:.2}",
            sum, mean, norm
        );
    }

    // Clean up
    std::fs::remove_file(&float32_file).ok();

    Ok(())
}
