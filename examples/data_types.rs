//! Example showing different data types and conversions
//!
//! Run with: cargo run --example data_types

use metrovector::{
    builder::MvfBuilder,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
    reader::MvfReader,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MVF Data Types Example ===\n");

    let temp_dir = std::env::temp_dir();

    println!("Float32 vectors (standard precision)");
    example_float32_vectors(&temp_dir)?;

    println!("\nFloat16 vectors (half precision - space efficient)");
    example_float16_vectors(&temp_dir)?;

    println!("\nMemory-efficient access patterns");
    example_efficient_access(&temp_dir)?;

    println!("\nSIMD and alignment features");
    example_simd_features(&temp_dir)?;

    Ok(())
}

fn example_float32_vectors(temp_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let float32_vectors = vec![
        vec![1.0, 2.5, 3.7, 4.2],
        vec![5.1, 6.8, 7.3, 8.9],
        vec![-1.2, 0.0, 2.4, -3.1],
    ];

    let mut builder = MvfBuilder::new();
    builder.add_vector_space(
        "float32_space",
        4,
        VectorType::Dense,
        DistanceMetric::Cosine,
        DataType::Float32,
    );
    builder.add_vectors("float32_space", &float32_vectors)?;
    let mvf = builder.build();

    let float32_file = temp_dir.join("float32_example.mvf");
    mvf.save(&float32_file)?;

    let reader = MvfReader::open(&float32_file)?;
    let space = reader.vector_space("float32_space")?;

    println!("   Created {} Float32 vectors", space.total_vectors());
    println!(
        "   Storage size: {} bytes per vector",
        space.dimension() * 4
    );

    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;

        // Stats
        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        println!("   Vector {}: {:?}", i, data);
        println!(
            "     Sum: {:.2}, Mean: {:.2}, L2 Norm: {:.2}",
            sum, mean, norm
        );
    }

    std::fs::remove_file(&float32_file).ok();
    Ok(())
}

fn example_float16_vectors(temp_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    // Same vectors but stored as Float16 for half precision
    let vectors = vec![
        vec![1.0, 2.5, 3.7, 4.2],
        vec![5.1, 6.8, 7.3, 8.9],
        vec![-1.2, 0.0, 2.4, -3.1],
    ];

    let mut builder = MvfBuilder::new();
    builder.add_vector_space(
        "float16_space",
        4,
        VectorType::Dense,
        DistanceMetric::Cosine,
        DataType::Float16,
    );
    builder.add_vectors("float16_space", &vectors)?;
    let mvf = builder.build();

    let float16_file = temp_dir.join("float16_example.mvf");
    mvf.save(&float16_file)?;

    let reader = MvfReader::open(&float16_file)?;
    let space = reader.vector_space("float16_space")?;

    println!("   Created {} Float16 vectors", space.total_vectors());
    println!(
        "   Storage size: {} bytes per vector (50% savings)",
        space.dimension() * 2
    );

    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?; // Float16 -> Float32 conversion

        let original = &vectors[i as usize];
        let max_error = original
            .iter()
            .zip(data.iter())
            .map(|(orig, stored)| (orig - stored).abs())
            .fold(0.0f32, f32::max);

        println!("   Vector {}: {:?}", i, data);
        println!("     Max error from Float16 conversion: {:.6}", max_error);
    }

    std::fs::remove_file(&float16_file).ok();
    Ok(())
}

fn example_efficient_access(temp_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    // Larger dataset, still efficient access
    let mut vectors = Vec::new();
    for i in 0..100 {
        let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 * 0.01).collect();
        vectors.push(vector);
    }

    let mut builder = MvfBuilder::new();
    builder.add_vector_space(
        "efficient_space",
        64,
        VectorType::Dense,
        DistanceMetric::L2,
        DataType::Float32,
    );
    builder.add_vectors("efficient_space", &vectors)?;
    let mvf = builder.build();

    let efficient_file = temp_dir.join("efficient_example.mvf");
    mvf.save(&efficient_file)?;

    let reader = MvfReader::open(&efficient_file)?;
    let space = reader.vector_space("efficient_space")?;

    println!(
        "   Created {} vectors for efficiency testing",
        space.total_vectors()
    );

    // Map a range of vectors for batch processing
    let _range_slice = space.map_vector_range(10, 20)?; // 20 vectors starting at index 10
    println!("   Mapped range: 20 vectors starting at index 10");

    // Access using optimized patterns
    let indices = vec![5, 15, 25, 35, 45];
    let access_pattern = space.prepare_access_pattern(&indices);
    println!("   Prepared access pattern for {} indices", indices.len());

    let pattern_vectors = space.get_vectors_with_pattern(&access_pattern)?;
    println!(
        "   Retrieved {} vectors using optimized pattern",
        pattern_vectors.len()
    );

    // Streaming support
    println!("   Streaming vectors in chunks:");
    let mut total_processed = 0;
    for (chunk_idx, chunk_result) in space.stream_vectors(0, 25).enumerate() {
        let chunk = chunk_result?;
        total_processed += chunk.len();
        println!(
            "     Chunk {}: {} vectors (total processed: {})",
            chunk_idx,
            chunk.len(),
            total_processed
        );

        if chunk_idx >= 3 {
            // Limit output for example
            break;
        }
    }

    std::fs::remove_file(&efficient_file).ok();
    Ok(())
}

fn example_simd_features(temp_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let vectors = vec![
        vec![1.0; 16], // 16-dimensional for SIMD demo
        vec![2.0; 16],
        vec![3.0; 16],
    ];

    let mut builder = MvfBuilder::new();
    builder.add_vector_space(
        "simd_space",
        16,
        VectorType::Dense,
        DistanceMetric::L2,
        DataType::Float32,
    );
    builder.add_vectors("simd_space", &vectors)?;
    let mvf = builder.build();

    let simd_file = temp_dir.join("simd_example.mvf");
    mvf.save(&simd_file)?;

    let reader = MvfReader::open(&simd_file)?;
    let space = reader.vector_space("simd_space")?;

    println!("   Created vectors for SIMD demonstration");

    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;

        // Try SIMD-aligned access
        match vector.as_simd_slice::<f32>() {
            Ok(simd_slice) => {
                println!("   Vector {}: SIMD-aligned access successful", i);
                println!("     First 4 elements: {:?}", &simd_slice[..4]);
            }
            Err(_) => {
                println!(
                    "   Vector {}: SIMD alignment not available, using regular access",
                    i
                );
                let data = vector.as_f32()?;
                println!("     First 4 elements: {:?}", &data[..4]);
            }
        }

        // // Raw vector slice for advanced operations
        // let vector_slice = vector.as_vector_slice()?;
        // let simd_width = 32; // 256-bit SIMD (AVX2)
        // if vector_slice.is_simd_aligned(32) {
        //     let chunk_size = vector_slice.chunk_size_for_simd(256);
        //     println!(
        //         "     SIMD ready: {} elements per SIMD operation",
        //         chunk_size
        //     );
        // } else {
        //     println!("     SIMD not ready");
        // }

        // Raw byte access for custom processing
        let bytes = vector.as_bytes();
        println!("     Raw data: {} bytes", bytes.len());
    }

    std::fs::remove_file(&simd_file).ok();
    Ok(())
}
