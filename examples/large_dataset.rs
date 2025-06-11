//! Large dataset example - creates multi-GB MVF files for performance testing
//!
//! Run with: cargo run --example large_dataset
//! Or with custom size: cargo run --example large_dataset -- --size 4gb
//! Or with custom params: cargo run --example large_dataset -- --vectors 1000000 --dimensions 768

use metrovector::{
    builder::MvfBuilder,
    errors::Result,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
    reader::MvfReader,
};
use std::env;
use std::time::Instant;

struct DatasetConfig {
    num_vectors: usize,
    dimensions: usize,
    data_type: DataType,
    target_size_gb: f32,
}

impl DatasetConfig {
    fn from_args() -> Self {
        let args: Vec<String> = env::args().collect();

        let mut num_vectors = 0;
        let mut dimensions = 0;
        let mut target_size_gb = 2.5; // Default 2.5GB

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--size" => {
                    if i + 1 < args.len() {
                        let size_str = &args[i + 1];
                        target_size_gb = if size_str.ends_with("gb") || size_str.ends_with("GB") {
                            size_str[..size_str.len() - 2].parse().unwrap_or(2.5)
                        } else {
                            size_str.parse().unwrap_or(2.5)
                        };
                        i += 1;
                    }
                }
                "--vectors" => {
                    if i + 1 < args.len() {
                        num_vectors = args[i + 1].parse().unwrap_or(0);
                        i += 1;
                    }
                }
                "--dimensions" => {
                    if i + 1 < args.len() {
                        dimensions = args[i + 1].parse().unwrap_or(0);
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        if num_vectors == 0 || dimensions == 0 {
            let target_bytes = (target_size_gb * 1024.0 * 1024.0 * 1024.0) as usize;
            let bytes_per_float = 4; // Float32

            if num_vectors == 0 && dimensions == 0 {
                dimensions = 768; // Common for sentence transformers
                num_vectors = target_bytes / (dimensions * bytes_per_float);
            } else if num_vectors == 0 {
                num_vectors = target_bytes / (dimensions * bytes_per_float);
            } else if dimensions == 0 {
                dimensions = target_bytes / (num_vectors * bytes_per_float);
            }
        }

        Self {
            num_vectors,
            dimensions,
            data_type: DataType::Float32,
            target_size_gb,
        }
    }

    fn estimated_size_gb(&self) -> f32 {
        let vector_data_bytes = self.num_vectors * self.dimensions * 4; // Float32
        let overhead_bytes = vector_data_bytes / 100; // ~1% overhead estimate
        (vector_data_bytes + overhead_bytes) as f32 / (1024.0 * 1024.0 * 1024.0)
    }
}

fn main() -> Result<()> {
    println!("=== MVF Large Dataset Example ===\n");

    let config = DatasetConfig::from_args();

    println!("Configuration:");
    println!("  Vectors: {:?}", config.num_vectors);
    println!("  Dimensions: {:?}", config.dimensions);
    println!("  Data type: {:?}", config.data_type);
    println!("  Estimated size: {:.2} GB", config.estimated_size_gb());
    println!("  Target size: {:.2} GB", config.target_size_gb);

    if config.estimated_size_gb() > 5.0 {
        println!(
            "\nWARNING: This will create a {:.1}GB file!",
            config.estimated_size_gb()
        );
        println!("Press Ctrl+C within 5 seconds to cancel...");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    let temp_file = create_large_dataset(&config)?;
    validate_and_benchmark(&temp_file, &config)?;

    Ok(())
}

fn create_large_dataset(config: &DatasetConfig) -> Result<std::path::PathBuf> {
    println!(
        "\nGenerating {} vectors with {} dimensions...",
        config.num_vectors, config.dimensions
    );

    let temp_file = std::env::temp_dir().join(format!(
        "large_dataset_{:.1}gb.mvf",
        config.estimated_size_gb()
    ));

    let total_start = Instant::now();
    let mut builder = MvfBuilder::new();

    builder.add_vector_space(
        "large_embeddings",
        config.dimensions as u32,
        VectorType::Dense,
        DistanceMetric::Cosine,
        config.data_type,
    );

    // Add metadata for tracking batch access
    #[derive(Copy, Clone)]
    struct BatchMetadata(u32);

    impl From<BatchMetadata> for Vec<u8> {
        fn from(val: BatchMetadata) -> Self {
            val.0.to_le_bytes().to_vec()
        }
    }

    let chunk_size = 10_000; // Generate 10k vectors at a time
    let mut vectors_added = 0;
    let mut metadata_batch = Vec::new();
    let generation_start = Instant::now();

    while vectors_added < config.num_vectors {
        let current_chunk_size = (config.num_vectors - vectors_added).min(chunk_size);

        let chunk_start = Instant::now();
        let chunk_vectors =
            generate_vector_chunk(vectors_added, current_chunk_size, config.dimensions);
        let generation_time = chunk_start.elapsed();

        let add_start = Instant::now();
        builder.add_vectors("large_embeddings", &chunk_vectors)?;
        let add_time = add_start.elapsed();

        let batch_id = vectors_added / chunk_size;
        for _ in 0..current_chunk_size {
            metadata_batch.push(BatchMetadata(batch_id as u32));
        }

        vectors_added += current_chunk_size;

        let progress = (vectors_added as f32 / config.num_vectors as f32) * 100.0;
        println!(
            "  Progress: {:.1}% ({:?}/{:?}) - Gen: {:.1}ms, Add: {:.1}ms",
            progress,
            vectors_added,
            config.num_vectors,
            generation_time.as_millis(),
            add_time.as_millis()
        );
    }

    // Add all metadata at once
    builder.add_metadata_column("batch_id", DataType::UInt32, &metadata_batch)?;

    let total_generation_time = generation_start.elapsed();
    println!(
        "  Total generation: {:.2}s ({:.0} vectors/sec)",
        total_generation_time.as_secs_f32(),
        config.num_vectors as f32 / total_generation_time.as_secs_f32()
    );

    println!("\nBuilding MVF structure...");
    let build_start = Instant::now();
    let built_mvf = builder.build();
    let build_time = build_start.elapsed();
    println!("  Built in {:.2}s", build_time.as_secs_f32());

    println!("\nWriting to file: {:?}", temp_file);
    let write_start = Instant::now();
    built_mvf.save(&temp_file)?;
    let write_time = write_start.elapsed();

    let file_size = std::fs::metadata(&temp_file)?.len();
    let file_size_gb = file_size as f32 / (1024.0 * 1024.0 * 1024.0);

    println!("  Written in {:.2}s", write_time.as_secs_f32());
    println!(
        "  File size: {:.2} GB ({:?} bytes)",
        file_size_gb, file_size
    );
    println!(
        "  Write throughput: {:.1} MB/s",
        file_size as f32 / write_time.as_secs_f32() / (1024.0 * 1024.0)
    );

    let total_time = total_start.elapsed();
    println!(
        "  Total creation time: {:.2}s ({:.1} minutes)",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / 60.0
    );

    Ok(temp_file)
}

fn validate_and_benchmark(temp_file: &std::path::Path, config: &DatasetConfig) -> Result<()> {
    println!("\nValidating file integrity...");
    let read_start = Instant::now();
    let reader = MvfReader::open(temp_file)?;
    let read_time = read_start.elapsed();
    println!("  Opened in {:.2}ms", read_time.as_millis());

    reader.validate()?;
    println!("  Structure validation passed");

    let space = reader.vector_space("large_embeddings")?;
    println!(
        "  Verified: {} vectors, {} dimensions",
        space.total_vectors(),
        space.dimension()
    );

    if reader.has_metadata() {
        let metadata_columns = reader.metadata_column_names();
        println!("  Metadata columns: {:?}", metadata_columns);
    }

    println!("\nPerformance benchmarks...");

    benchmark_random_access(&space, 1000)?;

    benchmark_sequential_access(&space, 100_000.min(space.total_vectors()))?;

    benchmark_batch_access(&space, 10_000)?;

    analyze_memory_usage(std::fs::metadata(temp_file)?.len())?;

    let file_size_gb = std::fs::metadata(temp_file)?.len() as f32 / (1024.0 * 1024.0 * 1024.0);
    println!("\n=== Summary ===");
    println!("File created: {:?}", temp_file);
    println!("File size: {:.2} GB", file_size_gb);
    println!("Vectors: {:?}", config.num_vectors);
    println!("Dimensions: {:?}", config.dimensions);

    println!("\nKeep the file? It's {:.2} GB. (y/n)", file_size_gb);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();

    if !input.trim().to_lowercase().starts_with('y') {
        std::fs::remove_file(temp_file).ok();
        println!("File deleted.");
    } else {
        println!("File kept at: {:?}", temp_file);
    }

    Ok(())
}

fn generate_vector_chunk(start_idx: usize, count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let vector_idx = start_idx + i;
            (0..dimensions)
                .map(|dim| {
                    // Generate realistic embedding-like data
                    let base = vector_idx as f32 * 0.1 + dim as f32 * 0.01;
                    let noise = ((vector_idx + dim) as f32 * 12345.0).sin() * 0.1;
                    let trend = (dim as f32 / dimensions as f32 - 0.5) * 2.0; // -1 to 1

                    // Normalize to reasonable embedding range
                    (base.sin() + noise + trend * 0.1).tanh()
                })
                .collect()
        })
        .collect()
}

fn benchmark_random_access(
    space: &metrovector::vectors::vector_space::VectorSpace,
    samples: usize,
) -> Result<()> {
    use rand::Rng;

    let mut rng = rand::rng();
    let total_vectors = space.total_vectors();

    let start = Instant::now();
    let mut total_elements = 0;

    for i in 0..samples {
        let random_idx = rng.random_range(0..total_vectors);
        let vector = space.get_vector(random_idx)?;
        let data = vector.as_f32()?;
        total_elements += data.len();

        if i % (samples / 10) == 0 {
            let progress = (i as f32 / samples as f32) * 100.0;
            print!("\r  Random access progress: {:.1}%", progress);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\r  Random access: {} samples in {:.2}s",
        samples,
        elapsed.as_secs_f32()
    );
    println!(
        "    Avg time per access: {:.2}μs",
        elapsed.as_micros() as f32 / samples as f32
    );
    println!("    Elements processed: {:?}", total_elements);

    Ok(())
}

fn benchmark_sequential_access(
    space: &metrovector::vectors::vector_space::VectorSpace,
    max_vectors: u64,
) -> Result<()> {
    let vectors_to_process = space.total_vectors().min(max_vectors);

    let start = Instant::now();
    let mut checksum = 0.0f64; // Prevent dead code elimination

    for i in 0..vectors_to_process {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;
        checksum += data[0] as f64; // <- This forces data access

        if i % (vectors_to_process / 20) == 0 {
            let progress = (i as f32 / vectors_to_process as f32) * 100.0;
            print!("\r  Sequential progress: {:.1}%", progress);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\r  Sequential access: {:?} vectors in {:.2}s",
        vectors_to_process,
        elapsed.as_secs_f32()
    );
    println!(
        "    Throughput: {:.0} vectors/sec",
        vectors_to_process as f32 / elapsed.as_secs_f32()
    );
    println!("    Checksum: {:.6} (verification)", checksum);

    Ok(())
}

fn benchmark_batch_access(
    space: &metrovector::vectors::vector_space::VectorSpace,
    batch_size: usize,
) -> Result<()> {
    let indices: Vec<u64> = (0..batch_size.min(space.total_vectors() as usize) as u64).collect();

    let start = Instant::now();
    let batch_vectors = space.get_vectors_batch(&indices)?;
    let batch_time = start.elapsed();

    let mut checksum = 0.0;
    for vector in batch_vectors {
        let data = vector.as_f32()?;
        checksum += data[0];
    }

    println!(
        "  Batch access ({} vectors): {:.2}ms",
        indices.len(),
        batch_time.as_millis()
    );
    println!("    Checksum: {:.6} (verification)", checksum);

    Ok(())
}

fn analyze_memory_usage(file_size: u64) -> Result<()> {
    println!("\n6. Memory analysis:");
    println!(
        "  File size on disk: {:.2} GB",
        file_size as f32 / (1024.0 * 1024.0 * 1024.0)
    );

    // Show OS page size impact
    if let Ok(page_size) = get_page_size() {
        let pages_needed = file_size.div_ceil(page_size);
        println!("  OS page size: {} KB", page_size / 1024);
        println!("  Pages needed: {:?}", pages_needed);

        // This is a rough estimate
        println!(
            "  Memory overhead: ~{:.1} MB (page tables, etc.)",
            pages_needed as f32 * 8.0 / (1024.0 * 1024.0)
        );
    }

    Ok(())
}

fn get_page_size() -> Result<u64> {
    #[cfg(unix)]
    {
        Ok(unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64)
    }
    #[cfg(not(unix))]
    {
        Ok(4096) // Default assumption for non-Unix platforms 
    }
}
