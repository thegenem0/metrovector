//! large dataset example - creates 2.5GB MVF file by default
//!
//! Run with: cargo run --example large_dataset
//! Or with custom size: cargo run --example large_dataset -- --size 4gb
//! Or with custom params: cargo run --example large_dataset -- --vectors 1000000 --dimensions 768

use metrovector::{
    builder::MvfBuilder,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
    reader::MvfReader,
    vector_space::VectorSpace,
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

        // Parse command line arguments
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

        // Calculate dimensions based on target size if not specified
        if num_vectors == 0 || dimensions == 0 {
            let target_bytes = (target_size_gb * 1024.0 * 1024.0 * 1024.0) as usize;
            let bytes_per_float = 4; // Float32

            if num_vectors == 0 && dimensions == 0 {
                // Modern embedding dimensions: 384, 512, 768, 1024, 1536
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    println!(
        "\n1. Generating {} vectors with {} dimensions...",
        config.num_vectors, config.dimensions
    );

    let chunk_size = 10_000; // Gen 10k vectors at a time
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

    let mut vectors_added = 0;
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

        vectors_added += current_chunk_size;

        let progress = (vectors_added as f32 / config.num_vectors as f32) * 100.0;
        println!(
            "  Progress: {:.1}% ({:?}/{:?}) - Gen: {:.3}ms, Add: {:.3}ms",
            progress,
            vectors_added,
            config.num_vectors,
            generation_time.as_millis(),
            add_time.as_millis()
        );
    }

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

    println!("\nVerifying file integrity...");
    let read_start = Instant::now();
    let mvf_file = MvfReader::open(&temp_file)?;
    let read_time = read_start.elapsed();
    println!("  Opened in {:.2}s", read_time.as_secs_f32());

    let space = mvf_file.vector_space(mvf_file.vector_space_names().first().unwrap())?;
    println!(
        "  Verified: {} vectors, {} dimensions",
        space.total_vectors(),
        space.dimension()
    );

    println!("\nBenchmarking random access...");
    benchmark_random_access(&space, 1000)?;

    println!("\nBenchmarking sequential access...");
    benchmark_sequential_access(&space, 100_000)?;

    println!("\nMemory analysis...");
    analyze_memory_usage(file_size)?;

    let total_time = total_start.elapsed();
    println!("\n=== Summary ===");
    println!(
        "Total time: {:.2}s ({:.1} minutes)",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / 60.0
    );
    println!("File created: {:?}", temp_file);
    println!("File size: {:.2} GB", file_size_gb);
    println!("Vectors: {:?}", config.num_vectors);
    println!("Dimensions: {:?}", config.dimensions);

    println!("\nKeep the file? It's {:.2} GB. (y/n)", file_size_gb);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();

    if !input.trim().to_lowercase().starts_with('y') {
        std::fs::remove_file(&temp_file).ok();
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
                    // Gen realistic embedding-like data
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
    space: &VectorSpace,
    samples: usize,
) -> Result<(), Box<dyn std::error::Error>> {
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
    space: &VectorSpace,
    max_vectors: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let vectors_to_process = space.total_vectors().min(max_vectors);

    let start = Instant::now();

    // Adding a dummy checksum to prevent dead code elimination
    // If we don't read a value out of the vector, the compiler could do any of a few things:
    // 1. Skip the as_f32() call as we don't access the data
    // 2. Skip reading the actual vector data entirely
    // 3. In extreme cases, it might just optimize out the entire loop
    // Either of which would make this benchmark useless
    // This way we force the compiler to leave this code in the binary,
    // and actually use CPU cycles to read the data
    // In a real application, you probably would use the vector data,
    // so this wouldn't be necessary
    let mut checksum = 0.0f64;
    for i in 0..vectors_to_process {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;
        checksum += data[0] as f64; // <-- This forces data access

        if i % (vectors_to_process / 20) == 0 {
            let progress = (i as f32 / vectors_to_process as f32) * 100.0;
            print!("\r  Sequential progress: {:.1}%", progress);
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
    println!("    Checksum: {:.6} (ignore this)", checksum);

    Ok(())
}

fn analyze_memory_usage(file_size: u64) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "  File size on disk: {:.2} GB",
        file_size as f32 / (1024.0 * 1024.0 * 1024.0)
    );

    // Show OS page size impact
    if let Ok(page_size) = get_page_size() {
        let pages_needed = file_size.div_ceil(page_size);
        println!("  OS page size: {} KB", page_size / 1024);
        println!("  Pages needed: {:?}", pages_needed);
    }

    Ok(())
}

fn get_page_size() -> Result<u64, Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        Ok(unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64)
    }
    #[cfg(not(any(unix)))]
    {
        Ok(4096) // Default assumption
    }
}
