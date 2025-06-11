//! Advanced similarity search example
//!
//! Run with: cargo run --example similarity_search

use metrovector::{
    builder::MvfBuilder,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
    reader::MvfReader,
};
use std::collections::BinaryHeap;
use std::{cmp::Ordering, time::Instant};

#[derive(PartialEq)]
struct ScoredVector {
    index: u64,
    score: f32,
    vector: Vec<f32>,
}

impl Eq for ScoredVector {}

impl Ord for ScoredVector {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (smallest distance first)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScoredVector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MVF Similarity Search Example ===\n");

    let dataset = create_clustered_dataset()?;
    println!(
        "Created dataset with {} vectors in 3 clusters",
        dataset.vectors.len()
    );

    let temp_file = build_search_index(&dataset)?;
    println!("Built search index: {:?}", temp_file);

    let reader = MvfReader::open(&temp_file)?;
    reader.validate()?;

    let space = reader.vector_space("clustered_data")?;
    println!("Loaded {} vectors for searching", space.total_vectors());

    example_searches(&space, &dataset)?;

    benchmark_search_performance(&space)?;

    std::fs::remove_file(&temp_file).ok();

    Ok(())
}

#[allow(dead_code)]
struct Dataset {
    vectors: Vec<Vec<f32>>,
    cluster_labels: Vec<usize>,
    cluster_centers: Vec<Vec<f32>>,
}

fn create_clustered_dataset() -> Result<Dataset, Box<dyn std::error::Error>> {
    let mut vectors = Vec::new();
    let mut cluster_labels = Vec::new();

    let cluster_centers = vec![
        vec![1.0, 1.0, 1.0, 1.0],  // Cluster 0
        vec![5.0, 5.0, 5.0, 5.0],  // Cluster 1
        vec![-2.0, 3.0, 0.0, 4.0], // Cluster 2
    ];

    // Gen vectors around each cluster center
    for (cluster_id, center) in cluster_centers.iter().enumerate() {
        for i in 0..30 {
            let noise_factor = (i as f32) * 0.05;
            let vector = center
                .iter()
                .enumerate()
                .map(|(dim, &val)| {
                    let noise = ((i + dim) as f32 * 12345.0).sin() * noise_factor;
                    val + noise
                })
                .collect();

            vectors.push(vector);
            cluster_labels.push(cluster_id);
        }
    }

    Ok(Dataset {
        vectors,
        cluster_labels,
        cluster_centers,
    })
}

fn build_search_index(dataset: &Dataset) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let mut builder = MvfBuilder::new();

    builder.add_vector_space(
        "clustered_data",
        4,
        VectorType::Dense,
        DistanceMetric::L2,
        DataType::Float32,
    );

    builder.add_vectors("clustered_data", &dataset.vectors)?;

    #[derive(Copy, Clone)]
    struct ClusterMetadata(u32);

    impl From<ClusterMetadata> for Vec<u8> {
        fn from(val: ClusterMetadata) -> Self {
            val.0.to_le_bytes().to_vec()
        }
    }

    let cluster_metadata: Vec<ClusterMetadata> = dataset
        .cluster_labels
        .iter()
        .map(|&label| ClusterMetadata(label as u32))
        .collect();

    builder.add_metadata_column("cluster_id", DataType::UInt32, &cluster_metadata)?;

    let built_mvf = builder.build();
    let temp_file = std::env::temp_dir().join("similarity_search.mvf");
    built_mvf.save(&temp_file)?;

    Ok(temp_file)
}

fn example_searches(
    space: &metrovector::vectors::vector_space::VectorSpace,
    dataset: &Dataset,
) -> Result<(), Box<dyn std::error::Error>> {
    let queries = vec![
        (vec![1.0, 1.0, 1.0, 1.0], "Near cluster 0", 0),
        (vec![5.0, 5.0, 5.0, 5.0], "Near cluster 1", 1),
        (vec![-2.0, 3.0, 0.0, 4.0], "Near cluster 2", 2),
        (vec![0.0, 0.0, 0.0, 0.0], "At origin", 999), // No expected cluster
    ];

    for (query, description, expected_cluster) in queries {
        println!("\n=== Query: {} ===", description);
        println!("Query vector: {:?}", query);

        // Sequential search with top-k
        let k = 5;
        let start = Instant::now();
        let top_k = find_top_k_similar(space, &query, k)?;
        let search_time = start.elapsed();

        println!(
            "Top {} most similar vectors (found in {:.2}ms):",
            k,
            search_time.as_millis()
        );
        for (rank, scored_vec) in top_k.iter().enumerate() {
            let actual_cluster = dataset.cluster_labels[scored_vec.index as usize];
            println!(
                "  {}. Vector {} (distance: {:.3}, cluster: {}): {:?}",
                rank + 1,
                scored_vec.index,
                scored_vec.score,
                actual_cluster,
                scored_vec.vector
            );
        }

        analyze_cluster_distribution(&top_k, dataset, expected_cluster);

        // Batch search (more efficient for multiple queries)
        if query[0] != 0.0 {
            // Skip origin query
            let similar_indices: Vec<u64> = top_k.iter().map(|sv| sv.index).collect();
            let start = Instant::now();
            let batch_vectors = space.get_vectors_batch(&similar_indices)?;
            let batch_time = start.elapsed();

            println!(
                "  Batch retrieval of {} vectors: {:.2}ms",
                batch_vectors.len(),
                batch_time.as_millis()
            );
        }
    }

    Ok(())
}

fn find_top_k_similar(
    space: &metrovector::vectors::vector_space::VectorSpace,
    query: &[f32],
    k: usize,
) -> Result<Vec<ScoredVector>, Box<dyn std::error::Error>> {
    let mut heap = BinaryHeap::new();

    // Streaming for large datasets
    for chunk_result in space.stream_vectors(0, 1000) {
        let chunk = chunk_result?;

        for (local_idx, vector) in chunk.iter().enumerate() {
            let data = vector.as_f32()?;
            let global_idx = local_idx as u64; // In a real implementation, you'd track the actual index

            // Euclidean distance
            let distance: f32 = query
                .iter()
                .zip(data.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            heap.push(ScoredVector {
                index: global_idx,
                score: distance,
                vector: data,
            });

            // Keep only top k
            if heap.len() > k {
                heap.pop();
            }
        }
    }

    let mut all_results = Vec::new();

    for i in 0..space.total_vectors() {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;

        let distance: f32 = query
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        all_results.push(ScoredVector {
            index: i,
            score: distance,
            vector: data,
        });
    }

    // Sort and take top k
    all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    all_results.truncate(k);

    Ok(all_results)
}

fn analyze_cluster_distribution(
    results: &[ScoredVector],
    dataset: &Dataset,
    expected_cluster: usize,
) {
    let mut cluster_counts = [0; 3];

    for result in results {
        if (result.index as usize) < dataset.cluster_labels.len() {
            let cluster = dataset.cluster_labels[result.index as usize];
            cluster_counts[cluster] += 1;
        }
    }

    println!(
        "  Cluster distribution: C0={}, C1={}, C2={}",
        cluster_counts[0], cluster_counts[1], cluster_counts[2]
    );

    if expected_cluster < 3 {
        let precision = cluster_counts[expected_cluster] as f32 / results.len() as f32;
        println!(
            "  Precision for expected cluster {}: {:.1}%",
            expected_cluster,
            precision * 100.0
        );
    }
}

fn benchmark_search_performance(
    space: &metrovector::vectors::vector_space::VectorSpace,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Performance Benchmarks ===");

    // Individual vector access
    let start = Instant::now();
    let mut checksum = 0.0;
    for i in 0..100.min(space.total_vectors()) {
        let vector = space.get_vector(i)?;
        let data = vector.as_f32()?;
        checksum += data[0]; // Prevent optimization
    }
    let individual_time = start.elapsed();

    println!(
        "Individual access (100 vectors): {:.2}μs (checksum: {:.2})",
        individual_time.as_micros(),
        checksum
    );

    // Batch access
    let indices: Vec<u64> = (0..100.min(space.total_vectors())).collect();
    let start = Instant::now();
    let batch_vectors = space.get_vectors_batch(&indices)?;
    let mut batch_checksum = 0.0;
    for vector in batch_vectors {
        let data = vector.as_f32()?;
        batch_checksum += data[0];
    }
    let batch_time = start.elapsed();

    println!(
        "Batch access (100 vectors): {:.2}μs (checksum: {:.2})",
        batch_time.as_micros(),
        batch_checksum
    );

    // Range mapping
    let vector_count = 100.min(space.total_vectors());
    let start = Instant::now();
    let range_slice = space.map_vector_range(0, vector_count)?;
    let range_time = start.elapsed();

    for _vector in range_slice.iter_elements::<f32>().flatten() {
        // Do something with the vector
    }

    println!(
        "Range mapping ({} vectors): {:.2}μs",
        vector_count,
        range_time.as_micros()
    );

    // This becomes more relevant for large datasets
    // with few vectors, it's unlikely to make a difference
    if batch_time < individual_time {
        let speedup = individual_time.as_secs_f32() / batch_time.as_secs_f32();
        println!(
            "Batch access is {:.1}x faster than individual access",
            speedup
        );
    }

    Ok(())
}
