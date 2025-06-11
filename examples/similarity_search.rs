//! Advanced similarity search example
//!
//! Run with: cargo run --example similarity_search

use metrovector::{
    builder::MvfBuilder,
    mvf_fbs::{DataType, DistanceMetric, VectorType},
    reader::MvfReader,
    vectors::vector_space::VectorSpace,
};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

    // Create a dataset with different clusters
    let mut vectors = Vec::new();

    // Cluster 1: around [1, 1, 1, 1]
    for i in 0..20 {
        let noise = (i as f32) * 0.1;
        vectors.push(vec![
            1.0 + noise,
            1.0 - noise,
            1.0 + noise * 0.5,
            1.0 - noise * 0.5,
        ]);
    }

    // Cluster 2: around [5, 5, 5, 5]
    for i in 0..20 {
        let noise = (i as f32) * 0.1;
        vectors.push(vec![
            5.0 + noise,
            5.0 - noise,
            5.0 + noise * 0.5,
            5.0 - noise * 0.5,
        ]);
    }

    // Cluster 3: around [-2, 3, 0, 4]
    for i in 0..20 {
        let noise = (i as f32) * 0.1;
        vectors.push(vec![
            -2.0 + noise,
            3.0 - noise,
            0.0 + noise,
            4.0 - noise * 0.5,
        ]);
    }

    println!(
        "Created dataset with {} vectors in 3 clusters",
        vectors.len()
    );

    // Build MVF file
    let mut builder = MvfBuilder::new();
    builder.add_vector_space(
        "clustered_data",
        4,
        VectorType::Dense,
        DistanceMetric::L2,
        DataType::Float32,
    );

    builder.add_vectors("clustered_data", &vectors)?;
    let built_mvf = builder.build();

    let temp_file = std::env::temp_dir().join("similarity_search.mvf");
    built_mvf.save(&temp_file)?;

    // Load for searching
    let mvf_file = MvfReader::open(&temp_file)?;
    let space = mvf_file.vector_space(mvf_file.vector_space_names().first().unwrap())?;

    // Define queries
    let queries = vec![
        (vec![1.0, 1.0, 1.0, 1.0], "Near cluster 1"),
        (vec![5.0, 5.0, 5.0, 5.0], "Near cluster 2"),
        (vec![-2.0, 3.0, 0.0, 4.0], "Near cluster 3"),
        (vec![0.0, 0.0, 0.0, 0.0], "At origin"),
    ];

    for (query, description) in queries {
        println!("\n=== Query: {} ===", description);
        println!("Query vector: {:?}", query);

        // Find top K similar vectors
        let k = 5;
        let top_k = find_top_k_similar(&space, &query, k)?;

        println!("Top {} most similar vectors:", k);
        for (rank, scored_vec) in top_k.iter().enumerate() {
            println!(
                "  {}. Vector {} (distance: {:.3}): {:?}",
                rank + 1,
                scored_vec.index,
                scored_vec.score,
                scored_vec.vector
            );
        }

        // Analyze which cluster the results belong to
        analyze_clusters(&top_k);
    }

    // Clean up
    std::fs::remove_file(&temp_file).ok();

    Ok(())
}

fn find_top_k_similar(
    space: &VectorSpace,
    query: &[f32],
    k: usize,
) -> Result<Vec<ScoredVector>, Box<dyn std::error::Error>> {
    let mut heap = BinaryHeap::new();

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

        heap.push(ScoredVector {
            index: i,
            score: distance,
            vector: data,
        });

        // Keep only top k
        if heap.len() > k {
            heap.pop();
        }
    }

    // Convert to sorted vector (best first)
    let mut result: Vec<_> = heap.into_iter().collect();
    result.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    Ok(result)
}

fn analyze_clusters(results: &[ScoredVector]) {
    let mut cluster_counts = [0; 3];

    for result in results {
        // Determine cluster based on vector values
        let cluster = if result.vector[0] < 2.0 {
            0 // Cluster 1
        } else if result.vector[0] > 4.0 {
            1 // Cluster 2  
        } else {
            2 // Cluster 3
        };

        cluster_counts[cluster] += 1;
    }

    println!(
        "  Cluster distribution: C1={}, C2={}, C3={}",
        cluster_counts[0], cluster_counts[1], cluster_counts[2]
    );
}
