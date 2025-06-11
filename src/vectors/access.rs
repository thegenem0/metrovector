/// Optimized access pattern for batch vector retrieval.
///
/// Pre-processes vector indices to optimize memory access patterns and
/// minimize cache misses during batch operations.
///
/// # Examples
///
/// ```
/// use metrovector::vectors::access::AccessPattern;
///
/// let pattern = AccessPattern::new(vec![100, 50, 75, 200]);
/// println!("Sorted indices: {:?}", pattern.indices());
/// println!("Block ranges: {:?}", pattern.block_ranges());
/// ```
pub struct AccessPattern {
    stored_indices: Vec<u64>,
    /// Block ranges as (start_idx, end_idx) pairs in sorted_indices.
    block_ranges: Vec<(usize, usize)>,
}

impl AccessPattern {
    /// Creates a new access pattern from unsorted indices.
    ///
    /// Automatically sorts and deduplicates indices, then groups them
    /// by storage blocks for optimal access patterns.
    ///
    /// # Arguments
    /// * `indices` - Vector indices to access (will be sorted and deduplicated)
    pub fn new(mut indices: Vec<u64>) -> Self {
        indices.sort_unstable();
        indices.dedup();

        /// Group indices by storage blocks (assuming 1024 vectors per block)
        const VECTORS_PER_BLOCK: u64 = 1024;

        let mut block_ranges = Vec::new();
        if !indices.is_empty() {
            let mut start_idx = 0;
            let mut current_block = indices[0] / VECTORS_PER_BLOCK;

            for (i, &idx) in indices.iter().enumerate().skip(1) {
                let block = idx / VECTORS_PER_BLOCK;
                if block != current_block {
                    block_ranges.push((start_idx, i));
                    start_idx = i;
                    current_block = block;
                }
            }
            block_ranges.push((start_idx, indices.len()));
        }

        Self {
            stored_indices: indices,
            block_ranges,
        }
    }

    /// Returns the sorted and deduplicated indices.
    pub fn indices(&self) -> &[u64] {
        &self.stored_indices
    }

    /// Returns block ranges for optimized batch access.
    ///
    /// Each range represents indices that belong to the same storage block,
    /// allowing for more efficient memory access patterns.
    pub fn block_ranges(&self) -> &[(usize, usize)] {
        &self.block_ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_pattern_empty() {
        let pattern = AccessPattern::new(vec![]);
        assert!(pattern.indices().is_empty());
        assert!(pattern.block_ranges().is_empty());
    }

    #[test]
    fn test_access_pattern_single_element() {
        let pattern = AccessPattern::new(vec![42]);
        assert_eq!(pattern.indices(), &[42]);
        assert_eq!(pattern.block_ranges().len(), 1);
        assert_eq!(pattern.block_ranges()[0], (0, 1));
    }

    #[test]
    fn test_access_pattern_duplicates() {
        let indices = vec![5, 1, 3, 1, 7, 3, 2, 5];
        let pattern = AccessPattern::new(indices);

        // Should be deduplicated and sorted
        assert_eq!(pattern.indices(), &[1, 2, 3, 5, 7]);
        assert!(!pattern.block_ranges().is_empty());
    }

    #[test]
    fn test_access_pattern_multiple_blocks() {
        // indices that span multiple blocks (assuming 1024 vectors per block)
        let indices = vec![10, 1025, 2050, 15, 1030];
        let pattern = AccessPattern::new(indices);

        assert_eq!(pattern.indices(), &[10, 15, 1025, 1030, 2050]);
        assert_eq!(pattern.block_ranges().len(), 3); // should be 3 different blocks
    }

    #[test]
    fn test_access_pattern_large_gaps() {
        // Test with large gaps between indices
        let indices = vec![0, 10000, 20000, 30000];
        let pattern = AccessPattern::new(indices);

        assert_eq!(pattern.indices(), &[0, 10000, 20000, 30000]);
        // Should create multiple block ranges due to large gaps
        assert!(pattern.block_ranges().len() > 1);
    }

    #[test]
    fn test_access_pattern_reverse_order() {
        // Test with indices in reverse order
        let indices = vec![100, 50, 75, 25];
        let pattern = AccessPattern::new(indices);

        // Should be sorted
        assert_eq!(pattern.indices(), &[25, 50, 75, 100]);
    }

    #[test]
    fn test_access_pattern_same_block() {
        // Test indices all in same block (< 1024)
        let indices = vec![10, 50, 100, 500];
        let pattern = AccessPattern::new(indices);

        assert_eq!(pattern.block_ranges().len(), 1);
        assert_eq!(pattern.block_ranges()[0], (0, 4));
    }

    #[test]
    fn test_access_pattern_block_boundaries() {
        // Test indices right at block boundaries
        let indices = vec![1023, 1024, 1025, 2047, 2048];
        let pattern = AccessPattern::new(indices);

        // Should create separate ranges for different blocks
        assert!(pattern.block_ranges().len() >= 2);
    }
}
