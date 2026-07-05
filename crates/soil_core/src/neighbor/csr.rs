//! CSR neighbor-pair iteration.

use super::Neighbor;

/// Iterator over `(i, j)` neighbor pairs from the CSR neighbor list.
///
/// Created by [`Neighbor::pairs()`]. Yields each pair exactly once, with `i` as a
/// local atom index and `j` as either local or ghost. Uses `unsafe` index access
/// for performance in the inner loop (validated by CSR construction invariants).
pub struct PairIter<'a> {
    offsets: &'a [u32],
    indices: &'a [u32],
    nlocal: usize,
    /// Current local atom index (row in the CSR).
    i: usize,
    /// Current position within `indices` for atom `i`'s neighbors.
    k: usize,
    /// End position within `indices` for atom `i`'s neighbors.
    end: usize,
}

impl<'a> Iterator for PairIter<'a> {
    type Item = (usize, usize);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.k >= self.end {
            self.i += 1;
            if self.i >= self.nlocal {
                return None;
            }
            // SAFETY: self.i < self.nlocal <= offsets.len() - 1, so self.i and self.i + 1 are in bounds.
            unsafe {
                self.k = *self.offsets.get_unchecked(self.i) as usize;
                self.end = *self.offsets.get_unchecked(self.i + 1) as usize;
            }
        }
        // SAFETY: self.k < self.end <= indices.len() (from CSR construction).
        let j = unsafe { *self.indices.get_unchecked(self.k) } as usize;
        self.k += 1;
        Some((self.i, j))
    }
}

impl Neighbor {
    /// Iterate over all (i, j) pairs from the CSR neighbor list.
    /// `nlocal` is the number of local atoms (only local atoms own neighbor lists).
    pub fn pairs(&self, nlocal: usize) -> PairIter<'_> {
        let end = if nlocal > 0 {
            self.neighbor_offsets[1] as usize
        } else {
            0
        };
        PairIter {
            offsets: &self.neighbor_offsets,
            indices: &self.neighbor_indices,
            nlocal,
            i: 0,
            k: if nlocal > 0 {
                self.neighbor_offsets[0] as usize
            } else {
                0
            },
            end,
        }
    }
}
