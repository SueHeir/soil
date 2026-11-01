//! Global tag → current row index lookup, shared by every tier that addresses
//! persistent partners by tag (DEM bonds, peridynamic families, ...).
//!
//! Local rows and the ghost set only change at a neighbor-list build (atom
//! sorting, migration and border exchange all happen then), so the map is
//! rebuilt lazily when [`Neighbor::build_count`] advances instead of on every
//! step. The lookup is a dense `Vec` indexed by tag when tags are compact
//! (the normal case) and a hash map otherwise.
//!
//! A tag can be present several times, as the owned atom and as one or more
//! periodic ghost images. The lookup returns the **lowest row**, which is the
//! owned copy when this rank owns the atom. Callers that need the geometric
//! separation should still pass it through
//! [`Domain::minimum_image`](crate::Domain::minimum_image), because the copy
//! found is not necessarily the nearest image.

use std::collections::HashMap;

use crate::{Atom, Neighbor};

const ABSENT: u32 = u32::MAX;

/// Lazily rebuilt global-tag → row-index map over local and ghost atoms.
///
/// Registered as a resource by [`NeighborPlugin`](crate::NeighborPlugin).
/// Systems that need it take `ResMut<TagIndex>` and call
/// [`refresh`](Self::refresh) once before looking tags up:
///
/// ```rust,ignore
/// fn my_bond_force(atoms: Res<Atom>, neighbor: Res<Neighbor>, mut tags: ResMut<TagIndex>) {
///     tags.refresh(&atoms, &neighbor);
///     if let Some(j) = tags.get(partner_tag) { /* ... */ }
/// }
/// ```
#[derive(Default)]
pub struct TagIndex {
    /// `(build_count, nlocal, len)` the map was built for.
    key: Option<(u64, usize, usize)>,
    dense: Vec<u32>,
    sparse: HashMap<u32, u32>,
    use_dense: bool,
}

impl TagIndex {
    /// An empty index; the first [`refresh`](Self::refresh) builds it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild the map if the neighbor list has been rebuilt (or the atom
    /// counts changed) since the last build. Returns `true` when it rebuilt.
    pub fn refresh(&mut self, atoms: &Atom, neighbor: &Neighbor) -> bool {
        let key = (neighbor.build_count, atoms.nlocal() as usize, atoms.len());
        if self.key == Some(key) {
            return false;
        }
        self.rebuild(atoms);
        self.key = Some(key);
        true
    }

    /// Unconditionally rebuild from the current rows.
    pub fn rebuild(&mut self, atoms: &Atom) {
        let n = atoms.len();
        let max_tag = (0..n).map(|i| atoms.tag[i]).max().unwrap_or(0) as usize;
        // Dense when tags are compact enough that the table stays within a small
        // multiple of the row count; otherwise fall back to hashing.
        self.use_dense = max_tag <= 4 * n + 1024;
        self.dense.clear();
        self.sparse.clear();
        if self.use_dense {
            self.dense.resize(max_tag + 1, ABSENT);
            // Reverse order so the lowest row (the owned copy) wins.
            for i in (0..n).rev() {
                self.dense[atoms.tag[i] as usize] = i as u32;
            }
        } else {
            self.sparse.reserve(n);
            for i in (0..n).rev() {
                self.sparse.insert(atoms.tag[i], i as u32);
            }
        }
        self.key = None;
    }

    /// Row index of `tag` (owned copy preferred), if present on this rank.
    #[inline]
    pub fn get(&self, tag: u32) -> Option<usize> {
        if self.use_dense {
            match self.dense.get(tag as usize) {
                Some(&i) if i != ABSENT => Some(i as usize),
                _ => None,
            }
        } else {
            self.sparse.get(&tag).map(|&i| i as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atoms_with_tags(tags: &[u32], nlocal: u32) -> Atom {
        let mut atoms = Atom::new();
        for (i, &tag) in tags.iter().enumerate() {
            atoms.push_test_atom(tag, [i as f64, 0.0, 0.0], 0.5, 1.0);
            atoms.is_ghost[i] = i as u32 >= nlocal;
        }
        let nghost = tags.len() as u32 - nlocal;
        // SAFETY: no extensions are registered and rows are ordered as given.
        unsafe { atoms.set_layout_unchecked(nlocal, nghost, nlocal as u64) };
        atoms
    }

    #[test]
    fn owned_copy_wins_over_periodic_ghost_image() {
        // Tag 7 is local at row 1 and a ghost image at row 3.
        let atoms = atoms_with_tags(&[5, 7, 9, 7], 3);
        let mut idx = TagIndex::new();
        idx.rebuild(&atoms);
        assert_eq!(idx.get(7), Some(1));
        assert_eq!(idx.get(5), Some(0));
        assert_eq!(idx.get(9), Some(2));
        assert_eq!(idx.get(8), None);
        assert_eq!(idx.get(1_000_000), None);
    }

    #[test]
    fn sparse_tags_use_the_hash_path() {
        let atoms = atoms_with_tags(&[3, 4_000_000_000, 12], 3);
        let mut idx = TagIndex::new();
        idx.rebuild(&atoms);
        assert!(!idx.use_dense);
        assert_eq!(idx.get(4_000_000_000), Some(1));
        assert_eq!(idx.get(12), Some(2));
        assert_eq!(idx.get(13), None);
    }

    #[test]
    fn refresh_only_rebuilds_after_a_neighbor_build() {
        let atoms = atoms_with_tags(&[1, 2], 2);
        let mut neighbor = Neighbor::new();
        let mut idx = TagIndex::new();
        assert!(idx.refresh(&atoms, &neighbor));
        assert!(!idx.refresh(&atoms, &neighbor));
        neighbor.build_count += 1;
        assert!(idx.refresh(&atoms, &neighbor));
    }
}
