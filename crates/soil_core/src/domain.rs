//! Simulation box geometry, domain decomposition, and periodic boundary conditions.
//!
//! This module provides:
//! - [`Domain`]: runtime state for box boundaries, sub-domain bounds, and periodicity
//! - [`DomainConfig`]: TOML `[domain]` section with boundary types and box extents
//! - [`decompose_domain`]: uniform Cartesian grid decomposition across MPI ranks
//! - [`DomainPlugin`]: registers setup, PBC wrapping, and shrink-wrap systems

use std::fmt;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    Atom, AtomDataRegistry, CommBackend, CommResource, CommState, Config, ParticleSimScheduleSet,
    Real, ScheduleSetupSet,
};
use grass_scheduler::prelude::CurrentState;

/// Type-level label for [`domain_read_input`].
pub struct DomainReadInputSystem;
impl SystemLabel for DomainReadInputSystem {
    const NAME: &'static str = "domain_read_input";
}
/// Stable scheduler key for [`domain_read_input`].
pub const DOMAIN_READ_INPUT: SystemKey<DomainReadInputSystem> = SystemKey::new();

/// Type-level label for [`shrink_wrap`].
pub struct ShrinkWrapSystem;
impl SystemLabel for ShrinkWrapSystem {
    const NAME: &'static str = "shrink_wrap";
}
/// Stable scheduler key for [`shrink_wrap`].
pub const SHRINK_WRAP: SystemKey<ShrinkWrapSystem> = SystemKey::new();

/// Type-level label for [`pbc`].
pub struct PbcSystem;
impl SystemLabel for PbcSystem {
    const NAME: &'static str = "pbc";
}
/// Stable scheduler key for [`pbc`].
pub const PBC: SystemKey<PbcSystem> = SystemKey::new();

fn default_one_f64() -> f64 {
    1.0
}
fn default_periodic() -> BoundaryType {
    BoundaryType::Periodic
}

/// Boundary condition type for a single axis.
///
/// - `Periodic`: atoms that exit one side re-enter from the opposite side.
/// - `Fixed`: the box boundary is static; atoms leaving are removed.
/// - `ShrinkWrap`: the box boundary automatically adjusts each step to
///   encompass all atoms (plus padding), like LAMMPS `boundary s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoundaryType {
    /// Atoms exiting one face re-enter from the opposite face.
    Periodic,
    /// Static box face; atoms that leave the box are removed.
    Fixed,
    /// Box face auto-adjusts each step to wrap all atoms (plus padding).
    ShrinkWrap,
}

impl Default for BoundaryType {
    fn default() -> Self {
        BoundaryType::Periodic
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[domain]` — simulation box boundaries and boundary types.
pub struct DomainConfig {
    /// Lower x boundary of the simulation box (simulation units, e.g. meters for DEM).
    #[serde(default, alias = "x_lo")]
    pub x_low: f64,
    /// Upper x boundary of the simulation box (simulation units).
    #[serde(default = "default_one_f64", alias = "x_hi")]
    pub x_high: f64,
    /// Lower y boundary of the simulation box (simulation units).
    #[serde(default, alias = "y_lo")]
    pub y_low: f64,
    /// Upper y boundary of the simulation box (simulation units).
    #[serde(default = "default_one_f64", alias = "y_hi")]
    pub y_high: f64,
    /// Lower z boundary of the simulation box (simulation units).
    #[serde(default, alias = "z_lo")]
    pub z_low: f64,
    /// Upper z boundary of the simulation box (simulation units).
    #[serde(default = "default_one_f64", alias = "z_hi")]
    pub z_high: f64,
    /// Boundary type for the x axis: "periodic", "fixed", or "shrink-wrap".
    #[serde(default = "default_periodic")]
    pub boundary_x: BoundaryType,
    /// Boundary type for the y axis: "periodic", "fixed", or "shrink-wrap".
    #[serde(default = "default_periodic")]
    pub boundary_y: BoundaryType,
    /// Boundary type for the z axis: "periodic", "fixed", or "shrink-wrap".
    #[serde(default = "default_periodic")]
    pub boundary_z: BoundaryType,
    /// Padding added on each side of shrink-wrap boundaries (simulation units).
    /// Defaults to 0.0 (auto-computed from max particle cutoff radius).
    #[serde(default)]
    pub shrink_wrap_padding: f64,
}

impl DomainConfig {
    /// Return the boundary types as a 3-element array `[x, y, z]`.
    /// Box edge lengths `[x_high - x_low, y_high - y_low, z_high - z_low]`.
    pub fn box_lengths(&self) -> [f64; 3] {
        [
            self.x_high - self.x_low,
            self.y_high - self.y_low,
            self.z_high - self.z_low,
        ]
    }

    pub fn boundary_types(&self) -> [BoundaryType; 3] {
        [self.boundary_x, self.boundary_y, self.boundary_z]
    }
}

impl Default for DomainConfig {
    fn default() -> Self {
        DomainConfig {
            x_low: 0.0,
            x_high: 1.0,
            y_low: 0.0,
            y_high: 1.0,
            z_low: 0.0,
            z_high: 1.0,
            boundary_x: BoundaryType::Periodic,
            boundary_y: BoundaryType::Periodic,
            boundary_z: BoundaryType::Periodic,
            shrink_wrap_padding: 0.0,
        }
    }
}

/// Validation failures for [`DomainConfig`] that depend on the active runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DomainValidationError {
    /// Shrink-wrap changes global box bounds dynamically; SOIL does not yet
    /// support recomputing MPI sub-domain bounds from those changes.
    ShrinkWrapWithMpi {
        /// Number of MPI ranks in the active communicator.
        ranks: i32,
    },
}

impl fmt::Display for DomainValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DomainValidationError::ShrinkWrapWithMpi { ranks } => write!(
                f,
                "[domain] shrink-wrap boundaries are not supported with MPI rank count {ranks}; \
                 use fixed or periodic boundaries, or run with one MPI rank"
            ),
        }
    }
}

impl std::error::Error for DomainValidationError {}

/// Simulation box geometry: global boundaries, sub-domain bounds, and periodicity.
pub struct Domain {
    /// Global lower box corner `[x, y, z]`.
    pub boundaries_low: [f64; 3],
    /// Global upper box corner `[x, y, z]`.
    pub boundaries_high: [f64; 3],
    /// Lower corner of this rank's sub-domain `[x, y, z]`.
    pub sub_domain_low: [f64; 3],
    /// Upper corner of this rank's sub-domain `[x, y, z]`.
    pub sub_domain_high: [f64; 3],
    /// Per-axis extent of this rank's sub-domain (`sub_domain_high - sub_domain_low`).
    pub sub_length: [f64; 3],
    /// Global box volume.
    pub volume: f64,
    /// Per-axis extent of the global box (`boundaries_high - boundaries_low`).
    pub size: [f64; 3],
    /// Per-axis boundary type (single source of truth).
    pub boundary_type: [BoundaryType; 3],
    /// Padding for shrink-wrap boundaries. If 0, uses ghost_cutoff as padding.
    pub shrink_wrap_padding: f64,
    /// Set to true whenever shrink-wrap updates domain bounds.
    /// Cleared by the neighbor system after it recomputes bins.
    pub bounds_changed: bool,
    /// Ghost atom communication cutoff. 0 = use per-atom skin * 4.0 (DEM default).
    pub ghost_cutoff: f64,
    /// Triclinic tilt factors `[xy, xz, yz]` of the box edge vectors:
    /// `a=(Lx,0,0)`, `b=(xy,Ly,0)`, `c=(xz,yz,Lz)`. All zero = orthogonal box.
    /// Only `xy` is exercised today (Lees–Edwards simple shear); `xz`/`yz` are
    /// reserved for generality. See [`Domain::x2lamda`] / [`Domain::lamda2x`].
    pub tilt: [f64; 3],
    /// `true` when the box is non-orthogonal (any tilt active). Gates the
    /// triclinic code paths in `pbc`, `borders`/`exchange`, and neighbor binning;
    /// when `false`, the orthogonal fast paths run unchanged.
    pub triclinic: bool,
    /// Velocity of the upper box edges relative to the lower ones, used for the
    /// Lees–Edwards streaming-velocity remap on periodic crossings. For simple
    /// shear `boundary_vel = (γ̇·Ly, 0, 0)`. Tied to the shear *rate*, not the
    /// tilt: a static tilted box has `boundary_vel = 0`.
    pub boundary_vel: [f64; 3],
    /// Sub-domain bounds in fractional (lamda) coordinates, `[p/n, (p+1)/n]` per
    /// dimension. Used to classify atom ownership for ghost/exchange comm when
    /// the box is triclinic (in lamda space the tilted box is a unit cube, so the
    /// orthogonal single-hop-per-dim decomposition still applies).
    pub sub_lamda_low: [f64; 3],
    /// Upper sub-domain bound in fractional (lamda) coordinates; see [`sub_lamda_low`](Self::sub_lamda_low).
    pub sub_lamda_high: [f64; 3],
    /// Every rank's cut planes per axis in fractional (lamda) coordinates:
    /// `p + 1` values from `0.0` to `1.0` for `p` ranks on that axis.
    /// Identical on all ranks; load balancing keeps it current. This is what
    /// [`ParticlePartitionDirectory::from_domain`](crate::ParticlePartitionDirectory::from_domain)
    /// reads to tell a coupler which rank owns a point.
    pub lamda_planes: [Vec<f64>; 3],
}

impl Default for Domain {
    fn default() -> Self {
        Self::new()
    }
}

impl Domain {
    /// Create a unit-cube [`Domain`] (box `[0,1]³`, single sub-domain, fully periodic).
    /// Real bounds are set later from [`DomainConfig`] during setup.
    pub fn new() -> Self {
        Domain {
            boundaries_high: [1.0; 3],
            boundaries_low: [0.0; 3],
            sub_domain_low: [0.0; 3],
            sub_domain_high: [1.0; 3],
            sub_length: [1.0; 3],
            size: [1.0; 3],
            boundary_type: [BoundaryType::Periodic; 3],
            shrink_wrap_padding: 0.0,
            bounds_changed: false,
            volume: 1.0,
            ghost_cutoff: 0.0,
            tilt: [0.0; 3],
            triclinic: false,
            boundary_vel: [0.0; 3],
            sub_lamda_low: [0.0; 3],
            sub_lamda_high: [1.0; 3],
            lamda_planes: [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]],
        }
    }

    /// Recompute derived fields (size, sub_length, volume) after bounds change.
    pub fn update_derived(&mut self) {
        for d in 0..3 {
            self.size[d] = self.boundaries_high[d] - self.boundaries_low[d];
            self.sub_length[d] = self.sub_domain_high[d] - self.sub_domain_low[d];
        }
        self.volume = self.size[0] * self.size[1] * self.size[2];
    }

    /// Whether axis `d` is periodic.
    #[inline]
    pub fn is_periodic(&self, d: usize) -> bool {
        self.boundary_type[d] == BoundaryType::Periodic
    }

    /// Whether axis `d` is shrink-wrap.
    #[inline]
    pub fn is_shrink_wrap(&self, d: usize) -> bool {
        self.boundary_type[d] == BoundaryType::ShrinkWrap
    }

    /// Return periodic flags as a `[bool; 3]` array (convenience for bulk assignment).
    pub fn periodic_flags(&self) -> [bool; 3] {
        [
            self.is_periodic(0),
            self.is_periodic(1),
            self.is_periodic(2),
        ]
    }

    /// Wrap a separation vector `d = x_j - x_i` to its **minimum image** on every
    /// periodic axis (orthogonal or triclinic box); non-periodic axes are left as
    /// they are.
    ///
    /// Use it for any persistent pair addressed by tag (bonds, peridynamic
    /// families): the same tag can be present as a local atom *and* as one or
    /// more periodic ghost images, so the copy a tag lookup returns is not
    /// necessarily the nearest one. Valid while the interaction range is below
    /// half the periodic box length.
    #[inline]
    pub fn minimum_image(&self, d: [f64; 3]) -> [f64; 3] {
        let pflags = self.periodic_flags();
        if !(pflags[0] || pflags[1] || pflags[2]) {
            return d;
        }
        if !self.triclinic {
            let mut d = d;
            for k in 0..3 {
                if pflags[k] && self.size[k] > 0.0 {
                    d[k] -= self.size[k] * (d[k] / self.size[k]).round();
                }
            }
            return d;
        }
        let [xy, xz, yz] = self.tilt;
        // Fractional components via back-substitution through H (upper
        // triangular), computed from the *unwrapped* separation, then wrapped.
        let lz = if self.size[2] != 0.0 {
            d[2] / self.size[2]
        } else {
            0.0
        };
        let ly = if self.size[1] != 0.0 {
            (d[1] - yz * lz) / self.size[1]
        } else {
            0.0
        };
        let lx = if self.size[0] != 0.0 {
            (d[0] - xy * ly - xz * lz) / self.size[0]
        } else {
            0.0
        };
        let wrap = |l: f64, axis: usize| if pflags[axis] { l - l.round() } else { l };
        let (lx, ly, lz) = (wrap(lx, 0), wrap(ly, 1), wrap(lz, 2));
        [
            if self.size[0] != 0.0 {
                self.size[0] * lx + xy * ly + xz * lz
            } else {
                d[0]
            },
            if self.size[1] != 0.0 {
                self.size[1] * ly + yz * lz
            } else {
                d[1]
            },
            if self.size[2] != 0.0 {
                self.size[2] * lz
            } else {
                d[2]
            },
        ]
    }

    /// Convert a real-space point to fractional (lamda) box coordinates, where the
    /// (possibly tilted) box maps onto the unit cube `[0,1)³`.
    ///
    /// With box origin `boundaries_low`, edge lengths `size = [Lx,Ly,Lz]`, and
    /// upper-triangular tilt `H = [[Lx, xy, xz],[0, Ly, yz],[0, 0, Lz]]`, the
    /// forward map is `r = origin + H·λ`. This inverts it (back-substitution):
    /// ```text
    /// λz = (z - z0) / Lz
    /// λy = (y - y0 - yz·λz) / Ly
    /// λx = (x - x0 - xy·λy - xz·λz) / Lx
    /// ```
    /// Reduces to the plain `(r - origin)/size` when all tilts are zero.
    #[inline]
    pub fn x2lamda(&self, r: [f64; 3]) -> [f64; 3] {
        let [xy, xz, yz] = self.tilt;
        let lz = (r[2] - self.boundaries_low[2]) / self.size[2];
        let ly = (r[1] - self.boundaries_low[1] - yz * lz) / self.size[1];
        let lx = (r[0] - self.boundaries_low[0] - xy * ly - xz * lz) / self.size[0];
        [lx, ly, lz]
    }

    /// Convert fractional (lamda) box coordinates back to real space:
    /// `r = origin + H·λ`. Inverse of [`x2lamda`](Self::x2lamda).
    #[inline]
    pub fn lamda2x(&self, lam: [f64; 3]) -> [f64; 3] {
        let [xy, xz, yz] = self.tilt;
        [
            self.boundaries_low[0] + self.size[0] * lam[0] + xy * lam[1] + xz * lam[2],
            self.boundaries_low[1] + self.size[1] * lam[1] + yz * lam[2],
            self.boundaries_low[2] + self.size[2] * lam[2],
        ]
    }
}

// ── Domain decomposition ─────────────────────────────────────────────────────

/// Cut planes SOIL must use instead of choosing its own decomposition.
///
/// Insert this resource before setup to make SOIL split its particles exactly
/// where another code splits its data, so a particle and the grid cells it
/// talks to live on the same rank (MPM, particle-in-cell). `planes[d]` holds
/// `p + 1` physical positions for `p` ranks along axis `d`, strictly
/// increasing. Only the interior planes are used; the outer two stand for the
/// box edges, which stay SOIL's own `[domain]` bounds. Ranks are numbered
/// x-major like the rest of SOIL.
///
/// With this resource present SOIL takes its processor grid from the planes
/// (ignoring `[comm] processors_*`) and load balancing leaves the cuts alone.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedCuts {
    /// Physical cut positions per axis, `p + 1` values for `p` ranks.
    pub planes: [Vec<f64>; 3],
}

impl FixedCuts {
    /// Ranks per axis implied by the planes.
    pub fn decomposition(&self) -> [i32; 3] {
        std::array::from_fn(|d| self.planes[d].len().saturating_sub(1) as i32)
    }

    /// Check the cuts against SOIL's box and the rank count.
    // Negated comparisons on purpose: they also reject NaN.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn validate(&self, box_low: [f64; 3], box_high: [f64; 3], nprocs: i32) -> Result<(), String> {
        let decomp = self.decomposition();
        for d in 0..3 {
            let p = &self.planes[d];
            if p.len() < 2 {
                return Err(format!("FixedCuts axis {d} needs at least two planes"));
            }
            if p.windows(2).any(|w| !(w[1] > w[0])) {
                return Err(format!("FixedCuts planes on axis {d} must increase strictly"));
            }
            if let Some(&x) = p[1..p.len() - 1]
                .iter()
                .find(|&&x| !(x > box_low[d] && x < box_high[d]))
            {
                return Err(format!(
                    "FixedCuts plane {x} on axis {d} is not inside SOIL's box ({}, {})",
                    box_low[d], box_high[d]
                ));
            }
        }
        let product = decomp[0] * decomp[1] * decomp[2];
        if product != nprocs {
            return Err(format!(
                "FixedCuts describe {product} ranks ({decomp:?}) but the run has {nprocs}"
            ));
        }
        Ok(())
    }
}

/// Apply [`FixedCuts`] to a freshly decomposed orthogonal [`Domain`]: this
/// rank's slab becomes `[plane[pos], plane[pos + 1])` on every axis, exactly
/// at the given positions, and `lamda_planes` records every rank's cuts.
pub fn apply_fixed_cuts(domain: &mut Domain, cuts: &FixedCuts, position: [i32; 3]) {
    assert!(!domain.triclinic, "FixedCuts support orthogonal boxes only");
    for d in 0..3 {
        let p = cuts.planes[d].len() - 1;
        let (blo, bhi) = (domain.boundaries_low[d], domain.boundaries_high[d]);
        let size = domain.size[d];
        let mut phys = cuts.planes[d].clone();
        phys[0] = blo;
        phys[p] = bhi;
        let mut lamda: Vec<f64> = phys.iter().map(|x| (x - blo) / size).collect();
        lamda[0] = 0.0;
        lamda[p] = 1.0;
        let r = position[d] as usize;
        domain.sub_domain_low[d] = phys[r];
        domain.sub_domain_high[d] = phys[r + 1];
        domain.sub_length[d] = phys[r + 1] - phys[r];
        domain.sub_lamda_low[d] = lamda[r];
        domain.sub_lamda_high[d] = lamda[r + 1];
        domain.lamda_planes[d] = lamda;
    }
}

/// Compute [`Domain`] from config using uniform Cartesian grid decomposition.
pub fn decompose_domain(config: &DomainConfig, comm: &dyn CommBackend) -> Domain {
    let boundaries_low = [config.x_low, config.y_low, config.z_low];
    let boundaries_high = [config.x_high, config.y_high, config.z_high];
    let size = [
        boundaries_high[0] - boundaries_low[0],
        boundaries_high[1] - boundaries_low[1],
        boundaries_high[2] - boundaries_low[2],
    ];
    let boundary_types = config.boundary_types();

    let proc_decomp = comm.processor_decomposition();
    let proc_pos = comm.processor_position();

    let delta_x = size[0] / proc_decomp[0] as f64;
    let delta_y = size[1] / proc_decomp[1] as f64;
    let delta_z = size[2] / proc_decomp[2] as f64;

    let sub_domain_low = [
        boundaries_low[0] + delta_x * proc_pos[0] as f64,
        boundaries_low[1] + delta_y * proc_pos[1] as f64,
        boundaries_low[2] + delta_z * proc_pos[2] as f64,
    ];
    let sub_domain_high = [
        boundaries_low[0] + delta_x * (1 + proc_pos[0]) as f64,
        boundaries_low[1] + delta_y * (1 + proc_pos[1]) as f64,
        boundaries_low[2] + delta_z * (1 + proc_pos[2]) as f64,
    ];
    let sub_length = [delta_x, delta_y, delta_z];

    // Sub-domain bounds in fractional (lamda) coordinates: a uniform Cartesian
    // split of the unit cube, independent of any box tilt. Used by the triclinic
    // comm/exchange paths.
    let sub_lamda_low = [
        proc_pos[0] as f64 / proc_decomp[0] as f64,
        proc_pos[1] as f64 / proc_decomp[1] as f64,
        proc_pos[2] as f64 / proc_decomp[2] as f64,
    ];
    let sub_lamda_high = [
        (1 + proc_pos[0]) as f64 / proc_decomp[0] as f64,
        (1 + proc_pos[1]) as f64 / proc_decomp[1] as f64,
        (1 + proc_pos[2]) as f64 / proc_decomp[2] as f64,
    ];

    Domain {
        boundaries_low,
        boundaries_high,
        sub_domain_low,
        sub_domain_high,
        sub_length,
        size,
        boundary_type: boundary_types,
        shrink_wrap_padding: config.shrink_wrap_padding,
        bounds_changed: false,
        volume: size[0] * size[1] * size[2],
        ghost_cutoff: 0.0,
        tilt: [0.0; 3],
        triclinic: false,
        boundary_vel: [0.0; 3],
        sub_lamda_low,
        sub_lamda_high,
        lamda_planes: std::array::from_fn(|d| uniform_lamda_planes(proc_decomp[d])),
    }
}

/// `p + 1` evenly spaced planes on `[0, 1]`, ending exactly at `1.0`.
pub(crate) fn uniform_lamda_planes(p: i32) -> Vec<f64> {
    let p = p.max(1) as usize;
    (0..=p)
        .map(|i| if i == p { 1.0 } else { i as f64 / p as f64 })
        .collect()
}

/// Validate a [`DomainConfig`] against the active communicator.
pub fn validate_domain_config(
    config: &DomainConfig,
    comm: &dyn CommBackend,
) -> Result<(), DomainValidationError> {
    let has_shrink_wrap = config.boundary_types().contains(&BoundaryType::ShrinkWrap);
    if comm.size() > 1 && has_shrink_wrap {
        return Err(DomainValidationError::ShrinkWrapWithMpi { ranks: comm.size() });
    }
    Ok(())
}

/// Validate and compute [`Domain`] from config using uniform Cartesian grid decomposition.
pub fn try_decompose_domain(
    config: &DomainConfig,
    comm: &dyn CommBackend,
) -> Result<Domain, DomainValidationError> {
    validate_domain_config(config, comm)?;
    Ok(decompose_domain(config, comm))
}

// ── Dynamic load balancing (roadmap step 5) ──────────────────────────────────

/// Equal-count split planes along one axis from a global particle histogram.
/// Returns `px + 1` planes spanning `[lo, hi]`: plane `r` is rank `r`'s low edge
/// and plane `r+1` its high edge, chosen so each rank owns ~`total/px` particles
/// (linear interpolation within the boundary bin). Pure — no MPI; the comm-driven
/// [`rebalance_x`] builds the global histogram and calls this. Degenerate inputs
/// (no particles, `px < 2`) fall back to an even geometric split.
pub fn balanced_planes(hist: &[f64], lo: f64, hi: f64, px: usize) -> Vec<f64> {
    let nbins = hist.len();
    let w = (hi - lo) / nbins.max(1) as f64;
    let total: f64 = hist.iter().sum();
    let mut planes = vec![lo; px + 1];
    planes[px] = hi;
    if total <= 0.0 || px < 2 || nbins == 0 {
        for (r, p) in planes.iter_mut().enumerate() {
            *p = lo + (hi - lo) * r as f64 / px as f64;
        }
        return planes;
    }
    let mut cum = 0.0;
    let mut b = 0usize;
    for r in 1..px {
        let target = total * r as f64 / px as f64;
        while b < nbins && cum + hist[b] < target {
            cum += hist[b];
            b += 1;
        }
        let frac = if b < nbins && hist[b] > 0.0 {
            (target - cum) / hist[b]
        } else {
            0.0
        };
        planes[r] = (lo + (b as f64 + frac) * w).clamp(lo, hi);
    }
    planes
}

/// Rebalance the decomposed x-axis so each rank owns ~equal particle counts, then
/// flag `bounds_changed` to trigger atom re-exchange + neighbour rebuild. Physics
/// is decomposition-invariant, so a periodically-rebalanced run must reproduce a
/// static one. The histogram is summed in one collective, so it works for any
/// rank count without an all-gather. Prefer [`rebalance_shift`], which balances
/// every decomposed axis and limits how far planes move per call.
pub fn rebalance_x(atoms: &Atom, domain: &mut Domain, comm: &dyn CommBackend, nbins: usize) {
    let px = comm.processor_decomposition()[0] as usize;
    if px < 2 {
        return;
    }
    let lo = domain.boundaries_low[0];
    let hi = domain.boundaries_high[0];
    let w = (hi - lo) / nbins.max(1) as f64;
    let mut hist = vec![0.0f64; nbins];
    for i in 0..atoms.nlocal as usize {
        let b = (((atoms.pos[i][0] as f64 - lo) / w).floor() as usize).min(nbins - 1);
        hist[b] += 1.0;
    }
    comm.all_reduce_sum_f64_slice(&mut hist);
    let planes = balanced_planes(&hist, lo, hi, px);
    let r = comm.processor_position()[0] as usize;
    domain.sub_domain_low[0] = planes[r];
    domain.sub_domain_high[0] = planes[r + 1];
    domain.sub_length[0] = planes[r + 1] - planes[r];
    domain.sub_lamda_low[0] = (planes[r] - lo) / domain.size[0];
    domain.sub_lamda_high[0] = (planes[r + 1] - lo) / domain.size[0];
    domain.lamda_planes[0] = planes.iter().map(|x| (x - lo) / domain.size[0]).collect();
    domain.bounds_changed = true;
}

/// Make every sub-domain along an axis at least `min_width` wide while
/// keeping the end planes fixed. Returns `false` if the axis is too short to
/// fit `planes.len() - 1` such slabs.
fn enforce_min_width(planes: &mut [f64], min_width: f64) -> bool {
    let p = planes.len() - 1;
    if (planes[p] - planes[0]) < p as f64 * min_width * (1.0 + 1e-12) {
        return false;
    }
    for r in 1..p {
        planes[r] = planes[r].max(planes[r - 1] + min_width);
    }
    for r in (1..p).rev() {
        planes[r] = planes[r].min(planes[r + 1] - min_width);
    }
    true
}

/// Move `old` planes toward `target` by one uniform fraction `a <= 1`, chosen
/// so no interior plane moves more than half of either neighboring old slab.
/// That keeps every atom within one rank of its old owner (exchange is
/// single-hop), and because every plane moves by the same fraction, any width
/// bound that both `old` and `target` satisfy still holds afterwards.
pub fn damped_planes(old: &[f64], target: &[f64]) -> Vec<f64> {
    let p = old.len() - 1;
    let mut a: f64 = 1.0;
    for r in 1..p {
        let delta = (target[r] - old[r]).abs();
        let limit = 0.5 * (old[r] - old[r - 1]).min(old[r + 1] - old[r]);
        if delta > limit {
            a = a.min(limit / delta);
        }
    }
    let mut planes: Vec<f64> = old
        .iter()
        .zip(target)
        .map(|(&o, &t)| o + a * (t - o))
        .collect();
    planes[0] = old[0];
    planes[p] = old[p];
    planes
}

/// Shift-style load balancing over every decomposed axis, like LAMMPS
/// `balance shift xyz`. The grid stays a brick (planes are per axis), so
/// the 6-swap ghost pattern is unchanged. Each axis gets an equal-count
/// target from a global 1D histogram, widened so no sub-domain is thinner
/// than the ghost cutoff allows for the current swap depth (`maxneed`), and
/// the planes then move toward it by at most half a slab (see
/// [`damped_planes`]); a badly skewed start converges over a few calls.
///
/// All histograms and the current plane positions are summed in a single
/// collective, so every rank computes identical planes. Orthogonal boxes
/// only: returns `false` without touching the domain when triclinic. Returns
/// `true` and sets `bounds_changed` when any plane moved.
pub fn rebalance_shift(
    atoms: &Atom,
    domain: &mut Domain,
    comm: &dyn CommBackend,
    nbins: usize,
    maxneed: [i32; 3],
) -> bool {
    if domain.triclinic || nbins == 0 {
        return false;
    }
    let decomp = comm.processor_decomposition();
    let position = comm.processor_position();
    let axes: Vec<usize> = (0..3).filter(|&d| decomp[d] > 1).collect();
    if axes.is_empty() {
        return false;
    }

    // Layout per balanced axis: [nbins histogram][p + 1 planes, in lamda].
    // Only the rank at grid position 0 on the other two axes writes planes,
    // so the sum reproduces each plane exactly once.
    let mut offsets = [(0usize, 0usize); 3];
    let mut len = 0;
    for &d in &axes {
        offsets[d] = (len, len + nbins);
        len += nbins + decomp[d] as usize + 1;
    }
    let mut buf = vec![0.0f64; len];
    for &d in &axes {
        let representative = (0..3).all(|e| e == d || position[e] == 0);
        if representative {
            let p = decomp[d] as usize;
            let r = position[d] as usize;
            let base = offsets[d].1;
            buf[base + r] = domain.sub_lamda_low[d];
            if r == p - 1 {
                buf[base + p] = domain.sub_lamda_high[d];
            }
        }
    }
    let scale = nbins as f64;
    for i in 0..atoms.nlocal as usize {
        for &d in &axes {
            let lam = (atoms.pos[i][d] as f64 - domain.boundaries_low[d]) / domain.size[d];
            let bin = ((lam * scale).floor().max(0.0) as usize).min(nbins - 1);
            buf[offsets[d].0 + bin] += 1.0;
        }
    }
    comm.all_reduce_sum_f64_slice(&mut buf);

    let mut moved = false;
    for &d in &axes {
        let p = decomp[d] as usize;
        let (h, b) = offsets[d];
        let hist = &buf[h..h + nbins];
        let old = &buf[b..b + p + 1];
        let mut target = balanced_planes(hist, 0.0, 1.0, p);
        // Narrowest slab the current swap depth can serve with ghosts.
        let size = domain.size[d];
        let need = if maxneed[d] > 0 {
            maxneed[d] as f64
        } else {
            (domain.ghost_cutoff * p as f64 / size).ceil().max(1.0)
        };
        let min_width = domain.ghost_cutoff / need / size;
        if !enforce_min_width(&mut target, min_width) {
            continue;
        }
        let planes = damped_planes(old, &target);
        // Every rank computed the same planes, so every rank records them,
        // including ranks whose own slab did not move.
        if planes != domain.lamda_planes[d] {
            domain.lamda_planes[d] = planes.clone();
        }
        let r = position[d] as usize;
        let (lo_l, hi_l) = (planes[r], planes[r + 1]);
        if lo_l == domain.sub_lamda_low[d] && hi_l == domain.sub_lamda_high[d] {
            continue;
        }
        let blo = domain.boundaries_low[d];
        let lo = if r == 0 { blo } else { blo + lo_l * size };
        let hi = if r + 1 == p {
            domain.boundaries_high[d]
        } else {
            blo + hi_l * size
        };
        domain.sub_lamda_low[d] = lo_l;
        domain.sub_lamda_high[d] = hi_l;
        domain.sub_domain_low[d] = lo;
        domain.sub_domain_high[d] = hi;
        domain.sub_length[d] = hi - lo;
        moved = true;
    }
    // Every rank sees the same sums, but only some ranks' own slabs change:
    // flag the change on all of them so bins are recomputed consistently.
    let any_moved = comm.all_reduce_sum_f64(moved as u8 as f64) > 0.0;
    if any_moved {
        domain.bounds_changed = true;
    }
    any_moved
}

/// Cadence, resolution and trigger for periodic dynamic load balancing.
pub struct LoadBalanceConfig {
    /// Rebalance every `every`-th step (0 disables). The balance runs on the
    /// first neighbor-rebuild step at or after it is due, so it never forces
    /// an extra rebuild.
    pub every: usize,
    /// Histogram bins per balanced axis.
    pub nbins: usize,
    /// Only rebalance when `max(nlocal) / mean(nlocal)` exceeds this.
    pub threshold: f64,
}

#[derive(Default)]
struct LoadBalanceCounter(usize);

/// Rebalance every decomposed axis by particle count (see
/// [`rebalance_shift`]), at `PreExchange` of a rebuild step so the following
/// `Exchange` migrates atoms to their new owners.
fn load_balance_system(
    atoms: Res<Atom>,
    mut domain: ResMut<Domain>,
    comm: Res<CommResource>,
    cfg: Res<LoadBalanceConfig>,
    topo: Option<Res<crate::CommTopology>>,
    comm_state: Res<CurrentState<CommState>>,
    mut counter: ResMut<LoadBalanceCounter>,
    fixed_cuts: Option<Res<FixedCuts>>,
) {
    if fixed_cuts.is_some() {
        // Another code owns the cuts; moving them would break the alignment.
        return;
    }
    counter.0 += 1;
    if cfg.every == 0 || counter.0 < cfg.every || comm_state.0 != CommState::FullRebuild {
        return;
    }
    counter.0 = 0;
    if comm.size() < 2 {
        return;
    }
    if cfg.threshold > 1.0 {
        let n = atoms.nlocal as f64;
        let max = -comm.all_reduce_min_f64(-n);
        let mean = comm.all_reduce_sum_f64(n) / comm.size() as f64;
        if mean <= 0.0 || max / mean <= cfg.threshold {
            return;
        }
    }
    let maxneed = topo.map(|t| t.maxneed).unwrap_or([0; 3]);
    rebalance_shift(&atoms, &mut domain, &**comm, cfg.nbins, maxneed);
}

/// Periodic dynamic load balancing. Add to the app to move the sub-domain
/// planes of every decomposed axis toward equal particle counts every `every`
/// steps (LAMMPS `fix balance ... shift xyz`). Physics is
/// decomposition-invariant, so a rebalanced run reproduces a static one.
///
/// ```ignore
/// app.add_plugins(LoadBalancePlugin { every: 100, ..Default::default() });
/// ```
pub struct LoadBalancePlugin {
    /// Rebalance every `every` steps (0 disables rebalancing).
    pub every: usize,
    /// Number of histogram bins per balanced axis.
    pub nbins: usize,
    /// Skip a due rebalance unless `max(nlocal) / mean(nlocal)` exceeds this
    /// (LAMMPS `thresh`). `1.0` always rebalances.
    pub threshold: f64,
}

impl Default for LoadBalancePlugin {
    fn default() -> Self {
        LoadBalancePlugin {
            every: 100,
            nbins: 256,
            threshold: 1.05,
        }
    }
}

impl Plugin for LoadBalancePlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(LoadBalanceConfig {
            every: self.every,
            nbins: self.nbins,
            threshold: self.threshold,
        });
        app.add_resource(LoadBalanceCounter(0));
        app.add_update_system(
            load_balance_system.after(PBC),
            ParticleSimScheduleSet::PreExchange,
        );
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Registers [`Domain`] resource and periodic boundary condition systems.
pub struct DomainPlugin;

impl Plugin for DomainPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[domain]
# Simulation box boundaries (also accepts x_lo/x_hi, y_lo/y_hi, z_lo/z_hi)
x_low = 0.0
x_high = 1.0
y_low = 0.0
y_high = 1.0
z_low = 0.0
z_high = 1.0
# Boundary type per axis: "periodic", "fixed", or "shrink-wrap"
boundary_x = "periodic"
boundary_y = "periodic"
boundary_z = "periodic"
# Padding for shrink-wrap boundaries [simulation units]. 0 = auto (use ghost cutoff).
# shrink_wrap_padding = 0.0"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<DomainConfig>(app, "domain");
        install_domain(app);
    }

    fn try_build(&self, app: &mut App) -> Result<(), AppError> {
        let config = Config::try_load::<DomainConfig>(app, "domain")
            .map_err(|error| AppError::message(error.to_string()))?;
        if let Some(comm) = app.get_resource_ref::<CommResource>() {
            validate_domain_config(&config, &**comm)
                .map_err(|error| AppError::message(error.to_string()))?;
        }
        install_domain(app);
        Ok(())
    }
}

fn install_domain(app: &mut App) {
    app.add_resource(Domain::new())
        .add_setup_system(
            domain_read_input
                .label(DOMAIN_READ_INPUT)
                .run_if(first_stage_only()),
            ScheduleSetupSet::Setup,
        )
        .add_update_system(
            shrink_wrap.label(SHRINK_WRAP).before(PBC),
            ParticleSimScheduleSet::PreExchange,
        )
        .add_update_system(
            pbc.requires(SHRINK_WRAP)
                .label(PBC)
                .run_if(in_state(CommState::FullRebuild)),
            ParticleSimScheduleSet::PreExchange,
        );
}

fn boundary_type_char(bt: BoundaryType) -> char {
    match bt {
        BoundaryType::Periodic => 'p',
        BoundaryType::Fixed => 'f',
        BoundaryType::ShrinkWrap => 's',
    }
}

/// Setup system: read `[domain]` config and initialize the [`Domain`] resource.
pub fn domain_read_input(
    config: Res<DomainConfig>,
    comm: Res<CommResource>,
    fixed_cuts: Option<Res<FixedCuts>>,
    mut domain: ResMut<Domain>,
) {
    let boundary_types = config.boundary_types();

    let has_shrink_wrap = boundary_types.contains(&BoundaryType::ShrinkWrap);

    validate_domain_config(&config, &**comm)
        .expect("DomainPlugin validates configuration before setup");

    if comm.rank() == 0 {
        println!(
            "Domain: {} {} {} {} {} {}",
            config.x_low, config.x_high, config.y_low, config.y_high, config.z_low, config.z_high
        );
        println!(
            "Domain: boundary {} {} {}",
            boundary_type_char(boundary_types[0]),
            boundary_type_char(boundary_types[1]),
            boundary_type_char(boundary_types[2]),
        );
        if has_shrink_wrap {
            if config.shrink_wrap_padding > 0.0 {
                println!(
                    "Domain: shrink-wrap padding = {}",
                    config.shrink_wrap_padding
                );
            } else {
                println!("Domain: shrink-wrap padding = auto (ghost cutoff)");
            }
        }
    }

    *domain = decompose_domain(&config, &**comm);
    if let Some(cuts) = fixed_cuts {
        apply_fixed_cuts(&mut domain, &cuts, comm.processor_position());
    }
}

/// Core shrink-wrap logic: update domain bounds to encompass all atom positions + padding.
///
/// Returns `true` if any bounds changed. This is extracted as a standalone function
/// so it can be unit-tested without the ECS resource wrappers.
///
/// **MPI note**: Shrink-wrap is currently single-process only (validated in
/// `domain_read_input`). Future MPI support would need `all_reduce_min/max` for
/// global extremes and correct sub-domain bound updates per rank.
pub fn shrink_wrap_update(domain: &mut Domain, positions: &[[f64; 3]], nlocal: usize) -> bool {
    let any_shrink =
        domain.is_shrink_wrap(0) || domain.is_shrink_wrap(1) || domain.is_shrink_wrap(2);
    if !any_shrink || nlocal == 0 {
        return false;
    }

    // Padding: use explicit value if > 0, otherwise fall back to ghost_cutoff.
    // ghost_cutoff encompasses the max pairwise interaction distance + skin buffer,
    // so it is a conservative but safe choice for any unit system.
    let padding = if domain.shrink_wrap_padding > 0.0 {
        domain.shrink_wrap_padding
    } else if domain.ghost_cutoff > 0.0 {
        domain.ghost_cutoff
    } else {
        // ghost_cutoff not yet set (shrink_wrap runs before neighbor_setup on first step).
        // Use 1% of the current domain size as a unit-independent fallback.
        let max_size = domain.size[0].max(domain.size[1]).max(domain.size[2]);
        if max_size > 0.0 {
            max_size * 0.01
        } else {
            1.0
        }
    };

    let mut changed = false;

    for d in 0..3 {
        if !domain.is_shrink_wrap(d) {
            continue;
        }

        // Find min/max positions on this axis
        let mut pos_min = f64::MAX;
        let mut pos_max = f64::MIN;
        for pos in &positions[..nlocal] {
            let p = pos[d];
            if p < pos_min {
                pos_min = p;
            }
            if p > pos_max {
                pos_max = p;
            }
        }

        let new_low = pos_min - padding;
        let new_high = pos_max + padding;

        // Check if bounds actually changed (with tolerance to avoid churn)
        let tol = 1e-12;
        if (new_low - domain.boundaries_low[d]).abs() > tol
            || (new_high - domain.boundaries_high[d]).abs() > tol
        {
            domain.boundaries_low[d] = new_low;
            domain.boundaries_high[d] = new_high;
            // Single-process: sub-domain = global domain.
            // TODO: For MPI, sub-domain bounds must be recomputed via domain decomposition.
            domain.sub_domain_low[d] = new_low;
            domain.sub_domain_high[d] = new_high;
            changed = true;
        }
    }

    if changed {
        domain.update_derived();
        domain.bounds_changed = true;
    }
    changed
}

/// ECS system wrapper: update shrink-wrap boundaries each step.
///
/// Runs at `PreExchange` before `pbc`. Delegates to [`shrink_wrap_update`].
pub fn shrink_wrap(atoms: Res<Atom>, mut domain: ResMut<Domain>) {
    // Nothing to do (and nothing to copy) unless an axis is shrink-wrapped; this
    // runs every step, so the common periodic/fixed box must not pay for a
    // position copy.
    if !(domain.is_shrink_wrap(0) || domain.is_shrink_wrap(1) || domain.is_shrink_wrap(2)) {
        return;
    }
    // shrink_wrap_update works in f64; convert the local `Real` positions at the
    // boundary (shrink-wrap is single-process and rare, so the copy is cheap).
    let nlocal = atoms.nlocal as usize;
    let positions_f64: Vec<[f64; 3]> = atoms.pos[..nlocal]
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    shrink_wrap_update(&mut domain, &positions_f64, nlocal);
}

/// Wrap a position into [low, low+size) with periodic boundaries.
/// Returns (new_pos, image_delta). The common case is one image (+1/-1/0);
/// a particle that crossed several periods since the last wrap (a fast
/// particle in a thin periodic direction, e.g. a DSMC slab) is wrapped by as
/// many images as it needs, at no cost to the common case.
#[inline]
fn wrap_periodic(mut pos: f64, low: f64, size: f64) -> (f64, i32) {
    let high = low + size;
    if pos < low {
        let mut delta = 0;
        while pos < low {
            pos += size;
            delta -= 1;
        }
        (pos, delta)
    } else if pos >= high {
        let mut delta = 0;
        while pos >= high {
            pos -= size;
            delta += 1;
        }
        (pos, delta)
    } else {
        (pos, 0)
    }
}

#[cfg(test)]
mod wrap_tests {
    use super::wrap_periodic;

    #[test]
    fn wrap_periodic_handles_one_and_several_images() {
        assert_eq!(wrap_periodic(0.5, 0.0, 1.0), (0.5, 0));
        assert_eq!(wrap_periodic(1.25, 0.0, 1.0), (0.25, 1));
        assert_eq!(wrap_periodic(-0.25, 0.0, 1.0), (0.75, -1));
        assert_eq!(wrap_periodic(3.25, 0.0, 1.0), (0.25, 3));
        assert_eq!(wrap_periodic(-2.75, 0.0, 1.0), (0.25, -3));
    }
}

/// Apply periodic boundary conditions: wrap positions on periodic axes,
/// remove out-of-bounds atoms on fixed/shrink-wrap axes.
///
/// Gated by `run_if(in_state(CommState::FullRebuild))` — only runs on full rebuild steps.
pub fn pbc(
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut comm_state: ResMut<CurrentState<CommState>>,
) {
    let low = domain.boundaries_low;
    let high = domain.boundaries_high;
    let size = domain.size;
    let periodic = domain.periodic_flags();

    if domain.triclinic {
        pbc_triclinic(&mut atoms, &domain, &registry, &mut comm_state);
        return;
    }

    if periodic[0] && periodic[1] && periodic[2] {
        // Fast path: fully periodic, no removals possible (local atoms only, ghosts live outside box)
        for i in 0..atoms.nlocal as usize {
            for d in 0..3 {
                // `pos` is `Real` (f32 in mixed/single); wrap in f64 then store back.
                let (new_pos, delta) = wrap_periodic(atoms.pos[i][d] as f64, low[d], size[d]);
                atoms.pos[i][d] = new_pos as Real;
                atoms.image[i][d] += delta;
            }
        }
    } else {
        // Slow path: non-periodic axes may require removal (local atoms only).
        // Shrink-wrap axes: atoms are always inside bounds (shrink_wrap ran first),
        // so the out-of-bounds check is a no-op — but it's harmless and correct.
        let mut removed = 0usize;
        'outer: for i in (0..atoms.nlocal as usize).rev() {
            for d in 0..3 {
                if periodic[d] {
                    let (new_pos, delta) = wrap_periodic(atoms.pos[i][d] as f64, low[d], size[d]);
                    atoms.pos[i][d] = new_pos as Real;
                    atoms.image[i][d] += delta;
                } else if (atoms.pos[i][d] as f64) < low[d] || (atoms.pos[i][d] as f64) >= high[d] {
                    crate::ParticleStore::new(&mut atoms, &registry)
                        .swap_remove(i)
                        .expect("pbc found misaligned particle columns");
                    removed += 1;
                    continue 'outer;
                }
            }
        }
        // Update nlocal and force full rebuild since sendlists are stale.
        if removed > 0 {
            comm_state.0 = CommState::FullRebuild;
        }
    }
}

/// Triclinic periodic boundary conditions (used when `domain.triclinic`).
///
/// Wraps each local atom in fractional (lamda) coordinates — where the tilted box
/// is the unit cube — so a crossing of the gradient (y) face automatically picks
/// up the box's x-tilt via [`Domain::lamda2x`]. On top of the geometric wrap it
/// applies the **Lees–Edwards streaming-velocity remap** (`remap v`): an atom that
/// wraps across the y face by `Δi_y` images has its velocity shifted by
/// `−Δi_y · boundary_vel` (the moving-edge velocity jump). Non-periodic axes fall
/// back to out-of-bounds removal, matching the orthogonal slow path.
fn pbc_triclinic(
    atoms: &mut Atom,
    domain: &Domain,
    registry: &AtomDataRegistry,
    comm_state: &mut CurrentState<CommState>,
) {
    let periodic = domain.periodic_flags();
    let bvel = domain.boundary_vel;
    let mut removed = 0usize;

    'outer: for i in (0..atoms.nlocal as usize).rev() {
        let pr = atoms.pos[i];
        let mut lam = domain.x2lamda([pr[0] as f64, pr[1] as f64, pr[2] as f64]);
        let mut delta = [0i32; 3];
        for d in 0..3 {
            if periodic[d] {
                // Single-step wrap: an atom moves < 1 box per rebuild interval
                // (same assumption as the orthogonal `wrap_periodic`).
                if lam[d] < 0.0 {
                    lam[d] += 1.0;
                    delta[d] = -1;
                } else if lam[d] >= 1.0 {
                    lam[d] -= 1.0;
                    delta[d] = 1;
                }
            } else if lam[d] < 0.0 || lam[d] >= 1.0 {
                // Fixed / shrink-wrap axis: atom left the box → remove it.
                crate::ParticleStore::new(atoms, registry)
                    .swap_remove(i)
                    .expect("triclinic pbc found misaligned particle columns");
                removed += 1;
                continue 'outer;
            }
        }
        // Geometric wrap (the y-image shift carries the x-tilt through lamda2x).
        let x = domain.lamda2x(lam);
        atoms.pos[i] = [x[0] as Real, x[1] as Real, x[2] as Real];
        for d in 0..3 {
            atoms.image[i][d] += delta[d];
        }
        // Lees–Edwards velocity remap: a y-face crossing jumps the streaming
        // velocity by the moving-edge differential. `delta[1] = +1` means the
        // atom wrapped down to the slower bottom edge → subtract boundary_vel.
        if delta[1] != 0 {
            let s = delta[1] as f64;
            atoms.vel[i][0] -= (s * bvel[0]) as Real;
            atoms.vel[i][1] -= (s * bvel[1]) as Real;
            atoms.vel[i][2] -= (s * bvel[2]) as Real;
        }
    }

    if removed > 0 {
        comm_state.0 = CommState::FullRebuild;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SingleProcessComm;

    fn make_comm(decomp: [i32; 3], pos: [i32; 3]) -> SingleProcessComm {
        let mut c = SingleProcessComm::new();
        c.set_processor_grid(decomp, pos);
        c
    }

    struct TestComm {
        size: i32,
        rank: i32,
        decomp: [i32; 3],
        pos: [i32; 3],
    }

    impl TestComm {
        fn new(size: i32, rank: i32, decomp: [i32; 3], pos: [i32; 3]) -> Self {
            Self {
                size,
                rank,
                decomp,
                pos,
            }
        }
    }

    impl CommBackend for TestComm {
        fn rank(&self) -> i32 {
            self.rank
        }

        fn size(&self) -> i32 {
            self.size
        }

        fn processor_decomposition(&self) -> [i32; 3] {
            self.decomp
        }

        fn processor_position(&self) -> [i32; 3] {
            self.pos
        }

        fn set_processor_grid(&mut self, decomp: [i32; 3], position: [i32; 3]) {
            self.decomp = decomp;
            self.pos = position;
        }

        fn all_reduce_sum_f64(&self, local: f64) -> f64 {
            local
        }

        fn all_reduce_min_f64(&self, local: f64) -> f64 {
            local
        }

        fn barrier(&self) {}

        fn send_f64(&self, _dest: i32, _buf: &[f64]) {
            unreachable!("TestComm point-to-point is not used in domain tests");
        }

        fn recv_f64(&self, _source: i32) -> Vec<f64> {
            unreachable!("TestComm point-to-point is not used in domain tests");
        }

        fn recv_f64_any(&self) -> Vec<f64> {
            unreachable!("TestComm point-to-point is not used in domain tests");
        }

        fn sendrecv_f64(&self, _dest: i32, _send_buf: &[f64], _source: i32) -> Vec<f64> {
            unreachable!("TestComm point-to-point is not used in domain tests");
        }

        fn sendrecv_f64_into(
            &self,
            _dest: i32,
            _send_buf: &[f64],
            _source: i32,
            _recv_buf: &mut [f64],
        ) {
            unreachable!("TestComm point-to-point is not used in domain tests");
        }

        fn sendrecv_batch_f64_into(&self, _ops: &mut [crate::SendRecvOp<'_>]) {
            unreachable!("TestComm point-to-point is not used in domain tests");
        }
    }

    #[test]
    fn minimum_image_wraps_only_periodic_axes() {
        let mut d = Domain::new();
        d.size = [10.0, 10.0, 10.0];
        d.boundary_type = [
            BoundaryType::Periodic,
            BoundaryType::Fixed,
            BoundaryType::Periodic,
        ];
        let w = d.minimum_image([9.0, 9.0, -6.0]);
        assert!((w[0] - -1.0).abs() < 1e-12);
        assert!((w[1] - 9.0).abs() < 1e-12);
        assert!((w[2] - 4.0).abs() < 1e-12);
        d.boundary_type = [BoundaryType::Fixed; 3];
        assert_eq!(d.minimum_image([9.0, 9.0, -6.0]), [9.0, 9.0, -6.0]);
    }

    #[test]
    fn minimum_image_triclinic_matches_lamda_wrap() {
        let mut d = Domain::new();
        d.size = [10.0, 8.0, 6.0];
        d.tilt = [2.0, 1.0, 0.5];
        d.triclinic = true;
        d.boundary_type = [BoundaryType::Periodic; 3];
        // A shift by one full lattice vector of H (column 2) must wrap to zero.
        let w = d.minimum_image([1.0 + 0.3, 0.5, 6.0]);
        assert!(
            (w[0] - 0.3).abs() < 1e-12 && w[1].abs() < 1e-12 && w[2].abs() < 1e-12,
            "{w:?}"
        );
    }

    #[test]
    fn damped_planes_keep_every_move_within_half_a_slab() {
        let old = [0.0, 0.25, 0.5, 0.75, 1.0];
        // Close target: moved all the way.
        let near = [0.0, 0.2, 0.45, 0.7, 1.0];
        assert_eq!(damped_planes(&old, &near), near.to_vec());
        // Far target: every plane moves by the same fraction, the largest
        // move is exactly half a slab, and planes stay ordered.
        let far = [0.0, 0.05, 0.1, 0.2, 1.0];
        let got = damped_planes(&old, &far);
        let max_move = (1..4).map(|r| (got[r] - old[r]).abs()).fold(0.0, f64::max);
        assert!((max_move - 0.125).abs() < 1e-12, "{got:?}");
        assert!(got.windows(2).all(|w| w[1] > w[0]));
        assert_eq!((got[0], got[4]), (0.0, 1.0));
        // Each atom can only change owner to an adjacent rank.
        for r in 1..4 {
            assert!(got[r] > old[r - 1] && got[r] < old[r + 1]);
        }
    }

    #[test]
    fn enforce_min_width_widens_thin_slabs() {
        let mut planes = vec![0.0, 0.01, 0.02, 0.5, 1.0];
        assert!(enforce_min_width(&mut planes, 0.2));
        assert!(
            planes.windows(2).all(|w| w[1] - w[0] >= 0.2 - 1e-12),
            "{planes:?}"
        );
        assert_eq!((planes[0], planes[4]), (0.0, 1.0));
        let mut tight = vec![0.0, 0.5, 1.0];
        assert!(!enforce_min_width(&mut tight, 0.6));
    }

    #[test]
    fn balanced_planes_equalizes_a_skewed_distribution() {
        // 10 bins over [0,10]; all mass piled in the left 3 bins (30,30,40 = 100).
        // Equal-volume split would put the plane at x=5 (0 particles right!);
        // the load-balanced plane must put ~50 particles each side.
        let mut hist = vec![0.0; 10];
        hist[0] = 30.0;
        hist[1] = 30.0;
        hist[2] = 40.0;
        let planes = balanced_planes(&hist, 0.0, 10.0, 2);
        assert_eq!(planes.len(), 3);
        assert_eq!(planes[0], 0.0);
        assert_eq!(planes[2], 10.0);
        // cum hits 50 inside bin 1: 30 + 20/30 of bin 1 -> x = 1 + 0.6667.
        assert!((planes[1] - 1.6667).abs() < 0.01, "plane = {}", planes[1]);
        // particle count left of the plane must be ~half the total.
        let left: f64 = (0..10)
            .map(|b| {
                let (bl, bh) = (b as f64, b as f64 + 1.0);
                if bh <= planes[1] {
                    hist[b]
                } else if bl >= planes[1] {
                    0.0
                } else {
                    hist[b] * (planes[1] - bl)
                }
            })
            .sum();
        assert!((left - 50.0).abs() < 1e-6, "left = {left}");
    }

    #[test]
    fn rebalance_x_conserves_the_global_atom_set() {
        // Substrate invariant: dynamic load balancing may only *re-partition* the
        // atoms across ranks — it must never lose or duplicate one. The existing
        // `balanced_planes_*` test only checks the plane-computation helper in
        // isolation; this drives the full `rebalance_x` → re-exchange cycle across
        // a real multi-rank decomposition and asserts the global atom set is
        // conserved (count + id/position). A gap between adjacent ranks' new
        // subdomains would silently drop atoms; an overlap would duplicate them —
        // both are corruption-class bugs any method on soil (DEM/MD/SPH/Peri)
        // would inherit.
        let px = 4usize;
        let (lo, hi) = (0.0f64, 40.0f64);
        let n: u32 = 1000;
        let nbins = 200usize;

        // Global atom set with a badly left-skewed x distribution (quadratic
        // pile-up near `lo`), so the initial uniform decomposition is imbalanced
        // and a rebalance genuinely has to move the planes. Positions stay
        // strictly inside (lo, hi) so no PBC remap is needed (u ∈ (0,1)).
        let mut atoms = Atom::new();
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            let x = lo + (hi - lo) * u * u;
            let y = (i % 10) as f64 / 10.0 + 0.005;
            let z = (i % 7) as f64 / 7.0 + 0.005;
            atoms.push_test_atom(i, [x, y, z], 0.1, 1.0);
        }
        atoms.nlocal = n;
        atoms.natoms = n as u64;

        // Snapshot the "before" set: tag → stored position (positions must be
        // carried through redecomposition untouched).
        let before: std::collections::HashMap<u32, [Real; 3]> = (0..n as usize)
            .map(|i| (atoms.tag[i], atoms.pos[i]))
            .collect();
        let global_count = n as usize;

        // Drive `rebalance_x` once per rank — exactly the call
        // `LoadBalancePlugin::load_balance_system` makes at `PreExchange`. In real
        // MPI each rank builds a *local* histogram and `all_reduce_sum`s it to the
        // identical global histogram, so every rank computes the same planes and
        // picks its own slice. Here we hand the full global atom set to an
        // identity-all_reduce `SingleProcessComm`: the per-rank local histogram
        // already equals the global one, so `rebalance_x` yields the same planes a
        // distributed run would (bin counts are exact integers — no rounding).
        let mut sub_lo = vec![0.0f64; px];
        let mut sub_hi = vec![0.0f64; px];
        for r in 0..px {
            let comm = make_comm([px as i32, 1, 1], [r as i32, 0, 0]);
            let config = DomainConfig {
                x_low: lo,
                x_high: hi,
                y_low: 0.0,
                y_high: 1.0,
                z_low: 0.0,
                z_high: 1.0,
                ..Default::default()
            };
            let mut domain = decompose_domain(&config, &comm);
            rebalance_x(&atoms, &mut domain, &comm, nbins);
            sub_lo[r] = domain.sub_domain_low[0];
            sub_hi[r] = domain.sub_domain_high[0];
            assert!(
                domain.bounds_changed,
                "rebalance_x must flag bounds_changed so the following Exchange re-migrates atoms",
            );
        }

        // The rebalanced subdomains must tile [lo, hi] with NO gap and NO overlap:
        // a gap loses atoms, an overlap duplicates them. This contiguity is the
        // conservation precondition the subsequent exchange relies on.
        assert_eq!(sub_lo[0], lo, "rank 0 must start at the global low edge");
        assert_eq!(
            sub_hi[px - 1],
            hi,
            "last rank must end at the global high edge"
        );
        for r in 0..px - 1 {
            assert_eq!(
                sub_hi[r],
                sub_lo[r + 1],
                "ranks {r}/{} subdomains must be contiguous (no gap/overlap)",
                r + 1,
            );
        }

        // Emulate the Exchange redecomposition: assign every atom to the rank whose
        // new subdomain [sub_lo, sub_hi) owns its x — the same half-open ownership
        // predicate `exchange()` uses. Counting owners per atom directly exposes
        // loss (0 owners) and duplication (>1 owner).
        let mut owned: Vec<Vec<usize>> = vec![Vec::new(); px];
        for i in 0..n as usize {
            let x = atoms.pos[i][0] as f64;
            let mut owners = 0;
            for r in 0..px {
                if x >= sub_lo[r] && x < sub_hi[r] {
                    owned[r].push(i);
                    owners += 1;
                }
            }
            assert_eq!(
                owners, 1,
                "atom tag {} (x={x}) has {owners} owners after rebalance (0 = lost, >1 = duplicated)",
                atoms.tag[i],
            );
        }

        // (1) Sum of owned local atom counts across ranks == original global count.
        let total_owned: usize = owned.iter().map(|v| v.len()).sum();
        assert_eq!(
            total_owned, global_count,
            "particle count changed across rebalance (lost or duplicated atoms)",
        );

        // (2) The set of (id, position) after rebalance == the set before. Rebuild
        // it from the per-rank owned lists and compare against the snapshot.
        let mut after: std::collections::HashMap<u32, [Real; 3]> = std::collections::HashMap::new();
        for r in 0..px {
            for &i in &owned[r] {
                let prev = after.insert(atoms.tag[i], atoms.pos[i]);
                assert!(
                    prev.is_none(),
                    "atom tag {} owned by two ranks",
                    atoms.tag[i]
                );
            }
        }
        assert_eq!(
            after.len(),
            before.len(),
            "atom id set size changed across rebalance"
        );
        for (tag, pos) in &before {
            assert_eq!(
                after.get(tag),
                Some(pos),
                "atom tag {tag} was lost or its position was disturbed by rebalance",
            );
        }

        // Sanity: the rebalance genuinely fired — the skewed input is heavily
        // imbalanced under the initial uniform decomposition (~half the atoms on
        // one rank) and must be materially more even afterwards. This guards
        // against a degenerate "did nothing" pass; the conservation asserts above
        // are the acceptance criterion.
        let ideal = global_count as f64 / px as f64;
        let max_after = owned.iter().map(|v| v.len()).max().unwrap() as f64;
        let mut uniform = vec![0usize; px];
        for i in 0..n as usize {
            let x = atoms.pos[i][0] as f64;
            let r = (((x - lo) / (hi - lo) * px as f64).floor() as usize).min(px - 1);
            uniform[r] += 1;
        }
        let max_uniform = *uniform.iter().max().unwrap() as f64;
        assert!(
            max_after < max_uniform,
            "rebalance did not improve balance (uniform max {max_uniform}, rebalanced max {max_after})",
        );
        assert!(
            max_after <= 1.5 * ideal,
            "rebalanced load still far from even (max {max_after}, ideal {ideal})",
        );
    }

    #[test]
    fn cartesian_single_proc_full_domain() {
        let config = DomainConfig {
            x_low: 0.0,
            x_high: 10.0,
            y_low: 0.0,
            y_high: 5.0,
            z_low: 0.0,
            z_high: 2.0,
            boundary_x: BoundaryType::Periodic,
            boundary_y: BoundaryType::Fixed,
            boundary_z: BoundaryType::Periodic,
            ..Default::default()
        };
        let comm = make_comm([1, 1, 1], [0, 0, 0]);
        let domain = decompose_domain(&config, &comm);

        assert_eq!(domain.boundaries_low, [0.0, 0.0, 0.0]);
        assert_eq!(domain.boundaries_high, [10.0, 5.0, 2.0]);
        assert_eq!(domain.sub_domain_low, [0.0, 0.0, 0.0]);
        assert_eq!(domain.sub_domain_high, [10.0, 5.0, 2.0]);
        assert!(domain.is_periodic(0));
        assert!(!domain.is_periodic(1));
        assert!(domain.is_periodic(2));
        assert!((domain.volume - 100.0).abs() < 1e-10);
    }

    #[test]
    fn cartesian_multi_proc_subdivides() {
        let config = DomainConfig {
            x_low: 0.0,
            x_high: 10.0,
            y_low: 0.0,
            y_high: 10.0,
            z_low: 0.0,
            z_high: 10.0,
            ..Default::default()
        };
        // Simulate proc at position (1,0,0) in a 2x1x1 decomposition
        let comm = make_comm([2, 1, 1], [1, 0, 0]);
        let domain = decompose_domain(&config, &comm);

        assert!((domain.sub_domain_low[0] - 5.0).abs() < 1e-10);
        assert!((domain.sub_domain_high[0] - 10.0).abs() < 1e-10);
        assert!((domain.sub_length[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn shrink_wrap_multi_rank_is_validation_error() {
        let config = DomainConfig {
            boundary_z: BoundaryType::ShrinkWrap,
            ..Default::default()
        };
        let comm = TestComm::new(2, 0, [2, 1, 1], [0, 0, 0]);

        let err = validate_domain_config(&config, &comm).unwrap_err();

        assert_eq!(err, DomainValidationError::ShrinkWrapWithMpi { ranks: 2 });
        let msg = err.to_string();
        assert!(msg.contains("shrink-wrap"), "{msg}");
        assert!(msg.contains("MPI rank count 2"), "{msg}");
    }

    #[test]
    fn shrink_wrap_single_rank_setup_is_valid() {
        let config = DomainConfig {
            x_low: -1.0,
            x_high: 3.0,
            boundary_z: BoundaryType::ShrinkWrap,
            shrink_wrap_padding: 0.25,
            ..Default::default()
        };
        let comm = make_comm([1, 1, 1], [0, 0, 0]);

        let domain = try_decompose_domain(&config, &comm).unwrap();

        assert_eq!(domain.boundary_type[2], BoundaryType::ShrinkWrap);
        assert_eq!(domain.shrink_wrap_padding, 0.25);
        assert_eq!(domain.sub_domain_low, [-1.0, 0.0, 0.0]);
        assert_eq!(domain.sub_domain_high, [3.0, 1.0, 1.0]);
    }

    #[test]
    fn fixed_and_periodic_multi_rank_setup_remains_valid() {
        let config = DomainConfig {
            x_low: 0.0,
            x_high: 12.0,
            boundary_x: BoundaryType::Periodic,
            boundary_y: BoundaryType::Fixed,
            boundary_z: BoundaryType::Periodic,
            ..Default::default()
        };
        let comm = TestComm::new(3, 1, [3, 1, 1], [1, 0, 0]);

        let domain = try_decompose_domain(&config, &comm).unwrap();

        assert_eq!(
            domain.boundary_type,
            [
                BoundaryType::Periodic,
                BoundaryType::Fixed,
                BoundaryType::Periodic
            ],
        );
        assert_eq!(domain.sub_domain_low, [4.0, 0.0, 0.0]);
        assert_eq!(domain.sub_domain_high, [8.0, 1.0, 1.0]);
    }

    // ── Boundary type tests ────────────────────────────────────────────────

    #[test]
    fn boundary_types_array() {
        let config = DomainConfig {
            boundary_x: BoundaryType::Periodic,
            boundary_y: BoundaryType::Fixed,
            boundary_z: BoundaryType::ShrinkWrap,
            ..Default::default()
        };
        let types = config.boundary_types();
        assert_eq!(
            types,
            [
                BoundaryType::Periodic,
                BoundaryType::Fixed,
                BoundaryType::ShrinkWrap
            ]
        );
    }

    #[test]
    fn shrink_wrap_decomposition() {
        let config = DomainConfig {
            x_low: 0.0,
            x_high: 10.0,
            y_low: 0.0,
            y_high: 10.0,
            z_low: 0.0,
            z_high: 20.0,
            boundary_z: BoundaryType::ShrinkWrap,
            ..Default::default()
        };
        let comm = make_comm([1, 1, 1], [0, 0, 0]);
        let domain = decompose_domain(&config, &comm);

        assert!(domain.is_periodic(0));
        assert!(domain.is_periodic(1));
        assert!(!domain.is_periodic(2));
        assert!(!domain.is_shrink_wrap(0));
        assert!(!domain.is_shrink_wrap(1));
        assert!(domain.is_shrink_wrap(2));
        assert_eq!(domain.boundary_type[2], BoundaryType::ShrinkWrap);
    }

    #[test]
    fn toml_parse_boundary_types() {
        let toml_str = r#"
            x_low = 0.0
            x_high = 1.0
            y_low = 0.0
            y_high = 1.0
            z_low = 0.0
            z_high = 1.0
            boundary_x = "periodic"
            boundary_y = "fixed"
            boundary_z = "shrink-wrap"
        "#;
        let config: DomainConfig = toml::from_str(toml_str).unwrap();
        let types = config.boundary_types();
        assert_eq!(
            types,
            [
                BoundaryType::Periodic,
                BoundaryType::Fixed,
                BoundaryType::ShrinkWrap
            ]
        );
    }

    // ── Shrink-wrap update tests ────────────────────────────────────────────

    // ── Shrink-wrap system tests ───────────────────────────────────────────

    #[test]
    fn shrink_wrap_expands_to_atom_positions() {
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [10.0, 10.0, 10.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [10.0, 10.0, 10.0];
        domain.size = [10.0, 10.0, 10.0];
        domain.sub_length = [10.0, 10.0, 10.0];
        domain.boundary_type = [
            BoundaryType::Periodic,
            BoundaryType::Periodic,
            BoundaryType::ShrinkWrap,
        ];
        domain.shrink_wrap_padding = 0.5;

        let positions = vec![[1.0, 2.0, 3.0], [5.0, 5.0, 8.0], [9.0, 1.0, 1.0]];

        let changed = super::shrink_wrap_update(&mut domain, &positions, 3);
        assert!(changed);
        // z bounds should wrap to [min_z - padding, max_z + padding] = [1.0 - 0.5, 8.0 + 0.5]
        assert!((domain.boundaries_low[2] - 0.5).abs() < 1e-10);
        assert!((domain.boundaries_high[2] - 8.5).abs() < 1e-10);
        // x and y should be unchanged (not shrink-wrap)
        assert!((domain.boundaries_low[0] - 0.0).abs() < 1e-10);
        assert!((domain.boundaries_high[0] - 10.0).abs() < 1e-10);
        assert!(domain.bounds_changed);
    }

    #[test]
    fn shrink_wrap_no_change_when_within_tolerance() {
        // Set bounds to exactly match what shrink_wrap would compute:
        // z positions are [3.0, 8.0, 1.0], so min=1.0, max=8.0
        // With padding=0.5: low=0.5, high=8.5
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.5];
        domain.boundaries_high = [10.0, 10.0, 8.5];
        domain.sub_domain_low = [0.0, 0.0, 0.5];
        domain.sub_domain_high = [10.0, 10.0, 8.5];
        domain.size = [10.0, 10.0, 8.0];
        domain.sub_length = [10.0, 10.0, 8.0];
        domain.boundary_type = [
            BoundaryType::Periodic,
            BoundaryType::Periodic,
            BoundaryType::ShrinkWrap,
        ];
        domain.shrink_wrap_padding = 0.5;
        domain.bounds_changed = false;

        let positions = vec![[1.0, 2.0, 3.0], [5.0, 5.0, 8.0], [9.0, 1.0, 1.0]];

        let changed = super::shrink_wrap_update(&mut domain, &positions, 3);
        assert!(!changed);
        assert!(!domain.bounds_changed);
    }

    #[test]
    fn shrink_wrap_no_atoms_is_noop() {
        let mut domain = Domain::new();
        domain.boundary_type = [
            BoundaryType::ShrinkWrap,
            BoundaryType::ShrinkWrap,
            BoundaryType::ShrinkWrap,
        ];
        domain.bounds_changed = false;
        let positions: Vec<[f64; 3]> = vec![];

        let changed = super::shrink_wrap_update(&mut domain, &positions, 0);
        assert!(!changed);
    }

    #[test]
    fn shrink_wrap_all_axes() {
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [100.0, 100.0, 100.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [100.0, 100.0, 100.0];
        domain.size = [100.0, 100.0, 100.0];
        domain.sub_length = [100.0, 100.0, 100.0];
        domain.boundary_type = [
            BoundaryType::ShrinkWrap,
            BoundaryType::ShrinkWrap,
            BoundaryType::ShrinkWrap,
        ];
        domain.shrink_wrap_padding = 1.0;

        let positions = vec![[10.0, 20.0, 30.0], [50.0, 60.0, 70.0]];

        let changed = super::shrink_wrap_update(&mut domain, &positions, 2);
        assert!(changed);
        // x: [10-1, 50+1] = [9, 51]
        assert!((domain.boundaries_low[0] - 9.0).abs() < 1e-10);
        assert!((domain.boundaries_high[0] - 51.0).abs() < 1e-10);
        // y: [20-1, 60+1] = [19, 61]
        assert!((domain.boundaries_low[1] - 19.0).abs() < 1e-10);
        assert!((domain.boundaries_high[1] - 61.0).abs() < 1e-10);
        // z: [30-1, 70+1] = [29, 71]
        assert!((domain.boundaries_low[2] - 29.0).abs() < 1e-10);
        assert!((domain.boundaries_high[2] - 71.0).abs() < 1e-10);
        // Derived fields should be updated
        assert!((domain.size[0] - 42.0).abs() < 1e-10);
        assert!((domain.volume - 42.0 * 42.0 * 42.0).abs() < 1e-6);
    }

    #[test]
    fn shrink_wrap_uses_ghost_cutoff_fallback() {
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [10.0, 10.0, 10.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [10.0, 10.0, 10.0];
        domain.size = [10.0, 10.0, 10.0];
        domain.sub_length = [10.0, 10.0, 10.0];
        domain.boundary_type = [
            BoundaryType::Periodic,
            BoundaryType::Periodic,
            BoundaryType::ShrinkWrap,
        ];
        domain.shrink_wrap_padding = 0.0; // no explicit padding
        domain.ghost_cutoff = 2.0; // should use this as fallback

        let positions = vec![[5.0, 5.0, 5.0]];
        super::shrink_wrap_update(&mut domain, &positions, 1);
        // z bounds: [5.0 - 2.0, 5.0 + 2.0] = [3.0, 7.0]
        assert!((domain.boundaries_low[2] - 3.0).abs() < 1e-10);
        assert!((domain.boundaries_high[2] - 7.0).abs() < 1e-10);
    }

    // ── Triclinic / lamda tests ─────────────────────────────────────────────

    fn tilted_domain() -> Domain {
        let mut d = Domain::new();
        d.boundaries_low = [0.0, 0.0, 0.0];
        d.boundaries_high = [4.0, 3.0, 2.0];
        d.size = [4.0, 3.0, 2.0];
        d.sub_domain_low = [0.0, 0.0, 0.0];
        d.sub_domain_high = [4.0, 3.0, 2.0];
        d.sub_length = [4.0, 3.0, 2.0];
        d.triclinic = true;
        d.tilt = [1.2, 0.0, 0.0]; // xy tilt
        d
    }

    #[test]
    fn lamda_roundtrip_with_tilt() {
        let d = tilted_domain();
        for &r in &[[0.3, 0.7, 0.9], [3.9, 2.9, 0.1], [2.0, 1.5, 1.0]] {
            let lam = d.x2lamda(r);
            let back = d.lamda2x(lam);
            for k in 0..3 {
                assert!(
                    (back[k] - r[k]).abs() < 1e-12,
                    "roundtrip {:?} -> {:?}",
                    r,
                    back
                );
            }
        }
        // A point on the y-high face maps to lamda y = 1.
        let lam = d.x2lamda([1.2, 3.0, 0.0]);
        assert!((lam[1] - 1.0).abs() < 1e-12);
        assert!((lam[0] - 0.0).abs() < 1e-12); // x - xy*λy = 1.2 - 1.2*1 = 0 -> λx = 0
    }

    #[test]
    fn triclinic_pbc_wraps_and_remaps_velocity() {
        let mut d = tilted_domain();
        d.boundary_vel = [5.0, 0.0, 0.0]; // Δv = γ̇·Ly streaming jump

        let mut atoms = Atom::new();
        // Atom just above the y-high face (λy ≈ 1.01) → should wrap down by b=(xy,Ly,0)
        // and have vx reduced by Δv (crossing to the slower bottom edge).
        atoms.push_test_atom(0, [2.0, 3.03, 1.0], 0.1, 1.0);
        atoms.vel[0] = [7.0, 0.0, 0.0];
        atoms.nlocal = 1;
        atoms.natoms = 1;

        let registry = AtomDataRegistry::new();
        let mut cs = CurrentState(CommState::FullRebuild);
        pbc_triclinic(&mut atoms, &d, &registry, &mut cs);

        // Position tolerance is precision-aware: under mixed/single, pos is stored
        // as f32 so the f64 wrap result round-trips with ~1e-7 error.
        let pos_tol = if cfg!(feature = "precision-double") {
            1e-9
        } else {
            1e-4
        };
        // y wrapped down by Ly = 3.0
        assert!(
            (atoms.pos[0][1] - 0.03).abs() < pos_tol,
            "y={}",
            atoms.pos[0][1]
        );
        // x shifted by -xy = -1.2 (the tilt carried by the b lattice vector)
        assert!(
            (atoms.pos[0][0] - 0.8).abs() < pos_tol,
            "x={}",
            atoms.pos[0][0]
        );
        // velocity remapped: vx -= Δv
        assert!(
            (atoms.vel[0][0] - 2.0).abs() < 1e-12,
            "vx={}",
            atoms.vel[0][0]
        );
        assert_eq!(atoms.image[0][1], 1);
    }

    #[test]
    fn domain_update_derived() {
        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [5.0, 10.0, 20.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [5.0, 10.0, 20.0];
        domain.update_derived();

        assert!((domain.size[0] - 5.0).abs() < 1e-10);
        assert!((domain.size[1] - 10.0).abs() < 1e-10);
        assert!((domain.size[2] - 20.0).abs() < 1e-10);
        assert!((domain.volume - 1000.0).abs() < 1e-10);
    }
}
