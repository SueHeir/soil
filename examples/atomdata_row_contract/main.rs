//! Executable validation for the `AtomData` row-count registration contract.
use soil_core::{Atom, AtomData, AtomDataRegistrationError, AtomDataRegistry, ParticleStore};
use soil_derive::AtomData as DeriveAtomData;
use std::{any::Any, env, fs};

#[derive(Default, DeriveAtomData)]
struct DerivedColumn {
    scalar: Vec<f64>,
    vector: Vec<[f64; 3]>,
}

struct ManualMultiColumn {
    scalar: Vec<f64>,
    vector: Vec<[f64; 3]>,
}
impl AtomData for ManualMultiColumn {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn snapshot(&self) -> Box<dyn AtomData> {
        Box::new(Self {
            scalar: self.scalar.clone(),
            vector: self.vector.clone(),
        })
    }
    fn len(&self) -> usize {
        self.scalar.len()
    }
    unsafe fn push_default(&mut self) {
        self.scalar.push(0.0);
        self.vector.push([0.0; 3]);
    }
    fn validate_rows(&self, expected: usize) -> Result<(), AtomDataRegistrationError> {
        for actual in [self.scalar.len(), self.vector.len()] {
            if actual != expected {
                return Err(AtomDataRegistrationError::RowCountMismatch {
                    type_name: std::any::type_name::<Self>(),
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
    unsafe fn truncate(&mut self, n: usize) {
        self.scalar.truncate(n);
        self.vector.truncate(n);
    }
    unsafe fn swap_remove(&mut self, i: usize) {
        self.scalar.swap_remove(i);
        self.vector.swap_remove(i);
    }
    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        buf.push(self.scalar[i]);
        buf.extend_from_slice(&self.vector[i]);
    }
    unsafe fn unpack(&mut self, buf: &[f64]) -> usize {
        self.scalar.push(buf[0]);
        self.vector.push([buf[1], buf[2], buf[3]]);
        4
    }
    unsafe fn apply_permutation(&mut self, _perm: &[usize], _n: usize) {}
}

fn main() {
    let config = env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/atomdata_row_contract/config.toml".to_string());
    let expected = fs::read_to_string(config)
        .expect("read declarative AtomData row-contract config")
        .lines()
        .find_map(|line| line.strip_prefix("atom_count = "))
        .expect("config contains atom_count")
        .parse::<usize>()
        .expect("atom_count is an unsigned integer");
    let mut atoms = Atom::new();
    let mut registry = AtomDataRegistry::new();
    for _ in 0..expected {
        ParticleStore::new(&mut atoms, &registry)
            .push_default_local(expected as u64)
            .unwrap();
    }
    registry
        .register_for_atoms(DerivedColumn::default(), &atoms)
        .unwrap();
    let accepted_rows = registry.get::<DerivedColumn>().unwrap().len();
    println!("ATOMDATA_ROW case=late_derived expected={expected} observed={accepted_rows} accepted=true pass={}", accepted_rows == expected);
    let malformed = ManualMultiColumn {
        scalar: vec![0.0; expected],
        vector: vec![[0.0; 3]; expected - 1],
    };
    let hidden_rows = malformed.vector.len();
    let rejected = matches!(AtomDataRegistry::new().register_for_atoms(malformed, &atoms), Err(AtomDataRegistrationError::RowCountMismatch { actual, .. }) if actual == hidden_rows);
    println!("ATOMDATA_ROW case=manual_multi_mismatch expected={expected} observed={hidden_rows} accepted=false pass={rejected}");
    assert_eq!(accepted_rows, expected);
    assert!(
        rejected,
        "the malformed manual multi-column store must be rejected"
    );
}
