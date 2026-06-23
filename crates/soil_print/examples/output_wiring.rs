//! Wiring SOIL output: registering a custom dump column, a custom dump format,
//! and configuring thermo columns.
//!
//! This shows the public extension API of [`soil_print`] end to end. It does not
//! drive a full run loop (that needs the framework's run/config plugins); it
//! exercises the registration surface a physics tier actually calls from its
//! plugin `build()`:
//!
//! - [`DumpRegistry::register_scalar`] / [`DumpRegistry::register_vector`] —
//!   add per-atom columns that appear in CSV/LAMMPS dumps and VTP output.
//! - [`DumpRegistry::register_format`] — add a whole new dump file format.
//! - [`Thermo::set`] — publish a named value that shows up as a thermo column
//!   when listed in `[thermo] columns`.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example output_wiring
//! ```

use std::any::TypeId;

use grass_app::prelude::*;
use soil_core::{Atom, AtomDataRegistry, AtomPlugin};
use soil_print::{DumpFrame, DumpRegistry, PrintPlugin, Thermo};

fn main() {
    let mut app = App::new();

    // `PrintPlugin` registers the `Thermo` and `DumpRegistry` resources (and the
    // built-in "text"/"binary"/"lammps" formats). `AtomPlugin` gives us the
    // `Atom` + `AtomDataRegistry` the dump callbacks receive.
    app.add_plugins(AtomPlugin).add_plugins(PrintPlugin);

    // ── Register a per-atom dump/VTP column ──────────────────────────────────
    // `register_scalar` takes `&self` (interior mutability), so this is exactly
    // what a plugin does from `build()` with `get_resource_ref::<DumpRegistry>()`.
    {
        let dump_reg = app.get_resource_ref::<DumpRegistry>().unwrap();

        // A scalar column: speed |v|. The callback returns one value per LOCAL
        // atom; ghosts (if dumped) are padded automatically.
        dump_reg.register_scalar("speed", |atoms: &Atom, _reg: &AtomDataRegistry| {
            let n = atoms.nlocal as usize;
            (0..n)
                .map(|i| {
                    let v = atoms.vel[i];
                    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
                })
                .collect()
        });

        // A 3-vector column: the force. Splits into force_x/_y/_z in CSV/LAMMPS,
        // appears as a 3-component array in VTP.
        dump_reg.register_vector("force", |atoms: &Atom, _reg: &AtomDataRegistry| {
            let n = atoms.nlocal as usize;
            (0..n).map(|i| atoms.force[i]).collect()
        });

        // A custom dump format, selected at runtime by `[dump] format = "xyz"`.
        // The writer gets a fully-built (and, when gathered, MPI-merged) frame.
        dump_reg.register_format("xyz", |frame: &DumpFrame| {
            let mut s = format!("{}\n\n", frame.n());
            for i in 0..frame.n() {
                let p = frame.pos[i];
                s.push_str(&format!("C {} {} {}\n", p[0], p[1], p[2]));
            }
            // A real writer would create a file from `frame.path_stem`; we just
            // show the shape of the data here.
            let _ = s;
            Ok(())
        });

        println!("registered dump formats: {:?}", dump_reg.format_names());
        println!("has per-atom callbacks: {}", dump_reg.has_callbacks());
    }

    // ── Publish a custom thermo value ────────────────────────────────────────
    // Anything pushed via `Thermo::set` becomes available as a thermo column when
    // its name is listed in `[thermo] columns` (e.g. `columns = ["step", "pe"]`).
    // Resources are stored as `RefCell<Box<dyn Any>>`; borrow and downcast.
    {
        let cell = app.get_mut_resource(TypeId::of::<Thermo>()).unwrap();
        let mut binder = cell.borrow_mut();
        let thermo = binder.downcast_mut::<Thermo>().unwrap();
        thermo.set("pe", 1.234e-3);
        println!("thermo interval = {} steps", thermo.interval);
    }
}
