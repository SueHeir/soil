use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::registry::DumpFrame;

// Registered by `PrintPlugin::build` via the same public `register_format` API
// that plugins use, so core's formats and plugin formats are peers. All operate
// on a `DumpFrame`, so they are agnostic to per-rank vs. gathered output.

/// Append registered scalar/vector/ghost column NAMES with the given separator
/// and per-vector-component suffixes — shared by the text and LAMMPS headers.
fn append_column_names(header: &mut String, frame: &DumpFrame, sep: char) {
    for (name, _) in &frame.scalars {
        header.push(sep);
        header.push_str(name);
    }
    for (name, _) in &frame.vectors {
        header.push_str(&format!("{sep}{name}_x{sep}{name}_y{sep}{name}_z"));
    }
    if frame.is_ghost.is_some() {
        header.push(sep);
        header.push_str("IsGhost");
    }
}

/// Text/CSV dump: one row per atom, header line of column names.
pub(crate) fn write_dump_text(frame: &DumpFrame) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(format!("{}.csv", frame.path_stem))?);

    let mut header = "tag,type,x,y,z,vx,vy,vz,fx,fy,fz,radius".to_string();
    append_column_names(&mut header, frame, ',');
    writeln!(w, "{}", header)?;

    for i in 0..frame.n() {
        write!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            frame.tag[i],
            frame.atom_type[i],
            frame.pos[i][0],
            frame.pos[i][1],
            frame.pos[i][2],
            frame.vel[i][0],
            frame.vel[i][1],
            frame.vel[i][2],
            frame.force[i][0],
            frame.force[i][1],
            frame.force[i][2],
            frame.radius[i],
        )?;
        for (_, data) in &frame.scalars {
            write!(w, ",{}", data[i])?;
        }
        for (_, data) in &frame.vectors {
            write!(w, ",{},{},{}", data[i][0], data[i][1], data[i][2])?;
        }
        if let Some(g) = &frame.is_ghost {
            write!(w, ",{}", g[i] as u8)?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Binary dump: little-endian u32 count, then packed per-atom records
/// (`tag`/`type` as `u32`, everything else `f64`; IsGhost as a trailing `u8`).
pub(crate) fn write_dump_binary(frame: &DumpFrame) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(format!("{}.bin", frame.path_stem))?);

    w.write_all(&(frame.n() as u32).to_le_bytes())?;
    for i in 0..frame.n() {
        w.write_all(&(frame.tag[i] as u32).to_le_bytes())?;
        w.write_all(&(frame.atom_type[i] as u32).to_le_bytes())?;
        for c in 0..3 {
            w.write_all(&frame.pos[i][c].to_le_bytes())?;
        }
        for c in 0..3 {
            w.write_all(&frame.vel[i][c].to_le_bytes())?;
        }
        for c in 0..3 {
            w.write_all(&frame.force[i][c].to_le_bytes())?;
        }
        w.write_all(&frame.radius[i].to_le_bytes())?;
        for (_, data) in &frame.scalars {
            w.write_all(&data[i].to_le_bytes())?;
        }
        for (_, data) in &frame.vectors {
            for c in 0..3 {
                w.write_all(&data[i][c].to_le_bytes())?;
            }
        }
        if let Some(g) = &frame.is_ghost {
            w.write_all(&[g[i] as u8])?;
        }
    }
    Ok(())
}

/// LAMMPS dump (`.lammpstrj`) — OVITO reads this natively: `radius` renders
/// spheres, `type` colors by species, and registered scalar/vector columns
/// (e.g. a `mode` flag) become per-particle properties you can color by.
pub(crate) fn write_dump_lammps(frame: &DumpFrame) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(format!("{}.lammpstrj", frame.path_stem))?);

    // Global box bounds (same on every rank). Periodic -> "pp", else "ff".
    let bx = &frame.box_info;
    let tok = |d: usize| if bx.periodic[d] { "pp" } else { "ff" };
    writeln!(w, "ITEM: TIMESTEP")?;
    writeln!(w, "{}", frame.step)?;
    writeln!(w, "ITEM: NUMBER OF ATOMS")?;
    writeln!(w, "{}", frame.n())?;
    writeln!(w, "ITEM: BOX BOUNDS {} {} {}", tok(0), tok(1), tok(2))?;
    for d in 0..3 {
        writeln!(w, "{} {}", bx.low[d], bx.high[d])?;
    }

    let mut cols = "id type x y z vx vy vz fx fy fz radius".to_string();
    append_column_names(&mut cols, frame, ' ');
    writeln!(w, "ITEM: ATOMS {}", cols)?;

    for i in 0..frame.n() {
        write!(
            w,
            "{} {} {} {} {} {} {} {} {} {} {} {}",
            frame.tag[i],
            frame.atom_type[i],
            frame.pos[i][0],
            frame.pos[i][1],
            frame.pos[i][2],
            frame.vel[i][0],
            frame.vel[i][1],
            frame.vel[i][2],
            frame.force[i][0],
            frame.force[i][1],
            frame.force[i][2],
            frame.radius[i],
        )?;
        for (_, data) in &frame.scalars {
            write!(w, " {}", data[i])?;
        }
        for (_, data) in &frame.vectors {
            write!(w, " {} {} {}", data[i][0], data[i][1], data[i][2])?;
        }
        if let Some(g) = &frame.is_ghost {
            write!(w, " {}", g[i] as u8)?;
        }
        writeln!(w)?;
    }
    Ok(())
}
