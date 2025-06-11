#[allow(dead_code)]
pub mod builder;

#[allow(dead_code)]
pub mod reader;

#[allow(dead_code)]
pub mod vectors;

#[cfg(test)]
mod tests;

pub mod errors;
pub mod io;

mod metrovector_fbs {
    #![allow(unused_imports)]
    #![allow(dead_code)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(non_upper_case_globals)]
    #![allow(unsafe_op_in_unsafe_fn)]
    #![allow(clippy::all)]

    include!("../generated/mod.rs");
}

pub use metrovector_fbs::metrovector::fbs as mvf_fbs;

const METRO_MAGIC: [u8; 4] = *b"MVF1";
const METRO_FOOTER_SIZE: usize = 4;
