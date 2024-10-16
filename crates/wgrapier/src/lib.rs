#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

#[cfg(feature = "dim2")]
extern crate wgparry2d as wgparry;
#[cfg(feature = "dim3")]
extern crate wgparry3d as wgparry;

pub mod dynamics;
