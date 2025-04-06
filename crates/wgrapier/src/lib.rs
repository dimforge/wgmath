#![doc = include_str!("../README.md")]
// #![warn(missing_docs)]

#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;
#[cfg(feature = "dim2")]
pub extern crate wgparry2d as wgparry;
#[cfg(feature = "dim3")]
pub extern crate wgparry3d as wgparry;

pub mod dynamics;
