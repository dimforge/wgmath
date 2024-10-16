# wgmath − GPU scientific computing on every platform

<p align="center">
  <img src="https://wgmath.rs/img/wgmath_logo_w_padding.svg" alt="crates.io" height="200px">
</p>
<p align="center">
    <a href="https://discord.gg/vt9DJSW">
        <img src="https://img.shields.io/discord/507548572338880513.svg?logo=discord&colorB=7289DA">
    </a>
</p>

-----

**wgmath** is a set of [Rust](https://www.rust-lang.org/) libraries exposing re-usable GPU shaders for scientific
computing including:

- Linear algebra with the **wgebra** crate.
- AI (Large Language Models) with the **wgml** crate.
- Collision-detection with the **wgparry2d** and **wgparry3d** crates (still very WIP).
- Rigid-body physics with the **wgrapier2d** and **wgrapier3d** crates( (still very WIP).
- Non-rigid physics with the **.
  By targeting WebGPU, these libraries run on most GPUs, including on mobile and on the web. It aims to promote open and
  cross-platform GPU computing for scientific applications, a field currently strongly dominated by proprietary
  solutions (like CUDA).

All of the libraries are still under heavy development and might be lacking some important features. Contributions are
welcome!

In particular, the **wgcore** crate part of the **wgmath** ecosystem exposes a set of proc-macros to facilitate sharing
and composing shaders across Rust libraries.

See the readme of each individual crate (on the `crates` directory) for additional details.
