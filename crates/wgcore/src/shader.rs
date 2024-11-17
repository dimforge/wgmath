//! Trait for reusable gpu shaders.

use crate::hot_reloading::HotReloadState;
use naga_oil::compose::{Composer, ComposerError};
use std::path::{Path, PathBuf};
use wgpu::Device;

/// A composable gpu shader (with or without associated compute pipelines).
///
/// This trait serves as the basis for the shader compatibility feature of `wgcore`. If the
/// implementor of this trait is a struct and has no fields of type other than `ComputePipeline`,
/// thin this trait can be automatically derive using the `Shader` proc-macro:
/// ```.ignore
/// #[derive(Shader)]
/// #[shader(src = "composable.wgsl")]
/// struct ComposableShader;
/// ```
pub trait Shader: Sized {
    /// Path of the shader’s `.wgsl` file.
    const FILE_PATH: &'static str;

    /// Instantiates this `Shader` from a gpu `device`.
    ///
    /// This is generally used to instantiate all the `ComputeShader` fields of `self`.
    fn from_device(device: &wgpu::Device) -> Result<Self, ComposerError>;

    /// This shader’s sources.
    fn src() -> String;

    /// Add to `composer` the composable module definition of `Self` (if there are any) and all its
    /// shader dependencies .
    fn compose(composer: &mut Composer) -> Result<(), ComposerError>;

    /// A composer filled with the module definition of `Self` (if there is any) and all its
    /// shader dependencies.
    fn composer() -> Result<Composer, ComposerError> {
        let mut composer = Composer::default();
        Self::compose(&mut composer)?;
        Ok(composer)
    }

    /*
     * For hot-reloading.
     */
    /// Loads this shader from a file on disk.
    fn src_from_disk() -> String;
    fn from_disk(device: &wgpu::Device) -> Result<Self, ComposerError>;

    /// The absolute path of this source file.
    fn absolute_path() -> Option<PathBuf>;

    fn watch_sources(state: &mut HotReloadState) -> notify::Result<()>;
    fn needs_reload(state: &HotReloadState) -> bool;

    fn compose_from_disk(composer: &mut Composer) -> Result<(), ComposerError>;

    fn composer_from_disk() -> Result<Composer, ComposerError> {
        let mut composer = Composer::default();
        Self::compose_from_disk(&mut composer)?;
        Ok(composer)
    }

    fn reload_if_changed(
        &mut self,
        device: &Device,
        state: &HotReloadState,
    ) -> Result<bool, ComposerError> {
        if Self::needs_reload(state) {
            *self = Self::from_disk(device)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
