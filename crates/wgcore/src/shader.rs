//! Trait for reusable gpu shaders.

use crate::hot_reloading::HotReloadState;
use dashmap::DashMap;
use naga_oil::compose::{Composer, ComposerError};
use std::any::TypeId;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use wgpu::naga::Module;
use wgpu::{Device, Label, ShaderModule};

/// The global shader registry used by various auto-implemented method of `Shader` for loading the
/// shader.
///
/// To access the global shader registry, call [`ShaderRegistry::get`].
/// Whenever a shader source is needed (e.g. as a dependency of another, for instantiating one
/// of its kernel, for hot-reloading), the path registered in this map will take precedence in the
/// automatically-generated implementation of [`Shader::wgsl_path`]. If no path is provided
/// by this registry, the absolute path detected automatically by the `derive(Shader)` will be
/// applied. If neither exist, the shader loading code will default to the shader sources that
/// were embedded at the time of compilation of the module.
#[derive(Debug, Default, Clone)]
pub struct ShaderRegistry {
    paths: DashMap<TypeId, PathBuf>,
}

impl ShaderRegistry {
    /// Gets the global shader registry used by various auto-implemented method of `Shader` for loading the
    /// shader.
    pub fn get() -> &'static ShaderRegistry {
        static SHADER_REGISTRY: OnceLock<ShaderRegistry> = OnceLock::new();
        SHADER_REGISTRY.get_or_init(ShaderRegistry::default)
    }

    /// Registers the path for the given shader.
    ///
    /// Whenever the shader sources is needed as a dependency or as a kernel, it will be loaded
    /// from disk from this file path. This overwrites any previously registered path.
    pub fn set_path<T: Shader>(&self, path: PathBuf) {
        self.paths.insert(TypeId::of::<T>(), path);
    }

    /// Gets the registered path, if any, for the shader `T`.
    pub fn get_path<T: Shader>(&self) -> Option<PathBuf> {
        self.paths.get(&TypeId::of::<T>()).map(|p| p.clone())
    }

    /// Unregisters the path for the given shader.
    pub fn remove_path<T: Shader>(&self) {
        self.paths.remove(&TypeId::of::<T>());
    }
}

/// A composable gpu shader (with or without associated compute pipelines).
///
/// This trait serves as the basis for the shader compatibility feature of `wgcore`. If the
/// implementor of this trait is a struct and has no fields of type other than `ComputePipeline`,
/// thin this trait can be automatically derive using the `Shader` proc-macro:
/// ```.ignore
/// #[derive(Shader)]
/// #[shader(src = "compose_dependency.wgsl")]
/// struct ComposableShader;
/// ```
pub trait Shader: Sized + 'static {
    /// Path of the shader’s `.wgsl` file.
    const FILE_PATH: &'static str;

    /// Instantiates this `Shader` from a gpu `device`.
    ///
    /// This is generally used to instantiate all the `ComputeShader` fields of `self`.
    fn from_device(device: &wgpu::Device) -> Result<Self, ComposerError>;

    /// This shader’s sources (before dependency and macro resolution).
    fn src() -> String;

    /// This shader’s WGSL sources as a single file (after dependency and macro resolution).
    fn flat_wgsl() -> Result<String, ComposerError> {
        let module = Self::naga_module()?;
        Ok(crate::utils::naga_module_to_wgsl(&module))
    }

    /// The naga [`Module`] built from this shader.
    fn naga_module() -> Result<Module, ComposerError>;

    /// The [`ShaderModule`] built from this shader.
    fn shader_module(device: &wgpu::Device, label: Label) -> Result<ShaderModule, ComposerError> {
        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(Self::naga_module()?)),
        }))
    }

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

    /// The path of this wgsl shader source file.
    ///
    /// This returns the path from the global [`ShaderRegistry`] if it was set. Otherwise, this returns
    /// the path automatically-computed by the `derive(Shader)`. If that failed too, returns `None`.
    fn wgsl_path() -> Option<PathBuf>;

    /// Registers in the global [`ShaderRegistry`] known path for this shader.
    ///
    /// Any function form `Self` relying on the shader’s path, including hot-reloading,
    /// will rely on this path. Note that calling [`Self::watch_sources`] is necessary for
    /// hot-reloading to automatically detect changes at the new path.
    fn set_wgsl_path<P: AsRef<Path>>(path: P) {
        ShaderRegistry::get()
            .paths
            .insert(TypeId::of::<Self>(), path.as_ref().to_path_buf());
    }

    /// Registers all the source files, for `Self` and all its shader dependencies, for change
    /// detection.
    fn watch_sources(state: &mut HotReloadState) -> notify::Result<()>;

    /// Checks if this shader (or any of its dependencies) need to be reloaded due to a change
    /// from disk.
    fn needs_reload(state: &HotReloadState) -> bool;

    /// Reloads this shader if it on any of its dependencies have been changed from disk.
    fn reload_if_changed(
        &mut self,
        device: &Device,
        state: &HotReloadState,
    ) -> Result<bool, ComposerError> {
        if Self::needs_reload(state) {
            *self = Self::from_device(device)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
