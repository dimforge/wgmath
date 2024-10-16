//! Trait for reusable gpu shaders.

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
pub trait Shader {
    /// Path of the shader’s `.wgsl` file.
    const FILE_PATH: &'static str;

    /// Instantiates this `Shader` from a gpu `device`.
    ///
    /// This is generally used to instantiate all the `ComputeShader` fields of `self`.
    fn from_device(device: &wgpu::Device) -> Self;
    /// This shader’s sources.
    fn src() -> String;
    /// Add to `composer` the composable module definition of `Self` (if there are any) and all its
    /// shader dependencies .
    fn compose(composer: &mut naga_oil::compose::Composer) -> &mut naga_oil::compose::Composer;
    /// A composer filled with the module definition of `Self` (if there is any) and all its
    /// shader dependencies.
    fn composer() -> naga_oil::compose::Composer {
        let mut composer = crate::re_exports::Composer::default();
        Self::compose(&mut composer);
        composer
    }
}
