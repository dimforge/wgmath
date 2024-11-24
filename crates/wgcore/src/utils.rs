//! Utilities for creating a ComputePipeline from source or from a naga module.

use wgpu::naga::Module;
use wgpu::{ComputePipeline, ComputePipelineDescriptor, Device};

/// Creates a compute pipeline from the shader sources `content` and the name of its `entry_point`.
pub fn load_shader(device: &Device, entry_point: &str, content: &str) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(content)),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: None,
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Creates a compute pipeline from the shader `module` and the name of its `entry_point`.
pub fn load_module(device: &Device, entry_point: &str, module: Module) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: None,
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Convents a naga module to its WGSL string representation.
pub fn naga_module_to_wgsl(module: &Module) -> String {
    use wgpu::naga;

    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), Default::default());
    let info = validator
        .validate(module)
        .unwrap();

    naga::back::wgsl::write_string(
        module,
        &info,
        naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
    ).unwrap()
}