//! Utilities for queueing and dispatching kernels.

use crate::timestamps::GpuTimestamps;
use std::sync::Arc;
use wgpu::{Buffer, CommandEncoder, ComputePass, ComputePassDescriptor, ComputePipeline, Device};

pub trait CommandEncoderExt {
    fn compute_pass<'encoder>(
        &'encoder mut self,
        label: &str,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> ComputePass<'encoder>;
}

impl CommandEncoderExt for CommandEncoder {
    fn compute_pass<'encoder>(
        &'encoder mut self,
        label: &str,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> ComputePass<'encoder> {
        let desc = ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: timestamps.and_then(|ts| ts.next_compute_pass_timestamp_writes()),
        };
        self.begin_compute_pass(&desc)
    }
}

/// Trait implemented for workgroup sizes in gpu kernel invocations.
///
/// The purpose of this trait is mainly to be able to pass both a single `u32` or an
/// array `[u32; 3]` as the workgorup size in [`KernelDispatch::dispatch`].
pub trait WorkgroupSize {
    /// Converts `self` into the actual workgroup sizes passed to the kernel invocation.
    fn into_workgroups_size(self) -> [u32; 3];
}

impl WorkgroupSize for u32 {
    fn into_workgroups_size(self) -> [u32; 3] {
        [self, 1, 1]
    }
}

impl WorkgroupSize for [u32; 3] {
    fn into_workgroups_size(self) -> [u32; 3] {
        self
    }
}

// TODO: remove the other KernelInvocation*.
pub struct KernelDispatch<'a, 'encoder> {
    device: &'a Device,
    pass: &'a mut ComputePass<'encoder>,
    pipeline: &'a ComputePipeline,
    queueable: bool,
}

impl<'a, 'encoder> KernelDispatch<'a, 'encoder> {
    pub fn new(
        device: &'a Device,
        pass: &'a mut ComputePass<'encoder>,
        pipeline: &'a ComputePipeline,
    ) -> Self {
        pass.set_pipeline(pipeline);
        Self {
            device,
            pass,
            pipeline,
            queueable: true,
        }
    }

    pub fn pass(&mut self) -> &mut ComputePass<'encoder> {
        self.pass
    }

    /// Binds `INPUTS` consecutive buffers to the bind group with id 0.
    ///
    /// This method is less versatile than [`Self::bind`] and [`Self::bind_at`] but covers one
    /// of the most common cases. This will bind `ipunts[i]` to the storage binding `i` of bind
    /// group 0.
    pub fn bind0<const INPUTS: usize>(self, inputs: [&Buffer; INPUTS]) -> Self {
        self.bind(0, inputs)
    }

    /// Binds `INPUTS` consecutive buffers to the bind group with id `bind_group_id`.
    ///
    /// This method is more versatile than [`Self::bind0`], but less than [`Self::bind_at`]. This
    /// will bind `ipunts[i]` to the storage binding `i` of bind group `bind_group_id`.
    pub fn bind<const INPUTS: usize>(self, bind_group_id: u32, inputs: [&Buffer; INPUTS]) -> Self {
        let mut inputs = inputs.map(|b| (b, 0));
        for (id, input) in inputs.iter_mut().enumerate() {
            input.1 = id as u32;
        }
        self.bind_at(bind_group_id, inputs)
    }

    /// Binds `INPUTS` buffers with arbitrary storage binding ids, to the bind group with id
    /// `bind_group_id`.
    ///
    /// This method is more versatile than [`Self::bind0`], and [`Self::bind`]. This
    /// will bind `inputs[i].0` to the storage binding `inputs[i].1` of bind group `bind_group_id`.
    pub fn bind_at<const INPUTS: usize>(
        mut self,
        bind_group_id: u32,
        inputs: [(&Buffer, u32); INPUTS],
    ) -> Self {
        let entries = inputs.map(|(input, binding)| {
            // TODO: 0 is not the only invalid binding size.
            //       See https://github.com/gfx-rs/wgpu/issues/253
            if input.size() == 0 {
                self.queueable = false;
            }

            wgpu::BindGroupEntry {
                binding,
                resource: input.as_entire_binding(),
            }
        });

        if !self.queueable {
            return self;
        }

        let bind_group_layout = self.pipeline.get_bind_group_layout(bind_group_id);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &entries,
        });
        self.pass.set_bind_group(bind_group_id, &bind_group, &[]);
        self
    }

    /// Queues the kernel invocation into the compute pass that was given to
    /// [`KernelDispatch::new`].
    ///
    /// The invocation will be configured with the given `workgroups` size (typically specified as
    /// a single `u32` or a `[u32; 3]`).
    pub fn dispatch(self, workgroups: impl WorkgroupSize) {
        let workgroup_size = workgroups.into_workgroups_size();

        // NOTE: we don’t need to queue if the workgroup is empty.
        if self.queueable && workgroup_size[0] * workgroup_size[1] * workgroup_size[2] > 0 {
            self.pass
                .dispatch_workgroups(workgroup_size[0], workgroup_size[1], workgroup_size[2]);
        }
    }

    /// Queues the indirect kernel invocation into the compute pass that was given to
    /// [`KernelDispatch::new`].
    ///
    /// The invocation will be configured with an indirect `workgroups` size specified with a
    /// `Buffer` that must contain exactly one instance of [`wgpu::util::DispatchIndirectArgs`].
    pub fn dispatch_indirect(self, workgroups: &Buffer) {
        if !self.queueable {
            return;
        }

        self.pass.dispatch_workgroups_indirect(workgroups, 0);
    }
}

/// Workgroup sizes for a direct or indirect dispatch.
pub enum Workgroups {
    /// Workgroup size for direct dispatch. Each element must be non-zero.
    Direct([u32; 3]),
    /// Workgroup for indirect dispatch. Must be a buffer containing exactly one instance of
    /// [`wgpu::util::DispatchIndirectArgs`].
    Indirect(Arc<Buffer>),
}
