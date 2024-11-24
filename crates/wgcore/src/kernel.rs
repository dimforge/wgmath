//! Utilities for queueing and dispatching kernels.

use crate::shapes::{ViewShape, ViewShapeBuffers};
use crate::timestamps::GpuTimestamps;
use std::sync::Arc;
use wgpu::{
    BindGroup, Buffer, CommandEncoder, ComputePass, ComputePassDescriptor, ComputePipeline, Device,
};

/// Trait implemented for workgroup sizes in gpu kernel invocations.
///
/// The purpose of this trait is mainly to be able to pass both a single `u32` or an
/// array `[u32; 3]` as the workgorup size in [`KernelInvocationBuilder::queue`].
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

/// A builder-pattern constructor for kernel invocations.
pub struct KernelInvocationBuilder<'a, 'b> {
    queue: &'b mut KernelInvocationQueue<'a>,
    pipeline: &'a ComputePipeline,
    bind_groups: Vec<(u32, BindGroup)>,
}

impl<'a, 'b> KernelInvocationBuilder<'a, 'b> {
    /// Initiates the creation of a kernel invocation for the given compute `pipeline`.
    ///
    /// Note that the invocation will not be queued into `queue` until
    /// [`KernelInvocationBuilder::queue`] or [`KernelInvocationBuilder::queue_indirect`] is called.
    pub fn new(queue: &'b mut KernelInvocationQueue<'a>, pipeline: &'a ComputePipeline) -> Self {
        Self {
            queue,
            pipeline,
            bind_groups: vec![],
        }
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
        let entries = inputs.map(|(input, binding)| wgpu::BindGroupEntry {
            binding,
            resource: input.as_entire_binding(),
        });
        let bind_group_layout = self.pipeline.get_bind_group_layout(bind_group_id);
        let bind_group = self
            .queue
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &entries,
            });
        self.bind_groups.push((bind_group_id, bind_group));

        self
    }

    /// Queues the kernel invocation into the `queue` that was given to
    /// [`KernelInvocationQueue::new`].
    ///
    /// The invocation will be configured with the given `workgroups` size (typically specified as
    /// a single `u32` or a `[u32; 3]`).
    pub fn queue(self, workgroups: impl WorkgroupSize) {
        let invocation = KernelInvocation {
            pipeline: self.pipeline,
            bind_groups: self.bind_groups,
            workgroups: Workgroups::Direct(workgroups.into_workgroups_size()),
        };
        self.queue.push(invocation)
    }

    /// Queues the indirect kernel invocation into the `queue` that was given to
    /// [`KernelInvocationQueue::new`].
    ///
    /// The invocation will be configured with an indirect `workgroups` size specified with a
    /// `Buffer` that must contain exactly one instance of [`wgpu::util::DispatchIndirectArgs`].
    pub fn queue_indirect(self, workgroups: Arc<Buffer>) {
        let invocation = KernelInvocation {
            pipeline: self.pipeline,
            bind_groups: self.bind_groups,
            workgroups: Workgroups::Indirect(workgroups),
        };
        self.queue.push(invocation)
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

/// A kernel invocation queued into a [`KernelInvocationQueue`].
///
/// See [`KernelInvocationBuilder`] for queueing a kernel invocation.
pub struct KernelInvocation<'a> {
    /// The compute pipeline to dispatch.
    pub pipeline: &'a ComputePipeline,
    /// The invocation’s bind groups.
    pub bind_groups: Vec<(u32, BindGroup)>,
    /// The dispatch’s direct or indirect workgroup sizes.
    pub workgroups: Workgroups,
}

impl<'a> KernelInvocation<'a> {
    /// Dispatch the kernel invocation defined by `self` as part of the given compute `pass`.
    pub fn dispatch<'b>(&'a self, pass: &mut ComputePass<'b>)
    where
        'a: 'b,
    {
        pass.set_pipeline(self.pipeline);
        for (id, bind_group) in &self.bind_groups {
            pass.set_bind_group(*id, bind_group, &[]);
        }

        match &self.workgroups {
            Workgroups::Direct(workgroups) => {
                pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
            }
            Workgroups::Indirect(workgroups) => {
                pass.dispatch_workgroups_indirect(workgroups, 0);
            }
        }
    }
}

enum Invocation<'a> {
    Kernel(KernelInvocation<'a>),
    Timestamp {
        query_index: u32,
    },
    ComputePass {
        label: &'a str,
        // TODO: this isn’t really good, as having the user know
        //       what timestamp id is associated to which compute
        //       pass is tricky.
        add_timestamps: bool,
    },
}

/// A kernel invocation queue.
///
/// This effectively describes a chain of Gpu operations waiting to be encoded.
/// The supported operations include kernel invocations queued with [`KernelInvocationBuilder`],
/// timestamp queries queued with [`KernelInvocationQueue::push_timestamp`]. It can also init
/// intermediary compute passes queued with [`KernelInvocationQueue::compute_pass`].
pub struct KernelInvocationQueue<'a> {
    device: &'a Device,
    // TODO: should this be stored separately so multiple queues share the same cache?
    shapes: ViewShapeBuffers,
    invocations: Vec<Invocation<'a>>,
}

impl<'a> KernelInvocationQueue<'a> {
    /// Inits a new invocation queue from the given `device`.
    ///
    /// This operation is cheap. Multiple invocation queues are allowed.
    pub fn new(device: &'a Device) -> Self {
        Self {
            device,
            shapes: ViewShapeBuffers::new(),
            invocations: vec![],
        }
    }

    /// The underlying wgpu device.
    pub fn device(&self) -> &Device {
        self.device
    }

    /// Gets or inits a uniform storage buffer containing a single [`ViewShape`] value equal to
    /// `shape`.
    ///
    /// Calling this method multiple times with the same `shape` will return the same buffer.
    pub fn shape_buffer(&self, shape: ViewShape) -> Arc<Buffer> {
        self.shapes.get(self.device, shape)
    }

    /// Queues a kernel dispatch.
    ///
    /// Note that the recommended way of queueing a kernel dispatch is with
    /// [`KernelInvocationBuilder::queue`] or [`KernelInvocationBuilder::queue_indirect`].
    pub fn push(&mut self, invocation: KernelInvocation<'a>) {
        self.invocations.push(Invocation::Kernel(invocation));
    }

    /// Queues a compute pass with the given label and optional timestamp queries.
    ///
    /// By queuing a compute pass, every subsequent queued kernel invocations will be dispatched
    /// in this queue.
    ///
    /// If `add_timestamps` is set to `true` then a pair of timestamp queries will
    /// be associated to this compute pass for measuring its runtime. See [`GpuTimestamps`] for
    /// details on timestamp queries. Note that these timestamp queries require the
    /// [`wgpu::Features::TIMESTAMP_QUERY`] feature to be supported and enabled.
    pub fn compute_pass(&mut self, label: &'a str, add_timestamps: bool) {
        self.invocations.push(Invocation::ComputePass {
            label,
            add_timestamps,
        });
    }

    // TODO: this isn’t very convenient
    /// Queues a timestamp query.
    ///
    /// This push a timestamp query in-between two kernel dispatches. Note that these timestamp
    /// queries require the [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`] to be supported and
    /// enabled.
    pub fn push_timestamp(&mut self, timestamps: &mut GpuTimestamps) -> Option<u32> {
        let query_index = timestamps.next_query_index();

        if let Some(query_index) = query_index {
            self.invocations.push(Invocation::Timestamp { query_index });
        }

        query_index
    }

    /// Encode everything from this queue into the given `encoder`.
    ///
    /// This will create compute passes, dispatch kernels, and setup timestamp queries. Note that
    /// if kernel dispatches were specified before any compute pass was queued with
    /// [`Self::compute_pass`], a default compute pass with no label and no timestamp query will
    /// be created first.
    ///
    /// If `timestamps` is set to `None`, all the timestamp queries in `self` will be ignored.
    /// If `timestamps` is `Some` but the remaining [`GpuTimestamps`]’s capacity is too small, any
    /// timestamp exceeding the capacity will be ignored.
    pub fn encode(&self, encoder: &mut CommandEncoder, mut timestamps: Option<&mut GpuTimestamps>) {
        if self.invocations.is_empty() {
            return;
        }

        let (mut pass, start) = if let Invocation::ComputePass {
            label,
            add_timestamps,
        } = &self.invocations[0]
        {
            let desc = ComputePassDescriptor {
                label: Some(*label),
                timestamp_writes: timestamps
                    .as_deref_mut()
                    .filter(|_| *add_timestamps)
                    .and_then(|ts| ts.next_compute_pass_timestamp_writes()),
            };

            (encoder.begin_compute_pass(&desc), 1)
        } else {
            (encoder.begin_compute_pass(&Default::default()), 0)
        };

        for invocation in &self.invocations[start..] {
            match invocation {
                Invocation::Kernel(kernel) => kernel.dispatch(&mut pass),
                Invocation::Timestamp { query_index } => {
                    if let Some(timestamps) = timestamps.as_deref_mut() {
                        timestamps.write_timestamp_at(&mut pass, *query_index);
                    }
                }
                Invocation::ComputePass {
                    label,
                    add_timestamps,
                } => {
                    drop(pass);
                    let desc = ComputePassDescriptor {
                        label: Some(*label),
                        timestamp_writes: timestamps
                            .as_deref_mut()
                            .filter(|_| *add_timestamps)
                            .and_then(|ts| ts.next_compute_pass_timestamp_writes()),
                    };
                    pass = encoder.begin_compute_pass(&desc);
                }
            }
        }
    }

    /// Dispatch avery kernels and timestamp queries from `self` into the given compute pass.
    ///
    /// Because the compute pass is specified explicitly, every compute pass queued with
    /// [`Self::compute_pass`] will be ignored.
    ///
    /// If `timestamps` is set to `None`, all the timestamp queries in `self` will be ignored.
    /// If `timestamps` is `Some` but the remaining [`GpuTimestamps`]’s capacity is too small, any
    /// timestamp exceeding the capacity will be ignored.
    pub fn dispatch<'b>(
        &'a self,
        pass: &mut ComputePass<'b>,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) where
        'a: 'b,
    {
        for invocation in &self.invocations {
            match invocation {
                Invocation::Kernel(kernel) => kernel.dispatch(pass),
                Invocation::Timestamp { query_index } => {
                    if let Some(timestamps) = timestamps.as_deref_mut() {
                        timestamps.write_timestamp_at(pass, *query_index);
                    }
                }
                Invocation::ComputePass { .. } => {
                    /* Don’t switch compute pass if one was provided by the user explicitly. */
                }
            }
        }
    }

    /// Clear everything from `self`.
    pub fn clear(&mut self) {
        self.invocations.clear();
    }
}
