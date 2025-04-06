use crate::linalg::shape::Shape;
use bytemuck::Pod;
use naga_oil::compose::{ComposerError, NagaModuleDescriptor};
use naga_oil::redirect::Redirector;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::GpuVectorView;
use wgcore::utils;
use wgcore::Shader;
use wgpu::{ComputePipeline, Device};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
/// The desired operation for the [`OpAssign`] kernel.
pub enum OpAssignVariant {
    /// Sum: `a += b`
    Add,
    /// Subtraction: `a -= b`
    Sub,
    /// Product: `a *= b`
    Mul,
    /// Division: `a /= b`
    Div,
    /// Copy: `a = b`
    Copy,
}

impl OpAssignVariant {
    fn kernel_fn(self) -> &'static str {
        match self {
            Self::Add => "add_f32",
            Self::Sub => "sub_f32",
            Self::Mul => "mul_f32",
            Self::Div => "div_f32",
            Self::Copy => "copy_f32",
        }
    }
}

// TODO: we could probably use proc-macros to specify the modules this
//       has to be composed with?
/// A GPU kernel for performing the operation described by [`OpAssignVariant`].
pub struct OpAssign(pub ComputePipeline, pub OpAssignVariant);

impl OpAssign {
    /// WGSL source file for `OpAssign`.
    pub const SRC: &'static str = include_str!("op_assign.wgsl");
    /// The WGSL file path.
    pub const FILE_PATH: &'static str = "wgebra/src/op_assign.wgsl";

    /// Creates the compute pipeline for the operation described by the given [`OpAssignVariant`].
    pub fn new(device: &Device, op: OpAssignVariant) -> Result<Self, ComposerError> {
        let module = Shape::composer()?.make_naga_module(NagaModuleDescriptor {
            source: Self::SRC,
            file_path: Self::FILE_PATH,
            ..Default::default()
        })?;
        let mut redirector = Redirector::new(module);
        redirector
            .redirect_function("placeholder", op.kernel_fn(), &Default::default())
            .unwrap();

        Ok(OpAssign(
            utils::load_module(device, "main", redirector.into_module().unwrap()),
            op,
        ))
    }

    /// Queues the operation for computing `in_out_a ?= in_b` where `?` depends on the
    /// [`OpAssignVariant`] selected when creating `self`.
    pub fn queue<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        in_out_a: impl Into<GpuVectorView<'b, T>>,
        in_b: impl Into<GpuVectorView<'b, T>>,
    ) {
        let in_out_a = in_out_a.into();
        let in_b = in_b.into();

        assert_eq!(
            in_out_a.shape().size[0],
            in_b.shape().size[0],
            "Op-assign: dimension mismatch."
        );

        let a_shape_buf = queue.shape_buffer(in_out_a.shape());
        let b_shape_buf = queue.shape_buffer(in_b.shape());

        KernelInvocationBuilder::new(queue, &self.0)
            .bind0([&a_shape_buf, &b_shape_buf, in_out_a.buffer(), in_b.buffer()])
            .queue(in_out_a.shape().size[0].div_ceil(64));
    }
}

#[cfg(test)]
mod test {
    use super::{OpAssign, OpAssignVariant};
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::TensorBuilder;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_op_assign() {
        let ops = [
            OpAssignVariant::Add,
            OpAssignVariant::Sub,
            OpAssignVariant::Mul,
            OpAssignVariant::Div,
        ];
        let gpu = GpuInstance::new().await.unwrap();

        for op in ops {
            let op_assign = OpAssign::new(gpu.device(), op).unwrap();
            let mut queue = KernelInvocationQueue::new(gpu.device());
            let mut encoder = gpu.device().create_command_encoder(&Default::default());

            const LEN: u32 = 1757;

            let v0 = DVector::from_fn(LEN as usize, |i, _| i as f32 + 0.1);
            let v1 = DVector::from_fn(LEN as usize, |i, _| i as f32 * 10.0 + 0.1);
            let gpu_v0 = TensorBuilder::vector(LEN, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
                .build_init(gpu.device(), v0.as_slice());
            let gpu_v1 = TensorBuilder::vector(LEN, BufferUsages::STORAGE)
                .build_init(gpu.device(), v1.as_slice());
            let staging =
                TensorBuilder::vector(LEN, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
                    .build(gpu.device());

            op_assign.queue(&mut queue, &gpu_v0, &gpu_v1);

            queue.encode(&mut encoder, None);
            staging.copy_from(&mut encoder, &gpu_v0);

            gpu.queue().submit(Some(encoder.finish()));

            let cpu_result = match op {
                OpAssignVariant::Add => v0 + v1,
                OpAssignVariant::Sub => v0 - v1,
                OpAssignVariant::Mul => v0.component_mul(&v1),
                OpAssignVariant::Div => v0.component_div(&v1),
                OpAssignVariant::Copy => v1.clone(),
            };

            approx::assert_relative_eq!(
                DVector::from(staging.read(gpu.device()).await.unwrap()),
                cpu_result,
                epsilon = 1.0e-7
            );
        }
    }
}
