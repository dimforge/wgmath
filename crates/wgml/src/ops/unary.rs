use bytemuck::Pod;
use naga_oil::compose::{ComposerError, NagaModuleDescriptor};
use naga_oil::redirect::Redirector;
use nalgebra::{Dyn, StorageMut, Vector, Vector4};
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuScalar, GpuVectorView};
use wgcore::utils;
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgpu::{ComputePipeline, Device};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
/// Listing of all unary operations that can be applied by the [`Unary`] kernel.
pub enum UnaryOp {
    Abs,
    Sgn,
    Neg,
    Step,
    Elu,
    Gelu,
    GeluQuick,
    Silu,
    Tanh,
    Relu,
    Sigmoid,
    HardSigmoid,
    HardSwish,
    Sqr,
    Sqrt,
    Log,
    // Unary ops with extra args.
    LeakyRelu,
    Clamp,
    Scale,
    AddScalar, // Named GGML_OP_ADD1 in ggml.
}

impl UnaryOp {
    const fn has_args(self) -> bool {
        match self {
            Self::Abs
            | Self::Sgn
            | Self::Neg
            | Self::Step
            | Self::Elu
            | Self::Gelu
            | Self::GeluQuick
            | Self::Silu
            | Self::Tanh
            | Self::Relu
            | Self::Sigmoid
            | Self::HardSigmoid
            | Self::HardSwish
            | Self::Sqr
            | Self::Sqrt
            | Self::Log => false,
            Self::LeakyRelu | Self::Clamp | Self::Scale | Self::AddScalar => true,
        }
    }
    const fn kernel_fn(self) -> &'static str {
        match self {
            Self::Abs => "abs_f32",
            Self::Sgn => "sgn_f32",
            Self::Neg => "neg_f32",
            Self::Step => "step_f32",
            Self::Elu => "elu_f32",
            Self::Gelu => "gelu_f32",
            Self::GeluQuick => "gelu_quick_f32",
            Self::Silu => "silu_f32",
            Self::Tanh => "tanh_f32",
            Self::Relu => "relu_f32",
            Self::Sigmoid => "sigmoid_f32",
            Self::HardSigmoid => "hard_sigmoid_f32",
            Self::HardSwish => "hard_swish_f32",
            Self::Sqr => "sqr_f32",
            Self::Sqrt => "sqrt_f32",
            Self::Log => "log_f32",
            Self::LeakyRelu => "leaky_relu_f32",
            Self::Clamp => "clamp_f32",
            Self::Scale => "scale_f32",
            Self::AddScalar => "add_scalar_f32",
        }
    }

    pub fn eval(self, x: f32, args: Vector4<f32>) -> f32 {
        match self {
            Self::Abs => x.abs(),
            Self::Sgn => x.signum(),
            Self::Neg => -x,
            Self::Step => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Self::Gelu => {
                const GELU_COEF_A: f32 = 0.044715;
                const SQRT_2_OVER_PI: f32 = 0.7978846;
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
            }
            Self::GeluQuick => {
                const GELU_QUICK_COEF: f32 = -1.702;
                x * (1.0 / (1.0 + (GELU_QUICK_COEF * x).exp()))
            }
            Self::Silu => x / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::Relu => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::HardSigmoid => 1.0f32.min(0.0f32.max((x + 3.0) / 6.0)),
            Self::HardSwish => x * 1.0f32.min(0.0f32.max((x + 3.0) / 6.0)),
            Self::Sqr => x * x,
            Self::Sqrt => x.sqrt(),
            Self::Log => x.ln(),
            Self::LeakyRelu => x.max(0.0) + x.min(0.0) * args.x,
            Self::Clamp => x.clamp(args.x, args.y),
            Self::Scale => x * args.x,
            Self::AddScalar => x + args.x,
        }
    }
}

/// Shader implementing various unary operations selected with [`UnaryOp`].
pub struct Unary(pub ComputePipeline, pub UnaryOp);

impl Unary {
    pub const SRC: &'static str = include_str!("unary.wgsl");
    pub const FILE_PATH: &'static str = "wgml/src/unary.wgsl";

    pub fn new(device: &Device, op: UnaryOp) -> Result<Self, ComposerError> {
        let module = Shape::composer()?.make_naga_module(NagaModuleDescriptor {
            source: Self::SRC,
            file_path: Self::FILE_PATH,
            ..Default::default()
        })?;
        let mut redirector = Redirector::new(module);
        redirector
            .redirect_function("placeholder", op.kernel_fn(), &Default::default())
            .unwrap();

        Ok(Self(
            utils::load_module(device, "main", redirector.into_module().unwrap()),
            op,
        ))
    }

    pub fn queue<'a, 'b, T: Pod + nalgebra::Scalar>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        dest: impl Into<GpuVectorView<'b, T>>,
        src: impl Into<GpuVectorView<'b, T>>,
        args: Option<&'b GpuScalar<Vector4<T>>>,
    ) {
        let dest = dest.into();
        let src = src.into();
        let shape_dest = queue.shape_buffer(dest.shape());
        let shape_src = queue.shape_buffer(src.shape());
        let workgroups = [dest.len().div_ceil(64), 1, 1];

        assert!(
            self.1.has_args() == args.is_some(),
            "Unary ops argument mismatch."
        );

        if let Some(args) = &args {
            let inputs = [
                &shape_dest,
                &shape_src,
                dest.buffer(),
                src.buffer(),
                args.buffer(),
            ];
            KernelInvocationBuilder::new(queue, &self.0)
                .bind0(inputs)
                .queue(workgroups);
        } else {
            let inputs = [&shape_dest, &shape_src, dest.buffer(), src.buffer()];
            KernelInvocationBuilder::new(queue, &self.0)
                .bind0(inputs)
                .queue(workgroups);
        };
    }

    pub fn run_cpu<S: StorageMut<f32, Dyn>>(
        &self,
        vals: &mut Vector<f32, Dyn, S>,
        args: Vector4<f32>,
    ) {
        vals.apply(|x| *x = self.1.eval(*x, args));
    }
}

#[cfg(test)]
mod test {
    use crate::ops::UnaryOp;
    use nalgebra::{DVector, Vector4};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::{GpuScalar, GpuVector};
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_unary_ops() {
        let ops = [
            UnaryOp::Abs,
            UnaryOp::Sgn,
            UnaryOp::Neg,
            UnaryOp::Step,
            UnaryOp::Elu,
            UnaryOp::Gelu,
            UnaryOp::GeluQuick,
            UnaryOp::Silu,
            UnaryOp::Tanh,
            UnaryOp::Relu,
            UnaryOp::Sigmoid,
            UnaryOp::HardSigmoid,
            UnaryOp::HardSwish,
            UnaryOp::Sqr,
            UnaryOp::Sqrt,
            UnaryOp::Log,
            UnaryOp::LeakyRelu,
            UnaryOp::Clamp,
            UnaryOp::Scale,
            UnaryOp::AddScalar,
        ];
        let gpu = GpuInstance::new().await.unwrap();

        for op in ops {
            println!("Checking {:?}", op);
            let unop = super::Unary::new(gpu.device(), op);
            let mut queue = KernelInvocationQueue::new(gpu.device());
            let mut encoder = gpu.device().create_command_encoder(&Default::default());

            const LEN: u32 = 1757;

            let device = gpu.device();
            let src = DVector::new_random(LEN as usize);
            let dst = DVector::new_random(LEN as usize);
            let mut args = Vector4::new_random();
            if args[1] < args[0] {
                args.swap_rows(0, 1); // Ensure min <= max for clamp.
            }
            let gpu_args = op
                .has_args()
                .then(|| GpuScalar::init(device, args, BufferUsages::UNIFORM));
            let gpu_src = GpuVector::init(device, &src, BufferUsages::STORAGE);
            let gpu_dst =
                GpuVector::init(device, &dst, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
            let staging =
                GpuVector::uninit(device, LEN, BufferUsages::MAP_READ | BufferUsages::COPY_DST);

            unop.queue(&mut queue, &gpu_dst, &gpu_src, gpu_args.as_ref());

            queue.encode(&mut encoder, None);
            staging.copy_from(&mut encoder, &gpu_dst);

            gpu.queue().submit(Some(encoder.finish()));

            let mut cpu_result = src;
            unop.run_cpu(&mut cpu_result, args);

            approx::assert_relative_eq!(
                DVector::from(staging.read(gpu.device()).await.unwrap()),
                cpu_result,
                epsilon = 1.0e-5
            );
        }
    }
}
