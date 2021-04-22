#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("MpmGrad")
  .Input("position: float")               //(batch_size, dim, particles)
  .Input("velocity: float")               //(batch_size, dim, particles)
  .Input("affine: float")                 //(batch_size, dim, dim, particles)
  .Input("deformation: float")            //(batch_size, dim, dim, particles)
  .Input("actuation: float")              //(batch_size, dim, dim, particles
  .Input("grid_normal: float")            //(batch_size, num_cells, dim + 1)
  .Input("position_out: float")           //(batch_size, dim, particles)
  .Input("velocity_out: float")           //(batch_size, dim, particles) 
  .Input("affine_out: float")             //(batch_size, dim, dim, particles) 
  .Input("deformation_out: float")        //(batch_size, dim, dim, particles)
  .Input("poly_out: float")               //(batch_size, dim, dim, particles)
  .Input("grid_out: float")               //(batch_size, num_cells, dim + 1)
  .Input("grid_star_out: float")               //(batch_size, num_cells, dim + 1)
  .Input("position_out_grad: float")      //(batch_size, dim, particles) 
  .Input("velocity_out_grad: float")      //(batch_size, dim, particles) 
  .Input("affine_out_grad: float")        //(batch_size, dim, dim, particles) 
  .Input("deformation_out_grad: float")   //(batch_size, dim, dim, particles) 
  .Input("poly_out_grad: float")          //(batch_size, dim, dim, particles)
  .Input("grid_out_grad: float")          //(batch_size, num_cells, dim + 1)
  .Input("grid_out_star_grad: float")     //(batch_size, num_cells, dim + 1)
  .Attr("dt: float")
  .Attr("dx: float")
  .Attr("E: float")
  .Attr("nu: float")
  .Attr("m_p: float")
  .Attr("V_p: float")
  .Attr("gravity: list(float)")
  .Attr("resolution: list(int)")
  .Output("position_grad: float")         //(batch_size, dim, particles)
  .Output("velocity_grad: float")         //(batch_size, dim, particles)
  .Output("affine_grad: float")           //(batch_size, dim, dim, particles)
  .Output("deformation_grad: float")      //(batch_size, dim, dim, particles)
  .Output("actuation_grad: float")        //(batch_size, dim, dim, particles)
  .Output("grid_normal_grad: float");     //(batch_size, num_cells, dim + 1)


void MPMGradKernelLauncher(
    int dim, int *res, int num_particles, float dx, float dt, float E, float nu,
    float m_p, float V_p,
    float *gravity,
    const float *inx, const float *inv, const float *inF, const float *inC,
    const float *inA, const float *ingrid,
    const float *outx, const float *outv, const float *outF, const float *outC,
    const float *outP, const float *outgrid, const float *outgrid_star,
    float *grad_inx, float *grad_inv, float *grad_inF, float *grad_inC,
    float *grad_inA, float *grad_ingrid,
    const float *grad_outx, const float *grad_outv, 
    const float *grad_outF, const float *grad_outC,
    const float *grad_outP, const float *grad_outgrid,
    const float *grad_outgrid_star);

class MPMGradOpGPU : public OpKernel {
 private:
  float dt_;
  float dx_;
  float E_, nu_, m_p_, V_p_;
  std::vector<float> gravity_;
  std::vector<int> res_;
 public:
  explicit MPMGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("dt", &dt_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dx", &dx_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("E", &E_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("nu", &nu_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("m_p", &m_p_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("V_p", &V_p_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("gravity", &gravity_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("resolution", &res_));
  }
  
  void Compute(OpKernelContext* context) override {
    //printf("MPMOpGPU\n");

    // get the x
    int cnt = 0;
    const Tensor& inx = context->input(cnt++);
    const Tensor& inv = context->input(cnt++);
    const Tensor& inF = context->input(cnt++);
    const Tensor& inC = context->input(cnt++);
    const Tensor& inA = context->input(cnt++);
    const Tensor& ingrid = context->input(cnt++);
    const Tensor& outx = context->input(cnt++);
    const Tensor& outv = context->input(cnt++);
    const Tensor& outF = context->input(cnt++);
    const Tensor& outC = context->input(cnt++);
    const Tensor& outP = context->input(cnt++);
    const Tensor& outgrid = context->input(cnt++);
    const Tensor& outgrid_star = context->input(cnt++);
    const Tensor& grad_outx = context->input(cnt++);
    const Tensor& grad_outv = context->input(cnt++);
    const Tensor& grad_outF = context->input(cnt++);
    const Tensor& grad_outC = context->input(cnt++);
    const Tensor& grad_outP = context->input(cnt++);
    const Tensor& grad_outgrid = context->input(cnt++);
    const Tensor& grad_outgrid_star = context->input(cnt++);

    const TensorShape& x_shape = inx.shape();
    const TensorShape& v_shape = inv.shape();
    const TensorShape& F_shape = inF.shape();
    const TensorShape& C_shape = inC.shape();
    const TensorShape& A_shape = inA.shape();
    const TensorShape& grid_shape = ingrid.shape();

    const int particles = x_shape.dim_size(2);

    const int dim = x_shape.dim_size(1);
    int res[dim];
    float gravity[dim];
    int num_cells = 1;
    for (int i = 0; i < dim; i++) {
      res[i] = res_[i];
      num_cells *= res[i];
      gravity[i] = gravity_[i];
    }

    // create output tensor
    Tensor* grad_inx= NULL;
    Tensor* grad_inv= NULL;
    Tensor* grad_inF= NULL;
    Tensor* grad_inC= NULL;
    Tensor* grad_inA= NULL;
    Tensor* grad_ingrid= NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &grad_inx));
    OP_REQUIRES_OK(context, context->allocate_output(1, v_shape, &grad_inv));
    OP_REQUIRES_OK(context, context->allocate_output(2, F_shape, &grad_inF));
    OP_REQUIRES_OK(context, context->allocate_output(3, C_shape, &grad_inC));
    OP_REQUIRES_OK(context, context->allocate_output(4, A_shape, &grad_inA));
    OP_REQUIRES_OK(context, context->allocate_output(5, grid_shape, &grad_ingrid));

    auto f_inx = inx.flat<float>();
    auto f_inv = inv.flat<float>();
    auto f_inF = inF.flat<float>();
    auto f_inC = inC.flat<float>();
    auto f_inA = inA.flat<float>();
    auto f_ingrid = ingrid.flat<float>();
    auto f_outx = outx.flat<float>();
    auto f_outv = outv.flat<float>();
    auto f_outF = outF.flat<float>();
    auto f_outC = outC.flat<float>();
    auto f_outP = outP.flat<float>();
    auto f_outgrid = outgrid.flat<float>();
    auto f_outgrid_star = outgrid_star.flat<float>();
    auto f_grad_outx = grad_outx.flat<float>();
    auto f_grad_outv = grad_outv.flat<float>();
    auto f_grad_outF = grad_outF.flat<float>();
    auto f_grad_outC = grad_outC.flat<float>();
    auto f_grad_outP = grad_outP.flat<float>();
    auto f_grad_outgrid = grad_outgrid.flat<float>();
    auto f_grad_outgrid_star = grad_outgrid_star.flat<float>();
    auto f_grad_inx = grad_inx->template flat<float>();
    auto f_grad_inv = grad_inv->template flat<float>();
    auto f_grad_inF = grad_inF->template flat<float>();
    auto f_grad_inC = grad_inC->template flat<float>();
    auto f_grad_inA = grad_inA->template flat<float>();
    auto f_grad_ingrid = grad_ingrid->template flat<float>();


    MPMGradKernelLauncher(dim, res, particles, dx_, dt_, E_, nu_, m_p_, V_p_, gravity,
        f_inx.data(), f_inv.data(), f_inF.data(), f_inC.data(), f_inA.data(), f_ingrid.data(),
        f_outx.data(), f_outv.data(), f_outF.data(), f_outC.data(),
        f_outP.data(), f_outgrid.data(), f_outgrid_star.data(),
        f_grad_inx.data(), f_grad_inv.data(),
        f_grad_inF.data(), f_grad_inC.data(),
        f_grad_inA.data(), f_grad_ingrid.data(),
        f_grad_outx.data(), f_grad_outv.data(),
        f_grad_outF.data(), f_grad_outC.data(),
        f_grad_outP.data(), f_grad_outgrid.data(),
        f_grad_outgrid_star.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("MpmGrad").Device(DEVICE_GPU), MPMGradOpGPU);
