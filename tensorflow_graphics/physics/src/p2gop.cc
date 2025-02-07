#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "config.h"

using namespace tensorflow;

/*
    Register P2G operation
*/

REGISTER_OP("P2g")
    .Input("position: float")     //(batch_size, dim, particles)
    .Input("velocity: float")     //(batch_size, dim, particles)
    .Input("affine: float")       //(batch_size, dim, dim, particles)
    .Input("deformation: float")  //(batch_size, dim, dim, particles)
    .Input("actuation: float")    //(batch_size, dim, dim, particles)
    .Attr("dt: float = 0.01")
    .Attr("dx: float = 0.01")
    .Attr("E: float = 50")
    .Attr("nu: float = 0.3")
    .Attr("m_p: float = 100")
    .Attr("V_p: float = 10")
    .Attr("gravity: list(float) = [0, 0, 0]")
    .Attr("resolution: list(int) = [100, 100, 100]")
    .Output("poly_out: float")  //(batch_size, dim, dim, particles)
    .Output("grid_out: float")  //(batch_size, num_cells, dim + 1)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {

      shape_inference::ShapeHandle x_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x_shape));
      shape_inference::ShapeHandle v_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &v_shape));
      shape_inference::ShapeHandle F_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &F_shape));
      shape_inference::ShapeHandle C_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &C_shape));
      shape_inference::ShapeHandle A_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 4, &A_shape));

      shape_inference::DimensionHandle temp;

      shape_inference::DimensionHandle batch_size = c->Dim(x_shape, 0);
      shape_inference::DimensionHandle batch_sizev = c->Dim(v_shape, 0);
      shape_inference::DimensionHandle batch_sizeF = c->Dim(F_shape, 0);
      shape_inference::DimensionHandle batch_sizeC = c->Dim(C_shape, 0);
      shape_inference::DimensionHandle batch_sizeA = c->Dim(A_shape, 0);
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizev, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeF, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeC, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeA, &temp));

      shape_inference::DimensionHandle dim = c->Dim(x_shape, 1);
      shape_inference::DimensionHandle dimv = c->Dim(v_shape, 1);
      shape_inference::DimensionHandle dimF1 = c->Dim(F_shape, 1);
      shape_inference::DimensionHandle dimF2 = c->Dim(F_shape, 2);
      shape_inference::DimensionHandle dimC1 = c->Dim(C_shape, 1);
      shape_inference::DimensionHandle dimC2 = c->Dim(C_shape, 2);
      shape_inference::DimensionHandle dimA1 = c->Dim(A_shape, 1);
      shape_inference::DimensionHandle dimA2 = c->Dim(A_shape, 2);
      TF_RETURN_IF_ERROR(c->Merge(dim, dimv, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimF1, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimF2, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimC1, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimC2, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimA1, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimA2, &temp));

      shape_inference::DimensionHandle particle = c->Dim(x_shape, 2);
      shape_inference::DimensionHandle particlev = c->Dim(v_shape, 2);
      shape_inference::DimensionHandle particleF = c->Dim(F_shape, 3);
      shape_inference::DimensionHandle particleC = c->Dim(C_shape, 3);
      shape_inference::DimensionHandle particleA = c->Dim(A_shape, 3);
      TF_RETURN_IF_ERROR(c->Merge(particle, particlev, &temp));
      TF_RETURN_IF_ERROR(c->Merge(particle, particleF, &temp));
      TF_RETURN_IF_ERROR(c->Merge(particle, particleC, &temp));
      TF_RETURN_IF_ERROR(c->Merge(particle, particleA, &temp));

      c->set_output(0, C_shape);
      auto dim_ = *((int *)dim.Handle());

      std::vector<int> res_;
      TF_RETURN_IF_ERROR(c->GetAttr("resolution", &res_));
      std::vector<float> gravity_;
      TF_RETURN_IF_ERROR(c->GetAttr("gravity", &gravity_));

      if ((int)gravity_.size() != dim_)
        return errors::InvalidArgument("Gravity length must be equal to ", dim_,
                                       ", but is ", gravity_.size());
      if ((int)res_.size() != dim_)
        return errors::InvalidArgument("Resolution length must be equal to ",
                                       dim_, ", but is ", res_.size());

      int res[3];
      int num_cells = 1;
      for (int i = 0; i < dim_; i++) {
        res[i] = res_[i];
        num_cells *= res[i];
      }
      std::vector<shape_inference::DimensionHandle> new_shape;
      new_shape.clear();
      new_shape.push_back(batch_size);
      new_shape.push_back(
          c->MakeDim(shape_inference::DimensionOrConstant(num_cells)));
      new_shape.push_back(
          c->MakeDim(shape_inference::DimensionOrConstant(dim_ + 1)));
      c->set_output(1, c->MakeShape(new_shape));

      return Status::OK();
    });

/*
    P2G Operation GPU
*/

void P2GKernelLauncher(int dim,
                       int *res,
                       int num_particles,
                       float dx,
                       float dt,
                       float E,
                       float nu,
                       float m_p,
                       float V_p,
                       float *gravity,
                       const float *inx,
                       const float *inv,
                       const float *inF,
                       const float *inC,
                       const float *inA,
                       float *outP,
                       float *outgrid);

class P2GOpGPU : public OpKernel {
 private:
  float dt_;
  float dx_;
  float m_p_, V_p_, E_, nu_;
  std::vector<float> gravity_;
  std::vector<int> res_;

 public:
  explicit P2GOpGPU(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dt", &dt_));
    OP_REQUIRES(context, dt_ > 0,
                errors::InvalidArgument("Need dt > 0, got ", dt_));
    OP_REQUIRES_OK(context, context->GetAttr("dx", &dx_));
    OP_REQUIRES(context, dx_ > 0,
                errors::InvalidArgument("Need dx > 0, got ", dx_));
    OP_REQUIRES_OK(context, context->GetAttr("gravity", &gravity_));
    OP_REQUIRES_OK(context, context->GetAttr("resolution", &res_));
    OP_REQUIRES_OK(context, context->GetAttr("E", &E_));
    OP_REQUIRES_OK(context, context->GetAttr("nu", &nu_));
    OP_REQUIRES_OK(context, context->GetAttr("m_p", &m_p_));
    OP_REQUIRES_OK(context, context->GetAttr("V_p", &V_p_));
    OP_REQUIRES(context, E_ > 0,
                errors::InvalidArgument("Need E > 0, got ", E_));
    OP_REQUIRES(context, nu_ > 0,
                errors::InvalidArgument("Need nu_p > 0, got ", nu_));
    OP_REQUIRES(context, m_p_ > 0,
                errors::InvalidArgument("Need m_p > 0, got ", m_p_));
    OP_REQUIRES(context, V_p_ > 0,
                errors::InvalidArgument("Need V_p > 0, got ", V_p_));
  }

  void Compute(OpKernelContext *context) override {
    // get the x
    const Tensor &inx = context->input(0);

    // get the v tensor
    const Tensor &inv = context->input(1);

    // get the F tensor
    const Tensor &inF = context->input(2);

    // get the C tensor
    const Tensor &inC = context->input(3);

    // get the A tensor
    const Tensor &inA = context->input(4);

    // check shapes of input and weights
    const TensorShape &x_shape = inx.shape();
    const TensorShape &v_shape = inv.shape();
    const TensorShape &F_shape = inF.shape();
    const TensorShape &C_shape = inC.shape();
    const TensorShape &A_shape = inA.shape();
    TensorShape P_shape = inC.shape();
    TensorShape grid_shape = inx.shape();

    // Check that inputs' dimensional
    DCHECK_EQ(x_shape.dims(), 3);
    DCHECK_EQ(v_shape.dims(), 3);
    DCHECK_EQ(F_shape.dims(), 4);
    DCHECK_EQ(C_shape.dims(), 4);
    DCHECK_EQ(A_shape.dims(), 4);

    const int batch_size = x_shape.dim_size(0);
    // printf("batch_size %d\n", batch_size);

    const int dim = x_shape.dim_size(1);
    // printf("dim %d\n", dim);

    // Check gravity
    int res[dim];
    float gravity[dim];
    int num_cells = 1;
    for (int i = 0; i < dim; i++) {
      res[i] = res_[i];
      num_cells *= res[i];
      gravity[i] = gravity_[i];
    }
    int grid_shape2 = dim + 1;
    // printf("P2GOpGPU\n");
    // float dx = 1.0f / res[0];

    const int particles = x_shape.dim_size(2);
    // printf("particles %d\n", particles);

    // Check input batch_size
    DCHECK_EQ(batch_size, v_shape.dim_size(0));
    DCHECK_EQ(batch_size, F_shape.dim_size(0));
    DCHECK_EQ(batch_size, C_shape.dim_size(0));
    DCHECK_EQ(batch_size, A_shape.dim_size(0));

    // Check input dim
    DCHECK_EQ(dim, v_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(2));
    DCHECK_EQ(dim, C_shape.dim_size(1));
    DCHECK_EQ(dim, C_shape.dim_size(2));
    DCHECK_EQ(dim, A_shape.dim_size(1));
    DCHECK_EQ(dim, A_shape.dim_size(2));

    // Check input particles
    DCHECK_EQ(particles, v_shape.dim_size(2));
    DCHECK_EQ(particles, F_shape.dim_size(3));
    DCHECK_EQ(particles, C_shape.dim_size(3));
    DCHECK_EQ(particles, A_shape.dim_size(3));

    // create output tensor
    Tensor *outP = NULL;
    Tensor *outgrid = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, P_shape, &outP));
    grid_shape.set_dim(1, num_cells);
    grid_shape.set_dim(2, grid_shape2);
    OP_REQUIRES_OK(context, context->allocate_output(1, grid_shape, &outgrid));

    auto f_inx = inx.flat<float>();
    auto f_inv = inv.flat<float>();
    auto f_inF = inF.flat<float>();
    auto f_inC = inC.flat<float>();
    auto f_inA = inA.flat<float>();
    auto f_outP = outP->template flat<float>();
    auto f_outgrid = outgrid->template flat<float>();

    P2GKernelLauncher(dim, res, particles, dx_, dt_, E_, nu_, m_p_, V_p_, gravity,
                      f_inx.data(), f_inv.data(), f_inF.data(), f_inC.data(),
                      f_inA.data(), f_outP.data(), f_outgrid.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("P2g").Device(DEVICE_GPU), P2GOpGPU);
