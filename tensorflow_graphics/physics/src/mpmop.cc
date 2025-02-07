#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "config.h"

using namespace tensorflow;

/*
    Register MPM operation
*/

REGISTER_OP("Mpm")
    .Input("position: float")     //(batch_size, dim, particles)
    .Input("velocity: float")     //(batch_size, dim, particles)
    .Input("affine: float")       //(batch_size, dim, dim, particles)
    .Input("deformation: float")  //(batch_size, dim, dim, particles)
    .Input("actuation: float")    //(batch_size, dim, dim, particles)
    .Input("grid_bc: float")  //(batch_size, num_cells, dim + 1)
    .Attr("dt: float = 0.01")
    .Attr("dx: float = 0.01")
    .Attr("E: float = 50")
    .Attr("nu: float = 0.3")
    .Attr("m_p: float = 100")
    .Attr("V_p: float = 10")
    .Attr("gravity: list(float) = [0, 0, 0]")
    .Attr("resolution: list(int) = [100, 100, 100]")
    .Output("position_out: float")
    .Output("velocity_out: float")
    .Output("affine_out: float")
    .Output("deformation_out: float")
    .Output("poly_out: float")  //(batch_size, dim, dim, particles)
    .Output("grid_out: float")  //(batch_size, num_cells, dim + 1)
    .Output("grid_star: float") //(batch_size, dim, dim, particles)
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
      shape_inference::ShapeHandle grid_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 3, &grid_shape));

      shape_inference::DimensionHandle temp;

      shape_inference::DimensionHandle batch_size = c->Dim(x_shape, 0);
      shape_inference::DimensionHandle batch_sizev = c->Dim(v_shape, 0);
      shape_inference::DimensionHandle batch_sizeF = c->Dim(F_shape, 0);
      shape_inference::DimensionHandle batch_sizeC = c->Dim(C_shape, 0);
      shape_inference::DimensionHandle batch_sizeA = c->Dim(A_shape, 0);
      shape_inference::DimensionHandle batch_sizegrid = c->Dim(grid_shape, 0);
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizev, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeF, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeC, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeA, &temp));
      TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizegrid, &temp));

      shape_inference::DimensionHandle dim = c->Dim(x_shape, 1);
      shape_inference::DimensionHandle dimv = c->Dim(v_shape, 1);
      shape_inference::DimensionHandle dimF1 = c->Dim(F_shape, 1);
      shape_inference::DimensionHandle dimF2 = c->Dim(F_shape, 2);
      shape_inference::DimensionHandle dimC1 = c->Dim(C_shape, 1);
      shape_inference::DimensionHandle dimC2 = c->Dim(C_shape, 2);
      shape_inference::DimensionHandle dimA1 = c->Dim(A_shape, 1);
      shape_inference::DimensionHandle dimA2 = c->Dim(A_shape, 2);
      shape_inference::DimensionHandle dimgrid = c->Dim(grid_shape, 2);
      TF_RETURN_IF_ERROR(c->Merge(dim, dimv, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimF1, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimF2, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimC1, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimC2, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimA1, &temp));
      TF_RETURN_IF_ERROR(c->Merge(dim, dimA2, &temp));
      auto dim_ = *((int *)dim.Handle());
      auto dim_1 = c->MakeDim(shape_inference::DimensionOrConstant(dim_ + 1));
      TF_RETURN_IF_ERROR(c->Merge(dim_1, dimgrid, &temp));

      shape_inference::DimensionHandle particle = c->Dim(x_shape, 2);
      shape_inference::DimensionHandle particlev = c->Dim(v_shape, 2);
      shape_inference::DimensionHandle particleF = c->Dim(F_shape, 3);
      shape_inference::DimensionHandle particleC = c->Dim(C_shape, 3);
      shape_inference::DimensionHandle particleA = c->Dim(A_shape, 3);
      TF_RETURN_IF_ERROR(c->Merge(particle, particlev, &temp));
      TF_RETURN_IF_ERROR(c->Merge(particle, particleF, &temp));
      TF_RETURN_IF_ERROR(c->Merge(particle, particleC, &temp));
      TF_RETURN_IF_ERROR(c->Merge(particle, particleA, &temp));


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
      auto num_cells_ = c->MakeDim(shape_inference::DimensionOrConstant(num_cells));

      shape_inference::DimensionHandle num_cells_grid = c->Dim(grid_shape, 1);
      TF_RETURN_IF_ERROR(c->Merge(num_cells_, num_cells_grid, &temp));

      c->set_output(0, x_shape);
      c->set_output(1, v_shape);
      c->set_output(2, F_shape);
      c->set_output(3, C_shape);
      c->set_output(4, C_shape);
      c->set_output(5, grid_shape);
      c->set_output(6, grid_shape);

      return Status::OK();
    });

/*
    MPM Operation GPU
*/

void MPMKernelLauncher(int dim,
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
                       const float *ingrid,
                       float *outx,
                       float *outv,
                       float *outF,
                       float *outC,
                       float *outP,
                       float *outgrid,
                       float *outgrid_star);

class MPMOpGPU : public OpKernel {
 private:
  float dt_;
  float dx_;
  float m_p_, V_p_, E_, nu_;
  std::vector<float> gravity_;
  std::vector<int> res_;

 public:
  explicit MPMOpGPU(OpKernelConstruction *context) : OpKernel(context) {
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
    OP_REQUIRES(context, E_ >= 0,
                errors::InvalidArgument("Need E >= 0, got ", E_));
    OP_REQUIRES(context, nu_ > 0,
                errors::InvalidArgument("Need nu_p > 0, got ", nu_));
    OP_REQUIRES(context, m_p_ > 0,
                errors::InvalidArgument("Need m_p > 0, got ", m_p_));
    OP_REQUIRES(context, V_p_ > 0,
                errors::InvalidArgument("Need V_p > 0, got ", V_p_));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &inx = context->input(0);
    const Tensor &inv = context->input(1);
    const Tensor &inF = context->input(2);
    const Tensor &inC = context->input(3);
    const Tensor &inA = context->input(4);
    const Tensor &ingrid = context->input(5);
    const TensorShape &x_shape = inx.shape();
    const TensorShape &v_shape = inv.shape();
    const TensorShape &F_shape = inF.shape();
    const TensorShape &C_shape = inC.shape();
    const TensorShape &A_shape = inA.shape();
    const TensorShape &grid_shape = ingrid.shape();
    const TensorShape &P_shape = inA.shape();

    // Check inputs' dimensional
    DCHECK_EQ(x_shape.dims(), 3);
    DCHECK_EQ(v_shape.dims(), 3);
    DCHECK_EQ(F_shape.dims(), 4);
    DCHECK_EQ(C_shape.dims(), 4);
    DCHECK_EQ(A_shape.dims(), 4);
    DCHECK_EQ(grid_shape.dims(), 3);

    const int batch_size = x_shape.dim_size(0);

    const int dim = x_shape.dim_size(1);

    // Check gravity
    int res[dim];
    float gravity[dim];
    int num_cells = 1;
    for (int i = 0; i < dim; i++) {
      res[i] = res_[i];
      num_cells *= res[i];
      gravity[i] = gravity_[i];
    }

    const int particles = x_shape.dim_size(2);
    // printf("particles %d\n", particles);

    // Check input batch_size
    DCHECK_EQ(batch_size, v_shape.dim_size(0));
    DCHECK_EQ(batch_size, F_shape.dim_size(0));
    DCHECK_EQ(batch_size, C_shape.dim_size(0));
    DCHECK_EQ(batch_size, A_shape.dim_size(0));
    DCHECK_EQ(batch_size, grid_shape.dim_size(0));

    // Check input dim
    DCHECK_EQ(dim, v_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(2));
    DCHECK_EQ(dim, C_shape.dim_size(1));
    DCHECK_EQ(dim, C_shape.dim_size(2));
    DCHECK_EQ(dim, A_shape.dim_size(1));
    DCHECK_EQ(dim, A_shape.dim_size(2));
    DCHECK_EQ(dim + 1, grid_shape.dim_size(2));

    // Check input particles
    DCHECK_EQ(particles, v_shape.dim_size(2));
    DCHECK_EQ(particles, F_shape.dim_size(3));
    DCHECK_EQ(particles, C_shape.dim_size(3));
    DCHECK_EQ(particles, A_shape.dim_size(3));

    // Check input num_cells
    DCHECK_EQ(num_cells, grid_shape.dim_size(1));

    // create output tensor
    Tensor *outx = NULL;
    Tensor *outv = NULL;
    Tensor *outF = NULL;
    Tensor *outC = NULL;
    Tensor *outP = NULL;
    Tensor *outgrid = NULL;
    Tensor *outgrid_star = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &outx));
    OP_REQUIRES_OK(context, context->allocate_output(1, v_shape, &outv));
    OP_REQUIRES_OK(context, context->allocate_output(2, F_shape, &outF));
    OP_REQUIRES_OK(context, context->allocate_output(3, C_shape, &outC));
    OP_REQUIRES_OK(context, context->allocate_output(4, P_shape, &outP));
    OP_REQUIRES_OK(context, context->allocate_output(5, grid_shape, &outgrid));
    OP_REQUIRES_OK(context, context->allocate_output(6, grid_shape, &outgrid_star));

    auto f_inx = inx.flat<float>();
    auto f_inv = inv.flat<float>();
    auto f_inF = inF.flat<float>();
    auto f_inC = inC.flat<float>();
    auto f_inA = inA.flat<float>();
    auto f_ingrid = ingrid.flat<float>();
    auto f_outx = outx->template flat<float>();
    auto f_outv = outv->template flat<float>();
    auto f_outF = outF->template flat<float>();
    auto f_outC = outC->template flat<float>();
    auto f_outP = outP->template flat<float>();
    auto f_outgrid = outgrid->template flat<float>();
    auto f_outgrid_star = outgrid_star->template flat<float>();

    MPMKernelLauncher(dim, res, particles, dx_, dt_, E_, nu_, m_p_, V_p_, gravity,
                      f_inx.data(), f_inv.data(), f_inF.data(), f_inC.data(),
                      f_inA.data(), f_ingrid.data(),
                      f_outx.data(), f_outv.data(), f_outF.data(),
                      f_outC.data(), f_outP.data(), f_outgrid.data(),
                      f_outgrid_star.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("Mpm").Device(DEVICE_GPU), MPMOpGPU);
