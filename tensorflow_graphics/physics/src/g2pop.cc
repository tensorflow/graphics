#if(0)
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

/*
    Register MPM operation
*/

    

REGISTER_OP("G2p")
  .Input("position: float")         //(batch_size, dim, particles)
  .Input("velocity: float")         //(batch_size, dim, particles)
  .Input("affine: float")           //(batch_size, dim, dim, particles)
  .Input("deformation: float")      //(batch_size, dim, dim, particles)
  .Input("poly: float")             //(batch_size, dim, dim, particles)
  .Input("grid: float")             //(batch_size, dim + 1, num_cells)
  .Output("position_out: float")
  .Output("velocity_out: float")
  .Output("affine_out: float")
  .Output("deformation_out: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle x_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x_shape));
    shape_inference::ShapeHandle v_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &v_shape));
    shape_inference::ShapeHandle F_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &F_shape));
    shape_inference::ShapeHandle C_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &C_shape));
    shape_inference::ShapeHandle P_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 4, &P_shape));
    shape_inference::ShapeHandle grid_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 3, &grid_shape));
    
    shape_inference::DimensionHandle temp;
    
    shape_inference::DimensionHandle batch_size = c->Dim(x_shape, 0);
    shape_inference::DimensionHandle batch_sizev = c->Dim(v_shape, 0);
    shape_inference::DimensionHandle batch_sizeF = c->Dim(F_shape, 0);
    shape_inference::DimensionHandle batch_sizeC = c->Dim(C_shape, 0);
    shape_inference::DimensionHandle batch_sizeP = c->Dim(P_shape, 0);
    shape_inference::DimensionHandle batch_sizegrid = c->Dim(grid_shape, 0);
    TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizev, &temp));
    TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeF, &temp));
    TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeC, &temp));
    TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizeP, &temp));
    TF_RETURN_IF_ERROR(c->Merge(batch_size, batch_sizegrid, &temp));
    
    shape_inference::DimensionHandle dim = c->Dim(x_shape, 1);
    shape_inference::DimensionHandle dimv = c->Dim(v_shape, 1);
    shape_inference::DimensionHandle dimF1 = c->Dim(F_shape, 1);
    shape_inference::DimensionHandle dimF2 = c->Dim(F_shape, 2);
    shape_inference::DimensionHandle dimC1 = c->Dim(C_shape, 1);
    shape_inference::DimensionHandle dimC2 = c->Dim(C_shape, 2);
    shape_inference::DimensionHandle dimP1 = c->Dim(P_shape, 1);
    shape_inference::DimensionHandle dimP2 = c->Dim(P_shape, 2);
    shape_inference::DimensionHandle dimgrid = c->Dim(grid_shape, 2);
    TF_RETURN_IF_ERROR(c->Merge(dim, dimv, &temp));
    TF_RETURN_IF_ERROR(c->Merge(dim, dimF1, &temp));
    TF_RETURN_IF_ERROR(c->Merge(dim, dimF2, &temp));
    TF_RETURN_IF_ERROR(c->Merge(dim, dimC1, &temp));
    TF_RETURN_IF_ERROR(c->Merge(dim, dimC2, &temp));
    TF_RETURN_IF_ERROR(c->Merge(dim, dimP1, &temp));
    TF_RETURN_IF_ERROR(c->Merge(dim, dimP2, &temp));
    // TF_RETURN_IF_ERROR(c->Merge(dim + 1, dimgrid, &temp)); TODO
    
    shape_inference::DimensionHandle particle = c->Dim(x_shape, 2);
    shape_inference::DimensionHandle particlev = c->Dim(v_shape, 2);
    shape_inference::DimensionHandle particleF = c->Dim(F_shape, 3);
    shape_inference::DimensionHandle particleC = c->Dim(C_shape, 3);
    shape_inference::DimensionHandle particleP = c->Dim(P_shape, 3);
    TF_RETURN_IF_ERROR(c->Merge(particle, particlev, &temp));
    TF_RETURN_IF_ERROR(c->Merge(particle, particleF, &temp));
    TF_RETURN_IF_ERROR(c->Merge(particle, particleC, &temp));
    TF_RETURN_IF_ERROR(c->Merge(particle, particleP, &temp));
    
    return Status::OK();
  });

/*
    MPM Operation GPU
*/

void G2PKernelLauncher(
    int res[3], int num_particles, float dx, float dt, float gravity[3],
    const float *inx, const float *inv, const float *inF, const float *inC,
    const float *inP, const float *ingrid, 
    float *outx, float *outv, float *outF, float *outC);

class G2POpGPU : public OpKernel {
public:
  explicit G2POpGPU(OpKernelConstruction* context) : OpKernel(context) {
  }
  
  void Compute(OpKernelContext* context) override {

    int res[3] = {100, 100, 100};
    float gravity[3] = {0, -0, 0};
    float dx = 1.0f / res[0];
    float dt = 1e-2f;
    int num_cells = res[0] * res[1] * res[2];

    // get the x
    const Tensor& inx = context->input(0);
    const Tensor& inv = context->input(1);
    const Tensor& inF = context->input(2);
    const Tensor& inC = context->input(3);
    const Tensor& inP = context->input(4);
    const Tensor& ingrid = context->input(5);
    
    // check shapes of input and weights
    const TensorShape& x_shape = inx.shape();
    const TensorShape& v_shape = inv.shape();
    const TensorShape& F_shape = inF.shape();
    const TensorShape& C_shape = inC.shape();
    const TensorShape& P_shape = inP.shape();
    const TensorShape& grid_shape = ingrid.shape();
    
    //Check that inputs' dimensional
    DCHECK_EQ(x_shape.dims(), 3);
    DCHECK_EQ(v_shape.dims(), 3);
    DCHECK_EQ(F_shape.dims(), 4);
    DCHECK_EQ(C_shape.dims(), 4);
    DCHECK_EQ(P_shape.dims(), 4);
    DCHECK_EQ(grid_shape.dims(), 4);

    const int batch_size = x_shape.dim_size(0);
    const int dim = x_shape.dim_size(1);
    const int particles = x_shape.dim_size(2);

    //Check input batch_size
    DCHECK_EQ(batch_size, v_shape.dim_size(0));
    DCHECK_EQ(batch_size, F_shape.dim_size(0));
    DCHECK_EQ(batch_size, C_shape.dim_size(0));
    DCHECK_EQ(batch_size, P_shape.dim_size(0));
    DCHECK_EQ(batch_size, grid_shape.dim_size(0));
    
    //Check input dim
    DCHECK_EQ(dim, v_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(1));
    DCHECK_EQ(dim, F_shape.dim_size(2));
    DCHECK_EQ(dim, C_shape.dim_size(1));
    DCHECK_EQ(dim, C_shape.dim_size(2));
    DCHECK_EQ(dim, P_shape.dim_size(1));
    DCHECK_EQ(dim, P_shape.dim_size(2));
    DCHECK_EQ(dim + 1, grid_shape.dim_size(2));
    
    //Check input particles
    DCHECK_EQ(particles, v_shape.dim_size(2));
    DCHECK_EQ(particles, F_shape.dim_size(3));
    DCHECK_EQ(particles, C_shape.dim_size(3));
    DCHECK_EQ(particles, P_shape.dim_size(3));
            
    // create output tensor
    Tensor* outx = NULL;
    Tensor* outv = NULL;
    Tensor* outF = NULL;
    Tensor* outC = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_shape, &outx));
    OP_REQUIRES_OK(context, context->allocate_output(1, v_shape, &outv));
    OP_REQUIRES_OK(context, context->allocate_output(2, F_shape, &outF));
    OP_REQUIRES_OK(context, context->allocate_output(3, C_shape, &outC));
    
    auto f_inx = inx.flat<float>();
    auto f_inv = inv.flat<float>();
    auto f_inF = inF.flat<float>();
    auto f_inC = inC.flat<float>();
    auto f_inP = inP.flat<float>();
    auto f_ingrid = ingrid.flat<float>();
    auto f_outx = outx->template flat<float>();
    auto f_outv = outv->template flat<float>();
    auto f_outF = outF->template flat<float>();
    auto f_outC = outC->template flat<float>();
    

    G2PKernelLauncher(
        res, particles, dx, dt, gravity,
        f_inx.data(),
        f_inv.data(),
        f_inF.data(),
        f_inC.data(),
        f_inP.data(),
        f_ingrid.data(),
        f_outx.data(),
        f_outv.data(),
        f_outF.data(),
        f_outC.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("G2p").Device(DEVICE_GPU), G2POpGPU);

#endif
