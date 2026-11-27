// Copyright 2004-present Facebook. All Rights Reserved.

#include <vector>

// NB: This differs from the GitHub version due to the different
// location of the nanoflann header when installing from source
#include <nanoflann/nanoflann.hpp>
#include <pangolin/geometry/geometry.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>

struct KdVertexList {
 public:
  KdVertexList(const std::vector<Eigen::Vector3f>& points) : points_(points) {}

  inline size_t kdtree_get_point_count() const {
    return points_.size();
  }

  inline float kdtree_distance(const float* p1, const size_t idx_p2, size_t /*size*/) const {
    Eigen::Map<const Eigen::Vector3f> p(p1);
    return (p - points_[idx_p2]).squaredNorm();
  }

  inline float kdtree_get_pt(const size_t idx, int dim) const {
    return points_[idx](dim);
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const {
    return false;
  }

 private:
  std::vector<Eigen::Vector3f> points_;
};

using KdVertexListTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, KdVertexList>,
    KdVertexList,
    3,
    int>;

std::vector<Eigen::Vector3f> EquiDistPointsOnSphere(const uint numSamples, const float radius);

std::vector<Eigen::Vector4f> ValidPointsFromIm(const pangolin::Image<Eigen::Vector4f>& verts);

std::vector<Eigen::Vector4f> ValidPointsAndTrisFromIm(
    const pangolin::Image<Eigen::Vector4f>& pixNorms,
    std::vector<Eigen::Vector4f>& tris,
    int& totalObs,
    int& wrongObs);

float TriangleArea(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c);

Eigen::Vector3f SamplePointFromTriangle(
    const Eigen::Vector3f& a,
    const Eigen::Vector3f& b,
    const Eigen::Vector3f& c);

std::pair<Eigen::Vector3f, float> ComputeNormalizationParameters(
    pangolin::Geometry& geom,
    const float buffer = 1.03);

float BoundingCubeNormalization(
    pangolin::Geometry& geom,
    const bool fitToUnitSphere,
    const float buffer = 1.03);
