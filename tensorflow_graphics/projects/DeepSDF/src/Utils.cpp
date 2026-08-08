// Copyright 2004-present Facebook. All Rights Reserved.

#include "Utils.h"

#include <random>

std::vector<Eigen::Vector3f> EquiDistPointsOnSphere(const uint numSamples, const float radius) {
  std::vector<Eigen::Vector3f> points(numSamples);
  const float offset = 2.f / numSamples;

  const float increment = static_cast<float>(M_PI) * (3.f - std::sqrt(5.f));

  for (uint i = 0; i < numSamples; i++) {
    const float y = ((i * offset) - 1) + (offset / 2);
    const float r = std::sqrt(1 - std::pow(y, 2.f));

    const float phi = (i + 1.f) * increment;

    const float x = cos(phi) * r;
    const float z = sin(phi) * r;

    points[i] = radius * Eigen::Vector3f(x, y, z);
  }

  return points;
}

std::vector<Eigen::Vector4f> ValidPointsFromIm(const pangolin::Image<Eigen::Vector4f>& verts) {
  std::vector<Eigen::Vector4f> points;
  Eigen::Vector4f v;

  for (unsigned int w = 0; w < verts.w; w++) {
    for (unsigned int h = 0; h < verts.h; h++) {
      v = verts(w, h);
      if (v[3] == 0.0f) {
        continue;
      }
      points.push_back(v);
    }
  }
  return points;
}

std::vector<Eigen::Vector4f> ValidPointsAndTrisFromIm(
    const pangolin::Image<Eigen::Vector4f>& pixNorms,
    std::vector<Eigen::Vector4f>& tris,
    int& totalObs,
    int& wrongObs) {
  std::vector<Eigen::Vector4f> points;
  Eigen::Vector4f n;

  for (unsigned int w = 0; w < pixNorms.w; w++) {
    for (unsigned int h = 0; h < pixNorms.h; h++) {
      n = pixNorms(w, h);
      if (n[3] == 0.0f)
        continue;
      totalObs++;
      const std::size_t triInd = static_cast<std::size_t>(n[3] + 0.01f) - 1;
      Eigen::Vector4f triTrack = tris[triInd];
      if (triTrack[3] == 0.0f)
        tris[triInd] = n;
      else if (triTrack[3] > 0.0f) {
        const float dot = triTrack.head<3>().dot(n.head<3>());
        if (dot < 0.0f) {
          tris[triInd][3] = -1.0f;
          wrongObs++;
        }
      } else if (triTrack[3] < 0.0f) {
        wrongObs++;
      }
      points.push_back(n);
    }
  }
  return points;
}

float TriangleArea(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) {
  const Eigen::Vector3f ab = b - a;
  const Eigen::Vector3f ac = c - a;

  float costheta = ab.dot(ac) / (ab.norm() * ac.norm());

  if (costheta < -1) // meaning theta is pi
    costheta = std::cos(static_cast<float>(M_PI) * 359.f / 360);
  else if (costheta > 1) // meaning theta is zero
    costheta = std::cos(static_cast<float>(M_PI) * 1.f / 360);

  const float sinTheta = std::sqrt(1 - costheta * costheta);

  return 0.5f * ab.norm() * ac.norm() * sinTheta;
}

Eigen::Vector3f SamplePointFromTriangle(
    const Eigen::Vector3f& a,
    const Eigen::Vector3f& b,
    const Eigen::Vector3f& c) {
  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);

  const float r1 = rand_dist(generator);
  const float r2 = rand_dist(generator);

  return Eigen::Vector3f(
      (1 - std::sqrt(r1)) * a + std::sqrt(r1) * (1 - r2) * b + r2 * std::sqrt(r1) * c);
}

// TODO: duplicated w/ below
std::pair<Eigen::Vector3f, float> ComputeNormalizationParameters(
    pangolin::Geometry& geom,
    const float buffer) {
  float xMin = 1000000, xMax = -1000000, yMin = 1000000, yMax = -1000000, zMin = 1000000,
        zMax = -1000000;

  pangolin::Image<float> vertices =
      pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

  const std::size_t numVertices = vertices.h;

  ///////// Only consider vertices that were used in some face
  std::vector<unsigned char> verticesUsed(numVertices, 0);
  // turn to true if the vertex is used
  for (const auto& object : geom.objects) {
    auto itVertIndices = object.second.attributes.find("vertex_indices");
    if (itVertIndices != object.second.attributes.end()) {
      pangolin::Image<uint32_t> ibo =
          pangolin::get<pangolin::Image<uint32_t>>(itVertIndices->second);

      for (uint i = 0; i < ibo.h; ++i) {
        for (uint j = 0; j < 3; ++j) {
          verticesUsed[ibo(j, i)] = 1;
        }
      }
    }
  }
  /////////

  // compute min max in each dimension
  for (size_t i = 0; i < numVertices; i++) {
    // pass when it's not used.
    if (verticesUsed[i] == 0)
      continue;
    xMin = fmin(xMin, vertices(0, i));
    yMin = fmin(yMin, vertices(1, i));
    zMin = fmin(zMin, vertices(2, i));
    xMax = fmax(xMax, vertices(0, i));
    yMax = fmax(yMax, vertices(1, i));
    zMax = fmax(zMax, vertices(2, i));
  }

  const Eigen::Vector3f center((xMax + xMin) / 2.0f, (yMax + yMin) / 2.0f, (zMax + zMin) / 2.0f);

  // make the mean zero
  float maxDistance = -1.0f;
  for (size_t i = 0; i < numVertices; i++) {
    // pass when it's not used.
    if (verticesUsed[i] == false)
      continue;

    const float dist = (Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(i)) - center).norm();
    maxDistance = std::max(maxDistance, dist);
  }

  // add some buffer
  maxDistance *= buffer;

  return {-1 * center, (1.f / maxDistance)};
}

float BoundingCubeNormalization(
    pangolin::Geometry& geom,
    bool fitToUnitSphere,
    const float buffer) {
  float xMin = 1000000, xMax = -1000000, yMin = 1000000, yMax = -1000000, zMin = 1000000,
        zMax = -1000000;

  pangolin::Image<float> vertices =
      pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

  const std::size_t numVertices = vertices.h;

  ///////// Only consider vertices that were used in some face
  std::vector<unsigned char> verticesUsed(numVertices, 0);
  // turn to true if the vertex is used
  for (const auto& object : geom.objects) {
    auto itVertIndices = object.second.attributes.find("vertex_indices");
    if (itVertIndices != object.second.attributes.end()) {
      pangolin::Image<uint32_t> ibo =
          pangolin::get<pangolin::Image<uint32_t>>(itVertIndices->second);

      for (uint i = 0; i < ibo.h; ++i) {
        for (uint j = 0; j < 3; ++j) {
          verticesUsed[ibo(j, i)] = 1;
        }
      }
    }
  }
  /////////

  // compute min max in each dimension
  for (size_t i = 0; i < numVertices; i++) {
    // pass when it's not used.
    if (verticesUsed[i] == 0)
      continue;
    xMin = fmin(xMin, vertices(0, i));
    yMin = fmin(yMin, vertices(1, i));
    zMin = fmin(zMin, vertices(2, i));
    xMax = fmax(xMax, vertices(0, i));
    yMax = fmax(yMax, vertices(1, i));
    zMax = fmax(zMax, vertices(2, i));
  }

  const float xCenter = (xMax + xMin) / 2.0f;
  const float yCenter = (yMax + yMin) / 2.0f;
  const float zCenter = (zMax + zMin) / 2.0f;

  // make the mean zero
  float maxDistance = -1.0f;
  for (size_t i = 0; i < numVertices; i++) {
    // pass when it's not used.
    if (verticesUsed[i] == false)
      continue;
    vertices(0, i) -= xCenter;
    vertices(1, i) -= yCenter;
    vertices(2, i) -= zCenter;

    const float dist = Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(i)).norm();
    maxDistance = std::max(maxDistance, dist);
  }

  // add some buffer
  maxDistance *= buffer;

  if (fitToUnitSphere) {
    for (size_t i = 0; i < numVertices; i++) {
      vertices(0, i) /= maxDistance;
      vertices(1, i) /= maxDistance;
      vertices(2, i) /= maxDistance;
    }
    maxDistance = 1;
  }

  return maxDistance;
}
