// Copyright 2004-present Facebook. All Rights Reserved.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>
#include <cnpy.h>

#include "Utils.h"

extern pangolin::GlSlProgram GetShaderProgram();

void SavePointsToPLY(const std::vector<Eigen::Vector3f>& verts, const std::string outputfile) {
  const std::size_t num_verts = verts.size();
  Eigen::Vector3f v;

  std::ofstream plyFile;
  plyFile.open(outputfile);
  plyFile << "ply\n";
  plyFile << "format ascii 1.0\n";
  plyFile << "element vertex " << num_verts << "\n";
  plyFile << "property float x\n";
  plyFile << "property float y\n";
  plyFile << "property float z\n";
  plyFile << "element face " << (num_verts / 3) << "\n";
  plyFile << "property list uchar int vertex_index\n";
  plyFile << "end_header\n";

  for (uint i = 0; i < num_verts; i++) {
    v = verts[i];
    plyFile << v[0] << " " << v[1] << " " << v[2] << "\n";
  }

  for (uint i = 0; i < num_verts; i += 3) {
    plyFile << "3 " << i << " " << (i + 1) << " " << (i + 2) << "\n";
  }

  plyFile.close();
}

void SaveNormalizationParamsToNPZ(
    const Eigen::Vector3f offset,
    const float scale,
    const std::string filename) {
  cnpy::npz_save(filename, "offset", offset.data(), {3ul}, "w");
  cnpy::npz_save(filename, "scale", &scale, {1ul}, "a");
}

void SampleFromSurfaceInside(
    pangolin::Geometry& geom,
    std::vector<Eigen::Vector3f>& surfpts,
    int num_sample,
    KdVertexListTree& kdTree,
    std::vector<Eigen::Vector3f>& surface_vertices,
    std::vector<Eigen::Vector3f>& surface_normals,
    float delta) {
  float total_area = 0.0f;

  std::vector<float> cdf_by_area;

  std::vector<Eigen::Vector3i> linearized_faces;

  for (const auto& object : geom.objects) {
    auto it_vert_indices = object.second.attributes.find("vertex_indices");
    if (it_vert_indices != object.second.attributes.end()) {
      pangolin::Image<uint32_t> ibo =
          pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

      for (uint i = 0; i < ibo.h; ++i) {
        linearized_faces.emplace_back(ibo(0, i), ibo(1, i), ibo(2, i));
      }
    }
  }

  pangolin::Image<float> vertices =
      pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

  for (const Eigen::Vector3i& face : linearized_faces) {
    float area = TriangleArea(
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2))));

    if (std::isnan(area)) {
      area = 0.f;
    }

    total_area += area;

    if (cdf_by_area.empty()) {
      cdf_by_area.push_back(area);

    } else {
      cdf_by_area.push_back(cdf_by_area.back() + area);
    }
  }

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, total_area);

  while ((int)surfpts.size() < num_sample) {
    float tri_sample = rand_dist(generator);
    std::vector<float>::iterator tri_index_iter =
        lower_bound(cdf_by_area.begin(), cdf_by_area.end(), tri_sample);
    int tri_index = tri_index_iter - cdf_by_area.begin();

    const Eigen::Vector3i& face = linearized_faces[tri_index];

    Eigen::Vector3f point = SamplePointFromTriangle(
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2))));

    // Now test if this point is on the shell
    int cl_index;
    float cl_distance;

    kdTree.knnSearch(point.data(), 1, &cl_index, &cl_distance);

    Eigen::Vector3f cl_vert = surface_vertices[cl_index];
    Eigen::Vector3f cl_normal = surface_normals[cl_index];

    Eigen::Vector3f ray_vec = cl_vert - point;
    float point_plane = fabs(cl_normal.dot(ray_vec));

    if (point_plane > delta)
      continue;

    surfpts.push_back(point);
  }
}

int main(int argc, char** argv) {
  std::string meshFileName;
  std::string plyOutFile;
  std::string normalizationOutputFile;
  int num_sample = 30000;

  CLI::App app{"SampleVisibleMeshSurface"};
  app.add_option("-m", meshFileName, "Mesh File Name for Reading")->required();
  app.add_option("-o", plyOutFile, "Save npy pc to here")->required();
  app.add_option("-n", normalizationOutputFile, "Save normalization");
  app.add_option("-s", num_sample, "Save ply pc to here");

  CLI11_PARSE(app, argc, argv);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

  pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);

  std::cout << geom.objects.size() << " objects" << std::endl;

  // linearize the object indices
  {
    int total_num_faces = 0;

    for (const auto& object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {
        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        total_num_faces += ibo.h;
      }
    }

    //      const int total_num_indices = total_num_faces * 3;
    pangolin::ManagedImage<uint8_t> new_buffer(3 * sizeof(uint32_t), total_num_faces);

    pangolin::Image<uint32_t> new_ibo =
        new_buffer.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);

    int index = 0;

    for (const auto& object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {
        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        for (int i = 0; i < ibo.h; ++i) {
          new_ibo.Row(index).CopyFrom(ibo.Row(i));
          ++index;
        }
      }
    }

    geom.objects.clear();
    auto faces = geom.objects.emplace(std::string("mesh"), pangolin::Geometry::Element());

    faces->second.Reinitialise(3 * sizeof(uint32_t), total_num_faces);

    faces->second.CopyFrom(new_buffer);

    new_ibo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);
    faces->second.attributes["vertex_indices"] = new_ibo;
  }

  // remove textures
  geom.textures.clear();

  pangolin::Image<uint32_t> modelFaces = pangolin::get<pangolin::Image<uint32_t>>(
      geom.objects.begin()->second.attributes["vertex_indices"]);

  // float max_dist = BoundingCubeNormalization(geom, true);

  pangolin::CreateWindowAndBind("Main", 1, 1);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_DITHER);
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_LINE_SMOOTH);
  glDisable(GL_POLYGON_SMOOTH);
  glHint(GL_POINT_SMOOTH, GL_DONT_CARE);
  glHint(GL_LINE_SMOOTH, GL_DONT_CARE);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);
  glDisable(GL_MULTISAMPLE_ARB);
  glShadeModel(GL_FLAT);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam2(
      pangolin::ProjectionMatrixOrthographic(-1, 1, 1, -1, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

  // Load geometry
  pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);

  pangolin::GlSlProgram prog = GetShaderProgram();

  // Create Framebuffer with attached textures
  size_t w = 400;
  size_t h = 400;
  pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
  pangolin::GlTexture normals(w, h, GL_RGBA32F);
  pangolin::GlTexture vertices(w, h, GL_RGBA32F);
  pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

  // View points around a sphere.
  std::vector<Eigen::Vector3f> views = EquiDistPointsOnSphere(100, 1.1);

  std::vector<Eigen::Vector4f> point_normals;
  std::vector<Eigen::Vector4f> point_verts;

  size_t num_tri = modelFaces.h;
  std::vector<Eigen::Vector4f> tri_id_normal_test(num_tri);
  for (size_t j = 0; j < num_tri; j++)
    tri_id_normal_test[j] = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
  int total_obs = 0;
  int wrong_obs = 0;

  for (unsigned int v = 0; v < views.size(); v++) {
    // change camera location
    s_cam2.SetModelViewMatrix(
        pangolin::ModelViewLookAt(views[v][0], views[v][1], views[v][2], 0, 0, 0, pangolin::AxisY));
    // Draw the scene to the framebuffer
    framebuffer.Bind();
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    prog.Bind();
    prog.SetUniform("MVP", s_cam2.GetProjectionModelViewMatrix());
    prog.SetUniform("V", s_cam2.GetModelViewMatrix());
    prog.SetUniform("ToWorld", s_cam2.GetModelViewMatrix().Inverse());
    prog.SetUniform("slant_thr", -1.0f, 1.0f);
    prog.SetUniform("ttt", 1.0, 0, 0, 1);
    pangolin::GlDraw(prog, gl_geom, nullptr);
    prog.Unbind();

    framebuffer.Unbind();

    pangolin::TypedImage img_normals;
    normals.Download(img_normals);
    std::vector<Eigen::Vector4f> im_norms = ValidPointsAndTrisFromIm(
        img_normals.UnsafeReinterpret<Eigen::Vector4f>(), tri_id_normal_test, total_obs, wrong_obs);
    point_normals.insert(point_normals.end(), im_norms.begin(), im_norms.end());

    pangolin::TypedImage img_verts;
    vertices.Download(img_verts);
    std::vector<Eigen::Vector4f> im_verts =
        ValidPointsFromIm(img_verts.UnsafeReinterpret<Eigen::Vector4f>());
    point_verts.insert(point_verts.end(), im_verts.begin(), im_verts.end());
  }

  std::vector<Eigen::Vector3f> vertices2;
  //    std::vector<Eigen::Vector3f> vertices_all;
  std::vector<Eigen::Vector3f> normals2;

  for (unsigned int v = 0; v < point_verts.size(); v++) {
    vertices2.push_back(point_verts[v].head<3>());
    normals2.push_back(point_normals[v].head<3>());
  }

  KdVertexList kdVerts(vertices2);
  KdVertexListTree kdTree_surf(3, kdVerts);
  kdTree_surf.buildIndex();

  std::vector<Eigen::Vector3f> surf_pts;
  SampleFromSurfaceInside(geom, surf_pts, num_sample, kdTree_surf, vertices2, normals2, 0.00001);
  SavePointsToPLY(surf_pts, plyOutFile);

  if (!normalizationOutputFile.empty()) {
    const std::pair<Eigen::Vector3f, float> normalizationParams =
        ComputeNormalizationParameters(geom);

    SaveNormalizationParamsToNPZ(
        normalizationParams.first, normalizationParams.second, normalizationOutputFile);
  }

  std::cout << "ended correctly" << std::endl;
  return 0;
}
