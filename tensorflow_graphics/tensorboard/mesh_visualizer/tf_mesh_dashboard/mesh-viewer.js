
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/**
 * @fileoverview MeshViewer aims to provide 3D rendering capabilities.
 */
var vz_mesh;
(function(vz_mesh) {

class MeshViewer extends THREE.EventDispatcher {
  /**
   * MeshViewer constructor. Initializes the component and underlying objects.
   * @param {string} runColor Run color to use in case when colors are absent.
   */
  constructor(runColor) {
    super();
    /** @type {!THREE.Mesh} Last rendered mesh. */
    this._lastMesh = null;
    this._clock = new THREE.Clock();
    /** @type {!Object} Contains width and height of the canvas. */
    this._canvasSize = null;
    this._runColor = runColor;
  }

  // TODO(b/130030314) replace with some thirdparty library call.
  /**
   * Returns true if the specified value is an object.
   * @param {?} val Variable to test.
   * @private
   * @return {boolean} Whether variable is an object.
   */
  _isObject(val) {
    var type = typeof val;
    // We're interested in objects representing dictionaries only. Everything
    // else is "not mergeable", so we consider it as primitive types.
    return type == 'object' && val != null && !Array.isArray(val);
  }

  /**
   * Merges two configs together.
   * @param {!Object} userConfig User configuration has higher priority.
   * @param {!Object} defaultConfig Default configuration has lower priority and
   *   will be overridden by any conflicting keys from userConfig.
   * @private
   * @return {!Object} Merged dictionary from two configuration dictionaries.
   */
  _applyDefaults(userConfig, defaultConfig) {
    let mergedConfig = {};
    const configs = [userConfig, defaultConfig];
    for (let i = 0; i < configs.length; i++) {
      const config = configs[i];
      for (let key in config) {
        const is_key_present = key in mergedConfig;
        if (this._isObject(config[key])) {
          mergedConfig[key] =
              this._applyDefaults(mergedConfig[key] || {}, config[key]);
        } else if (!is_key_present) {
          mergedConfig[key] = config[key];
        }
      }
    }
    return mergedConfig;
  }

  /**
   * Creates scene, camera and renderer.
   * @param {!Object} config Scene rendering configuration.
   * @param {!HTMLDOMElement} domElement The HTML element used for event listeners.
   * @private
   */
  _createWorld(config, domElement) {
    if (this.isReady()) {  // keep world objects as singleton objects.
      return;
    }
    this._scene = new THREE.Scene();
    var camera = new THREE[config.camera.cls](
        config.camera.fov, this._canvasSize.width / this._canvasSize.height,
        config.camera.near, config.camera.far);
    this._camera = camera;

    var camControls = new THREE.OrbitControls(camera, domElement);
    camControls.lookSpeed = 0.4;
    camControls.movementSpeed = 20;
    camControls.noFly = true;
    camControls.lookVertical = true;
    camControls.constrainVertical = true;
    camControls.verticalMin = 1.0;
    camControls.verticalMax = 2.0;
    camControls.addEventListener(
        'change', this._onCameraPositionChange.bind(this));

    this._cameraControls = camControls;

    this._renderer = new THREE.WebGLRenderer({antialias: true});
    this._renderer.setPixelRatio(window.devicePixelRatio);
    this._renderer.setSize(this._canvasSize.width, this._canvasSize.height);
    this._renderer.setClearColor(0xffffff, 1);
  }

  /**
   * Clears scene from any 3D geometry.
   */
  _clearScene() {
    while (this._scene.children.length > 0) {
      this._scene.remove(this._scene.children[0]);
    }
  }

  /**
   * Returns underlying renderer.
   * @public
   */
  getRenderer() {
    return this._renderer;
  }

  /**
   * Returns underlying camera controls.
   * @public
   */
  getCameraControls() {
    return this._cameraControls;
  }

  /**
   * Returns true when all underlying components were initialized.
   * @public
   */
  isReady() {
    return !!this._camera && !!this._cameraControls;
  }

  /**
   * Returns current camera position.
   * @public
   */
  getCameraPosition() {
    return {
      far: this._camera.far,
      position: this._camera.position.clone(),
      target: this._cameraControls.target.clone()
    };
  }

  /**
   * Sets new canvas size.
   * @param {!Object} canvasSize Contains current canvas width and height.
   * @public
   */
  setCanvasSize(canvasSize) {
    this._canvasSize = canvasSize;
  }

  /**
   * Renders component into the browser.
   * @public
   */
  draw() {
    // Cancel any previous requests to perform redraw.
    if (this._animationFrameIndex) {
      cancelAnimationFrame(this._animationFrameIndex);
    }
    this._camera.aspect = this._canvasSize.width / this._canvasSize.height;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(this._canvasSize.width, this._canvasSize.height);
    const animate = function () {
      var delta = this._clock.getDelta();
      this._cameraControls.update(delta);
      this._animationFrameIndex = requestAnimationFrame(animate);
      this._renderer.render(this._scene, this._camera);
    }.bind(this);
    animate();
  }

  /**
   * Updates the scene.
   * @param {!Object} currentStep Step datum.
   * @param {!HTMLDOMElement} domElement The HTML element used for event listeners.
   * @public
   */
  updateScene(currentStep, domElement) {
    let config = {};
    if ('config' in currentStep && currentStep.config) {
      config = JSON.parse(currentStep.config);
    }
    // This event is an opportunity for UI-responsible component (parent) to set
    // proper canvas size.
    this.dispatchEvent({type:'beforeUpdateScene'});
    const default_config = {
      camera: {cls: 'PerspectiveCamera', fov: 75, near: 0.1, far: 1000},
      lights: [
        {cls: 'AmbientLight', color: '#ffffff', intensity: 0.75}, {
          cls: 'DirectionalLight',
          color: '#ffffff',
          intensity: 0.75,
          position: [0, -1, 2]
        }
      ]
    };
    config = this._applyDefaults(config, default_config);
    this._createWorld(config, domElement);
    this._clearScene();
    this._createLights(this._scene, config);
    this._createGeometry(currentStep, config);
    this.draw();
  }

  /**
   * Sets camera to default position and zoom.
   * @param {?THREE.Mesh} mesh Mesh to fit into viewport.
   * @public
   */
  resetView(mesh) {
    if (!this.isReady()) return;
    this._cameraControls.reset();

    if (!mesh && this._lastMesh) {
      mesh = this._lastMesh;
    }

    if (mesh) {
      this._fitObjectToViewport(mesh);
      // Store last mesh in case of resetView method called due to some events.
      this._lastMesh = mesh;
    }

    this._cameraControls.update();
  }

  /**
   * Creates geometry for current step data.
   * @param {!Object} currentStep Step datum.
   * @param {!Object} config Scene rendering configuration.
   * @private
   */
  _createGeometry(currentStep, config) {
    if (currentStep.vertices && currentStep.faces) {
      this._createMesh(currentStep, config);
    } else {
      this._createPointCloud(currentStep, config);
    }
  }

  /**
   * Creates point cloud geometry for current step data.
   * @param {!Object} currentStep Step datum.
   * @param {!Object} config Scene rendering configuration.
   * @private
   */
  _createPointCloud(currentStep, config) {
    const points = currentStep.vertices;
    const colors = currentStep.colors;
    let defaultConfig = {
      material: {
        cls: 'PointsMaterial', size: 0.005
      }
    };
    // Determine what colors will be used.
    if (colors && colors.length == points.length) {
      defaultConfig.material['vertexColors'] = THREE.VertexColors;
    } else {
      defaultConfig.material['color'] = this._runColor;
    }
    const pc_config = this._applyDefaults(config, defaultConfig);

    var geometry = new THREE.Geometry();
    points.forEach(function(point) {
      var p = new THREE.Vector3(point[0], point[1], point[2]);

      const scale = 1.;
      p.x = point[0] * scale;
      p.y = point[1] * scale;
      p.z = point[2] * scale;

      geometry.vertices.push(p);
    });

    colors.forEach(function (color) {
      const c = new THREE.Color(
          color[0] / 255., color[1] / 255., color[2] / 255.);
      geometry.colors.push(c);
    });

    var material = new THREE[pc_config.material.cls](pc_config.material);
    var mesh = new THREE.Points(geometry, material);
    this._scene.add(mesh);
    this._lastMesh = mesh;
  }

  /**
   * Creates mesh geometry for current step data.
   * @param {!THREE.Vector3} position Position of the camera.
   * @param {number} far Camera frustum far plane.
   * @param {!THREE.Vector3} target Point in space for camera to look at.
   * @public
   */
  setCameraViewpoint(position, far, target) {
    this._silent = true;
    this._camera.far = far;
    this._camera.position.set(position.x, position.y, position.z);
    this._camera.lookAt(target.clone());
    this._camera.updateProjectionMatrix();
    this._cameraControls.target = target.clone();
    this._cameraControls.update();
    this._silent = false;
  }

   /**
   * Triggered when camera position changed.
   * @private
   */
  _onCameraPositionChange(event) {
    if (this._silent) return;
    this.dispatchEvent({type:'cameraPositionChange', event: event});
  }

  /**
   * Positions camera on such distance from the object that the whole object is
   * visible.
   * @param {!THREE.Mesh} mesh Mesh to fit into viewport.
   * @private
   */
  _fitObjectToViewport(mesh) {
    // Small offset multiplicator to avoid edges of mesh touching edges of
    // viewport.
    const offset = 1.25;
    const boundingBox = new THREE.Box3();
    boundingBox.setFromObject(mesh);
    const center = boundingBox.center();
    const size = boundingBox.size();
    const max_dim = Math.max(size.x, size.y, size.z);
    const fov = this._camera.fov * (Math.PI / 180);
    let camera_z = Math.abs(max_dim / (2 * Math.tan(fov / 2))) * offset;
    const min_z = boundingBox.min.z;
    // Make sure that even after arbitrary rotation mesh won't be clipped.
    const camera_to_far_edge =
        (min_z < 0) ? -min_z + camera_z : camera_z - min_z;
    // Set camera position and orientation.
    this.setCameraViewpoint(
        {x: center.x, y: center.y, z: camera_z}, camera_to_far_edge * 3,
        center);
  }

  /**
   * Creates mesh geometry for current step data.
   * @param {!Object} currentStep Step datum.
   * @param {!Object} config Scene rendering configuration.
   * @private
   */
  _createMesh(currentStep, config) {
    const vertices = currentStep.vertices;
    const faces = currentStep.faces;
    const colors = currentStep.colors;
    const mesh_config = this._applyDefaults(config, {
      material: {
        cls: 'MeshStandardMaterial',
        color: '#a0a0a0',
        roughness: 1,
        metalness: 0,
      }
    });

    // TODO(podlipensky): use BufferGeometry for performance reasons!
    let geometry = new THREE.Geometry();

    vertices.forEach(function(point) {
      let p = new THREE.Vector3(point[0], point[1], point[2]);

      const scale = 1.;
      p.x = point[0] * scale;
      p.y = point[1] * scale;
      p.z = point[2] * scale;

      geometry.vertices.push(p);
    });

    faces.forEach(function(face_indices) {
      let face =
          new THREE.Face3(face_indices[0], face_indices[1], face_indices[2]);
      if (colors) {
        const face_colors = [
          colors[face_indices[0]], colors[face_indices[1]],
          colors[face_indices[2]]
        ];
        for (let i = 0; i < face_colors.length; i++) {
          const vertex_color = face_colors[i];
          let color = new THREE.Color(
              vertex_color[0] / 255., vertex_color[1] / 255.,
              vertex_color[2] / 255.);
          face.vertexColors.push(color);
        }
      }
      geometry.faces.push(face);
    });

    if (colors) {
      mesh_config.material = mesh_config.material || {};
      mesh_config.material.vertexColors = THREE.VertexColors;
    }

    geometry.center();
    geometry.computeBoundingSphere();
    geometry.computeVertexNormals();

    let material = new THREE[mesh_config.material.cls](mesh_config.material);

    let mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    this._scene.add(mesh);
    this._lastMesh = mesh;
  }

  /**
   * Creates lights for a given scene based on passed configuration.
   * @param {!Scene} scene Scene object to add lights to.
   * @param {!Object} config Scene rendering configuration.
   * @private
   */
  _createLights(scene, config) {
    for (let i = 0; i < config.lights.length; i++) {
      const light_config = config.lights[i];
      let light = new THREE[light_config.cls](
          light_config.color, light_config.intensity);
      if (light_config.position) {
        light.position.set(
            light_config.position[0], light_config.position[1],
            light_config.position[2]);
      }
      scene.add(light);
    }
  }
}  // end of MeshViewer class.

vz_mesh.MeshViewer = MeshViewer;

})(vz_mesh || (vz_mesh = {}));
