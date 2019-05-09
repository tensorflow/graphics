
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
 * @fileoverview ArrayBufferProvider responsible for making requests to server,
 * receive and parse response.
 */

// TODO: this class must be refactored into base DataProvider and
// subclass ArrayBufferDataProvider later.
var vz_mesh;
(function(vz_mesh) {

/**
* Types of content displayed by the plugin.
* @enum {number}
*/
const ContentType = {
  VERTEX: 1,
  FACE: 2,
  COLOR: 3
};

/**
* Types of content displayed by the plugin mapped to underlying data types.
* @enum {string}
*/
const ContentTypeToItemType = {
  VERTEX: 'float32',
  FACE: 'int32',
  COLOR: 'uint8'
};

class ArrayBufferDataProvider {

  /**
   * ArrayBufferDataProvider constructor, initializes everything needed for
   * future requests to the server.
   * @param {!Object} requestManager Request manager to communicate with the
   *  server.
   */
  constructor(requestManager) {
    this._requestManager = requestManager;
    this._canceller = new tf_backend.Canceller();
  }

  /**
   * Requests new data from the server.
   */
  reload(run, tag, sample) {
    this._canceller.cancelAll();
    return this._fetchMetadata(run, tag, sample);
  }

  /**
   * Requests new data of some particular type from the server.
   * @param {string} run Name of the run to get data for.
   * @param {string} tag Name of the tug to get data for.
   * @param {string} content_type Type of the content to retrieve.
   * @param {!array} metadata List of metadata to complete with data from the
   *  server.
   * @param {number} sample Sample index from a batch of data.
   * @return {!Object} Promise object representing server request.
   * @private
   */
  _fetchDataByType(run, tag, content_type, metadata, sample) {
    const url = tf_backend.getRouter().pluginRoute(
        'mesh', '/data', new URLSearchParams({tag, run, content_type, sample}));

    const processData = this._canceller.cancellable(response => {
      if (response.cancelled) {
        return;
      }
      let buffer = response.value;
      let data;
      switch(content_type) {
        case 'VERTEX':
          data = new Float32Array(buffer);
          break;
        case 'FACE':
          data = new Int32Array(buffer);
          break;
        case 'COLOR':
          data = new Uint8Array(buffer);
          break;
      }
      // TODO: handle empty metadata.
      // itemShape expected to be of shape BxNxK, where B stands for batch,
      // N for number of points and K is either number of items representing
      // coordinate (x,y,z) or face indices or color (rgb).
      const itemShape = metadata[0].data_shape;
      if (itemShape.length != 3) {
        throw 'Data item shape expected to have 3 dimensions BxNxK';
      }
      const itemsCount = itemShape.slice(1).reduce((a, b) => a * b);
      for (let i = 0; i < metadata.length; i++) {
        let itemsFlat = data.slice(itemsCount * i, (i + 1) * itemsCount);
        let items = [];
        for (let n = 0; n < itemShape[1]; n++) {
          items.push([]);
          for (let m = 0; m < itemShape[2]; m++) {
            items[n].push(itemsFlat[n * itemShape[2] + m]);
          }
        }
        metadata[i].data = items;
      }
    });
    return this._requestManager
        .fetch(
            url, null, 'arraybuffer',
            ContentTypeToItemType[content_type])
        .then(response => response.arrayBuffer())
        .then(processData);
  }

  /**
   * Requests new data for each type of metadata from the server.
   * @param {string} run Name of the run to get data for.
   * @param {string} tag Name of the tug to get data for.
   * @param {!array} metadata List of metadata to complete with data from the
   *  server.
   * @param {number} sample Sample index from a batch of data.
   * @private
   */
  _fetchData(run, tag, metadata, sample) {
    let promises = [];
    Object.keys(ContentType).forEach(contentType => {
      let metadataType = [];
      for (let i = 0; i < metadata.length; i++) {
        if (metadata[i].content_type == ContentType[contentType]) {
          metadataType.push(metadata[i]);
        }
      }
      if (metadataType.length) {
        promises.push(this._fetchDataByType(
            run, tag, contentType, metadataType, sample));
      }
    });
    return Promise.all(promises);
  }

  /**
   * Requests new metadata from the server
   * @param {string} run Name of the run to get data for.
   * @param {string} tag Name of the tug to get data for.
   * @param {number} sample Sample index from a batch of data.
   * completion.
   * @private
   */
  _fetchMetadata(run, tag, sample) {
    this._canceller.cancelAll();
    const url = tf_backend.getRouter().pluginRoute(
        'mesh', '/meshes', new URLSearchParams({tag, run, sample}));
    const requestData = this._canceller.cancellable(response => {
      if (response.cancelled) {
        return;
      }
      let metadata = response.value;
      return this._fetchData(run, tag, metadata, sample).then(() => {
        return this._processMetadata(metadata);
      });
    });
    return this._requestManager.fetch(url)
        .then(response => response.json())
        .then(requestData);
  }

  /**
   * Process server raw data into frontend friendly format.
   * @param {!Array} data list of raw server records.
   * @return {!Array} list of step datums.
   * @private
   */
  _processMetadata(data) {
    const timestampToData = new Map();
    for (let i = 0; i < data.length; i++) {
      let dataEntry = data[i];
      if (!timestampToData.has(dataEntry.wall_time)) {
        timestampToData.set(dataEntry.wall_time, []);
      }
      timestampToData.get(dataEntry.wall_time).push(dataEntry);
    }
    let datums = [];
    timestampToData.forEach((data) => {
      let datum;
      let vertices = data.filter(i => i.content_type == ContentType.VERTEX);
      let faces = data.filter(i => i.content_type == ContentType.FACE);
      let colors = data.filter(i => i.content_type == ContentType.COLOR);
      datum = this._createStepDatum(data[0], vertices, faces, colors);
      datums.push(datum);
    });
    return datums;
  }

  /**
   * Process single row of server-side data and puts it in more structured form.
   * @param {!Object} metadata Object describing step summary.
   * @param {!Array} vertices List of 3D coordinates of vertices.
   * @param {?Array} faces List of indices of coordinates which form mesh faces.
   * @param {?Array} colors List of colors for each vertex.
   * @private
   * @return {!Object} with wall_time, step number and data for the step.
   */
  _createStepDatum(metadata, vertices, faces, colors) {
    // TODO: add data validation to make sure frontend is
    // compatible with backend.
    return {
      // The wall time within the metadata is in seconds. The Date
      // constructor accepts a time in milliseconds, so we multiply by 1000.
      wall_time: new Date(metadata.wall_time * 1000),
      step: metadata.step,
      vertices: vertices[0] && vertices[0].data,
      faces: faces && faces[0] && faces[0].data,
      colors: colors && colors[0] && colors[0].data,
      config: metadata.config
    };
  }
}

vz_mesh.ArrayBufferDataProvider = ArrayBufferDataProvider;

})(vz_mesh || (vz_mesh = {}));  // end of vz_mesh namespace
