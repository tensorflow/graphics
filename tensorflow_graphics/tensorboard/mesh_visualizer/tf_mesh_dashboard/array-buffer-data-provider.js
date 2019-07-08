
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
* Types of errors during network data roundtrip.
* @enum {number}
*/
vz_mesh.ErrorCodes = {
  CANCELLED: 1  // Happens when the request was cancelled before it finished.
};

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
   * @param {number} timestamp Point in time when data was recorded.
   * @param {!Object} meshData Map to populate with mesh data.
   * @return {!Object} Promise object representing server request.
   * @private
   */
  _fetchDataByTimestamp(run, tag, content_type, sample, timestamp, meshData) {
    const url = tf_backend.getRouter().pluginRoute(
        'mesh', '/data',
        new URLSearchParams({tag, run, content_type, sample, timestamp}));

    const reshapeTo1xNx3 = function (data) {
      const channelsCount = 3;
      let items = [];
      for (let i = 0; i < data.length / channelsCount; i++) {
        let dataEntry = [];
        for (let j = 0; j < channelsCount; j++) {
          dataEntry.push(data[i * channelsCount + j]);
        }
        items.push(dataEntry);
      }
      return items;
    };

    const processData = this._canceller.cancellable(response => {
      if (response.cancelled) {
        return Promise.reject({
          code: vz_mesh.ErrorCodes.CANCELLED,
          message: 'Response was invalidated.'
        });
      }
      let buffer = response.value;
      switch(content_type) {
        case 'VERTEX':
          meshData.vertices = reshapeTo1xNx3(new Float32Array(buffer));
          break;
        case 'FACE':
          meshData.faces = reshapeTo1xNx3(new Int32Array(buffer));
          break;
        case 'COLOR':
          meshData.colors = reshapeTo1xNx3(new Uint8Array(buffer));
          break;
      }
      return meshData;
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
   * Mesh may conists of vertices and optionally faces and colors, each data
   * type must be requested separately due to performance reasons.
   * @param {!Object} stepDatum Dictionary with mesh data for the current step.
   * @param {string} run Name of the run to get data for.
   * @param {string} tag Name of the tug to get data for.
   * @param {number} sample Sample index from a batch of data.
   * @private
   */
  fetchData(stepDatum, run, tag, sample) {
    let promises = [];
    // Map to populate with mesh data, i.e. vertices, faces, etc.
    let meshData = new Map();
    Object.keys(ContentType).forEach(contentType => {
      const component = (1 << ContentType[contentType]);
      if (stepDatum.components & component) {
        promises.push(this._fetchDataByTimestamp(
            run, tag, contentType, sample, stepDatum.wall_time_sec,
            meshData));
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
        return Promise.reject({
          code: vz_mesh.ErrorCodes.CANCELLED,
          message: 'Response was invalidated.'
        });
      }
      return response.value;
    });
    return this._requestManager.fetch(url)
        .then(response => response.json())
        .then(requestData)
        .then(this._processMetadata.bind(this));
  }

  /**
   * Process server raw data into frontend friendly format.
   * @param {!Array|undefined} data list of raw server records.
   * @return {!Array} list of step datums.
   * @private
   */
  _processMetadata(data) {
    if (!data) return;
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
      let datum = this._createStepDatum(data[0]);
      datums.push(datum);
    });
    return datums;
  }

  /**
   * Process single row of server-side data and puts it in more structured form.
   * @param {!Object} metadata Object describing step summary.
   * @private
   * @return {!Object} with wall_time, step number and data for the step.
   */
  _createStepDatum(metadata) {
    // TODO: add data validation to make sure frontend is
    // compatible with backend.
    return {
      // The wall time within the metadata is in seconds. The Date
      // constructor accepts a time in milliseconds, so we multiply by 1000.
      wall_time: new Date(metadata.wall_time * 1000),
      wall_time_sec: metadata.wall_time,
      step: metadata.step,
      config: metadata.config,
      content_type: metadata.content_type,
      components: metadata.components
    };
  }
}

vz_mesh.ArrayBufferDataProvider = ArrayBufferDataProvider;

})(vz_mesh || (vz_mesh = {}));  // end of vz_mesh namespace
