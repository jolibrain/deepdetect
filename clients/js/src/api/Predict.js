/**
 * DeepDetect JS client
 *
 * Copyright © 2017 Alexandre Girard. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.txt file in the root directory of this source tree.
 */
import axios from 'axios';

class Predict {

  constructor(config) {
    this.config = config;
  }

  make(params) {
    const url = `${this.config.apiEndpoint}/predict`;
    return new Promise((resolve, reject) => {
      axios.post(url, params).then((response) => {
        resolve(response.data);
      }, reject);
    });
  }

}

export default Predict;
