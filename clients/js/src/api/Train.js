/**
 * DeepDetect JS client
 *
 * Copyright © 2017 Alexandre Girard. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.txt file in the root directory of this source tree.
 */
import axios from 'axios';

class Train {

  constructor(config) {
    this.config = config;
  }

  launch(params) {
    const url = `${this.config.apiEndpoint}/train`;
    return new Promise((resolve, reject) => {
      axios.post(url, params).then((response) => {
        resolve(response.data);
      }, reject);
    });
  }

}

export default Train;
