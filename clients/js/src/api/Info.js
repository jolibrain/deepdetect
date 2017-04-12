/**
 * DeepDetect JS client
 *
 * Copyright © 2017 Alexandre Girard. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.txt file in the root directory of this source tree.
 */
import axios from 'axios';

class Info {

  constructor(config) {
    this.config = config;
  }

  get() {
    const url = `${this.config.apiEndpoint}/info`;
    return new Promise((resolve, reject) => {
      axios.get(url).then((response) => {
        resolve(response.data);
      }, reject);
    });
  }

}

export default Info;
