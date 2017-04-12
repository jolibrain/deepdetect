/**
 * DeepDetect JS client
 *
 * Copyright © 2017 Alexandre Girard. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.txt file in the root directory of this source tree.
 */
import axios from 'axios';

class Services {

  constructor(config) {
    this.config = config;
  }

  create(name, params) {
    const url = `${this.config.apiEndpoint}/services/${name}`;
    return new Promise((resolve, reject) => {
      axios.put(url, params).then((response) => {
        resolve(response.data);
      }, reject);
    });
  }

  delete(name) {
    const url = `${this.config.apiEndpoint}/services/${name}`;
    return new Promise((resolve, reject) => {
      axios.delete(url).then((response) => {
        resolve(response.data);
      }, reject);
    });
  }

}

export default Services;
