/**
 * DeepDetect JS client
 *
 * Copyright © 2017 Alexandre Girard. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.txt file in the root directory of this source tree.
 */

import Info from './api/Info';
import Predict from './api/Predict';
import Services from './api/Services';
import Train from './api/Train';

class DeepDetect {

  constructor(apiEndpoint) {
    this.validate(apiEndpoint);
    this.init(apiEndpoint);
  }

  validate(apiEndpoint) {
    if (!apiEndpoint) {
      throw new Error('apiEndpoint param is required');
    }
  }

  init(apiEndpoint) {
    this.config = {
      apiEndpoint,
    };
    this.info = new Info(this.config);
    this.predict = new Predict(this.config);
    this.services = new Services(this.config);
    this.train = new Train(this.config);
  }

}

export default DeepDetect;
