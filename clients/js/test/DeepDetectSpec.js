/**
 * DeepDetect Js client
 *
 * Copyright © 2017 Alexandre Girard. All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.txt file in the root directory of this source tree.
 */

import { expect } from 'chai';
import DeepDetect from '../src/DeepDeptect';

var app;

beforeEach(function() {
  app = new DeepDetect(
    process.env.API_ENDPOINT
  );
});

describe('DeepDetect JS client', function() {

  describe('Predict', function() {

    it('Call predict', function(done) {
      app.predict.post({
        service:"generic",
        parameters:{mllib:{gpu:true},output:{best:3}},
        data:["http://i.ytimg.com/vi/0vxOhd4qlnA/maxresdefault.jpg"]
      }).then(
          function(response) {
            expect(response.status.code).to.be.equal(200);
            expect(response.status.msg).to.be.equal('OK');
            expect(response.head.method).to.be.equal('/predict');
            expect(response.head.service).to.be.equal('generic');
            expect(response.body.predictions.length).to.be.equal(1);
          },
          errorHandler.bind(done)
      );
    });
  });
});

function errorHandler(err) {
  log(err);
  this();
};

function log(obj) {
  console.log('[ERROR]', JSON.stringify(obj));
};
