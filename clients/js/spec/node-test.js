var DeepDetect = require('./../src/index');
var Predict = require('./../src/Predict');
const assert = require('assert');

var app;

beforeEach(function() {
  app = new DeepDetect.Service(
    process.env.API_ENDPOINT
  );
});

describe('DeepDetect JS SDK', function() {

  describe('Predict', function() {

    it('Call predict', function(done) {
      app.predict.post({
        service:"generic",
        parameters:{mllib:{gpu:true},output:{best:3}},
        data:["http://i.ytimg.com/vi/0vxOhd4qlnA/maxresdefault.jpg"]
      }).then(
          function(response) {
            assert.equal(response.status.code, 200);
            assert.equal(response.status.msg, 'OK');
            assert.equal(response.head.method, '/predict');
            assert.equal(response.head.service, 'generic');
            assert.equal(response.body.predictions.length, 1);
            done();
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
