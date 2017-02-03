let Predict = require('./Predict');
let Info = require('./Info');
let {ERRORS} = require('./constants');

/**
* top-level class that allows access to predict
* @class
*/

class App {
  constructor(apiEndpoint, options) {
    this._validate(apiEndpoint, options);
    this._init(apiEndpoint, options);
  }
  _validate(apiEndpoint, options) {
    if (!apiEndpoint) {
      throw ERRORS.paramsRequired(['apiEndpoint']);
    }
  }
  _init(apiEndpoint, options={}) {
    this._config = {
      apiEndpoint: apiEndpoint
    };
    this.predict = new Predict(this._config);
    this.info = new Info(this._config);
  }
};

module.exports = App;
