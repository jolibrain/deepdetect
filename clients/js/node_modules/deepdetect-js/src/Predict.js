let axios = require('axios');
let {API} = require('./constants');

class Predict {
  constructor(_config) {
    this._config = _config;
  }
  post(params) {
    let url = `${this._config.apiEndpoint}${API.PREDICT}`;
    return new Promise((resolve, reject)=> {
      axios.post(url, params).then((response)=> {
        resolve(response.data);
      }, reject);
    });
  }
};

module.exports = Predict;
