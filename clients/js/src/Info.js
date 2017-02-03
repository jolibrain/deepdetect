let axios = require('axios');
let {API} = require('./constants');

class Predict {
  constructor(_config) {
    this._config = _config;
  }
  get() {
    let url = `${this._config.apiEndpoint}${API.INFO}`;
    return new Promise((resolve, reject)=> {
      axios.get(url).then((response)=> {
        resolve(response.data);
      }, reject);
    });
  }
};

module.exports = Predict;
