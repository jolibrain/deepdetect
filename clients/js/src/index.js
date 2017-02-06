let Service = require('./Service');
let {version} = require('./../package.json');

module.exports = global.DeepDetect = {
  version,
  Service
};
