module.exports = {
  API: {
    INFO: '/info',
    SERVICES: '/services',
    TRAIN: '/train',
    PREDICT: '/predict'
  },
  ERRORS: {
    paramsRequired: function paramsRequired(param) {
      let paramList = Array.isArray(param) ? param: [param];
      return new Error(`The following ${paramList.length > 1? 'params are': 'param is'} required: ${paramList.join(', ')}`);
    }
  }
};
