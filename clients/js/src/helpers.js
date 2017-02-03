const SUCCESS_CODES = [200, 201];

module.exports = {
  isSuccess: (response)=> {
    return SUCCESS_CODES.indexOf(response.status) > -1;
  },
  deleteEmpty: (obj, strict=false)=> {
    Object.keys(obj).forEach((key) => {
      if (obj[key] === null ||
          obj[key] === undefined ||
          strict === true && (
          obj[key] === '' ||
          obj[key].length === 0 ||
          Object.keys(obj[key]).length === 0)) {
        delete obj[key];
      }
    });
  },
  clone: (obj)=> {
    let keys = Object.keys(obj);
    let copy = {};
    keys.forEach((k) => {
      copy[k] = obj[k];
    });
    return copy;
  },
  checkType: (regex, val)=> {
    if ((regex instanceof RegExp) === false) {
      regex = new RegExp(regex);
    }
    return regex.test(Object.prototype.toString.call(val));
  }
};
