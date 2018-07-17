/*eslint-env node, mocha */
/*global expect */
/*eslint no-console: 0*/
'use strict';

import config from 'config';

describe('appEnvConfigTests', function () {
  it('should load app config file depending on current --env', function () {
    expect(config.appEnv).to.equal('test');
  });
});
