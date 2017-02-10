/*eslint-env node, mocha */
/*global expect */
/*eslint no-console: 0*/
'use strict';

// Uncomment the following lines to use the react test utilities
// import React from 'react/addons';
// const TestUtils = React.addons.TestUtils;
import createComponent from 'helpers/shallowRenderHelper';

import Main from 'components/Main';

describe('MainComponent', function () {

  beforeEach(function () {
    this.MainComponent = createComponent(Main);
  });

  it('should have its component name as default className', function () {
    expect(this.MainComponent.props.className).to.equal('index');
  });
});
