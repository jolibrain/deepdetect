require('normalize.css/normalize.css');
require('styles/App.css');

import React from 'react';
import { Container } from 'semantic-ui-react';
import { observer } from 'mobx-react';

import ImageComponent from './ImageComponent';

import Store from '../stores/Store';

let store;

@observer
class AppComponent extends React.Component {

  constructor (props) {
    super(props);
    store = new Store();
  }

  getChildContext() {
    return {
      store
    };
  }

  render() {
    return (
      <Container fluid>
        <ImageComponent />
      </Container>
    );
  }
}

AppComponent.childContextTypes = {
  store: React.PropTypes.object
};

AppComponent.defaultProps = {
};

export default AppComponent;
