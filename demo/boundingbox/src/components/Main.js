require('normalize.css/normalize.css');
require('styles/App.css');

import React from 'react';
import { Divider, Container, Header } from 'semantic-ui-react';
import { observer } from 'mobx-react';

import FormComponent from './FormComponent';
import ImageListComponent from './ImageListComponent';

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
      <Container text>
        <Header as='h2'>Boundingbox Demo</Header>
        <Divider/>
        <FormComponent />
        <Divider/>
        <ImageListComponent />
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
