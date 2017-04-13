require('normalize.css/normalize.css');
require('styles/App.css');

import React from 'react';
import { Container, Divider } from 'semantic-ui-react';
import { observer } from 'mobx-react';

import FormComponent from './FormComponent';
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
      <Container>
        <FormComponent service={this.props.service}
                       imageList={this.props.imageList}
                       confidenceThreshold={this.props.confidenceThreshold}
                       best={this.props.best}
                       thresholdControl={this.props.thresholdControl}
                       thresholdControlSteps={this.props.thresholdControlSteps}
        />
        <Divider hidden/>
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
