'use strict';

import React from 'react';
import axios from 'axios';
import { observer } from 'mobx-react';
import { Button, Form } from 'semantic-ui-react'

require('styles//Form.css');

@observer
class FormComponent extends React.Component {

  constructor (props) {
    super(props);
  }

  push = (e, { formData }) => {
    e.preventDefault();
    const self = this;
    axios.post('/api/predict', {
      service: 'faces',
      parameters: {
        output: {
          bbox: true,
          confidence_threshold: 0.1
        }
      },
      data: [ formData.url ]
    })
    .then(function (response) {
      self.context.store.images.push(response.data);
    })
    .catch(function () {
      //console.log(error);
    });
  }

  render() {
    return (
      <Form onSubmit={this.push}>
        <Form.Field>
          <label>Image URL</label>
          <input name='url' placeholder='URL' />
        </Form.Field>
        <Button type='submit'>Submit</Button>
      </Form>
    );
  }
}

FormComponent.displayName = 'FormComponent';

FormComponent.contextTypes = {
  store: React.PropTypes.object
};

// Uncomment properties you need
// FormComponent.propTypes = {};
// FormComponent.defaultProps = {};

export default FormComponent;
