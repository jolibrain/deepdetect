'use strict';

import React from 'react';
import axios from 'axios';
import { observer } from 'mobx-react';
import { Button, Form, Grid, Image, Input } from 'semantic-ui-react'

require('styles//Form.css');

@observer
class FormComponent extends React.Component {

  constructor (props) {
    super(props);
    this.state = { demos: [
      'http://i.imgur.com/QiksDIs.jpg',
      'https://www.nbc.com/sites/nbcunbc/files/files/styles/1080xauto/public/scet/photos/269/7924/PAA_267.JPG',
      'http://wiinoob.walyou.netdna-cdn.com/wp-content/uploads/2009/03/mii-cupcakes.jpg'
    ]};
  }

  push = (e) => {
    e.preventDefault();
    console.log(e.target.value);
    this.request(e.target.value);
  }

  demo = (e) => {
    e.preventDefault();
    this.request(e.target.src);
  }

  request = (url) => {
    const self = this;
    const params = {
      service: 'faces',
      parameters: {
        output: {
          bbox: true,
          confidence_threshold: 0.1
        }
      },
      data: [ url ]
    };

    axios.post('/api/predict', params)
    .then(function (response) {

      const prediction = response.data.body.predictions[0];
      self.context.store.image = {
        curl: params,
        body: response.data.body,
        uri: prediction.uri,
        classes: prediction.classes
      }

    })
    .catch(function (error) {
      self.context.store.image = {
        curl: params,
        body: error.response.data.body,
        uri: params.data[0],
        classes: []
      }
    });
  }

  componentDidMount() {
    this.request(this.state.demo[0]);
  }

  render() {
    return (
      <div>
        {this.state.demos.map(demo => {
         return <Image className='demo' src={demo} onClick={this.demo} size='tiny'/>
        })}
        <Input ref={(ref) => this.input = ref} onChange={this.push} name='url' placeholder='Image URL' />
      </div>
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
