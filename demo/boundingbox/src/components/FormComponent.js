'use strict';

import React from 'react';
import axios from 'axios';
import { observer } from 'mobx-react';
import { Button, Grid, Image, Input, Segment } from 'semantic-ui-react'

require('styles//Form.css');

@observer
class FormComponent extends React.Component {

  constructor (props) {
    super(props);
    this.state = {
      imageList: [],
      confidenceThreshold: this.props.confidenceThreshold,
      url: ''
    };
  }

  handleThresholdClick = (value) => {
    this.setState({confidenceThreshold: value}, () => {
      this.request(this.state.url);
    });
  }

  validateURL(textval) {
    const urlregex = /^(https?|ftp):\/\/([a-zA-Z0-9.-]+(:[a-zA-Z0-9.&%$-]+)*@)*((25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])){3}|([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.(com|edu|gov|int|mil|net|org|biz|arpa|info|name|pro|aero|coop|museum|[a-zA-Z]{2}))(:[0-9]+)*(\/($|[a-zA-Z0-9.,?'\\+&%$#=~_-]+))*$/;
    return urlregex.test(textval);
  }

  cleanInput = (e) => {
    e.target.parentElement.parentElement.firstChild.value= '';
  }

  push = (e) => {
    e.preventDefault();
    const input = e.target.value;
    if(this.validateURL(input)) {
      this.setState({imageList: this.state.imageList.concat([input])});
      this.request(input);
    }
  }

  demo = (e) => {
    e.preventDefault();
    this.request(e.target.src);
  }

  request = (url) => {
    this.setState({url: url});
    const self = this;
    const params = {
      service: this.props.service,
      parameters: {
        output: {
          bbox: true,
          best: (this.props.best ? 1 : -1),
          confidence_threshold: this.state.confidenceThreshold
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
        time: response.data.head.time,
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
    this.setState({imageList: this.props.imageList});
    this.request(this.props.imageList[0]);
  }

  render() {
    return (
      <Grid>
        <Grid.Row>
          <Grid.Column width={10} only='tablet mobile'>
            <Image.Group size='tiny'>
            {this.state.imageList.slice(-4).map((demo, n) => {
             return <Image key={n} className='demo' src={demo} onClick={this.demo} size='tiny'  shape='rounded'/>
            })}
            </Image.Group>
          </Grid.Column>
          <Grid.Column computer={12} only='computer'>
            <Image.Group size='tiny'>
            {this.state.imageList.slice(-8).map((demo, n) => {
             return <Image key={n} className='demo' src={demo} onClick={this.demo} size='tiny'  shape='rounded'/>
            })}
            </Image.Group>
          </Grid.Column>
          <Grid.Column computer={4} mobile={6}>
            <Segment basic>
              <Input ref={(ref) => this.input = ref} onChange={this.push} name='url' placeholder='Image URL' action={{icon: 'remove', onClick: this.cleanInput }}/>
              <Button.Group style={{paddingTop: '4px', display: this.props.thresholdControl ? 'inline-flex' : 'none'}}>
                <Button toggle active={this.state.confidenceThreshold == 0.8} onClick={this.handleThresholdClick.bind(this, 0.8)}>Salient</Button>
                <Button.Or />
                <Button toggle active={this.state.confidenceThreshold == 0.5} onClick={this.handleThresholdClick.bind(this, 0.5)}>Medium</Button>
                <Button.Or />
                <Button toggle active={this.state.confidenceThreshold == 0.3} onClick={this.handleThresholdClick.bind(this, 0.3)}>Detailed</Button>
              </Button.Group>
            </Segment>
          </Grid.Column>
        </Grid.Row>
      </Grid>
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
