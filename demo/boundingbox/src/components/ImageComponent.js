'use strict';

import React from 'react';
import { observer } from 'mobx-react';
import { Card, Grid, Label } from 'semantic-ui-react';
import Boundingbox from 'react-bounding-box';

require('styles//Image.css');

@observer
class ImageComponent extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      selected: -1,
      jsonDisplay: false
    };
  }

  toggleJsonDisplay = () => {
    this.setState({jsonDisplay: !this.state.jsonDisplay});
  }

  onSelected = (index) => {
    //console.log(`onSelected ${index}`);
    this.setState({selected: index});
  }

  onIconLeave = () => {
    this.setState({selected: -1});
  }

  render() {
    const image = this.context.store.image;
    let description = '';
    let boxes = [];
    if(image.classes != null) {
      image.classes.forEach(category => {
        if(category.bbox) {
          boxes.push({
            coord: [category.bbox.xmin, category.bbox.ymax,
                   category.bbox.xmax - category.bbox.xmin,
                   category.bbox.ymin - category.bbox.ymax],
            label: ''
          });
        }
      });
      description = image.classes.map((category, n) => {
        let bottomClass = 'fa fa-stack-2x ' + category.cat;
        bottomClass += this.state.selected == n ? ' fa-square' : ' fa-circle';
        let opacity = this.state.selected == n ? 1 : category.prob;

        let topClass = 'fa fa-stack-1x fa-inverse fa-' + category.cat;
        return (
          <span key={n} className='fa-stack fa-lg' onMouseOver={this.onSelected.bind(this, n)} onMouseLeave={this.onIconLeave}>
            <i className={bottomClass} style={{opacity: opacity}}/>
            <i className={topClass} style={{opacity: opacity}}/>
          </span>
        );
      });
    } else {
      description = 'Request failed';
    }

    let code ='';
    if(this.state.jsonDisplay) {
      code = JSON.stringify(image.body, null, 2);
    } else {
      code = `curl -X POST 'http://localhost:8000/predict' -d '${JSON.stringify(image.curl, null, 2)}'`;
    }

    return (
      <Grid >
        <Grid.Column width={12} only='computer' textAlign='center'>
          <Boundingbox key={image.uri}
                       image={image.uri}
                       boxes={boxes}
                       selectedIndex={this.state.selected}
                       onSelected={this.onSelected}
          />
        </Grid.Column>
        <Grid.Column width={16} only='tablet mobile' textAlign='center'>
          <Boundingbox key={image.uri}
                       image={image.uri}
                       boxes={boxes}
                       selectedIndex={this.state.selected}
                       onSelected={this.onSelected}
                       options={{
                         colors: {
                          normal: 'rgba(255,225,255,1)',
                          selected: 'rgba(0,225,204,1)',
                          unselected: 'rgba(100,100,100,1)'
                         },
                         style:{maxWidth: '100%', maxHeight: '50vh'}
                       }}
          />
        </Grid.Column>
        <Grid.Column computer={4} mobile={16}>
          <Card fluid>
            <Card.Content>
              <Card.Description>
                {description}
                <p>{this.state.selected == -1 ? ' ' : image.classes[this.state.selected].cat + ' - ' + image.classes[this.state.selected].prob.toFixed(2)}</p>
              </Card.Description>
            </Card.Content>
            <Card.Content extra>
              <Label onClick={this.toggleJsonDisplay} color={this.state.jsonDisplay ? 'grey' : 'blue'}>CURL</Label>
              <Label onClick={this.toggleJsonDisplay} color={this.state.jsonDisplay ? 'blue' : 'grey'}>JSON Response</Label>
              <pre>{code}</pre>
            </Card.Content>
          </Card>
        </Grid.Column>
      </Grid>
    );
  }
}

ImageComponent.displayName = 'ImageComponent';

ImageComponent.contextTypes = {
  store: React.PropTypes.object
};
// Uncomment properties you need
ImageComponent.propTypes = {
  image: React.PropTypes.object
};
// ImageComponent.defaultProps = {};

export default ImageComponent;
