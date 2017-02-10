'use strict';

import React from 'react';
import { Card, Grid, Label, Progress } from 'semantic-ui-react';
import BoundingboxComponent from './BoundingboxComponent';

require('styles//Image.css');

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
    this.setState({selected: index});
  }

  render() {
    const prediction = this.props.image.body.predictions[0];
    const boxes = prediction.classes.map(category => {
      return [category.bbox.xmin, category.bbox.ymax,
              category.bbox.xmax - category.bbox.xmin,
              category.bbox.ymin - category.bbox.ymax];
    });
    return (
      <Card>
        <BoundingboxComponent image={prediction.uri}
                              boxes={boxes}
                              onSelected={this.onSelected}
        />
        <Card.Content>
          <Card.Description>
            <Grid columns={2}>
              {prediction.classes.map((category, index) => {
                return (<Grid.Row key={'item-' + index} className="cardRow">
                  <Grid.Column>
                    {category.cat}
                  </Grid.Column>
                  <Grid.Column>
                    <Progress color={this.state.selected == index ? 'blue' : 'grey'} percent={parseInt(category.prob * 100)} progress />
                  </Grid.Column>
                </Grid.Row>);
              })}
            </Grid>
          </Card.Description>
        </Card.Content>
        <Card.Content extra>
          <Label onClick={this.toggleJsonDisplay}>{this.state.jsonDisplay ? "Hide" : "Show"} JSON</Label>
          <pre className={this.state.jsonDisplay ? "" : "hidden"}>
            {JSON.stringify(this.props.image, null, 2)}
          </pre>
        </Card.Content>
      </Card>
    );
  }
}

ImageComponent.displayName = 'ImageComponent';

// Uncomment properties you need
ImageComponent.propTypes = {
  image: React.PropTypes.object
};
// ImageComponent.defaultProps = {};

export default ImageComponent;
