'use strict';

import React from 'react';
import { observer } from 'mobx-react';
import { Card } from 'semantic-ui-react';
import ImageComponent from './ImageComponent';

require('styles//ImageList.css');

@observer
class ImageListComponent extends React.Component {
  render() {
    let images = this.context.store.images.map((image, index) => {
      return <ImageComponent image={image} key={`image-${index}`} />;
    });

    return (
      <Card.Group>
        { images }
      </Card.Group>
    );
  }
}

ImageListComponent.displayName = 'ImageListComponent';

ImageListComponent.contextTypes = {
  store: React.PropTypes.object
};

// Uncomment properties you need
// ImageListComponent.propTypes = {};
// ImageListComponent.defaultProps = {};

export default ImageListComponent;
