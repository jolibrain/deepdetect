import React, { Component, PropTypes } from "react";
import ImageItem                       from "./ImageItem";
import { observer }                    from "mobx-react";

@observer
export default class ImageList extends Component {
  constructor(props) {
    super(props);
  }

  render () {
    let images = this.context.store.images.map(image => {
      //console.log(image);
      return <ImageItem image={image} key={`image-${image.id}`} />;
    });

    return (
      <div>
      { images }
      </div>
    );
  }
}

ImageList.contextTypes = {
  store: PropTypes.object
};
