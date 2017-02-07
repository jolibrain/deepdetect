import React, { Component } from "react";

export default class ImageItem extends Component {
  constructor (props) {
    super(props);
  }

  render () {
    return (
      <div className='image'>
      <p>{this.props.image.title}</p>
      <p>{this.props.image.description}</p>
      </div>
    );
  }
}
