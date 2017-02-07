import React, { Component, PropTypes } from "react";
import axios                           from "axios";
import { observer }                    from "mobx-react";

@observer
export default class ImageForm extends Component {
  constructor (props) {
    super(props);
  }

  push = (e) => {
    e.preventDefault();
    let self = this;
    axios.post("/api/predict", {
      service: "faces",
      parameters: {
        output: {
          bbox: true,
          confidence_threshold: 0.1
        }
      },
      data: [ self.refs.urlInput.value ]
    })
    .then(function (response) {
      self.context.store.images.push(response.data.data);
    })
    .catch(function () {
      //console.log(error);
    });
  }

  render () {
    return (
      <div>
        <form ref="form" onSubmit={this.push}>
          <input ref='urlInput' type='text' name='push-images' placeholder='Image url' />
          <button type="submit">Submit</button>
        </form>
      </div>
    );
  }
}

ImageForm.contextTypes = {
  store: PropTypes.object
};
