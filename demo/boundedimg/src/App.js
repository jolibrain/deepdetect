import React, { Component, PropTypes } from "react";
import { observer }                    from "mobx-react";

import Store from "./Store";

import ImageForm from "./image/ImageForm";
import ImageList from "./image/ImageList";

var store;

@observer
class App extends Component {
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
      <div>
        <ImageForm />
        <ImageList />
      </div>
    );
  }
}

App.childContextTypes = {
  store: PropTypes.object
};

export default App;
