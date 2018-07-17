import 'core-js/fn/object/assign';
import React from 'react';
import ReactDOM from 'react-dom';
import App from './components/Main';

// Render the main component into the dom
const x = document.getElementsByClassName('boundingBox');

for (let i = 0; i < x.length; i++) {

  const service = x[i].dataset.service;
  const imageList = x[i].dataset.imageList ? x[i].dataset.imageList.split(',') : [];
  const confidenceThreshold = x[i].dataset.confidenceThreshold ? parseFloat(x[i].dataset.confidenceThreshold) : 0.1;
  const thresholdControl = x[i].dataset.thresholdControl && x[i].dataset.thresholdControl == 'true';
  const thresholdControlSteps = x[i].dataset.thresholdControlSteps ? x[i].dataset.thresholdControlSteps.split(',').map(value => parseFloat(value)) : [0.8, 0.5, 0.3];

  let displayBoundingBox = true;
  if(x[i].dataset.boundingbox) {
    displayBoundingBox = x[i].dataset.boundingbox == 'true';
  }

  ReactDOM.render(<App
    service={service}
    imageList={imageList}
    confidenceThreshold={confidenceThreshold}
    best={x[i].dataset.best}
    thresholdControl={thresholdControl}
    thresholdControlSteps={thresholdControlSteps}
    displayBoundingBox={displayBoundingBox}
  />, x[i]);

}
