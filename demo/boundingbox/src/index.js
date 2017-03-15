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

  ReactDOM.render(<App service={service} imageList={imageList} confidenceThreshold={confidenceThreshold}/>, x[i]);

}
