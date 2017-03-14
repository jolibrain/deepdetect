import 'core-js/fn/object/assign';
import React from 'react';
import ReactDOM from 'react-dom';
import App from './components/Main';

// Render the main component into the dom
const x = document.getElementsByClassName('boundingBox');

for (let i = 0; i < x.length; i++) {

  const service = x[i].dataset.service;
  const imageList = x[i].dataset.imageList ? x[i].dataset.imageList.split(',') : [];
  const confidence = x[i].dataset.confidence ? parseFloat(x[i].dataset.confidence) : 0.1;

  ReactDOM.render(<App service={service} imageList={imageList} confidence={confidence}/>, x[i]);

}
