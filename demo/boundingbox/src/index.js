import 'core-js/fn/object/assign';
import React from 'react';
import ReactDOM from 'react-dom';
import App from './components/Main';

// Render the main component into the dom
const x = document.getElementsByClassName('boundingBox');
for (let i = 0; i < x.length; i++) {
  console.log(x[i].dataset);
  const service = x[i].dataset.service;
  const imageList = x[i].dataset.imagelist ? x[i].dataset.imagelist.split(',') : [];
  ReactDOM.render(<App service={service} imageList={imageList}/>, x[i]);
}
