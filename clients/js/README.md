# deepdetect-js

The official JavaScript client for interacting with the [DeepDetect](http://deepdetect.com) API.

## Basic use

You can use DeepDetect js client by adding this script to your HTML:

```js

<script type="text/javascript" src="https://deepdetect.com/js/deepdetect.min.js"></script>
<script>
  var app = new DeepDetect.Service('http://localhost/api');
</script>
```

## Build

To build DeepDetect js client:

```
cd deepdetect/clients/js
yarn install
yarn build
```

The files `deepdetect.js` and `deepdetect.min.js` will be available in the `dist` folder.

## Docs

Learn the basics — predicting the contents of an image, searching across a collection and creating your own models with our [Guide](https://deepdetect.com/tutorials/imagenet-classifier/).

Looking for a different client? We have many languages available with lots of documentation [API Reference](https://deepdetect.com/api/)
