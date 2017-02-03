# deepdetect-js

The official JavaScript client for interacting with the [DeepDetect](http://deepdetect.com) API.

## Basic use

To start, install the SDK via NPM: ```npm install deepdetect-js```


*This will work in node.js and browsers via [Browserify](http://browserify.org/)*

```js
var DeepDetect = require('deepdetect-js');
var app = new DeepDetect.App(
);

```

You can also use the SDK by adding this script to your HTML:

```js

<script type="text/javascript" src="https://deepdetect.com/js/deepdetect.js"></script>
<script>
  var app = new DeepDetect.App(
  );
</script>
```

## Docs

Learn the basics — predicting the contents of an image, searching across a collection and creating your own models with our [Guide](https://deepdetect.com/tutorials/imagenet-classifier/).

Looking for a different client? We have many languages available with lots of documentation [API Reference](https://deepdetect.com/api/)
