# Bounding Box demo

## Requirements

* [yarn](https://yarnpkg.com)

## Installation

```
yarn install
```

## Demo server

```
yarn serve
```

A proxy to a deepdetect server is needed on */api/* endpoint.

the current webpack configuration for dev environment - *./cfg/dev.js* - redirects */api/* requests to *http://localhost/* :

```
devServer: {
  proxy: {
    '/api/*': {
      target: 'http://localhost/',
    }
  },
```

## Build dist file

```
yarn dist
```
