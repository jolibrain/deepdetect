var webpack = require('webpack');

module.exports = {
  entry: {
    "deepdetect": "./src/index.js",
    "deepdetect.min": "./src/index.js",
  },
  output: {
    path: "./dist",
    filename: "[name].js"
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        use: 'babel-loader?presets[]=es2015',
        exclude: /(node_modules|bower_components)/,
      },
    ],
  },
  plugins: [
    new webpack.optimize.UglifyJsPlugin({
      include: /\.min\.js$/,
      minimize: true
    })
  ]
};
