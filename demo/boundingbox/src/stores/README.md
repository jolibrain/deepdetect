# About this folder
This folder will hold all of your **flux** stores.
You can include them into your components like this:

```javascript
let react = require('react/addons');
let MyStore = require('stores/MyStore');
class MyComponent extends React.Component {
  constructor(props) {
    super(props);
    MyStore.doSomething();
  }
}
```
