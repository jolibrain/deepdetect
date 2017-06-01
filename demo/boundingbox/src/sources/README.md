# About this folder
This folder will hold all of your **flux** datasources.
You can include them into your components or stores like this:

```javascript
let react = require('react/addons');
let MySource = require('sources/MyAction');
class MyComponent extends React.Component {
  constructor(props) {
    super(props);
    MySource.getRemoteData();
  }
}
```
