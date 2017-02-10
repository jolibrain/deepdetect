'use strict';

import React from 'react';
import ReactDOM from 'react-dom';

class BoundingboxComponent extends React.Component {

  state = {
    canvasCreated: false,
    hoverIndex: -1
  };

  renderBox(index, box, selected) {

    let canvas = ReactDOM.findDOMNode(this.refs.canvasImage);
    let ctx = canvas.getContext('2d');

    let [x, y, width, height] = box;

    let colorStyle = this.props.options.colors.normal;
    if(this.state.hoverIndex >= 0) {
      colorStyle = this.props.options.colors.unselected;
    }
    if(selected) {
      colorStyle = this.props.options.colors.selected;
    }

    let lineWidth = 2;
    if(canvas.width > 600)
      lineWidth = 3;
    if(canvas.width > 1000)
      lineWidth = 5;

    if(x < lineWidth / 2)
      x = lineWidth / 2;
    if(y < lineWidth / 2)
      y = lineWidth / 2;

    if((x + width) > canvas.width)
      width = canvas.width - lineWidth - x;
    if((y + height) > canvas.height)
      height = canvas.height - lineWidth - y;

    // Left segment
    ctx.strokeStyle = colorStyle;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(x + width / 10, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + height);
    ctx.lineTo(x + width / 10, y + height);
    ctx.stroke();

    // Right segment
    ctx.beginPath();
    ctx.moveTo(x + 9 * width / 10, y);
    ctx.lineTo(x + width, y);
    ctx.lineTo(x + width, y + height);
    ctx.lineTo(x + 9 * width / 10, y + height);
    ctx.stroke();

    /* uncomment to DEBUG
    ctx.font = "30px Arial";
    ctx.fillStyle = 'rgba(225,0,0,1)';
    //ctx.fillText(this.props.boxids[index].map(i => i.slice(0, 2)).join(','), x,y+height);
    ctx.fillStyle = 'rgba(0,0,225,1)';
    ctx.fillText(index,x+width,y);
  */
  }

  renderBoxes() {
    this.props.boxes
      .map((box, index) => {
        const selected = index == this.state.hoverIndex;
        return {box: box, index: index, selected: selected};
      })
      .sort((a) => {
        return a.selected ? 1 : -1;
      })
      .forEach(box => this.renderBox(box.index, box.box, box.selected));
  }

  componentWillReceiveProps() {
    let canvas = ReactDOM.findDOMNode(this.refs.canvasImage);
    let ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let background = new Image();
    background.src = this.props.image;
    ctx.drawImage(background,0,0);
    return true;
  }

  shouldComponentUpdate() {
      return true;
  }

  componentDidMount() {
    let canvas = ReactDOM.findDOMNode(this.refs.canvasImage);
    let ctx = canvas.getContext('2d');

    let background = new Image();
    background.src = this.props.image;

    // Make sure the image is loaded first otherwise nothing will draw.
    background.onload = (() => {

      canvas.width = background.width;
      canvas.height = background.height;

      ctx.drawImage(background,0,0);
      this.renderBoxes();

      canvas.onmousemove = ((e) => {

        // Get the current mouse position
        const r = canvas.getBoundingClientRect();
        const scaleX = canvas.width / r.width;
        const scaleY = canvas.height / r.height;
        const x = (e.clientX - r.left) * scaleX;
        const y = (e.clientY - r.top) * scaleY;

        //ctx.clearRect(0, 0, canvas.width, canvas.height);

        let selectedBox = {index: -1, dimensions: null};
        for(var i = this.props.boxes.length - 1, b; b = this.props.boxes[i]; i--) {
          const [bx, by, bw, bh] = b;

          if(x >= bx && x <= bx + bw &&
             y >= by && y <= by + bh) {
              // The mouse honestly hits the rect
              const insideBox = !selectedBox.dimensions || (
                bx >= selectedBox.dimensions[0] &&
                bx <= selectedBox.dimensions[0] + selectedBox.dimensions[2] &&
                by >= selectedBox.dimensions[1] &&
                by <= selectedBox.dimensions[1] + selectedBox.dimensions[3]
              );
              if(insideBox) {
                selectedBox.index = i;
                selectedBox.dimensions = b;
              }
          }
        }

        this.props.onSelected(selectedBox.index);
        this.setState({hoverIndex: selectedBox.index});
      });

      canvas.onmouseout = () => {
        this.props.onSelected(-1);
        this.setState({hoverIndex: -1});
        //this.renderBoxes();
      };
    });
  }

  componentDidUpdate() {
    this.renderBoxes();
  }

  render() {
    return (
      <div className="boundingbox-component">
        <canvas ref="canvasImage"/>
      </div>
    );
  }
}

BoundingboxComponent.displayName = 'BoundingboxComponent';

// Uncomment properties you need
BoundingboxComponent.propTypes = {
  image: React.PropTypes.string,
  boxes: React.PropTypes.oneOfType([
    React.PropTypes.arrayOf(React.PropTypes.array),
    React.PropTypes.arrayOf(React.PropTypes.object)
  ]),
  onSelected: React.PropTypes.func,
  options: React.PropTypes.shape({
    colors: React.PropTypes.shape({
      normal: React.PropTypes.string,
      selected: React.PropTypes.string,
      unselected: React.PropTypes.string
    })
  })
};

BoundingboxComponent.defaultProps = {
  options: {
    colors: {
      normal: 'rgba(255,225,255,1)',
      selected: 'rgba(0,225,204,1)',
      unselected: 'rgba(100,100,100,1)'
    }
  }
};

export default BoundingboxComponent;
