export default class DrawCanvas{
    constructor(canvas=null){
        this.canvas = canvas || $('#canvas')[0];
        this.ctx = this.canvas.getContext('2d');
        this.lastPos = null;
        this.radius = 3;

        this.undo = [];

        $(this.canvas).on('mousemove click', (event) => {
            if(event.which == 1){
                this.ctx.beginPath();
                this.ctx.lineWidth = 0;
                this.ctx.arc(event.offsetX, event.offsetY, this.radius, 0, 2* Math.PI);
                this.ctx.fillStyle = 'black';
                this.ctx.fill();
                if(this.lastPos != null) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.lastPos.x, this.lastPos.y);
                    //console.log(this.lastPos);
                    this.ctx.lineTo(event.offsetX, event.offsetY);
                    this.ctx.lineWidth = 2 * this.radius;
                    this.ctx.stroke();
                }
                console.log('!');
                this.lastPos = {
                    x: event.offsetX,
                    y: event.offsetY
                };
            }
        });

        $(this.canvas).on('mousedown mouseup mouseleave', () => {
            this.lastPos = null;
            if (this.callback != undefined) {
                this.callback();
            }
        });

        $(this.canvas).on('mouseup mousedown', () => {
            this.undo.push(this.canvas.toDataURL());
        });
        this.clearCanvas();
    }

    setCallback(callback) {
        this.callback = callback;
    }

    clearCanvas() {
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.undo = [];
    }

    undoCanvas() {
      if (this.undo.length > 0) {
        let draw = () => this.ctx.drawImage(image,0,0,512,512);
        console.log(this.undo);
        //console.log(this.undo.pop());
        this.undo.pop();
        console.log(this.undo);
        var image=new Image();
        image.src=this.undo[this.undo.length-1];
        image.onload = draw;
      }
    }
  }
