export default class DisplayCanvas {
    constructor(canvas) {
        this.canvas = canvas || $('#display-canvas')[0];
        window.ca = this.canvas;
        this.ctx = this.canvas.getContext('2d');
    }
    
    setImage(image_dat) {
        console.log('setting image');
        let img = new Image();
        img.onload = () => {
            let canvasRatio = this.canvas.width/this.canvas.height;
            let imgRatio = img.width/img.height;
            let offsetX = imgRatio > canvasRatio ?
                img.height * (imgRatio - canvasRatio) / 2 : 0;
            let offsetY = imgRatio < canvasRatio ?
                img.width * (imgRatio - canvasRatio) / 2 : 0;
            console.log(offsetX, offsetY,
                img.width - (2 * offsetX),
                img.height - (2 * offsetY),
                0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(
                img, offsetX, offsetY,
                img.width - (2 * offsetX),
                img.height - (2 * offsetY),
                0, 0, this.canvas.width, this.canvas.height
            );
        };
        img.src = image_dat;
    }
}
