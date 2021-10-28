import DrawCanvas from './DrawCanvas';
import DisplayCanvas from './DisplayCanvas';

const {ipcRenderer} = window.require('electron');

let canvas = new DrawCanvas();
let displayCanvas = new DisplayCanvas();

let imgName = '';

function getCombinedImage() {
    let combined = document.createElement('canvas');
    combined.width = 2 * canvas.canvas.width;
    combined.height = canvas.canvas.height;
    let ctx = combined.getContext('2d');
    ctx.drawImage(displayCanvas.canvas, 0, 0);
    ctx.drawImage(canvas.canvas, canvas.canvas.width, 0);
    return combined.toDataURL('image/jpeg').split('base64,')[1];
}

ipcRenderer.on('image-response', (event, img, name) => {
    let src = `data:image/jpg;base64,${img}`;
    imgName = name
    displayCanvas.setImage(src);
    canvas.clearCanvas();
});

ipcRenderer.on('save-response', (event) => {
    ipcRenderer.send('fetch-image')
});

$('#submit-button').on('click', () => {
    let imgData = getCombinedImage();
    ipcRenderer.send('save-image', imgData, imgName);
});

$('#clear-button').on('click', () => {
    canvas.clearCanvas();
});

window.onload = () => {
    console.log('fetching image');
    ipcRenderer.send('fetch-image');
}

