import DrawCanvas from './DrawCanvas';
import DisplayCanvas from './DisplayCanvas';

const {ipcRenderer} = window.require('electron');

let canvas = new DrawCanvas();

let displayCanvas = new DisplayCanvas();

ipcRenderer.on('image-response', (event, arg) => {
    let src = `data:image/jpg;base64,${arg}`;
    displayCanvas.setImage(src);
});

console.log('fetching image');
ipcRenderer.send('fetch-image');

