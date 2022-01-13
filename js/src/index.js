import DrawCanvas from './DrawCanvas';
import DisplayCanvas from './DisplayCanvas';

console.log('wtf update much?');

function isElectron() {
    // Renderer process
    if (typeof window !== 'undefined' && typeof window.process === 'object' && window.process.type === 'renderer') {
        return true;
    }

    // Main process
    if (typeof process !== 'undefined' && typeof process.versions === 'object' && !!process.versions.electron) {
        return true;
    }

    // Detect the user agent when the `nodeIntegration` option is set to true
    if (typeof navigator === 'object' && typeof navigator.userAgent === 'string' && navigator.userAgent.indexOf('Electron') >= 0) {
        return true;
    }

    return false;
}

if (isElectron()) {
    const {ipcRenderer} = window.require('electron');
}

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

function setImage(img, name) {
    let src = `data:image/jpg;base64,${img}`;
    imgName = name
    displayCanvas.setImage(src);
    canvas.clearCanvas();
}

if (isElectron()) {
    ipcRenderer.on('image-response', (event, img, name) => {
        setImage(img, name);
    });

    ipcRenderer.on('save-response', (event) => {
        ipcRenderer.send('fetch-image')
    });
}

function fetchImage() {
    console.log('fetching image');
    if (isElectron()) {
        ipcRenderer.send('fetch-image');
    } else {
        $.get('webscripts/get_image.py', (data) => {
            console.log(data);
            data = JSON.parse(data);
            console.log(data);
            setImage(data['img'], data['name']);
        });
    }
}

$('#submit-button').on('click', () => {
    let imgData = getCombinedImage();
    if (isElectron()) {
        ipcRenderer.send('save-image', imgData, imgName);
    } else {
        window.imgData = imgData;
        $.post('webscripts/save_image.py',
            {img: imgData, name: imgName},
            () => {
                fetchImage();
            }
        );
        
    }
});

$('#skip-button').on('click', () => {
    let imgData = getCombinedImage();
    ipcRenderer.send('save-image', imgData, imgName, 'skip');
});

$('#clear-button').on('click', () => {
    canvas.clearCanvas();
});

window.onload = () => {
    fetchImage();
}

