const {app, BrowserWindow, ipcMain} = require('electron');
const path = require('path');
const fs = require('fs/promises');

const IMAGES_FOLDER ='../images/images/images/Michelangelo';

function createWindow() {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });
    
    win.loadFile('dist/index.html');
}

app.whenReady().then(() => {
    createWindow();
    
    app.on('activate', ()=> {
        if(BrowserWindow.getAllWindows().length == 0) {
            createWindow()
        }
    });
});

app.on('window-all-closed', () => {
    if(process.platform !== 'darwin'){
        app.quit();
    }
});

async function getRandomImage(event){
    let files = await fs.readdir(IMAGES_FOLDER);
    let image = await fs.readFile(`${IMAGES_FOLDER}/${files[0]}`);
    image = image.toString('base64');
    
    event.reply('image-response', image);
}

ipcMain.on('fetch-image', (event) => {
    console.log('fetching image');
    getRandomImage(event);
});
