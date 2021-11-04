const {app, BrowserWindow, ipcMain} = require('electron');
const path = require('path');
const fs = require('fs/promises');

const IMAGES_FOLDER = '../images/raw/images/Michelangelo';
const GENERATED_FOLDER = '../images/generated/';
const TRAIN_FOLDER = '../images/generated/train';
const TEST_FOLDER = '../images/generated/test';
const SKIP_FOLDER = '../images/generated/skip';

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

async function readDirOrEmpty(dir) {
    let stats;
    let exists = true;
    try {
        stats = await fs.stat(dir);
    } catch (e) {
        exists = false;
    }
        
    if (exists) {
        return await fs.readdir(dir);
    } else {
        return [];
    }
}

async function getAllGenerated() {
    let generated = await readDirOrEmpty(TRAIN_FOLDER);
    let test = await readDirOrEmpty(TEST_FOLDER);
    let skip = await readDirOrEmpty(SKIP_FOLDER);
    return [generated, test, skip].flat();
}

async function getRandomImage(event) {
    let files = await fs.readdir(IMAGES_FOLDER);
    let generated = await getAllGenerated();
    generated = new Set(generated);
    files = files.filter(file => !generated.has(file));
    let image = await fs.readFile(`${IMAGES_FOLDER}/${files[0]}`);
    image = image.toString('base64');
    
    event.reply('image-response', image, files[0]);
}

async function writeImage(event, image, saveName, folder='train') {
    let imageData = new Buffer(image, 'base64');
    await fs.writeFile(`${GENERATED_FOLDER}/${folder}/${saveName}`, imageData);
    event.reply('save-response');
}

ipcMain.on('fetch-image', (event) => {
    getRandomImage(event);
});

ipcMain.on('save-image', (event, image, saveName) => {
    writeImage(event, image, saveName);
});
