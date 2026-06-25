const { app, BrowserWindow } = require("electron")

function createWindow() {
    const win = new BrowserWindow({width: 300, height: 300});
    win.loadFile('index.html')
}

app.whenReady().then(createWindow)

