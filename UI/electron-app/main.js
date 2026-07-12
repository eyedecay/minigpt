import { app, BrowserWindow } from "electron"
import { spawn } from "child_process"
import path from "path"
import fs from "fs"
import { fileURLToPath } from "url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const MODEL_URL = "https://github.com/eyedecay/horizons/releases/download/v1.0.0/finetuned-1.pth"

let backendProcess

function getBackendDir() {
  return app.isPackaged
    ? path.join(process.resourcesPath, "backend")
    : path.join(__dirname, "..", "..", "backend")
}

function getModelDir() {
  return app.getPath("userData")
}

function getModelPath() {
  return path.join(getModelDir(), "finetuned-1.pth")
}

function startBackend() {
  const cwd = getModelDir()
  const binaryPath = app.isPackaged
    ? path.join(getBackendDir(), "minigpt-backend")
    : path.join(getBackendDir(), "dist", "minigpt-backend")

  backendProcess = spawn(binaryPath, [], { cwd })
  backendProcess.stdout.on("data", (d) => console.log(`[backend] ${d}`))
  backendProcess.stderr.on("data", (d) => console.error(`[backend] ${d}`))
  backendProcess.on("exit", (code) => console.log(`[backend] exited with code ${code}`))
}

function createWindow() {
  const win = new BrowserWindow({ width: 1000, height: 1000 })

  if (app.isPackaged) {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"))
  } else {
    win.loadURL("http://localhost:5173")
  }
}

async function ensureModel() {
  console.log(`[main] model path: ${getModelPath()}`)
  console.log(`[main] model exists: ${fs.existsSync(getModelPath())}`)
  if (fs.existsSync(getModelPath())) return

  const win = new BrowserWindow({ width: 400, height: 120, autoHideMenuBar: true, title: "Downloading model..." })
  win.loadURL("data:text/html,<h3 style='font-family:sans-serif;text-align:center;padding:20px'>Downloading model (1.9 GB)...</h3>")

  fs.mkdirSync(getModelDir(), { recursive: true })

  console.log(`[main] downloading from ${MODEL_URL}`)
  const response = await fetch(MODEL_URL)
  if (!response.ok) throw new Error(`download failed: ${response.status} ${response.statusText}`)

  const total = parseInt(response.headers.get("content-length"), 10)
  let downloaded = 0

  const reader = response.body.getReader()
  const writer = fs.createWriteStream(getModelPath())

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    writer.write(Buffer.from(value))
    downloaded += value.length
    const pct = total ? Math.round((downloaded / total) * 100) : 0
    win.setTitle(`Downloading model... ${pct}%`)
  }

  writer.end()
  win.close()
}

app.whenReady().then(async () => {
  try {
    await ensureModel()
  } catch (e) {
    console.error(`[main] download error: ${e.message}`)
  }
  startBackend()
  createWindow()
})

app.on("will-quit", () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill()
  }
})
