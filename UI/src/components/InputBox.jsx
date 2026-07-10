import { useState } from "react"
import "./InputBox.css"

export default function InputBox( { onSend }) {
    const [input, setInput] = useState("")

    function handleSend() {
        if (!input.trim()) return

        onSend(input)
        setInput("")
    }
    return (
        <div className = "input-container">
            <textarea 
                value = {input} 
                onChange = {(e) => setInput(e.target.value)} onKeyDown = {(e) => {
                    if (e.key === "Enter") { handleSend()
                }}} placeholder = "Ask Anything..."
            />
            <button onClick = {handleSend} > Send </button>
        </div>
    )
}