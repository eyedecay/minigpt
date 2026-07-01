import { useState } from "react"

export default function InputBox( { onSend }) {
    const [input, setInput] = useState("")

    function handleSend() {
        if (!input.trim()) return

        onSend(input)
        setInput("")
    }
    return (
        <div>
            <input 
                value = {input} 
                onChange = {(e) => setInput(e.target.value)} onKeyDown = {(e) => {
                    if (e.key === "Enter") onSend()
            }} placeholder = "Message."
            />
            <button onClick = {handleSend} > Send </button>
        </div>
    )
}