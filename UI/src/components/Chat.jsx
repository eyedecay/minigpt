import { useState } from "react"
import InputBox from "./InputBox"

export default function Chat() {
    const [messages, setMessages] = useState([])

    async function sendMessage(text) {
        if (!text.trim()) return

        //sets the user message
        setMessages(prev => [
            ...prev,
            { role: "user", text}
        ])

        const response = await fetch("http://127.0.0.1:8000/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            

            },
            body: JSON.stringify({
                prompt: text
            })
        })
        const data = await response.json()

        setMessages(prev => [
            ...prev, {role: "assistant", text: data.response}
        ])
    }

    return (
        <div>
            <h2> the chat component</h2>
            <div>
                {messages.map((message, i) => (
                    <div key = {i}>
                        <b>{message.role}: </b> {message.text}
                    </div>
                ))}
            </div>
            <InputBox onSend = {sendMessage}/>
            
        </div>
    )
}