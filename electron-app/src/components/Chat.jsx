import { useState } from "react"
import InputBox from "./InputBox"

export default function Chat() {
    const [messages, setMessages] = useState([])

    function sendMessage(text) {
        if (!text.trim()) return
        setMessages(prev => [
            ...prev,
            { role: "user", text}
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