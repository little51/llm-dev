import './App.css';
import React from 'react';
import '@chatui/core/es/styles/index.less';
import Chat, { Bubble, useMessages } from '@chatui/core';
import '@chatui/core/dist/index.css';
import OpenAI from 'openai';
import './chatui-theme.css';
const openai = new OpenAI({
  apiKey: 'EMPTY', dangerouslyAllowBrowser: true,
  baseURL: "http://server-llm-dev:8000/v1"
});
var message_history = [];
function App() {
  const { messages, appendMsg, setTyping, updateMsg } = useMessages([]);
  async function chat_stream(prompt, _msgId) {
    message_history.push({ role: 'user', content: prompt });
    const stream = openai.beta.chat.completions.stream({
      model: 'ChatGLM3-6B',
      messages: message_history,
      stream: true,
    });
    var snapshot = "";
    for await (const chunk of stream) {
      snapshot = snapshot + chunk.choices[0]?.delta?.content || '';
      updateMsg(_msgId, {
        type: "text",
        content: { text: snapshot.trim() }
      });
    }
    message_history.push({ "role": "assistant", "content": snapshot });
  }
  function handleSend(type, val) {
    if (type === 'text' && val.trim()) {
      appendMsg({
        type: 'text',
        content: { text: val },
        position: 'right',
      });
      const msgID = new Date().getTime();
      setTyping(true);
      appendMsg({
        _id: msgID,
        type: 'text',
        content: { text: '' },
      });
      chat_stream(val, msgID);
    }
  }
  function renderMessageContent(msg) {
    const { content } = msg;
    return <Bubble content={content.text} />;
  }
  return (
    <Chat
      navbar={{ title: 'chat-app' }}
      messages={messages}
      renderMessageContent={renderMessageContent}
      onSend={handleSend}
    />
  );
}
export default App;