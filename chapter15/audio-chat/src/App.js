import './App.css';
import React, { useState } from 'react';
import '@chatui/core/es/styles/index.less';
import './chatui-theme.css';
import Chat, { Bubble, useMessages } from '@chatui/core';
import '@chatui/core/dist/index.css';
import OpenAI from 'openai';
import localStorage from "localStorage";
import Recorder from 'js-audio-recorder';

const apiurl = "http://server-llm-dev:8000"
const openai = new OpenAI({
  apiKey: '0000',
  dangerouslyAllowBrowser: true, baseURL: apiurl + "/v1"
});
var message_history = [];
const recorder = new Recorder();
const defaultQuickReplies = [
  {
    icon: 'message',
    name: '文字',
    isNew: false,
    isHighlight: true,
  },
  {
    icon: 'file',
    name: '语音',
    isNew: true,
    isHighlight: true,
  }
];

function App() {
  const { messages, appendMsg, setTyping, updateMsg } = useMessages([]);
  var _inputType = localStorage.getItem("inputType");
  if (_inputType == null) {
    _inputType = "text";
  }
  const [inputType] = useState(_inputType);

  async function chat_stream(prompt, _msgId) {
    message_history.push({ role: 'user', content: prompt });
    const stream = openai.beta.chat.completions.stream({
      model: 'ChatGLM3-6B',
      messages: message_history,
      stream: true,
    });
    var snapshot = "";
    for await (const chunk of stream) {
      if (chunk.choices[0]?.delta?.content === undefined) {
        continue;
      }
      snapshot = snapshot + chunk.choices[0]?.delta?.content || '';
      updateMsg(_msgId, {
        type: "text",
        content: { text: snapshot.trim() }
      });
    }
    message_history.push({ "role": "assistant", "content": snapshot });
  }

  function handleSend(type, val) {
    var prompt = "";
    if (type === 'text' && val.trim()) {
      appendMsg({
        type: 'text',
        content: { text: val },
        position: 'right',
      });
      prompt = val.trim();
    } else if (type === "voice") {
      appendMsg({
        type: 'text',
        content: { text: val.payTime + "秒语音" },
        position: 'right',
      });
      prompt = "voice:" + val.clientid;
    }
    const msgID = new Date().getTime();
    setTyping(true);
    appendMsg({
      _id: msgID,
      type: 'text',
      content: { text: '' },
    });
    chat_stream(prompt, msgID);
  }

  function renderMessageContent(msg) {
    const { content } = msg;
    return <Bubble content={content.text} />;
  }

  function handleQuickReplyClick(item) {
    if (item.name.startsWith("文字")) {
      localStorage.setItem("inputType", "text")
    } else {
      localStorage.setItem("inputType", "voice")
    }
    window.location.reload();
  }

  function onVoiceStart() {
    recorder.start();
  }

  function onVoiceEnd() {
    recorder.stop();
    uploadVoice();
  }

  function uploadVoice() {
    var fileId = new Date().getTime() + '.wav';
    var payTime = Math.round(recorder.duration);
    var wavBlob = recorder.getWAVBlob();
    var formData = new FormData();
    const newbolb = new Blob([wavBlob], { type: 'audio/wav' });
    const fileOfBlob = new File([newbolb], fileId);
    formData.append('file', fileOfBlob);
    formData.append('clientid', fileId);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', apiurl + '/upload', true);
    xhr.onload = function () {
      if (xhr.status === 200) {
        console.log('File uploaded successfully');
        handleSend("voice", { "clientid": fileId, "payTime": payTime });
      } else {
        console.error('File upload failed');
      }
    };
    xhr.send(formData);
  }

  return (
    <Chat
      navbar={{ title: 'voice-chat' }}
      messages={messages}
      inputType={inputType}
      quickReplies={defaultQuickReplies}
      onQuickReplyClick={handleQuickReplyClick}
      recorder={{ canRecord: true, onStart: onVoiceStart, onEnd: onVoiceEnd }}
      renderMessageContent={renderMessageContent}
      onSend={handleSend}
    />
  );
}

export default App;
