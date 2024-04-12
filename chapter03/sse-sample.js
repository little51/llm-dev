async function chat_stream(prompt) {
  message_history.push({ role: 'user', content: prompt });
  const stream = openai.beta.chat.completions.stream({
    model: '模型名称',
    messages: message_history,
    stream: true,  //流方式（服务器使用SSE响应）
  });
  var snapshot = "";
  for await (const chunk of stream) {
    //content是SSE推送的增量内容
    var content = chunk.choices[0]?.delta?.content;
    snapshot = snapshot + content;
    //更新界面显示
  }
  message_history.push({ "role": "assistant", "content": snapshot });
}