import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: 'EMPTY', dangerouslyAllowBrowser: true,
    baseURL: "http://server-llm-dev:8000/v1"
});
/*var conversationHistory = {
    "npc_01": [
        { role: 'user', content: '当在小镇遇到熟人，聊天气，随机写一个开始话题' },
        { role: "assistant", content: '天气不错，准备去哪？' },
        { role: 'user', content: '当在小镇遇到熟人，聊天气，熟人说：" 天气不错，准备去哪？"，随机写一个回答' },
        { role: "assistant", content: '去健身啊！' },
        { role: 'user', content: '当在小镇遇到熟人，聊天气，熟人说："去健身啊！"，随机写一个回答' },
        { role: "assistant", content: '好的，再见' }
    ]
};*/
var conversationHistory = {};
const startWords = "当在小镇遇到熟人，聊{topic}，随机写一个开始话题";
const chatTemplate = '当在小镇遇到熟人，聊{topic}，熟人说：" {prevanswer}"，随机写一个回答'

export const initConversation = (characterName, topic) => {
    if (!conversationHistory[characterName]) {
        conversationHistory[characterName] = [];
        const prompt = startWords.replace('{topic}', topic);
        conversationHistory[characterName].push(
            { role: 'user', content: prompt });
    }
}

export const addConversation = (characterName, prompts) => {
    if (!conversationHistory[characterName]) {
        conversationHistory[characterName] = [];
    }
    conversationHistory[characterName].push(prompts);
}

export const getChatHistory = () => {
    var history = "";
    Object.keys(conversationHistory).forEach(characterName => {
        var npc = false;
        for (let item of conversationHistory[characterName]) {
            if (item.role === "assistant") {
                if (npc) {
                    history = `${history}\n${characterName}:${item.content}`
                } else {
                    history = `${history}\nyou:${item.content}`
                }
                npc = !npc;
            }
        }
    });
    return history;
}

export const callGpt = async (characterName, topic, i) => {
    initConversation(characterName, topic);
    const stream = openai.beta.chat.completions.stream({
        model: 'ChatGLM3-6B',
        messages: conversationHistory[characterName],
        stream: true,
    });
    var snapshot = "";
    var text = "";
    for await (const chunk of stream) {
        snapshot = snapshot + chunk.choices[0]?.delta?.content || '';
        if (i % 2 === 0) {
            text = `<span style='color:yellow'>you:${snapshot}</span>`;
        } else {
            text = `<span style='color:red'>
            ${characterName}:${snapshot}</span>`;
        }
        window.dispatchEvent(new CustomEvent('show-dialog', {
            detail: {
                "characterName": characterName,
                "message": text
            },
        }));
    }
    addConversation(characterName,
        { "role": "assistant", "content": snapshot });
    const prompt = chatTemplate.replace('{topic}', topic)
        .replace('{prevanswer}', snapshot);
    conversationHistory[characterName].push(
        { role: 'user', content: prompt });
}
