import * as vscode from 'vscode';
import OpenAI from 'openai';

const openai = new OpenAI({
	apiKey: 'EMPTY', dangerouslyAllowBrowser: true,
	baseURL: "http://server-llm-dev:8000/v1"
});
let cache: { [key: string]: string } = {};

export function activate(context: vscode.ExtensionContext) {
	let statusBarItem = vscode.window.createStatusBarItem
		(vscode.StatusBarAlignment.Right, 100);
	statusBarItem.text = `$(chip) stable-coder`;
	statusBarItem.show();
	context.subscriptions.push(statusBarItem);

	const provider = new PromptProvider(statusBarItem, context);
	let disposable = vscode.languages.
		registerInlineCompletionItemProvider
		({ pattern: '**', }, provider);
	context.subscriptions.push(disposable);
}

export function deactivate() { }

export class PromptProvider implements
	vscode.InlineCompletionItemProvider {

	statusbar: vscode.StatusBarItem;
	context: vscode.ExtensionContext;

	constructor(statusbar: vscode.StatusBarItem,
		context: vscode.ExtensionContext) {
		this.statusbar = statusbar;
		this.context = context;
		console.log("stable-coder init");
	}

	async provideInlineCompletionItems(
		document: vscode.TextDocument,
		position: vscode.Position,
		context: vscode.InlineCompletionContext,
		token: vscode.CancellationToken):
		Promise<vscode.InlineCompletionItem[] |
			vscode.InlineCompletionList | undefined | null> {
		// 获取当前行文本		
		const textBeforeCursor = document.getText(
			new vscode.Range(position.with(undefined, 0), position)
		);
		// 只有在编辑器中输入冒号时，才获取补全信息
		if (!textBeforeCursor.endsWith(":")) {
			return;
		}
		const items = new Array<vscode.InlineCompletionItem>();
		var llm_result = "";
		// 判断是否命中缓存，未命中则调大模型
		if (textBeforeCursor in cache) {
			llm_result = cache[textBeforeCursor];
		} else {
			llm_result = (await chat_stream(textBeforeCursor))
				.replace(textBeforeCursor, "");
			cache[textBeforeCursor] = llm_result;
		}
		items.push({ insertText: llm_result });
		return items;
	}
}

async function chat_stream(prompt: string) {
	const stream = openai.beta.chat.completions.stream({
		model: 'stable-code-3b',
		messages: [{ role: 'user', content: prompt }],
		stream: true,
	});
	var snapshot = "";
	for await (const chunk of stream) {
		snapshot = snapshot + chunk.choices[0]?.delta?.content || '';
	}
	return snapshot;
}
