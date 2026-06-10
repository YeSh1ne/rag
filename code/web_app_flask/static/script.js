const chatContainer = document.getElementById('chatContainer');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const status = document.getElementById('status');

sendBtn.addEventListener('click', sendMessage);

questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question) return;

    questionInput.disabled = true;
    sendBtn.disabled = true;

    addUserMessage(question);
    questionInput.value = '';

    // 立即创建助手消息占位符
    const assistantMsg = createAssistantMessage();
    status.className = 'status loading';
    status.textContent = '🔍 正在处理...';

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let steps = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'start') {
                            // 服务器已接收请求，立即显示处理中状态
                            assistantMsg.stepsContainer.style.display = 'block';
                        } else if (data.type === 'step') {
                            const existingIndex = steps.findIndex(s => s.label === data.label);
                            if (existingIndex >= 0) {
                                steps[existingIndex] = data;
                            } else {
                                steps.push(data);
                            }
                            updateSteps(assistantMsg, steps);
                        } else if (data.type === 'answer') {
                            updateAnswer(assistantMsg, data.content);
                        } else if (data.type === 'done') {
                            status.textContent = '';
                            status.className = 'status';
                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        }
                    } catch (e) {
                        // 忽略解析错误
                    }
                }
            }
        }
    } catch (error) {
        status.className = 'status error';
        status.textContent = `❌ 错误: ${error.message}`;
        setTimeout(() => {
            status.textContent = '';
            status.className = 'status';
        }, 5000);
    } finally {
        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.focus();
    }
}

function addUserMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = `<p>${escapeHtml(content)}</p>`;

    messageDiv.appendChild(bubble);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function createAssistantMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // 处理步骤容器
    const stepsContainer = document.createElement('div');
    stepsContainer.className = 'steps-container';
    stepsContainer.style.display = 'none';

    const stepsHeader = document.createElement('div');
    stepsHeader.className = 'steps-header';
    stepsHeader.innerHTML = '<span class="arrow">▼</span> <span class="steps-status">⏳ 正在处理...</span>';
    stepsHeader.onclick = () => toggleSteps(stepsHeader, stepsBody);

    const stepsBody = document.createElement('div');
    stepsBody.className = 'steps-body';

    stepsContainer.appendChild(stepsHeader);
    stepsContainer.appendChild(stepsBody);
    contentDiv.appendChild(stepsContainer);

    // 回答内容
    const answerDiv = document.createElement('div');
    answerDiv.className = 'answer-content';
    contentDiv.appendChild(answerDiv);

    bubble.appendChild(contentDiv);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(bubble);

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return { messageDiv, stepsContainer, stepsHeader, stepsBody, answerDiv };
}

function updateSteps(msg, steps) {
    msg.stepsContainer.style.display = 'block';
    msg.stepsBody.innerHTML = '';

    steps.forEach(step => {
        const stepItem = document.createElement('div');
        stepItem.className = 'step-item';
        stepItem.innerHTML = `
            <span class="step-icon">${step.icon}</span>
            <span class="step-label">${step.label}</span>
            ${step.detail ? `<span class="step-detail">${step.detail}</span>` : ''}
            ${step.time ? `<span class="step-time">${step.time}</span>` : ''}
        `;
        msg.stepsBody.appendChild(stepItem);
    });

    // 更新状态文本
    const statusSpan = msg.stepsHeader.querySelector('.steps-status');
    const allDone = steps.every(s => s.time);
    if (allDone) {
        statusSpan.textContent = '✅ 回答生成完成';
    } else {
        statusSpan.textContent = '⏳ 正在处理...';
    }

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateAnswer(msg, content) {
    msg.answerDiv.innerHTML = renderMarkdown(content);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function toggleSteps(header, body) {
    header.classList.toggle('collapsed');
    body.classList.toggle('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderMarkdown(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
        .replace(/---/g, '<hr>');
}