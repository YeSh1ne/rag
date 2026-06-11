const chatContainer = document.getElementById('chatContainer');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const status = document.getElementById('status');
const historyList = document.getElementById('historyList');

// 历史记录存储 - 内存存储，重启后清空
let messages = [];
let sidebarCollapsed = false;

sendBtn.addEventListener('click', sendMessage);

questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// 初始化
if (sidebarCollapsed) {
    document.getElementById('sidebar').classList.add('collapsed');
}
renderHistoryList();

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('collapsed');
    sidebarCollapsed = sidebar.classList.contains('collapsed');
    
    const btn = sidebar.querySelector('.toggle-sidebar-btn');
    btn.textContent = sidebarCollapsed ? '▶' : '◀';
}

function renderHistoryList() {
    historyList.innerHTML = '';
    
    // 按提问顺序从上到下排列
    messages.forEach((msg, index) => {
        if (msg.role === 'user') {
            const item = document.createElement('div');
            item.className = 'history-item';
            item.onclick = () => scrollToMessage(index);
            
            const question = document.createElement('div');
            question.className = 'question';
            question.textContent = msg.content;
            
            const time = document.createElement('div');
            time.className = 'time';
            time.textContent = formatTime(msg.timestamp);
            
            item.appendChild(question);
            item.appendChild(time);
            historyList.appendChild(item);
        }
    });
}

function scrollToMessage(index) {
    const msgEl = document.getElementById(`msg-${index}`);
    if (msgEl) {
        msgEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // 高亮效果
        msgEl.style.transition = 'background 0.3s';
        msgEl.style.background = 'rgba(0, 101, 255, 0.1)';
        setTimeout(() => {
            msgEl.style.background = '';
        }, 1500);
    }
}

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question) return;

    questionInput.disabled = true;
    sendBtn.disabled = true;

    // 保存用户消息到数组
    const userMsgIndex = messages.length;
    messages.push({
        role: 'user',
        content: question,
        timestamp: new Date().toISOString()
    });
    
    addUserMessage(question);
    questionInput.value = '';

    // 立即创建助手消息占位符
    const assistantMsg = createAssistantMessage(userMsgIndex);
    status.className = 'status loading';
    status.textContent = '🔍 正在处理...';

    const conversation = {
        answer: '',
        steps: []
    };

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
                            assistantMsg.stepsContainer.style.display = 'block';
                            assistantMsg.stepsHeader.classList.remove('collapsed');
                            assistantMsg.stepsBody.classList.remove('hidden');
                        } else if (data.type === 'step') {
                            const existingIndex = steps.findIndex(s => s.label === data.label);
                            if (existingIndex >= 0) {
                                steps[existingIndex] = data;
                            } else {
                                steps.push(data);
                            }
                            conversation.steps = [...steps];
                            updateSteps(assistantMsg, steps, false);
                        } else if (data.type === 'answer') {
                            conversation.answer = data.content;
                            updateAnswer(assistantMsg, data.content, data.sources || []);
                        } else if (data.type === 'done') {
                            updateSteps(assistantMsg, steps, true);
                            status.textContent = '';
                            status.className = 'status';
                            // 保存助手消息到数组
                            messages.push({
                                role: 'assistant',
                                ...conversation,
                                timestamp: new Date().toISOString()
                            });
                            localStorage.removeItem('chatMessages');
                            localStorage.removeItem('sidebarCollapsed');
                            renderHistoryList();
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

function addUserMessage(content, scroll = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.id = `msg-${messages.length - 1}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = `<p>${escapeHtml(content)}</p>`;

    messageDiv.appendChild(bubble);
    chatContainer.appendChild(messageDiv);
    
    if (scroll) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

function formatTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`;
    
    return date.toLocaleDateString('zh-CN', { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function createAssistantMessage(userMsgIndex) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = `msg-${userMsgIndex + 1}`;

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

function updateSteps(msg, steps, isDone = false) {
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

    // 更新状态文本和展开/收起状态
    const statusSpan = msg.stepsHeader.querySelector('.steps-status');
    if (isDone) {
        statusSpan.textContent = '📋 处理详情';
        // 处理完成后收起
        msg.stepsHeader.classList.add('collapsed');
        msg.stepsBody.classList.add('hidden');
    } else {
        statusSpan.textContent = '⏳ 正在处理...';
        // 处理中保持展开
        msg.stepsHeader.classList.remove('collapsed');
        msg.stepsBody.classList.remove('hidden');
    }

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateAnswer(msg, content, sources = []) {
    msg.answerDiv.innerHTML = renderMarkdown(content);
    
    // 处理引用链接
    if (sources && sources.length > 0) {
        const html = msg.answerDiv.innerHTML;
        
        // 将引用文本替换为可点击链接
        let newHtml = html;
        sources.forEach(source => {
            const citationText = source.full_citation;
            if (source.pdf_url) {
                const linkHtml = `来自: <a href="${source.pdf_url}" target="_blank" class="citation-link">[${citationText}]</a>`;
                newHtml = newHtml.replace(
                    `来自: [${citationText}]`,
                    linkHtml
                );
            }
        });
        
        msg.answerDiv.innerHTML = newHtml;
    }
    
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