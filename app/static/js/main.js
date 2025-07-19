document.addEventListener('DOMContentLoaded', () => {
    // 获取页面上的所有元素
    const chatbox = document.getElementById('agent-chatbox');
    const startBtn = document.getElementById('start-btn');
    const configForm = document.getElementById('config-form');

    if (startBtn) {
        startBtn.addEventListener('click', startAgentDialog);
    }

    async function startAgentDialog() {
        startBtn.disabled = true;
        startBtn.textContent = '对话进行中...';
        chatbox.innerHTML = '';

        const formData = new FormData(configForm);
        const config = {
            llm_model: formData.get('llm_model'),
            asker: {
                name: formData.get('asker_name'),
                system_message: formData.get('asker_system_message')
            },
            respondent: {
                name: formData.get('respondent_name'),
                system_message: formData.get('respondent_system_message'),
                tools: formData.getAll('tools')
            },
            max_turns: formData.get('max_turns'),
            initial_prompt: formData.get('initial_prompt')
        };

        addMessage('系统',
            `对话开始，使用模型: ${config.llm_model}, 最大轮数: ${config.max_turns}`,
            'system');
        addMessage('你 (指令)', config.initial_prompt, 'user');

        try {
            const response = await fetch('api/start_dialog', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) throw new Error(`服务器错误：${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    addMessage('系统', '对话已结束', 'system');
                    break;
                }

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n\n').filter(line =>
                    line.trim().startsWith('data: '));

                for (const line of lines) {
                    const jsonData = line.substring(5).trim()
                    try {
                        const data = JSON.parse(jsonData);
                        if (data.event_type && data.event_type === 'END') continue;
                        let senderClass = data.sender_type || 'agent';
                        addMessage(data.sender_name, data.content, senderClass);
                    } catch (e) {
                        console.error('解析JSON失败', jsonData, e);
                    }
                }
            }

        } catch (error) {
            addMessage('系统', `发生错误: ${error.message}`, 'system')
        } finally {
            startBtn.disabled = false;
            startBtn.textContent = '开始对话';
        }

    }

    function addMessage(sender, content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        const contentContainer = document.createElement('div');
        contentContainer.innerHTML = marked.parse(content || '', { trusted: true });

        messageDiv.innerHTML = `<strong>${sender}</strong>`;
        messageDiv.appendChild(contentContainer);

        chatbox.appendChild(messageDiv);

        renderMathInElement(contentContainer, {
            delimiters: [
                {left: '$$', right: '$$', display: true},  // 块级公式
                {left: '$', right: '$', display: false},   // 行内公式
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ]
        });

        chatbox.scrollTop = chatbox.scrollHeight;
    }
    
})