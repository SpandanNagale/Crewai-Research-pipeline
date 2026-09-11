document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('research-form');
    const topicInput = document.getElementById('topic');
    const evaluateInput = document.getElementById('evaluate');
    const btnText = document.querySelector('.btn-text');
    const loaderIcon = document.querySelector('.loader-icon');
    const startBtn = document.getElementById('start-btn');
    const consoleOutput = document.getElementById('console-output');
    const systemStatus = document.getElementById('system-status');
    const agentItems = document.querySelectorAll('.agent-item');
    
    // Modal elements
    const modalOverlay = document.getElementById('report-modal');
    const closeModal = document.getElementById('close-modal');
    const finalReportContent = document.getElementById('final-report-content');
    const finalEvalContent = document.getElementById('final-eval-content');
    const evalScoresPre = document.getElementById('eval-scores-pre');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const topic = topicInput.value.trim();
        const evaluate = evaluateInput.checked;
        
        if (!topic) return;

        // UI Updates for starting
        startBtn.disabled = true;
        btnText.textContent = "Agents Deployed";
        loaderIcon.classList.remove('hidden');
        systemStatus.textContent = "Pipeline Running";
        systemStatus.classList.add('running');
        consoleOutput.innerHTML = ''; // Clear console
        
        appendLog(`[SYSTEM] Initializing task: "${topic}"...`, 'system-msg');
        
        try {
            const response = await fetch('/research/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ topic, evaluate })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            // Named SSE frames ("event: <type>\ndata: <json>\n\n"), not the old
            // sentinel-string protocol -- see app.py::stream_research.
            let buffer = '';
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                let sepIndex;
                while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
                    const rawFrame = buffer.slice(0, sepIndex);
                    buffer = buffer.slice(sepIndex + 2);
                    handleSSEFrame(rawFrame);
                }
            }

        } catch (error) {
            appendLog(`[SYSTEM ERROR] Connection failed: ${error}`, 'system-msg control red');
        } finally {
            // Revert UI updates
            startBtn.disabled = false;
            btnText.textContent = "Deploy Agents";
            loaderIcon.classList.add('hidden');
            systemStatus.textContent = "System Idle";
            systemStatus.classList.remove('running');
            resetAgentStates();
            appendLog(`[SYSTEM] Pipeline execution terminated.`, 'system-msg');
        }
    });

    closeModal.addEventListener('click', () => {
        modalOverlay.classList.add('hidden');
    });

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag])
        );
    }

    function appendLog(text, className) {
        if (!text.trim()) return;
        const line = document.createElement('div');
        line.className = `console-line ${className}`;
        line.innerHTML = text; // innerHTML: labels can carry simple markup
        consoleOutput.appendChild(line);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    function handleSSEFrame(rawFrame) {
        let eventType = 'message';
        const dataLines = [];
        for (const line of rawFrame.split('\n')) {
            if (line.startsWith('event: ')) eventType = line.slice(7).trim();
            else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
            // lines starting with ':' are heartbeat comments -- ignore
        }
        if (dataLines.length === 0) return;

        let payload;
        try {
            payload = JSON.parse(dataLines.join('\n'));
        } catch (e) {
            return;
        }

        if (eventType === 'stage_update') {
            renderStageUpdate(payload);
        } else if (eventType === 'error') {
            appendLog(`[SYSTEM ERROR] ${escapeHTML(payload.detail || 'Pipeline error')}`, 'system-msg control red');
        } else if (eventType === 'done') {
            if (payload.report) {
                finalReportContent.innerHTML = marked.parse(payload.report);
                modalOverlay.classList.remove('hidden');
            }
            if (payload.eval) {
                finalEvalContent.classList.remove('hidden');
                evalScoresPre.textContent = JSON.stringify(payload.eval, null, 2);
            }
        }
    }

    function renderStageUpdate(event) {
        const stage = event.stage || '';
        if (stage.includes('planner')) setActiveAgent('agent-planner');
        else if (stage.includes('researcher')) setActiveAgent('agent-researcher');
        else if (stage.includes('writer')) setActiveAgent('agent-writer');
        else if (stage.includes('critic')) setActiveAgent('agent-critic');

        let className = 'log-thought';
        if (event.event_type === 'stage_failed') className = 'system-msg control red';
        else if (event.event_type === 'stage_started' || event.event_type === 'stage_finished') className = 'system-msg';

        const label = `[${stage || 'system'}] ${event.event_type}` + (event.message ? `: ${event.message}` : '');
        appendLog(escapeHTML(label), className);
    }

    function setActiveAgent(agentId) {
        agentItems.forEach(el => el.classList.remove('active'));
        const active = document.getElementById(agentId);
        if (active) active.classList.add('active');
    }

    function resetAgentStates() {
        agentItems.forEach(el => el.classList.remove('active'));
    }
});
