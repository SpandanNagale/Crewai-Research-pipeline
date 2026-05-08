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
            
            let isFinalReportPhase = false;
            let reportMarkdown = [];
            let evalMarkdown = [];
            let isEvalPhase = false;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (let line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        
                        if (data === '[DONE]') {
                            break;
                        }

                        if (data === '--- FINAL REPORT ---') {
                            isFinalReportPhase = true;
                            appendLog(`[SYSTEM] Research complete. Generating report...`, 'system-msg');
                            continue;
                        }

                        if (data === '--- EVALUATING REPORT ---') {
                            isEvalPhase = true;
                            appendLog(`[SYSTEM] Starting critical evaluation of report...`, 'system-msg');
                            continue;
                        }

                        if (isFinalReportPhase && !isEvalPhase) {
                            reportMarkdown.push(data);
                        } else if (isEvalPhase) {
                            evalMarkdown.push(data);
                        } else {
                            // Console stream during execution
                            processConsoleLog(data);
                        }
                    }
                }
            }
            
            // Render final report
            if (reportMarkdown.length > 0) {
                const mdContent = reportMarkdown.join('\n');
                finalReportContent.innerHTML = marked.parse(mdContent);
                modalOverlay.classList.remove('hidden');
            }

            if (isEvalPhase && evalMarkdown.length > 0) {
                finalEvalContent.classList.remove('hidden');
                evalScoresPre.textContent = evalMarkdown.join('\n').trim();
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
        line.innerHTML = text; // Used innerHTML because some processConsoleLog returns bold tags
        consoleOutput.appendChild(line);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    function processConsoleLog(data) {
        if (!data.trim()) return;
        
        let formattedStr = escapeHTML(data);
        let className = 'log-thought';
        
        // CrewAI Specific Parsing
        const lowerData = data.toLowerCase();
        
        // Detect which agent is acting to update UI highlighting
        if (lowerData.includes('planner:')) { setActiveAgent('agent-planner'); className = 'log-planner log-header'; }
        else if (lowerData.includes('researcher:')) { setActiveAgent('agent-researcher'); className = 'log-researcher log-header'; }
        else if (lowerData.includes('writer:')) { setActiveAgent('agent-writer'); className = 'log-writer log-header'; }
        else if (lowerData.includes('critic:')) { setActiveAgent('agent-critic'); className = 'log-critic log-header'; }
        else if (lowerData.includes('[0m')) {
            // Strip ANSI codes if any exist
            formattedStr = formattedStr.replace(/\[\d+m/g, "");
            // Heuristic coloring
            if (lowerData.includes('action:')) className = 'log-action';
            if (lowerData.includes('observation:')) className = 'log-observation';
        }

        appendLog(formattedStr, className);
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
