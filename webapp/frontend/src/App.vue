<template>
  <div class="chat-container">
    <div class="sidebar">
      <div class="sidebar-header"><h2>RAG Chat</h2></div>
      <button class="new-chat" @click="clearChat">+ New Chat</button>
    </div>

    <div class="main">
      <div class="messages" ref="messagesContainer">
        <div v-if="messages.length === 0" class="welcome">
          <h1>What can I help you find?</h1>
          <p>Ask questions about your technical documentation</p>
        </div>

        <div v-for="(msg, i) in messages" :key="i" :class="['message', msg.role]">
          <div class="message-content">
            <div class="avatar">{{ msg.role === 'user' ? 'You' : 'AI' }}</div>
            <div class="text">
              <vue-markdown :source="msg.content" />
              <div v-if="msg.sources && msg.sources.length" class="sources">
                <details>
                  <summary>{{ msg.sources.length }} sources</summary>
                  <ul>
                    <li v-for="(s, j) in msg.sources" :key="j">
                      <span class="source-file">{{ s.file }}</span>
                      <span class="source-meta">Page {{ s.page }} · {{ s.score }}</span>
                    </li>
                  </ul>
                </details>
              </div>
            </div>
          </div>
        </div>

        <div v-if="loading" class="message assistant">
          <div class="message-content">
            <div class="avatar">AI</div>
            <div class="text"><div class="thinking">Thinking...</div></div>
          </div>
        </div>
      </div>

      <div class="input-area">
        <div class="input-wrapper">
          <textarea v-model="question" @keydown.enter.exact.prevent="submitQuery"
            placeholder="Message RAG Chat..." rows="1"></textarea>
          <button @click="submitQuery" :disabled="loading || !question.trim()">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path fill="currentColor" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import VueMarkdown from 'vue-markdown-render'

const API_URL = `${window.location.protocol}//${window.location.hostname}/api`
const question = ref('')
const messages = ref([])
const loading = ref(false)
const messagesContainer = ref(null)

async function submitQuery() {
  if (!question.value.trim() || loading.value) return
  const userQuestion = question.value
  messages.value.push({ role: 'user', content: userQuestion })
  question.value = ''
  loading.value = true
  scrollToBottom()

  try {
    const res = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: userQuestion })
    })
    const data = await res.json()
    messages.value.push({ role: 'assistant', content: data.answer, sources: data.sources })
  } catch (e) {
    messages.value.push({ role: 'assistant', content: 'Error: ' + e.message })
  } finally {
    loading.value = false
    scrollToBottom()
  }
}

function clearChat() { messages.value = [] }
function scrollToBottom() {
  nextTick(() => { if (messagesContainer.value) messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight })
}
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, #app { height: 100%; width: 100%; overflow: hidden; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #343541; }
.chat-container { display: flex; height: 100vh; width: 100vw; color: #ececf1; position: fixed; top: 0; left: 0; }
.sidebar { width: 260px; background: #202123; padding: 10px; display: flex; flex-direction: column; }
.sidebar-header h2 { padding: 12px; font-size: 14px; }
.new-chat { padding: 12px; border: 1px solid #565869; border-radius: 6px; background: transparent; color: #fff; cursor: pointer; }
.new-chat:hover { background: #2a2b32; }
.main { flex: 1; display: flex; flex-direction: column; background: #343541; }
.messages { flex: 1; overflow-y: auto; padding: 20px 0; }
.welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #8e8ea0; }
.welcome h1 { font-size: 32px; color: #ececf1; margin-bottom: 10px; }
.message { padding: 20px 0; }
.message.assistant { background: #444654; }
.message-content { max-width: 900px; margin: auto; padding: 0 40px; display: flex; gap: 20px; }
.avatar { width: 36px; height: 36px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; }
.message.user .avatar { background: #5436da; }
.message.assistant .avatar { background: #19c37d; }
.text { flex: 1; line-height: 1.8; }
.text h1, .text h2, .text h3 { margin: 24px 0 16px; color: #ececf1; }
.text p { margin-bottom: 16px; text-align: left;}
.text ul, .text ol { margin-left: 24px; margin-bottom: 16px; }
.text li { margin-bottom: 8px; }
.text code { background: #1f2028; padding: 3px 6px; border-radius: 3px; font-family: monospace; color: #a5b4fc; }
.text pre { background: #1f2028; padding: 16px; border-radius: 6px; overflow-x: auto; margin-bottom: 20px; }
.thinking { color: #8e8ea0; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.sources { margin-top: 15px; font-size: 13px; }
.sources summary { cursor: pointer; color: #8e8ea0; padding: 8px 0; }
.sources ul { list-style: none; margin-top: 8px; }
.sources li { padding: 8px 12px; background: #3a3b44; border-radius: 6px; margin-bottom: 6px; display: flex; justify-content: space-between; gap: 8px; }
.source-file { font-weight: 500; font-size: 12px; }
.source-meta { color: #8e8ea0; font-size: 12px; }
.input-area { padding: 20px 40px; background: #343541; }
.input-wrapper { max-width: 900px; margin: 0 auto; display: flex; align-items: flex-end; background: #40414f; border-radius: 12px; border: 1px solid #565869; padding: 12px 16px; }
.input-wrapper textarea { flex: 1; background: transparent; border: none; outline: none; color: #fff; font-size: 16px; resize: none; font-family: inherit; }
.input-wrapper textarea::placeholder { color: #8e8ea0; }
.input-wrapper button { background: transparent; border: none; color: #8e8ea0; cursor: pointer; padding: 4px; }
.input-wrapper button:hover:not(:disabled) { color: #fff; }
.input-wrapper button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
