<script lang="ts">
  import { tick } from 'svelte'

  type Message = {
    role: 'user' | 'assistant'
    content: string
    streaming?: boolean
  }

  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8090'

  let input = $state('')
  let isSending = $state(false)
  let error = $state('')
  let messagesElement: HTMLDivElement
  let messages = $state<Message[]>([
    {
      role: 'assistant',
      content: 'Chào bạn, mình sẵn sàng trả lời qua backend streaming.',
    },
  ])

  async function scrollToLatest() {
    await tick()
    messagesElement?.scrollTo({ top: messagesElement.scrollHeight, behavior: 'smooth' })
  }

  async function appendAssistantChunk(index: number, chunk: string) {
    if (!chunk) return
    messages[index] = {
      ...messages[index],
      content: messages[index].content + chunk,
      streaming: true,
    }
    messages = [...messages]
    await scrollToLatest()
  }

  async function streamAssistantResponse(response: Response, assistantIndex: number) {
    const reader = response.body?.getReader()
    if (!reader) throw new Error('Backend response is not streamable')

    const decoder = new TextDecoder()
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      await appendAssistantChunk(assistantIndex, decoder.decode(value, { stream: true }))
    }

    await appendAssistantChunk(assistantIndex, decoder.decode())
    messages[assistantIndex] = { ...messages[assistantIndex], streaming: false }
    messages = [...messages]
  }

  async function sendMessage() {
    const text = input.trim()
    if (!text || isSending) return

    const assistantIndex = messages.length + 1
    messages = [...messages, { role: 'user', content: text }, { role: 'assistant', content: '', streaming: true }]
    input = ''
    error = ''
    isSending = true
    await scrollToLatest()

    try {
      const response = await fetch(`${apiBaseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`)
      }

      await streamAssistantResponse(response, assistantIndex)
    } catch (err) {
      error = err instanceof Error ? err.message : 'Không gọi được backend'
      messages[assistantIndex] = {
        role: 'assistant',
        content: 'Xin lỗi, hiện tại mình chưa kết nối được backend.',
        streaming: false,
      }
      messages = [...messages]
    } finally {
      isSending = false
      await scrollToLatest()
    }
  }
</script>

<main class="shell">
  <section class="chat-panel" aria-label="Chatbot">
    <header class="topbar">
      <div>
        <p class="eyebrow">VDT chatbot</p>
        <h1>Simple LangChain Chat</h1>
      </div>
      <a class="graph-link" href="http://localhost:7200" target="_blank" rel="noreferrer">GraphDB</a>
    </header>

    <div class="messages" aria-live="polite" bind:this={messagesElement}>
      {#each messages as message}
        <article class:assistant={message.role === 'assistant'} class:user={message.role === 'user'}>
          <span>{message.role === 'assistant' ? 'AI' : 'You'}</span>
          <p>{message.content || 'Đang trả lời...'}{#if message.streaming && message.content}<span class="stream-cursor" aria-hidden="true"></span>{/if}</p>
        </article>
      {/each}
    </div>

    {#if error}
      <p class="error">{error}</p>
    {/if}

    <form class="composer" onsubmit={(event) => { event.preventDefault(); sendMessage() }}>
      <input bind:value={input} placeholder="Nhập câu hỏi..." aria-label="Tin nhắn" disabled={isSending} />
      <button type="submit" disabled={isSending || !input.trim()}>
        {isSending ? 'Đang gửi' : 'Gửi'}
      </button>
    </form>
  </section>
</main>