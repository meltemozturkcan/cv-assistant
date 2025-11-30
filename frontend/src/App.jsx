import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import ReactMarkdown from 'react-markdown'

// Avatar resmi - kendi resmin için URL'yi değiştir
const AVATAR_URL = "/meltem.png"

// Hızlı sorular
const QUICK_QUESTIONS = [
"Teknik yetkinlikleriniz neler?",
"İş deneyiminizi özetleyebilir misiniz?",
"Yer aldığınız projelerden bahsedebilir misiniz?",
"Eğitim geçmişinizi paylaşabilir misiniz?"
]

// Teknik yetenekler
const SKILLS = ["C#", ".NET Core", "AI/ML", "Python","Web API", "Microservices", "React", "Azure", "Docker","PostgreSQL","MSSQL","GitHub"]

function App() {
const [messages, setMessages] = useState([
  {
    role: 'assistant',
    content: 'Merhaba. Ben Meltem Öztürkcan’ın profesyonel özgeçmiş asistanıyım. Bilgi almak istediğiniz konuyla ilgili sorunuzu paylaşabilirsiniz.'
  }
])
const [input, setInput] = useState('')
const [isLoading, setIsLoading] = useState(false)
const [sidebarOpen, setSidebarOpen] = useState(false)
const messagesEndRef = useRef(null)

const scrollToBottom = () => {
  messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
}

useEffect(() => {
  scrollToBottom()
}, [messages])

const sendMessage = async (e, questionText = null) => {
  if (e) e.preventDefault()
  
  const userMessage = questionText || input.trim()
  if (!userMessage || isLoading) return

  setInput('')
  setSidebarOpen(false) // Mobilde sidebar'ı kapat
  
  // Kullanıcı mesajını ekle
  setMessages(prev => [...prev, { role: 'user', content: userMessage }])
  setIsLoading(true)

  try {
const response = await axios.post(
'https://cv-assistant-duuy.onrender.com/ask',
{
  question: userMessage,
}
)
    
    // Bot cevabını ekle
    const cleanedAnswer = response.data.answer
.replace(/\n{3,}/g, '\n\n')
.replace(/^\s+/gm, '')
.trim();

setMessages(prev => [...prev, { role: 'assistant', content: cleanedAnswer }])
  } catch (error) {
    console.error('Hata:', error)
    setMessages(prev => [...prev, { 
      role: 'assistant', 
      content: 'Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.' 
    }])
  } finally {
    setIsLoading(false)
  }
}

const handleQuickQuestion = (question) => {
  sendMessage(null, question)
}

return (
  <div className="app">
    {/* Mobile Header */}
    <div className="mobile-header">
      <button className="menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
        ☰
      </button>
      <div className="mobile-title">
        <img src={AVATAR_URL} alt="Meltem Öztürkcan" className="mobile-avatar" />
        <span>Meltem Öztürkcan</span>
      </div>
    </div>

    {/* Sidebar */}
    <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
      <div className="sidebar-content">
        {/* Profil Bölümü */}
        <div className="profile-section">
          <img src={AVATAR_URL} alt="Meltem Öztürkcan" className="avatar-img" />
          <h2>Meltem Öztürkcan</h2>
          <p className="title">Full Stack .NET & AI Developer</p>
          
          <div style={{ display: 'flex', flexDirection: 'row', gap: '10px', justifyContent: 'center', marginBottom: '15px' }}>
<span className="badge"> İstanbul, Türkiye</span>
<span className="badge">2+ Yıl Deneyim</span>
</div>
          
          <p className="bio">
            Mikroservis mimarileri ve AI destekli sistemlerde uzman, modern teknolojilerle ölçeklenebilir çözümler üreten full-stack geliştirici            </p>
        </div>

        {/* Teknik Yetenekler */}
        <div className="skills-section">
          <h3>Teknik Yetenekler</h3>
          <div className="skills-grid">
            {SKILLS.map((skill, index) => (
              <span key={index} className="skill-tag">{skill}</span>
            ))}
          </div>
        </div>

        {/* Uzmanlık Alanları */}
<div className="expertise-section">
<h3>Uzmanlık Alanları</h3>
<div className="expertise-list">
  <div className="expertise-item">Full Stack .NET Development</div>
  <div className="expertise-item">AI/ML Integration</div>
  <div className="expertise-item">DevOps & Cloud</div>
  <div className="expertise-item">Frontend & UI/UX</div>
</div>
</div>

        {/* CV İndir */}
        <a href="/cv.pdf" download className="download-btn">
          📄 Özgeçmişi İncele
        </a>
      </div>

      {/* Sidebar Footer */}
      <div className="sidebar-footer">
        <div className="social-links">
          <a href="https://www.linkedin.com/in/meltem-öztürkcan" target="_blank" rel="noopener noreferrer">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
            </svg>
          </a>
          <a href="https://github.com/meltemozturkcan" target="_blank" rel="noopener noreferrer">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
          </a>
        </div>
        <p className="copyright">© 2025 Meltem Öztürkcan <br/>Designed with ❤️</p>
      </div>
    </aside>

    {/* Overlay for mobile */}
    {sidebarOpen && <div className="overlay" onClick={() => setSidebarOpen(false)}></div>}

    {/* Main Chat Area */}
    <main className="main-content">
      {/* Chat Header */}
      <header className="chat-header">
        <div className="header-info">
          <h1>Meltem AI - Özgeçmiş Asistanı</h1>
          <p>Meltem’in kariyeri, teknik uzmanlıkları ve projeleri hakkında sorularınızı yanıtlamaya hazırım</p>
        </div>
      </header>

      {/* Messages */}
      <div className="chat-container">
        <div className="messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              {message.role === 'assistant' && (
                <img src={AVATAR_URL} alt="Bot" className="message-avatar" />
              )}
              <div className="message-content">
<ReactMarkdown>{message.content}</ReactMarkdown>
</div>
            </div>
          ))}
          {isLoading && (
            <div className="message assistant">
              <img src={AVATAR_URL} alt="Bot" className="message-avatar" />
              <div className="message-content loading">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick Questions - Bottom */}
      <div className="bottom-questions">
        <p>Hızlı Sorular:</p>
        <div className="quick-btns">
          {QUICK_QUESTIONS.map((q, i) => (
            <button key={i} onClick={() => handleQuickQuestion(q)} disabled={isLoading}>
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Input Area */}
      <footer className="input-area">
        <form onSubmit={sendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Sorunuzu yazın..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </form>
      </footer>

      {/* Footer */}
      <div className="main-footer">
        <div className="footer-links">
          <a href="https://www.linkedin.com/in/meltem-öztürkcan" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          <span>•</span>
          <a href="https://github.com/meltemozturkcan" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
        <p>© 2025 Meltem Öztürkcan — Full Stack .NET & AI Developer</p>
      </div>
    </main>
  </div>
)
}

export default App