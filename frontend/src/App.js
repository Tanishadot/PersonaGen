import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [persona, setPersona] = useState('friendly');
  const [empathyThreshold, setEmpathyThreshold] = useState(0.6);
  const [personas, setPersonas] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [backendStatus, setBackendStatus] = useState('ready');
  const [modelsLoading, setModelsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize app
  useEffect(() => {
    checkBackendHealth();
    loadPersonas();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await axios.get('http://localhost:8000/health');
      const status = response.data.status;
      const modelsLoaded = response.data.models_loaded;
      
      setBackendStatus(status);
      setIsBackendReady(modelsLoaded);
      setModelsLoading(status === 'loading');
      
      if (status === 'healthy') {
        console.log('Backend is ready!');
        loadMetrics(); // Load metrics only when backend is ready
      } else if (status === 'loading') {
        console.log('Models are loading...');
        // Retry after 2 seconds
        setTimeout(checkBackendHealth, 2000);
      } else {
        console.log('Backend is ready, models will load on first request');
        loadMetrics();
      }
    } catch (error) {
      console.error('Backend not ready:', error);
      setBackendStatus('error');
      setTimeout(checkBackendHealth, 3000);
    }
  };

  const loadPersonas = async () => {
    try {
      const response = await axios.get('http://localhost:8000/personas');
      setPersonas(response.data.personas);
    } catch (error) {
      console.error('Error loading personas:', error);
    }
  };

  const loadMetrics = async () => {
    try {
      const response = await axios.get('http://localhost:8000/metrics');
      setMetrics(response.data);
    } catch (error) {
      console.error('Error loading metrics:', error);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        message: inputMessage,
        persona: persona,
        empathy_threshold: empathyThreshold,
        max_retries: 3
      });

      // Debug logging for backend response
      console.log("Backend response:", response.data);

      // Safety check for response data
      const responseData = response.data;
      if (!responseData || typeof responseData.response !== 'string') {
        throw new Error('Invalid response format from backend');
      }

      const botMessage = {
        type: 'bot',
        content: responseData.response || 'I apologize, but I could not generate a response.',
        detected_emotion: responseData.detected_emotion || 'neutral',
        emotion_confidence: responseData.emotion_confidence || 0.0,
        empathy_score: responseData.empathy_score || 0.0,
        retry_count: responseData.retry_count || 0,
        persona: responseData.persona || persona,
        context_used: responseData.context_used || false,
        bleu_score: responseData.bleu_score || 0.0,
        retrieved_context: responseData.retrieved_context || null,
        retrieval_used: responseData.retrieval_used || false,
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, botMessage]);
      
      // Refresh metrics after each message
      loadMetrics();
      
      // Update backend status after successful interaction
      if (!isBackendReady) {
        setIsBackendReady(true);
        setBackendStatus('healthy');
      }
      
    } catch (error) {
      console.error('Error sending message:', error);
      
      let errorMessage = {
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toLocaleTimeString()
      };

      // Handle specific loading errors with safety checks
      if (error.response && error.response.status === 503) {
        const errorDetail = error.response.detail;
        // Safety check before using .includes()
        if (typeof errorDetail === 'string' && errorDetail.includes('loading')) {
          errorMessage.content = 'Models are currently loading. Please wait a moment and try again.';
          setModelsLoading(true);
          // Retry health check after a delay
          setTimeout(checkBackendHealth, 3000);
        } else {
          errorMessage.content = errorDetail || 'Models are not available. Please try again.';
        }
      }
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const clearMemory = async () => {
    try {
      await axios.delete('http://localhost:8000/memory');
      loadMetrics();
      console.log('Memory cleared successfully');
    } catch (error) {
      console.error('Error clearing memory:', error);
    }
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      'joy': '#22C55E',
      'sadness': '#3B82F6',
      'anger': '#EF4444',
      'fear': '#F59E0B',
      'surprise': '#8B5CF6',
      'disgust': '#6B7280',
      'neutral': '#6B7280'
    };
    return colors[emotion] || '#6B7280';
  };

  const getEmpathyColor = (score) => {
    if (score >= 0.8) return '#22C55E';
    if (score >= 0.6) return '#F59E0B';
    return '#EF4444';
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>PersonaGen</h1>
        <p>Emotion-aware conversational AI</p>
      </header>

      <div className="main-container">
        {/* Controls Panel */}
        <div className="controls-panel">
          <div className="control-group">
            <label htmlFor="persona">Persona:</label>
            <select
              id="persona"
              value={persona}
              onChange={(e) => setPersona(e.target.value)}
              disabled={isLoading}
            >
              {personas.map(p => (
                <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="threshold">Empathy Threshold: {empathyThreshold.toFixed(2)}</label>
            <input
              id="threshold"
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={empathyThreshold}
              onChange={(e) => setEmpathyThreshold(parseFloat(e.target.value))}
              disabled={isLoading}
            />
          </div>

          <div className="control-group">
            <button onClick={clearChat} disabled={isLoading}>Clear Chat</button>
            <button onClick={clearMemory} disabled={isLoading}>Clear Memory</button>
          </div>

          {/* Metrics Display */}
          {metrics && (
            <div className="metrics">
              <h3>Conversation Metrics</h3>
              <p>Avg Empathy: {metrics.average_empathy_score.toFixed(3)}</p>
              <p>Total Messages: {metrics.total_conversations}</p>
              <p>Response Diversity: {metrics.response_diversity.toFixed(3)}</p>
              {metrics.recent_emotions.length > 0 && (
                <p>Recent Emotions: {metrics.recent_emotions.join(', ')}</p>
              )}
            </div>
          )}

          {/* Backend Status */}
          <div className={`status ${backendStatus === 'healthy' ? 'ready' : backendStatus === 'loading' ? 'loading' : backendStatus === 'error' ? 'error' : 'ready'}`}>
            {backendStatus === 'healthy' ? 'Backend Ready' : 
             backendStatus === 'loading' ? 'Loading Models...' :
             backendStatus === 'error' ? 'Backend Error' :
             'Backend Ready (Models will load on first use)'}
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-container">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h2>Welcome to PersonaGen</h2>
                <p>I'm an emotionally intelligent chatbot that adapts my responses based on your emotions.</p>
                <p>Try asking me something and see how I detect emotions and adjust my empathy!</p>
              </div>
            )}
            
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.type}`}>
                <div className="message-content">
                  <strong>{message.type === 'user' ? 'You' : 'PersonaGen'}:</strong>
                  <p>{message.content}</p>
                  <span className="timestamp">{message.timestamp}</span>
                  
                  {message.type === 'bot' && (
                    <div className="message-metadata">
                      <div className="metadata-item">
                        <span 
                          className="emotion-badge"
                          style={{ backgroundColor: getEmotionColor(message.detected_emotion) }}
                        >
                          {message.detected_emotion} ({(message.emotion_confidence * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="metadata-item">
                        <span 
                          className="empathy-score"
                          style={{ color: getEmpathyColor(message.empathy_score) }}
                        >
                          Empathy: {message.empathy_score.toFixed(3)}
                        </span>
                      </div>
                      {message.retry_count > 0 && (
                        <div className="metadata-item">
                          <span className="retry-count">
                            Retries: {message.retry_count}
                          </span>
                        </div>
                      )}
                      {message.context_used && (
                        <div className="metadata-item">
                          <span className="context-used">
                            Context Used
                          </span>
                        </div>
                      )}
                      {message.retrieval_used && (
                        <div className="metadata-item">
                          <span className="retrieval-used">
                            Dataset Retrieval Used
                          </span>
                        </div>
                      )}
                      {message.retrieved_context && (
                        <div className="metadata-item">
                          <span className="retrieved-context">
                            Retrieved: {message.retrieved_context.substring(0, 50)}...
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <strong>PersonaGen:</strong>
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="input-container">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                modelsLoading ? "Models are loading, please wait..." :
                backendStatus === 'error' ? "Backend connection error..." :
                "Type your message here..."
              }
              disabled={isLoading || modelsLoading || backendStatus === 'error'}
              rows={2}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !inputMessage.trim() || modelsLoading || backendStatus === 'error'}
            >
              {isLoading ? 'Sending...' : 
               modelsLoading ? 'Loading...' :
               'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
