import React, { useState } from 'react';
import axios from 'axios';

const ChatbotComponent = () => {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState([]);

  const handleSubmit = async () => {
    try {
      const response = await axios.post('/chat', { input });
      const { answer, source_documents } = response.data;

      setHistory([
        ...history,
        { origin: '🗣️ Human', message: input },
        { origin: '🧑‍⚖️ AI Lawyer', message: answer },
      ]);

      // Print the source documents to the console
      console.log('Source Documents:');
      source_documents.forEach((doc) => {
        console.log(`- ${doc.metadata.source || 'Unknown source'}: ${doc.page_content.slice(0, 100)}...`);
      });

      setInput('');
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <div>
        {history.map((message, index) => (
          <div key={index}>
            <strong>{message.origin}</strong>: {message.message}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter your question here..."
      />
      <button onClick={handleSubmit}>Send</button>
    </div>
  );
};

export default ChatbotComponent;