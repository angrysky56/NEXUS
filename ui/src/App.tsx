import { useState, useEffect } from 'react';
import { Layout } from './components/layout/Layout';
import { ChatArea } from './components/chat/ChatArea';
import { CognitiveDashboard } from './components/dashboard/CognitiveDashboard';
import { api } from './api';
import { v4 as uuidv4 } from 'uuid';

function App() {
  const [sessionId, setSessionId] = useState<string>(() => {
    return localStorage.getItem('NEXUS_SESSION_ID') || uuidv4();
  });
  const [sessions, setSessions] = useState<string[]>([]);

  useEffect(() => {
    localStorage.setItem('NEXUS_SESSION_ID', sessionId);
    refreshSessions();
  }, [sessionId]);

  const refreshSessions = async () => {
    try {
      const s = await api.listSessions();
      setSessions(s);
    } catch (e) {
      console.error("Failed to list sessions", e);
    }
  };

  const handleNewChat = () => {
    const newId = uuidv4();
    setSessionId(newId);
  };

  return (
    <Layout
      rightPanel={<CognitiveDashboard />}
      sessionId={sessionId}
      sessions={sessions}
      onSelectSession={setSessionId}
      onNewChat={handleNewChat}
    >
        <ChatArea sessionId={sessionId} />
    </Layout>
  );
}

export default App;
