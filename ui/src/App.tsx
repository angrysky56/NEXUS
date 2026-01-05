import { Layout } from './components/layout/Layout';
import { ChatArea } from './components/chat/ChatArea';
import { CognitiveDashboard } from './components/dashboard/CognitiveDashboard';

function App() {
  return (
    <Layout rightPanel={<CognitiveDashboard />}>
        <ChatArea />
    </Layout>
  );
}

export default App;
