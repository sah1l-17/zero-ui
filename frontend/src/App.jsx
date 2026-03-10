import './App.css';
import AssistantLine from './components/AssistantLine';
import OrbStage from './components/OrbStage';
import { useAssistantState } from './hooks/useAssistantState';

export default function App() {
  const { state, transport } = useAssistantState({ pollMs: 500 });

  return (
    <main className="shell">
      <section className="center">
        <OrbStage speaking={state.speaking} />
        <AssistantLine text={state.text || (state.speaking ? 'Speaking…' : 'Starting assistant…')} />
        <div className="transport" aria-hidden="true">
          {transport === 'ws' ? 'Connected' : transport === 'polling' ? 'Syncing' : 'Connecting'}
        </div>
      </section>
    </main>
  );
}
