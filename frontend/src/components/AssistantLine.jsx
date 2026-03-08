import { useEffect, useRef, useState } from 'react';

export default function AssistantLine({ text }) {
  const [active, setActive] = useState(text);
  const [incoming, setIncoming] = useState(null);
  const [phase, setPhase] = useState('idle'); // idle | fading
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (text === active) return;

    setIncoming(text);
    setPhase('fading');

    window.clearTimeout(timeoutRef.current);
    timeoutRef.current = window.setTimeout(() => {
      setActive(text);
      setIncoming(null);
      setPhase('idle');
    }, 260);

    return () => window.clearTimeout(timeoutRef.current);
  }, [text, active]);

  return (
    <div className="assistantLine" aria-live="polite" aria-atomic="true">
      <div className={`assistantLineLayer ${phase === 'fading' ? 'fadeOut' : 'visible'}`}>{active}</div>
      <div className={`assistantLineLayer ${phase === 'fading' ? 'fadeIn' : 'hidden'}`}>{incoming ?? ''}</div>
    </div>
  );
}
