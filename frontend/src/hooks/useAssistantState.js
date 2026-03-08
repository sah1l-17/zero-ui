import { useEffect, useMemo, useRef, useState } from 'react';

const DEFAULT_STATE = { speaking: false, text: '' };

export function useAssistantState({ pollMs = 500 } = {}) {
  const [state, setState] = useState(DEFAULT_STATE);
  const [transport, setTransport] = useState('connecting');
  const pollTimerRef = useRef(null);
  const wsRef = useRef(null);

  const wsUrl = useMemo(() => {
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${scheme}://${window.location.host}/ui/ws`;
  }, []);

  useEffect(() => {
    let cancelled = false;

    const stopPolling = () => {
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };

    const startPolling = () => {
      stopPolling();
      setTransport('polling');

      const fetchOnce = async () => {
        try {
          const res = await fetch('/ui/state', { cache: 'no-store' });
          if (!res.ok) return;
          const data = await res.json();
          if (cancelled) return;
          if (typeof data?.speaking === 'boolean' && typeof data?.text === 'string') {
            setState(data);
          }
        } catch {
          // ignore
        }
      };

      fetchOnce();
      pollTimerRef.current = window.setInterval(fetchOnce, pollMs);
    };

    const connectWs = () => {
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (cancelled) return;
          setTransport('ws');
          stopPolling();
        };

        ws.onmessage = (ev) => {
          if (cancelled) return;
          try {
            const data = JSON.parse(ev.data);
            if (typeof data?.speaking === 'boolean' && typeof data?.text === 'string') {
              setState(data);
            }
          } catch {
            // ignore
          }
        };

        ws.onclose = () => {
          if (cancelled) return;
          startPolling();
        };

        ws.onerror = () => {
          try {
            ws.close();
          } catch {
            // ignore
          }
        };
      } catch {
        startPolling();
      }
    };

    connectWs();

    return () => {
      cancelled = true;
      stopPolling();
      try {
        wsRef.current?.close();
      } catch {
        // ignore
      }
    };
  }, [pollMs, wsUrl]);

  return { state, transport };
}
