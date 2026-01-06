import React, { useEffect, useState, useRef } from "react";
import { type CognitiveState } from "../../api";
import { CognitiveCompass } from "./CognitiveCompass";
import { ManifoldGauge } from "./ManifoldGauge";
import { RegulationLog } from "./RegulationLog";
import type { LogEntry } from "./RegulationLog";
import { v4 as uuidv4 } from "uuid";

export const CognitiveDashboard: React.FC = () => {
  const [state, setState] = useState<CognitiveState>({
    valence: 0,
    arousal: 0,
    intrinsic_dimension: 0,
    gate_value: 0.5,
    primary_manifold: "neutral",
  });

  const [logs, setLogs] = useState<LogEntry[]>([]);

  // Refs for tracking changes to generate logs
  const prevManifold = useRef(state.primary_manifold);
  const prevID = useRef(state.intrinsic_dimension);

  useEffect(() => {
    const eventSource = new EventSource(
      "http://localhost:8000/api/v1/cognitive/status",
    );

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setState((prev) => ({ ...prev, ...data }));
      } catch (e) {
        console.error("Dashboard parse error", e);
      }
    };

    return () => eventSource.close();
  }, []);

  // Generate Logs on meaningful state changes
  useEffect(() => {
    // Did Manifold change?
    if (
      state.primary_manifold !== prevManifold.current &&
      prevManifold.current !== "neutral"
    ) {
      addLog(
        "system",
        `Switched Manifold: ${prevManifold.current.toUpperCase()} -> ${state.primary_manifold.toUpperCase()}`,
      );
    }
    prevManifold.current = state.primary_manifold;

    // Did ID spike/drop significantly? (e.g. > 0.5 change)
    if (Math.abs(state.intrinsic_dimension - prevID.current) > 0.5) {
      const type =
        state.intrinsic_dimension > prevID.current
          ? "Chaos Increases"
          : "Order Increases";
      addLog(
        "internal",
        `${type} (ID: ${state.intrinsic_dimension.toFixed(2)})`,
      );
    }
    prevID.current = state.intrinsic_dimension;

    // Did Valence go critical?
    if (state.valence < -0.8) {
      addLog(
        "alert",
        `Critical Negative Valence: ${state.valence.toFixed(2)}. Engaging Dampeners.`,
      );
    }
  }, [state]);

  const addLog = (type: LogEntry["type"], text: string) => {
    const entry: LogEntry = {
      id: uuidv4(),
      timestamp: Date.now(),
      type,
      text,
    };
    setLogs((prev) => [entry, ...prev].slice(0, 50));
  };

  return (
    <div className="p-4 flex flex-col gap-4 h-full text-xs font-mono overflow-y-auto custom-scrollbar">
      {/* 1. Compass */}
      <CognitiveCompass valence={state.valence} arousal={state.arousal} />

      {/* 2. Manifold/Gate */}
      <ManifoldGauge
        intrinsicDimension={state.intrinsic_dimension}
        gateValue={state.gate_value}
        primaryManifold={state.primary_manifold}
      />

      {/* 3. Logs */}
      <div className="flex-1 min-h-0">
        <RegulationLog logs={logs} />
      </div>
    </div>
  );
};
