import React, { useMemo } from "react";
import { Compass, Activity } from "lucide-react";

interface CognitiveCompassProps {
  valence: number;
  arousal: number;
}

export const CognitiveCompass: React.FC<CognitiveCompassProps> = ({
  valence,
  arousal,
}) => {
  // Determine Quadrant and Description
  const stateInfo = useMemo(() => {
    const intensity = Math.sqrt(valence * valence + arousal * arousal);

    if (intensity < 0.2)
      return {
        label: "Homeostasis",
        color: "text-slate-400",
        bg: "bg-slate-500",
      };
    if (valence > 0 && arousal > 0)
      return {
        label: "High Positive",
        color: "text-neon-blue",
        bg: "bg-neon-blue",
      }; // Joy/Flow
    if (valence < 0 && arousal > 0)
      return {
        label: "High Negative",
        color: "text-alert-yellow",
        bg: "bg-alert-yellow",
      }; // Anxiety/Fight
    if (valence < 0 && arousal < 0)
      return {
        label: "Low Negative",
        color: "text-neon-pink",
        bg: "bg-neon-pink",
      }; // Depressed/Freeze
    if (valence > 0 && arousal < 0)
      return {
        label: "Low Positive",
        color: "text-emerald-400",
        bg: "bg-emerald-400",
      }; // Calm/Rest
    return { label: "Neutral", color: "text-slate-400", bg: "bg-slate-500" };
  }, [valence, arousal]);

  // Map -1..1 to 0..100%
  const left = ((valence + 1) / 2) * 100;
  const bottom = ((arousal + 1) / 2) * 100;

  return (
    <div className="glass-panel p-4 rounded-xl relative overflow-hidden group">
      <div className="flex items-center justify-between mb-4 relative z-10">
        <span className="text-slate-300 font-bold tracking-widest flex items-center gap-2 uppercase text-[10px]">
          <Compass size={14} className="text-neon-blue" /> Affective State
        </span>
        <span
          className={`text-[10px] font-mono ${stateInfo.color} border border-current px-1.5 rounded-sm opacity-80`}
        >
          {stateInfo.label}
        </span>
      </div>

      <div className="aspect-square w-full bg-slate-900/50 rounded-lg border border-slate-700 relative overflow-hidden">
        {/* Axes */}
        <div className="absolute top-1/2 left-0 w-full h-px bg-slate-700/50"></div>
        <div className="absolute left-1/2 top-0 h-full w-px bg-slate-700/50"></div>

        {/* Labels */}
        <div className="absolute top-1 left-1/2 -translate-x-1/2 text-[8px] text-slate-600 uppercase">
          Arousal +
        </div>
        <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[8px] text-slate-600 uppercase">
          Arousal -
        </div>
        <div className="absolute left-1 top-1/2 -translate-y-1/2 text-[8px] text-slate-600 uppercase -rotate-90">
          Valence -
        </div>
        <div className="absolute right-1 top-1/2 -translate-y-1/2 text-[8px] text-slate-600 uppercase rotate-90">
          Valence +
        </div>

        {/* The Dot */}
        <div
          className="absolute w-4 h-4 -ml-2 -mb-2 z-20 transition-all duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)]"
          style={{ left: `${left}%`, bottom: `${bottom}%` }}
        >
          <div
            className={`w-full h-full rounded-full shadow-[0_0_10px_currentColor] animate-pulse ${stateInfo.bg} ${stateInfo.color}`}
          />
          <div className="absolute inset-0 bg-white/50 rounded-full animate-ping opacity-20" />
        </div>

        {/* Gravity/History Trail (Simplified) */}
        <div className="absolute inset-0 pointer-events-none opacity-20">
          <div
            className={`absolute w-12 h-12 -ml-6 -mb-6 rounded-full border border-current transition-all duration-1000 ${stateInfo.color}`}
            style={{ left: `${left}%`, bottom: `${bottom}%`, opacity: 0.3 }}
          />
        </div>
      </div>

      <div className="mt-2 text-[10px] text-slate-500 font-mono flex justify-between">
        <span>V: {valence.toFixed(2)}</span>
        <span>A: {arousal.toFixed(2)}</span>
      </div>
    </div>
  );
};
