import React from "react";
import { Layers, Brain, Zap, Activity } from "lucide-react";
import { motion } from "framer-motion";

interface ManifoldGaugeProps {
  intrinsicDimension: number;
  gateValue: number; // 0 (Creative) to 1 (Logic)
  primaryManifold: string;
}

export const ManifoldGauge: React.FC<ManifoldGaugeProps> = ({
  intrinsicDimension,
  gateValue,
  primaryManifold,
}) => {
  const isLogic = primaryManifold === "logic";

  // Aesthetic mapping based on AdaptiveSynthesizer snippet
  const activeColor = isLogic ? "text-neon-blue" : "text-neon-pink";
  const activeBg = isLogic ? "bg-neon-blue" : "bg-neon-pink";
  const activeLabel = isLogic ? "LOGIC MANIFOLD" : "CREATIVE MANIFOLD";
  const description = isLogic
    ? "Prioritizing structural integrity, facts, and minimal speculation."
    : "Prioritizing exploration, metaphors, and novel connections.";

  return (
    <div className="glass-panel p-4 rounded-xl space-y-4">
      {/* Header / Current Mode */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-slate-300 font-bold tracking-widest flex items-center gap-2 uppercase text-[10px]">
            <Layers size={14} className={activeColor} /> Bicameral Gate
          </span>
          <span
            className={`text-[10px] font-bold ${activeColor} glow-text uppercase border border-current px-2 py-0.5 rounded-full bg-opacity-10`}
          >
            {activeLabel}
          </span>
        </div>
        <p className="text-[10px] text-slate-500 leading-tight">
          {description}
        </p>
      </div>

      {/* The Gate Slider Visualization */}
      <div className="relative h-12 bg-slate-900/50 rounded-lg border border-slate-700/50 p-1 flex items-center gap-2">
        <Brain
          size={14}
          className={`${isLogic ? "text-neon-blue" : "text-slate-600"} transition-colors duration-500`}
        />

        <div className="flex-1 h-2 bg-slate-800 rounded-full relative overflow-hidden shadow-inner">
          {/* Background Gradients */}
          <div className="absolute inset-0 bg-gradient-to-r from-neon-pink/20 to-neon-blue/20" />

          {/* The Knob / Indicator */}
          <motion.div
            className="absolute top-0 bottom-0 w-2 bg-white rounded-full shadow-[0_0_10px_white] z-10"
            animate={{ left: `${gateValue * 100}%` }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
          />

          {/* Active Fill */}
          <motion.div
            className={`absolute top-0 bottom-0 ${activeBg} opacity-50`}
            style={{
              left: isLogic ? "50%" : `${gateValue * 100}%`,
              right: isLogic ? `${100 - gateValue * 100}%` : "50%",
            }}
          />
        </div>

        <Zap
          size={14}
          className={`${!isLogic ? "text-neon-pink" : "text-slate-600"} transition-colors duration-500`}
        />
      </div>

      {/* Fractal Entropy / ID */}
      <div className="pt-2 border-t border-slate-800">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] uppercase text-slate-500 flex items-center gap-1">
            <Activity size={10} /> Fractal ID
          </span>
          <span className="text-[10px] font-mono text-neon-purple">
            {intrinsicDimension.toFixed(2)}
          </span>
        </div>
        <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-neon-purple shadow-[0_0_8px_currentColor]"
            animate={{
              width: `${Math.min(100, (intrinsicDimension / 3) * 100)}%`,
            }}
          />
        </div>
      </div>
    </div>
  );
};
