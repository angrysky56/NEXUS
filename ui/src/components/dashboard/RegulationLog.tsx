import React, { useRef, useEffect } from "react";
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  Activity,
  Brain,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

export interface LogEntry {
  id: string;
  timestamp: number;
  type: "system" | "internal" | "ai" | "alert";
  text: string;
}

interface RegulationLogProps {
  logs: LogEntry[];
}

export const RegulationLog: React.FC<RegulationLogProps> = ({ logs }) => {
  // Show only the last 5-7 logs
  const recentLogs = logs.slice(0, 7);

  return (
    <div className="glass-panel rounded-xl p-4 flex flex-col gap-3 min-h-[200px]">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="text-slate-300 font-bold tracking-widest flex items-center gap-2 uppercase text-[10px]">
          <Shield size={14} className="text-emerald-400" /> Regulation Logs
        </span>
        <span className="text-[10px] text-slate-500 font-mono">LIVE</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2 custom-scrollbar relative">
        <AnimatePresence initial={false}>
          {recentLogs.map((log) => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, x: -20, height: 0 }}
              animate={{ opacity: 1, x: 0, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className={`p-2 rounded border text-[10px] font-mono leading-relaxed ${
                log.type === "alert"
                  ? "bg-red-500/10 border-red-500/30 text-red-400"
                  : log.type === "internal"
                    ? "bg-blue-500/10 border-blue-500/20 text-blue-300"
                    : log.type === "system"
                      ? "bg-slate-800/50 border-slate-700 text-slate-400"
                      : "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
              }`}
            >
              <div className="flex items-start gap-2">
                <span className="shrink-0 mt-0.5">
                  {log.type === "alert" && <AlertTriangle size={10} />}
                  {log.type === "internal" && <Brain size={10} />}
                  {log.type === "system" && <Activity size={10} />}
                  {log.type === "ai" && <CheckCircle size={10} />}
                </span>
                <div>
                  <span className="opacity-50 mr-2">
                    [
                    {new Date(log.timestamp).toLocaleTimeString([], {
                      hour12: false,
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    })}
                    ]
                  </span>
                  {log.text}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        {logs.length === 0 && (
          <div className="text-center text-slate-600 text-[10px] italic py-8">
            System stable. Monitoring...
          </div>
        )}
      </div>
    </div>
  );
};
