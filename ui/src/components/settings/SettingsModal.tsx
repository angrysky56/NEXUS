import React, { useState, useEffect } from "react";
import { X, Save, Key } from "lucide-react";
import { motion } from "framer-motion";
import { SystemPromptEditor } from "./SystemPromptEditor";
import { api } from "../../api";

interface SettingsModalProps {
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ onClose }) => {
  const [apiKey, setApiKey] = useState("");
  const [maxIterations, setMaxIterations] = useState(50);
  const [allowedPaths, setAllowedPaths] = useState<string[]>([]);
  const [workspaceDir, setWorkspaceDir] = useState("./");
  const [activeTab, setActiveTab] = useState<"general" | "prompt">("general");

  // Load from LocalStorage on mount
  useEffect(() => {
    const storedKey = localStorage.getItem("OPENROUTER_API_KEY");
    if (storedKey) setApiKey(storedKey);

    const storedIterations = localStorage.getItem("NEXUS_MAX_ITERATIONS");
    if (storedIterations) setMaxIterations(parseInt(storedIterations));

    // Load config from backend on mount too, to get workspace/paths
    api
      .getConfig()
      .then((cfg) => {
        if (cfg.allowed_paths) setAllowedPaths(cfg.allowed_paths);
        if (cfg.workspace_dir) setWorkspaceDir(cfg.workspace_dir);
        if (cfg.max_tool_iterations) setMaxIterations(cfg.max_tool_iterations);
      })
      .catch(console.error);
  }, []);

  const handleSave = () => {
    localStorage.setItem("OPENROUTER_API_KEY", apiKey);
    // Save to backend
    api
      .updateConfig({
        allowed_paths: allowedPaths,
        workspace_dir: workspaceDir,
        max_tool_iterations: maxIterations,
      })
      .then(() => {
        onClose();
      })
      .catch(console.error);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-black/60 backdrop-blur-md"
        onClick={onClose}
      />

      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="relative bg-nexus-gray border border-nexus-border rounded-xl w-full max-w-xl shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden"
      >
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b border-nexus-border bg-nexus-black/30">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <SettingsIcon className="text-neon-blue" />
              System Configuration
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Configure Neural Interface & API Gateways
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-nexus-border bg-nexus-black/30 px-6">
          <button
            onClick={() => setActiveTab("general")}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "general"
                ? "border-neon-blue text-neon-blue"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            General Configuration
          </button>
          <button
            onClick={() => setActiveTab("prompt")}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "prompt"
                ? "border-neon-purple text-neon-purple"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            System Context
          </button>
        </div>

        {/* Body */}
        <div className="p-6 h-[500px] overflow-y-auto custom-scrollbar">
          {activeTab === "general" ? (
            <div className="space-y-8">
              {/* OpenRouter Key Section */}
              <div className="space-y-3">
                <label className="text-sm font-medium text-neon-blue flex items-center gap-2">
                  <Key size={14} />
                  OpenRouter API Key
                </label>
                <div className="relative group">
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-or-..."
                    className="w-full bg-nexus-black border border-nexus-border rounded-lg px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-neon-blue transition-all"
                  />
                  <div className="absolute inset-0 rounded-lg bg-neon-blue/5 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity" />
                </div>
                <p className="text-[10px] text-slate-500">
                  Key is stored locally in your browser. Not transmitted to any
                  server other than OpenRouter.
                </p>
              </div>

              {/* Cognitive Configuration */}
              <div className="space-y-3 pt-4 border-t border-nexus-border/30">
                <label className="text-sm font-medium text-neon-purple flex items-center gap-2">
                  <SettingsIcon className="w-3.5 h-3.5" />
                  Cognitive Configuration
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">
                      Max Recursive Turns
                    </label>
                    <input
                      type="number"
                      value={maxIterations}
                      onChange={(e) =>
                        setMaxIterations(parseInt(e.target.value) || 50)
                      }
                      min="1"
                      max="1000"
                      className="w-full bg-nexus-black border border-nexus-border rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-neon-purple transition-all"
                    />
                    <p className="text-[9px] text-slate-500 italic">
                      Maximum number of reasoning steps (LLM calls) per user
                      request.
                    </p>
                  </div>
                </div>
              </div>

              {/* Workspace & Security */}
              <div className="space-y-3 pt-4 border-t border-nexus-border/30">
                <label className="text-sm font-medium text-neon-green flex items-center gap-2">
                  <SettingsIcon className="w-3.5 h-3.5" />
                  Workspace & Security
                </label>
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">
                      Workspace Root
                    </label>
                    <input
                      type="text"
                      value={workspaceDir}
                      onChange={(e) => setWorkspaceDir(e.target.value)}
                      placeholder="./"
                      className="w-full bg-nexus-black border border-nexus-border rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-neon-green transition-all"
                    />
                    <p className="text-[9px] text-slate-500 italic">
                      Root directory for relative paths (e.g. ./ ).
                    </p>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">
                      Allowed Paths (one per line)
                    </label>
                    <textarea
                      value={allowedPaths.join("\n")}
                      onChange={(e) =>
                        setAllowedPaths(e.target.value.split("\n"))
                      }
                      rows={3}
                      placeholder="/home/user/project"
                      className="w-full bg-nexus-black border border-nexus-border rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-neon-green transition-all font-mono"
                    />
                    <p className="text-[9px] text-slate-500 italic">
                      Security sandbox whitelist. Tools can only access files
                      here.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <SystemPromptEditor />
          )}
        </div>

        {/* Footer - Only show generic save if in general tab (Prompt editor has its own save) */}
        {activeTab === "general" && (
          <div className="p-6 border-t border-nexus-border bg-nexus-black/30 flex justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-slate-400 hover:text-white transition-colors text-sm"
            >
              Close
            </button>
            <button
              onClick={handleSave}
              className="px-6 py-2 bg-neon-blue/10 hover:bg-neon-blue/20 text-neon-blue border border-neon-blue/50 rounded-lg flex items-center gap-2 transition-all hover:shadow-[0_0_15px_rgba(0,243,255,0.2)]"
            >
              <Save size={16} />
              Save Configuration
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
};

const SettingsIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.72v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);
