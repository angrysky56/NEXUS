import React from 'react';
import { Plus, MessageSquare } from 'lucide-react';

interface SidebarProps {
    onNewChat?: () => void;
    sessionId?: string;
    sessions?: string[];
    onSelectSession?: (id: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ onNewChat, sessionId, sessions = [], onSelectSession }) => {
    return (
        <div className="w-64 bg-nexus-gray/50 h-screen border-r border-nexus-border flex flex-col backdrop-blur-sm">
            <div className="p-4">
                <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple mb-6 font-mono tracking-wider glow-text">
                    NEXUS // VER.0.2
                </h1>

                <button
                    onClick={onNewChat}
                    className="w-full flex items-center justify-center gap-2 bg-neon-blue/10 hover:bg-neon-blue/20 text-neon-blue border border-neon-blue/50 hover:border-neon-blue p-2 rounded-lg transition-all duration-300 shadow-[0_0_10px_rgba(0,243,255,0.1)] hover:shadow-[0_0_15px_rgba(0,243,255,0.3)]"
                >
                    <Plus size={18} />
                    <span className="font-mono text-sm">INITIALIZE_SESSION</span>
                </button>
            </div>

            <div className="flex-1 overflow-y-auto px-2 space-y-1">
                <div className="text-[10px] font-bold text-slate-500 mb-2 px-2 uppercase tracking-widest">Active Threads</div>

                {sessions.length === 0 && !sessionId && (
                    <div className="px-4 py-2 text-xs text-slate-600 italic">No previous sessions</div>
                )}

                {/* Always show current if not in list */}
                {sessionId && !sessions.includes(sessionId) && (
                    <div className="group flex items-center gap-3 p-2 rounded-lg bg-neon-purple/10 text-white cursor-default border border-neon-purple/30 shadow-[0_0_10px_rgba(157,0,255,0.1)]">
                        <MessageSquare size={16} className="text-neon-purple" />
                        <span className="truncate text-sm font-medium">Current Session</span>
                    </div>
                )}

                {sessions.map(s => (
                    <div
                        key={s}
                        onClick={() => onSelectSession?.(s)}
                        className={`group flex items-center gap-3 p-2 rounded-lg transition-all border ${
                            s === sessionId
                                ? 'bg-neon-purple/10 text-white border-neon-purple/30'
                                : 'hover:bg-white/5 text-slate-400 hover:text-white border-transparent hover:border-white/10'
                        } cursor-pointer`}
                    >
                        <MessageSquare size={16} className={`${s === sessionId ? 'text-neon-purple' : 'group-hover:text-neon-purple'} transition-colors`} />
                        <span className={`truncate text-sm ${s === sessionId ? 'font-medium' : 'font-light'}`}>
                            {s.slice(0, 8)}...
                        </span>
                    </div>
                ))}
            </div>

            <div className="p-4 border-t border-nexus-border bg-nexus-black/30 space-y-2">
                <div className="text-[10px] text-slate-600 font-mono text-center pt-2">
                    HYBRID ENGINE ONLINE
                </div>
            </div>
        </div>
    );
};
