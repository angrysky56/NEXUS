import React from 'react';
import { Sidebar } from './Sidebar';


interface LayoutProps {
    children: React.ReactNode;
    rightPanel?: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children, rightPanel }) => {
    return (
        <div className="flex h-screen bg-nexus-black text-slate-100 font-sans overflow-hidden selection:bg-neon-blue/30">
            {/* Background Ambient Glow */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-neon-blue/5 blur-[120px] rounded-full" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-neon-purple/5 blur-[120px] rounded-full" />
            </div>

            {/* Sidebar Pane (Z-10 to stay above ambient glow) */}
            <div className="relative z-10 shrink-0">
                <Sidebar
                    onNewChat={() => window.location.reload()}
                />
            </div>

            {/* Main Content Pane */}
            <main className="flex-1 flex flex-col relative z-10 min-w-0 bg-nexus-black/50 backdrop-blur-sm">
                 <div className="absolute inset-0 bg-[linear-gradient(rgba(18,18,26,0.5)_1px,transparent_1px),linear-gradient(90deg,rgba(18,18,26,0.5)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />
                {children}
            </main>

            {/* Right Telemetry Pane */}
            {rightPanel && (
                <div className="w-[400px] shrink-0 border-l border-nexus-border bg-nexus-gray/30 backdrop-blur-md relative z-10 hidden xl:flex flex-col">
                    <div className="p-3 border-b border-nexus-border bg-nexus-black/40">
                        <h2 className="text-xs font-mono text-neon-blue uppercase tracking-widest flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-neon-blue animate-pulse" />
                            Cognitive Telemetry
                        </h2>
                    </div>
                    <div className="flex-1 overflow-y-auto custom-scrollbar">
                        {rightPanel}
                    </div>
                </div>
            )}

            {/* Modals */}
            {/* Settings managed by ChatArea now */}
        </div>
    );
};
