import React, { useEffect, useState } from 'react';
import { Activity, Brain, Compass, Layers, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceLine, Tooltip } from 'recharts';

import { type CognitiveState } from '../../api';

export const CognitiveDashboard: React.FC = () => {
    const [state, setState] = useState<CognitiveState>({
        valence: 0,
        arousal: 0,
        intrinsic_dimension: 0,
        gate_value: 0.5,
        primary_manifold: "neutral"
    });

    useEffect(() => {
        const eventSource = new EventSource('http://localhost:8000/api/v1/cognitive/status');

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setState(prev => ({ ...prev, ...data }));
            } catch (e) {
                console.error("Dashboard parse error", e);
            }
        };

        return () => eventSource.close();
    }, []);

    const emotionData = [{ x: state.valence, y: state.arousal }];
    const isLogic = state.primary_manifold === 'logic';

    return (
        <div className="p-4 flex flex-col gap-6 h-full text-xs font-mono">
            {/* Emotional Compass */}
            <div className="glass-panel rounded-xl p-4 relative overflow-hidden group">
                <div className="absolute inset-0 bg-neon-blue/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />

                <div className="flex items-center justify-between mb-4 relative z-10">
                    <span className="text-neon-blue font-bold tracking-widest flex items-center gap-2 uppercase text-[10px]">
                        <Compass size={14} /> Affective State
                    </span>
                    <span className="text-slate-400">{state.valence.toFixed(2)} / {state.arousal.toFixed(2)}</span>
                </div>

                <div className="h-48 w-full relative z-10">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.5} />
                            <XAxis type="number" dataKey="x" domain={[-1, 1]} hide />
                            <YAxis type="number" dataKey="y" domain={[-1, 1]} hide />
                            <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                            <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
                            <Scatter name="State" data={emotionData} fill="#00f3ff">
                                <animate attributeName="r" values="3;6;3" dur="2s" repeatCount="indefinite" />
                            </Scatter>
                            <Tooltip
                                contentStyle={{ backgroundColor: '#0a0a0f', borderColor: '#2a2a35' }}
                                itemStyle={{ color: '#00f3ff' }}
                            />
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Fractal Dimension */}
            <div className="glass-panel rounded-xl p-4 relative overflow-hidden">
                <div className="flex items-center justify-between mb-2">
                     <span className="text-neon-purple font-bold tracking-widest flex items-center gap-2 uppercase text-[10px]">
                        <Activity size={14} /> Fractal Entropy
                    </span>
                    <span className="text-neon-purple glow-text">{state.intrinsic_dimension.toFixed(2)} ID</span>
                </div>

                <div className="w-full bg-nexus-black border border-nexus-border h-3 rounded-full overflow-hidden relative mt-2">
                    <motion.div
                        className="bg-neon-purple h-full shadow-[0_0_10px_rgba(189,0,255,0.8)]"
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(100, (state.intrinsic_dimension / 3) * 100)}%` }}
                        transition={{ type: "spring", stiffness: 50 }}
                    />
                    {/* Edge of Chaos Marker */}
                    <div className="absolute top-0 bottom-0 w-0.5 bg-alert-yellow/50 z-20" style={{ left: '60%' }} title="Criticality" />
                </div>
                <div className="flex justify-between text-[10px] text-slate-600 mt-1.5 uppercase tracking-wider">
                    <span>Ordered</span>
                    <span className="text-alert-yellow">Critical</span>
                    <span>Chaotic</span>
                </div>
            </div>

            {/* Manifold Gate */}
             <div className="glass-panel rounded-xl p-4 relative overflow-hidden">
                <div className="flex items-center justify-between mb-3">
                     <span className="text-slate-200 font-bold tracking-widest flex items-center gap-2 uppercase text-[10px]">
                        <Layers size={14} /> Bicameral Gate
                    </span>
                    <motion.span
                        key={state.primary_manifold}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`font-bold ${isLogic ? 'text-neon-blue' : 'text-neon-pink'} glow-text`}
                    >
                        {state.primary_manifold.toUpperCase()}
                    </motion.span>
                </div>

                <div className="flex items-center gap-2">
                    <Brain size={16} className={isLogic ? 'text-neon-blue' : 'text-slate-700'} />

                    <div className="flex-1 bg-nexus-black border border-nexus-border h-6 rounded-md overflow-hidden flex relative">
                        <motion.div
                            className="bg-neon-blue h-full flex items-center justify-center relative overflow-hidden"
                            animate={{ width: `${state.gate_value * 100}%` }}
                            transition={{ type: "spring", bounce: 0, duration: 0.5 }}
                        >
                            <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,0.1)_50%,transparent_75%,transparent_100%)] bg-[length:10px_10px]" />
                        </motion.div>
                        <motion.div
                            className="bg-neon-pink h-full flex items-center justify-center relative overflow-hidden"
                            animate={{ width: `${(1 - state.gate_value) * 100}%` }}
                            transition={{ type: "spring", bounce: 0, duration: 0.5 }}
                        >
                             <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,0.1)_50%,transparent_75%,transparent_100%)] bg-[length:10px_10px]" />
                        </motion.div>

                        {/* Slider Handle */}
                        <motion.div
                            className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_10px_white] z-10"
                            animate={{ left: `${state.gate_value * 100}%` }}
                        />
                    </div>

                    <Zap size={16} className={!isLogic ? 'text-neon-pink' : 'text-slate-700'} />
                </div>

                 <div className="flex justify-between text-[10px] text-slate-500 mt-2 px-6">
                    <span>LOGIC</span>
                    <span>CREATIVE</span>
                </div>
            </div>
        </div>
    );
};
