import React, { useMemo, useState } from 'react';
import { type Model } from '../../api';
import { ChevronDown } from 'lucide-react';

interface ModelSelectorProps {
    models: Model[];
    currentModel: string;
    onSelect: (modelId: string) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ models, currentModel, onSelect }) => {
    const [isOpen, setIsOpen] = useState(false);

    const selected = models.find(m => m.id === currentModel) || models[0];

    // Group models by provider
    const modelsByProvider = useMemo(() => {
        const grouped: Record<string, Model[]> = {};
        models.forEach(model => {
            const provider = model.id.split('/')[0];
            if (!grouped[provider]) {
                grouped[provider] = [];
            }
            grouped[provider].push(model);
        });
        return grouped;
    }, [models]);

    const formatPrice = (price: string) => {
        const num = parseFloat(price);
        if (num === 0) return 'Free';
        if (num < 0.000001) return `$${(num * 1000000).toFixed(2)}/1M`;
        return `$${num.toFixed(6)}`;
    };

    const handleSelect = (modelId: string) => {
        onSelect(modelId);
        setIsOpen(false);
    };

    return (
        <div className="relative">
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                onBlur={(e) => {
                    // Only close if focus moved outside dropdown
                    if (!e.currentTarget.parentElement?.contains(e.relatedTarget)) {
                        setTimeout(() => setIsOpen(false), 150);
                    }
                }}
                className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-md text-xs font-medium text-slate-200 transition-colors border border-slate-700"
            >
                <span>{selected?.name || "Select Model"}</span>
                <ChevronDown size={12} />
            </button>

            {isOpen && (
                <div
                    className="absolute top-full left-0 mt-1 w-96 bg-slate-800 border border-slate-700 rounded-md shadow-xl z-50"
                    style={{ maxHeight: '400px', overflowY: 'scroll' }}
                >
                    {models.length === 0 ? (
                        <div className="px-4 py-3 text-xs text-slate-500 text-center">
                            No models found.<br/>
                            <span className="opacity-75">Check API Key in Settings.</span>
                        </div>
                    ) : (
                        Object.entries(modelsByProvider).map(([provider, providerModels]) => (
                            <div key={provider}>
                                <div className="sticky top-0 bg-slate-900 px-4 py-2 text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-700">
                                    {provider}
                                </div>

                                {providerModels.map(model => (
                                    <div
                                        key={model.id}
                                        onClick={() => handleSelect(model.id)}
                                        className={`w-full text-left px-4 py-2.5 hover:bg-slate-700 text-xs border-b border-slate-700/30 cursor-pointer ${
                                            model.id === currentModel ? 'bg-slate-700/50' : ''
                                        }`}
                                    >
                                        <div className="flex items-start justify-between gap-2 mb-1">
                                            <div className="font-bold text-slate-200">{model.name}</div>
                                            {model.id === currentModel && (
                                                <span className="text-[9px] text-cyan-400">●</span>
                                            )}
                                        </div>

                                        <div className="flex items-center gap-3 text-[10px] text-slate-500 mb-1">
                                            <span>{Math.round(model.context_length / 1000)}k ctx</span>
                                            <span>Tools: {model.supports_tools ? '✅' : '❌'}</span>
                                        </div>

                                        <div className="flex items-center gap-2 text-[9px]">
                                            <span className="text-green-500">
                                                In: {formatPrice(model.pricing.prompt)}
                                            </span>
                                            <span className="text-slate-600">|</span>
                                            <span className="text-orange-500">
                                                Out: {formatPrice(model.pricing.completion)}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
};
