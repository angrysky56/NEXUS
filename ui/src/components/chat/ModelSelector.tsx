import React, { useMemo, useState, useEffect } from "react";
import { type Model } from "../../api";
import { ChevronDown, ChevronRight } from "lucide-react";

interface ModelSelectorProps {
  models: Model[];
  currentModel: string;
  onSelect: (modelId: string) => void;
  label?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  currentModel,
  onSelect,
  label,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [openProviders, setOpenProviders] = useState<Record<string, boolean>>(
    {},
  );

  const selected = models.find((m) => m.id === currentModel) || models[0];

  // Group models by provider
  const modelsByProvider = useMemo(() => {
    const grouped: Record<string, Model[]> = {};
    models.forEach((model) => {
      const provider = model.id.split("/")[0];
      if (!grouped[provider]) {
        grouped[provider] = [];
      }
      grouped[provider].push(model);
    });
    return grouped;
  }, [models]);

  // Auto-open the provider of the current model
  useEffect(() => {
    if (currentModel) {
      const provider = currentModel.split("/")[0];
      setOpenProviders((prev) => ({ ...prev, [provider]: true }));
    }
  }, [currentModel, isOpen]); // Re-check when dropdown opens

  const toggleProvider = (provider: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenProviders((prev) => ({ ...prev, [provider]: !prev[provider] }));
  };

  const formatPrice = (price: string) => {
    const num = parseFloat(price);
    if (num === 0) return "Free";
    // Threshold changed to 0.001 (e.g. $1000/1M) to capture almost all paid models in /1M format
    if (num < 0.001) return `$${(num * 1000000).toFixed(2)}/1M`;
    return `$${num.toFixed(6)}`;
  };

  const handleSelect = (modelId: string) => {
    onSelect(modelId);
    setIsOpen(false);
  };

  return (
    <div className="relative flex flex-col gap-1">
      {label && (
        <span className="text-[10px] text-slate-500 font-mono tracking-wider ml-1">
          {label}
        </span>
      )}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        onBlur={(e) => {
          // Only close if focus moved outside dropdown
          if (!e.currentTarget.parentElement?.contains(e.relatedTarget)) {
            setTimeout(() => setIsOpen(false), 150);
          }
        }}
        className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-md text-xs font-medium text-slate-200 transition-colors border border-slate-700 max-w-[200px]"
      >
        <span className="truncate">{selected?.name || "Select Model"}</span>
        <ChevronDown size={12} className="shrink-0" />
      </button>

      {isOpen && (
        <div
          className="absolute top-full left-0 mt-1 w-96 bg-slate-800 border border-slate-700 rounded-md shadow-xl z-50"
          style={{ maxHeight: "400px", overflowY: "scroll" }}
        >
          {models.length === 0 ? (
            <div className="px-4 py-3 text-xs text-slate-500 text-center">
              No models found.
              <br />
              <span className="opacity-75">Check API Key in Settings.</span>
            </div>
          ) : (
            Object.entries(modelsByProvider).map(
              ([provider, providerModels]) => (
                <div
                  key={provider}
                  className="border-b border-slate-700/50 last:border-0"
                >
                  <div
                    onClick={(e) => toggleProvider(provider, e)}
                    className="sticky top-0 bg-slate-900/95 backdrop-blur px-4 py-2 text-[10px] font-bold text-slate-400 uppercase tracking-wider cursor-pointer hover:bg-slate-800 flex items-center justify-between transition-colors"
                  >
                    <span>{provider}</span>
                    {openProviders[provider] ? (
                      <ChevronDown size={10} />
                    ) : (
                      <ChevronRight size={10} />
                    )}
                  </div>

                  {openProviders[provider] && (
                    <div className="bg-slate-800/50">
                      {providerModels.map((model) => (
                        <div
                          key={model.id}
                          onClick={() => handleSelect(model.id)}
                          className={`w-full text-left px-4 py-2 hover:bg-slate-700/80 text-xs border-b border-slate-700/30 cursor-pointer transition-colors ${
                            model.id === currentModel ? "bg-slate-700/60" : ""
                          }`}
                        >
                          <div className="flex items-start justify-between gap-2 mb-0.5">
                            <div className="font-medium text-slate-200 truncate pr-2">
                              {model.name.split("/").pop()}{" "}
                              {/* Show only model name part for compactness */}
                            </div>
                            {model.id === currentModel && (
                              <span className="text-[9px] text-cyan-400 shrink-0">
                                ●
                              </span>
                            )}
                          </div>

                          <div className="flex items-center justify-between text-[9px] text-slate-500">
                            <div className="flex items-center gap-2">
                              <span>
                                {Math.round(model.context_length / 1000)}k
                              </span>
                              <span>{model.supports_tools ? "Tools" : ""}</span>
                            </div>
                            <div className="flex items-center gap-1.5 font-mono opacity-80">
                              <span className="text-green-500/80">
                                In: {formatPrice(model.pricing.prompt)}
                              </span>
                              <span className="text-slate-700">|</span>
                              <span className="text-orange-500/80">
                                Out: {formatPrice(model.pricing.completion)}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ),
            )
          )}
        </div>
      )}
    </div>
  );
};
