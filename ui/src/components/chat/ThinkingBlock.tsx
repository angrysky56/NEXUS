import React, { useState } from 'react';
import { Lightbulb, ChevronDown, ChevronRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface ThinkingBlockProps {
    content: string;
}

export const ThinkingBlock: React.FC<ThinkingBlockProps> = ({ content }) => {
    const [isOpen, setIsOpen] = useState(true);

    if (!content) return null;

    return (
        <div className="my-2 border border-blue-900/40 bg-blue-900/10 rounded-md overflow-hidden">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-2 px-3 py-2 bg-blue-900/20 text-blue-300 text-xs font-medium hover:bg-blue-900/30 transition-colors"
            >
                <Lightbulb size={14} />
                <span>Thinking Process</span>
                <span className="ml-auto opacity-50">
                    {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </span>
            </button>

            {isOpen && (
                <div className="p-3 text-slate-300 text-sm italic font-mono bg-black/20">
                    <ReactMarkdown>{content}</ReactMarkdown>
                </div>
            )}
        </div>
    );
};
