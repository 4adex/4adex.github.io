'use client';

import { useState, useRef, ReactNode } from 'react';

interface CodeBlockProps {
    children?: ReactNode;
    className?: string;
}

export default function CodeBlock({ children, className }: CodeBlockProps) {
    const [copied, setCopied] = useState(false);
    const preRef = useRef<HTMLPreElement>(null);

    const handleCopy = async () => {
        const code = preRef.current?.textContent || '';
        await navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <pre ref={preRef} className={className}>
            <button 
                className={`copy-button ${copied ? 'copied' : ''}`}
                onClick={handleCopy}
                aria-label="Copy code"
            >
                {copied ? 'Copied!' : 'Copy'}
            </button>
            {children}
        </pre>
    );
}
