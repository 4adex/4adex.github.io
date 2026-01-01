'use client';

import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import CodeBlock from './CodeBlock';
import { Citation, ReferencesList, Reference } from './Citation';

interface MarkdownRendererProps {
  content: string;
  references?: Reference[];
}

export default function MarkdownRenderer({ content, references = [] }: MarkdownRendererProps) {
  // Extract citation IDs in order of appearance
  const citedIds = useMemo(() => {
    const citationRegex = /\[cite:([^\]]+)\]/g;
    const ids: string[] = [];
    let match;
    while ((match = citationRegex.exec(content)) !== null) {
      const id = match[1];
      if (!ids.includes(id)) {
        ids.push(id);
      }
    }
    return ids;
  }, [content]);

  // Process content to replace citation syntax with placeholders
  const processedContent = useMemo(() => {
    return content.replace(/\[cite:([^\]]+)\]/g, '%%CITE:$1%%');
  }, [content]);

  // Custom component to render text with citations
  const renderTextWithCitations = (text: string) => {
    const parts = text.split(/(%%CITE:[^%]+%%)/g);
    return parts.map((part, index) => {
      const match = part.match(/%%CITE:([^%]+)%%/);
      if (match) {
        const id = match[1];
        return <Citation key={`${id}-${index}`} id={id} references={references} />;
      }
      return part;
    });
  };

  // Process children recursively to handle citations in text
  const processChildren = (children: React.ReactNode): React.ReactNode => {
    return React.Children.map(children, (child) => {
      if (typeof child === 'string') {
        return renderTextWithCitations(child);
      }
      if (React.isValidElement<{ children?: React.ReactNode }>(child)) {
        if (child.props.children) {
          return React.cloneElement(child, {
            children: processChildren(child.props.children)
          });
        }
      }
      return child;
    });
  };

  return (
    <>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          pre: ({ children, ...props }) => (
            <CodeBlock {...props}>{children}</CodeBlock>
          ),
          p: ({ children, ...props }) => {
            return <p {...props}>{processChildren(children)}</p>;
          },
          li: ({ children, ...props }) => {
            return <li {...props}>{processChildren(children)}</li>;
          },
          td: ({ children, ...props }) => {
            return <td {...props}>{processChildren(children)}</td>;
          }
        }}
      >
        {processedContent}
      </ReactMarkdown>
      
      {references.length > 0 && citedIds.length > 0 && (
        <ReferencesList references={references} citedIds={citedIds} />
      )}
    </>
  );
}
