'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface Reference {
  id: string;
  authors: string;
  title: string;
  venue?: string;
  year?: string;
  url?: string;
}

interface CitationContextType {
  references: Reference[];
  registerReference: (ref: Reference) => number;
  getNumber: (id: string) => number;
}

const CitationContext = createContext<CitationContextType | null>(null);

export function CitationProvider({ 
  children, 
  references: initialReferences = [] 
}: { 
  children: ReactNode; 
  references?: Reference[];
}) {
  const [references, setReferences] = useState<Reference[]>(initialReferences);
  const [citationOrder, setCitationOrder] = useState<string[]>([]);

  const registerReference = (ref: Reference): number => {
    if (!citationOrder.includes(ref.id)) {
      setCitationOrder(prev => [...prev, ref.id]);
    }
    const existingIndex = citationOrder.indexOf(ref.id);
    return existingIndex !== -1 ? existingIndex + 1 : citationOrder.length + 1;
  };

  const getNumber = (id: string): number => {
    const index = citationOrder.indexOf(id);
    return index !== -1 ? index + 1 : -1;
  };

  useEffect(() => {
    setReferences(initialReferences);
  }, [initialReferences]);

  return (
    <CitationContext.Provider value={{ references, registerReference, getNumber }}>
      {children}
    </CitationContext.Provider>
  );
}

export function useCitations() {
  const context = useContext(CitationContext);
  if (!context) {
    throw new Error('useCitations must be used within a CitationProvider');
  }
  return context;
}

interface CitationProps {
  id: string;
  references: Reference[];
}

export function Citation({ id, references }: CitationProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [number, setNumber] = useState<number | null>(null);
  
  const reference = references.find(r => r.id === id);
  
  useEffect(() => {
    // Find the citation number based on order of appearance
    const allCitations = document.querySelectorAll('[data-citation-id]');
    const ids: string[] = [];
    allCitations.forEach(el => {
      const citId = el.getAttribute('data-citation-id');
      if (citId && !ids.includes(citId)) {
        ids.push(citId);
      }
    });
    const index = ids.indexOf(id);
    setNumber(index !== -1 ? index + 1 : 1);
  }, [id]);

  if (!reference) {
    return <span className="citation citation-error">[?]</span>;
  }

  return (
    <span 
      className="citation-wrapper"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <a 
        href={`#ref-${id}`}
        className="citation"
        data-citation-id={id}
      >
        [{number ?? '?'}]
      </a>
      {isHovered && (
        <span className="citation-tooltip">
          <span className="citation-tooltip-authors">{reference.authors}</span>
          <span className="citation-tooltip-title">"{reference.title}"</span>
          {reference.venue && (
            <span className="citation-tooltip-venue">{reference.venue}</span>
          )}
          {reference.year && (
            <span className="citation-tooltip-year">({reference.year})</span>
          )}
        </span>
      )}
    </span>
  );
}

interface ReferencesListProps {
  references: Reference[];
  citedIds: string[];
}

export function ReferencesList({ references, citedIds }: ReferencesListProps) {
  // Only show references that were actually cited, in order of citation
  const orderedRefs = citedIds
    .map(id => references.find(r => r.id === id))
    .filter((r): r is Reference => r !== undefined);

  if (orderedRefs.length === 0) return null;

  return (
    <section className="references-section">
      <h2>References</h2>
      <ol className="references-list">
        {orderedRefs.map((ref, index) => (
          <li key={ref.id} id={`ref-${ref.id}`} className="reference-item">
            <span className="reference-number">[{index + 1}]</span>
            <span className="reference-content">
              <span className="reference-authors">{ref.authors}.</span>{' '}
              {ref.url ? (
                <a href={ref.url} target="_blank" rel="noopener noreferrer" className="reference-title">
                  "{ref.title}"
                </a>
              ) : (
                <span className="reference-title">"{ref.title}"</span>
              )}
              {ref.venue && <span className="reference-venue">. {ref.venue}</span>}
              {ref.year && <span className="reference-year">, {ref.year}</span>}
              .
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
