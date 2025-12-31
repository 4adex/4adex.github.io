'use client';

import { usePathname } from 'next/navigation';
import { useEffect, useState, useRef, ReactNode } from 'react';

interface PageTransitionProps {
    children: ReactNode;
}

export default function PageTransition({ children }: PageTransitionProps) {
    const pathname = usePathname();
    const [isVisible, setIsVisible] = useState(true);
    const previousPathname = useRef(pathname);

    useEffect(() => {
        // Only animate if pathname actually changed
        if (previousPathname.current !== pathname) {
            setIsVisible(false);
            const timeout = setTimeout(() => {
                setIsVisible(true);
            }, 50);
            previousPathname.current = pathname;
            return () => clearTimeout(timeout);
        }
    }, [pathname]);

    return (
        <div className={`page-transition ${isVisible ? 'page-enter' : 'page-exit'}`}>
            {children}
        </div>
    );
}
