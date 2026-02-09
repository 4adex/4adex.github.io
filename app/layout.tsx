import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Link from 'next/link';
import ThemeToggle from './components/ThemeToggle';
import PageTransition from './components/PageTransition';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });

export const metadata: Metadata = {
  title: 'Portfolio',
  description: 'Student developer and ML researcher portfolio',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                var theme = localStorage.getItem('theme') || 'dark';
                document.body.setAttribute('data-theme', theme);
              })();
            `,
          }}
        />
      </head>
      <body className={`${inter.variable}`} data-theme="dark">
        <header className="header">
          <nav>
            <ul className="nav-list">
              <li><Link href="/">Home</Link></li>
              <li><Link href="/projects">Projects</Link></li>
              <li><Link href="/publications">Publications</Link></li>
              <li><Link href="/blog">Blog</Link></li>
              <li><Link href="/resume">Resume</Link></li>
            </ul>
            <ThemeToggle />
          </nav>
        </header>
        <main>
          <PageTransition>{children}</PageTransition>
        </main>
        <footer className="footer">
          <p>© {new Date().getFullYear()} Adesh Gupta</p>
        </footer>
      </body>
    </html>
  );
}
