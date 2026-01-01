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
      <body className={`${inter.variable}`}>
        <header className="header">
          <nav>
            <ul className="nav-list">
              <li><Link href="/">Home</Link></li>
              <li><Link href="/projects">Projects</Link></li>
              <li><Link href="/blog">Blog</Link></li>
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
