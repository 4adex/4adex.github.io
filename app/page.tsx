import Link from 'next/link';
import { getSortedPostsData } from '@/lib/posts';
import GitHubGraph from './components/GitHubGraph';

const GitHubIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
    </svg>
);

const LinkedInIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
    </svg>
);

const TwitterIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
    </svg>
);

const EmailIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect width="20" height="16" x="2" y="4" rx="2"/>
        <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
    </svg>
);

export default function Home() {
    const allPostsData = getSortedPostsData();
    const recentPosts = allPostsData.slice(0, 3); // Top 3

    return (
        <section>
            {/* Hero Section */}
            <div className="hero-section">
                <div className="hero-photo">
                    <img src="/assets/photo.png" alt="Profile photo" />
                </div>
                <div className="hero-info">
                    <h1 className="hero-name">Adesh Gupta</h1>
                    <p className="hero-title">Developer & ML Researcher</p>
                    <div className="hero-socials">
                        <a href="https://github.com/4adex" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                            <GitHubIcon />
                        </a>
                        <a href="https://www.linkedin.com/in/adesh-g/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                            <LinkedInIcon />
                        </a>
                        <a href="https://x.com/a4dex" target="_blank" rel="noopener noreferrer" aria-label="Twitter">
                            <TwitterIcon />
                        </a>
                        <a href="mailto:adeshgupta101@gmail.com" aria-label="Email">
                            <EmailIcon />
                        </a>
                    </div>
                </div>
            </div>

            <div style={{ marginBottom: '4rem' }}>
                <h1>Thinking, Building, Sharing.</h1>
                <p>
                    Hi, I am a student developer and machine learning researcher.
                </p>
                <p>
                    I work at the intersection of systems and AI, exploring how efficient compute can unlock new capabilities.
                    This site is a collection of my technical notes, research logs, and occasional rants.
                </p>
            </div>

            <section style={{ marginBottom: '4rem' }}>
                <h2 style={{ fontSize: '1.2rem', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1.5rem', fontFamily: 'var(--font-sans)' }}>Achievements</h2>
                <div className="timeline">
                    <div className="timeline-item">
                        <h3 className="timeline-title">Silver in Inter IIT Techmeet 14.0</h3>
                        <p className="timeline-description">1st runner up in ISRO Geospatial VLM problem statement among 24 IITs.</p>
                    </div>
                    <div className="timeline-item">
                        <h3 className="timeline-title">HiLabs AIQuest Winner</h3>
                        <p className="timeline-description">Won first prize in HiLabs AIQuest hackathon, organized in IIT Roorkee.</p>
                    </div>
                    <div className="timeline-item">
                        <h3 className="timeline-title">Google Summer of Code 2025</h3>
                        <p className="timeline-description">Selected for GSoC 2025 and contributed in Graphite.</p>
                    </div>
                    <div className="timeline-item">
                        <h3 className="timeline-title">ICLR 2025 BlogPost Track</h3>
                        <p className="timeline-description">Blog on positional embeddings selected for ICLR 2025.</p>
                    </div>
                    <div className="timeline-item">
                        <h3 className="timeline-title">Calimero x Starknet Hackathon</h3>
                        <p className="timeline-description">2nd runner up in privacy focused hackathon, made a game in Rust.</p>
                    </div>
                    <div className="timeline-item">
                        <h3 className="timeline-title">ETHOnline 2024</h3>
                        <p className="timeline-description">Won the DIMO and Sign protocol sponsor tracks.</p>
                    </div>
                </div>
            </section>

            <section>
                <h2 style={{ fontSize: '1.2rem', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1.5rem', fontFamily: 'var(--font-sans)' }}>Recent Writing</h2>
                <ul className="post-list">
                    {recentPosts.map(({ id, date, title, description }) => (
                        <li className="post-item" key={id}>
                            <span className="post-date">{date}</span>
                            <h3 className="post-title">
                                <Link href={`/blog/${id}`}>{title}</Link>
                            </h3>
                            <p className="post-description">{description}</p>
                        </li>
                    ))}
                </ul>
                <div style={{ marginTop: '2rem' }}>
                    <Link href="/blog" style={{ fontFamily: 'var(--font-sans)', fontSize: '0.9rem' }}>View all posts &rarr;</Link>
                </div>
            </section>

            {/* GitHub Contribution Graph */}
            <GitHubGraph username="4adex" />
        </section>
    );
}
