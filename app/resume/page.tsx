import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Resume | Portfolio',
    description: 'Resume and CV',
};

export default function Resume() {
    return (
        <section>
            <h1 style={{ marginBottom: '2rem' }}>Resume</h1>
            <div style={{
                textAlign: 'center',
                padding: '4rem 2rem',
                color: 'var(--accent)',
            }}>
                <p style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Coming Soon</p>
                <p style={{ fontSize: '1rem' }}>My resume will be available here shortly.</p>
            </div>
        </section>
    );
}
