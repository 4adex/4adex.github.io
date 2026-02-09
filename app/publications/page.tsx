import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Publications | Portfolio',
    description: 'Research publications and academic work.',
};

const publications = [
    {
        title: 'Positional Embeddings in Transformer Models: Evolution from Text to Vision Domains',
        venue: 'ICLR 2025 Blogpost Track',
        description: 'This blog explores the role of positional embeddings in transformers, analyzing their transition from text-based models to Vision Transformers (ViTs). It presents an in-depth study of RoPE and ALiBi, focusing on their mathematical foundations, impact on sequence length extrapolation, and performance in ViTs.',
        link: 'https://iclr-blogposts.github.io/2025/blog/positional-embedding/',
        linkText: 'blog',
    },
    {
        title: 'Adaptive Urban Planning: A Hybrid Framework for Balanced City Development',
        venue: 'AAAI AI4UP Workshop',
        description: 'Developed a Multi-Agent urban planning framework leveraging the reasoning and collaborative of different LLM agents. Used Genetic Algorithms and specialized agents for optimization and regional customization and achieved significant improvement in livability and accessibility compared to previous methods.',
        link: 'https://arxiv.org/abs/2412.15349',
        linkText: 'arxiv',
    },
];

export default function Publications() {
    return (
        <section>
            <h1 style={{ marginBottom: '3rem' }}>Publications</h1>
            <div className="publications-list">
                {publications.map((pub, index) => (
                    <div key={index} className="publication-card">
                        <div className="publication-header">
                            <h3 className="publication-title">{pub.title}</h3>
                            <a 
                                href={pub.link} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="publication-link"
                            >
                                {pub.linkText} ↗
                            </a>
                        </div>
                        <p className="publication-venue">Accepted in {pub.venue}</p>
                        <p className="publication-description">{pub.description}</p>
                    </div>
                ))}
            </div>
        </section>
    );
}
