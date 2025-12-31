import Link from 'next/link';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Projects | Portfolio',
    description: 'Selected projects and research.',
};

const projects = [
    {
        title: 'Tinker',
        description: 'An experiment in efficient LLM inference and reasoning chains.',
        link: 'https://github.com/example/tinker' // Placeholder
    },
    {
        title: 'Thinking Machines',
        description: 'A minimal portfolio inspired by academic aesthetics. Built with Next.js.',
        link: '#' // Current site
    },
    {
        title: 'Distributed Training Utils',
        description: 'Python library for optimizing PyTorch distributed training jobs on slurm clusters.',
        link: 'https://github.com/example/dist-utils' // Placeholder
    },
    {
        title: 'Vision-Language Adapter',
        description: 'Research code for adapting CLIP models to novel tasks via prompt tuning.',
        link: 'https://github.com/example/vla'
    }
];

export default function Projects() {
    return (
        <section>
            <h1 style={{ marginBottom: '3rem' }}>Projects</h1>
            <div className="project-grid">
                {projects.map((project, index) => (
                    <div key={index} className="project-card">
                        <h3 className="project-title">
                            <a href={project.link} target="_blank" rel="noopener noreferrer">
                                {project.title}
                            </a>
                        </h3>
                        <p className="project-description">{project.description}</p>
                    </div>
                ))}
            </div>
        </section>
    );
}
