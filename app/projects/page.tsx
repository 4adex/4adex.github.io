import Link from 'next/link';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Projects | Portfolio',
    description: 'Selected projects and research.',
};

const projects = [
    {
        title: 'VortexDB',
        tech: 'Rust',
        description: 'A lightweight vector database built from scratch in Rust, featuring HNSW and KD-Tree indexing algorithms for efficient similarity search.',
        link: 'https://github.com/sdslabs/VortexDB'
    },
    {
        title: 'Drishti',
        tech: 'RayServe, Docker, Huggingface',
        description: 'Scalable remote sensing pipeline integrating EarthMind-4B, RemoteSAM & BERT classifier for unified satellite imagery analysis.',
        link: 'https://github.com/4adex/drishti'
    },
    {
        title: 'Erdos',
        tech: 'PHP, Nginx, Docker',
        description: 'A math problem-solving platform with 1500+ active users. People solve problems, track progress, and compete in annual contests. I help keep it running and add new features.',
        link: 'https://erdos.sdslabs.co/'
    },
    {
        title: 'Library Management System',
        tech: 'Golang, MySQL',
        description: 'Built this to learn Go properly. It\'s a full MVC app with JWT auth, password hashing, and all the things you\'d expect from a library system.',
        link: 'https://github.com/4adex/mvc-golang'
    },
    // {
    //     title: 'Collision Detection using GJK Algorithm',
    //     tech: 'Python, OpenCV, Numpy, Scikit-learn',
    //     description: 'Extracts 3D wireframes from 2D urban images using depth maps and segmentation. The interesting part was implementing vertex and line detection with connected component analysis.',
    //     link: 'https://github.com/4adex/collision-detection'
    // }
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
                        <p className="project-tech">{project.tech}</p>
                        <p className="project-description">{project.description}</p>
                    </div>
                ))}
            </div>
        </section>
    );
}
