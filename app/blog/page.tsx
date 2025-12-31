import Link from 'next/link';
import { getSortedPostsData } from '@/lib/posts';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Blog | Portfolio',
    description: 'Writings on ML and Systems',
};

export default function BlogIndex() {
    const allPostsData = getSortedPostsData();

    return (
        <section>
            <h1 style={{ marginBottom: '3rem' }}>All Posts</h1>
            <ul className="post-list">
                {allPostsData.map(({ id, date, title, description }) => (
                    <li className="post-item" key={id}>
                        <span className="post-date">{date}</span>
                        <h3 className="post-title">
                            <Link href={`/blog/${id}`}>{title}</Link>
                        </h3>
                        <p className="post-description">{description}</p>
                    </li>
                ))}
            </ul>
        </section>
    );
}
