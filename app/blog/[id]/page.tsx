import { getAllPostIds, getPostData, PostData } from '@/lib/posts';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github-dark.css';
import CodeBlock from '@/app/components/CodeBlock';

type Params = Promise<{ id: string }>;

export async function generateMetadata({ params }: { params: Params }) {
    const { id } = await params;
    const postData = await getPostData(id);
    return {
        title: `${postData.title} | Portfolio`,
        description: postData.description,
    };
}


export default async function Post({ params }: { params: Params }) {
    const { id } = await params;
    const postData = await getPostData(id);

    return (
        <article>
            <div style={{ marginBottom: '2rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '2rem' }}>
                <h1 style={{ marginTop: 0, fontSize: '2.5rem', marginBottom: '0.5rem' }}>{postData.title}</h1>
                <div style={{ fontFamily: 'var(--font-sans)', color: 'var(--accent)', fontSize: '0.9rem' }}>
                    <time>{postData.date}</time>
                </div>
            </div>
            <div className="markdown-content">
                <ReactMarkdown 
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex, rehypeHighlight]}
                    components={{
                        pre: ({ children, ...props }) => (
                            <CodeBlock {...props}>{children}</CodeBlock>
                        )
                    }}
                >
                    {postData.content}
                </ReactMarkdown>
            </div>
        </article>
    );
}

export async function generateStaticParams() {
    const paths = getAllPostIds();
    return paths.map((path) => ({
        id: path.params.id,
    }));
}
