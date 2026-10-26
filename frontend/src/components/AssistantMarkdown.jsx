import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./AssistantMarkdown.css";

const plugins = [remarkGfm];
const components = {
  a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
  img: ({ alt }) => <span>{alt || "Image"}</span>,
  table: ({ children }) => <div className="agent-markdown-table"><table>{children}</table></div>,
};

export default function AssistantMarkdown({ children }) {
  return <div className="agent-message-text agent-markdown">
    <ReactMarkdown remarkPlugins={plugins} components={components} skipHtml>{children}</ReactMarkdown>
  </div>;
}
