import { RecentSessions } from "@/components/RecentSessions";
import { UploadForm } from "@/components/UploadForm";

const PILLARS = [
  {
    title: "Deliberative",
    text: "Four agents state positions on the candidate pipelines, a critic attacks each one, and a fixed resolution table decides. Every claim needs a tool run.",
  },
  {
    title: "Evidence and confidence",
    text: "Each recommendation carries the executed evidence and a confidence computed from experiments, never stated by the language model.",
  },
  {
    title: "Adaptive memory",
    text: "Dataset fingerprints suggest pipelines that worked on similar past data. Suggestions are advice the critic can reject.",
  },
];

export default function Home() {
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Upload a CSV and describe the problem</h1>
        <p className="mt-1 text-sm text-muted">
          DS Cortex audits the data, picks the task, builds and debates candidate pipelines, and returns the best model with plots, a decision
          ledger and code that reproduces the score.
        </p>
        <div className="mt-5">
          <UploadForm />
        </div>
      </div>
      <aside className="space-y-4">
        <section className="card p-4">
          <h2 className="text-sm font-semibold">How a run works</h2>
          <ul className="mt-3 space-y-3">
            {PILLARS.map((p) => (
              <li key={p.title}>
                <p className="text-sm font-semibold">{p.title}</p>
                <p className="text-sm text-muted">{p.text}</p>
              </li>
            ))}
          </ul>
        </section>
        <RecentSessions />
      </aside>
    </div>
  );
}
