import { ArrowRight, Bot, CircleCheckBig, GitBranch, RefreshCcw } from "lucide-react";
import Link from "next/link";

const STAGES = [
  {
    icon: Bot,
    title: "Generate",
    copy: "Adaptive attack",
  },
  {
    icon: CircleCheckBig,
    title: "Validate",
    copy: "Payment plausibility",
  },
  {
    icon: GitBranch,
    title: "Detect",
    copy: "Network behavior",
  },
  {
    icon: RefreshCcw,
    title: "Learn",
    copy: "Fidelity-gated",
  },
];

export function JudgeContext() {
  return (
    <section id="method" className="judge-context" aria-labelledby="judge-context-title">
      <div className="judge-context__intro">
        <div>
          <h2 id="judge-context-title">One closed loop, two independent gates</h2>
          <p>Payment plausibility runs live. Batch fidelity runs before any synthetic escape can influence retraining.</p>
        </div>
        <Link href="/evidence">Inspect the method <ArrowRight aria-hidden="true" /></Link>
      </div>

      <ol className="judge-flow" aria-label="Experiment stages">
        {STAGES.map(({ icon: Icon, title, copy }, index) => (
          <li key={title}>
            <div className="judge-flow__icon"><Icon aria-hidden="true" /></div>
            <div>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <h3>{title}</h3>
              <p>{copy}</p>
            </div>
            {index < STAGES.length - 1 && <ArrowRight className="judge-flow__arrow" aria-hidden="true" />}
          </li>
        ))}
      </ol>

    </section>
  );
}
