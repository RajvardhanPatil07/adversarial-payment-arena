import { ArrowRight, Bot, CircleCheckBig, GitBranch, RefreshCcw, ShieldCheck } from "lucide-react";
import Link from "next/link";

const STAGES = [
  {
    icon: Bot,
    title: "Generate the attack",
    copy: "An AI red team executes one of 14 fraud families and adapts from defense feedback.",
  },
  {
    icon: CircleCheckBig,
    title: "Check each payment",
    copy: "The plausibility gate rejects transactions that break fraud economics, metadata, or payment-rail rules.",
  },
  {
    icon: GitBranch,
    title: "Expose the network",
    copy: "Velocity, anomaly, and entity-graph signals reveal coordinated behavior that looks normal in isolation.",
  },
  {
    icon: RefreshCcw,
    title: "Learn without poisoning",
    copy: "Before retraining, a separate fidelity gate decides whether a batch of synthetic escapes resembles real fraud.",
  },
];

export function JudgeContext() {
  return (
    <section className="judge-context" aria-labelledby="judge-context-title">
      <div className="judge-context__intro">
        <div>
          <h2 id="judge-context-title">What this project is testing</h2>
          <p>
            Can an autonomous red team improve payment defenses without teaching the model to win only against unrealistic synthetic fraud?
          </p>
        </div>
        <div className="judge-context__watch">
          <ShieldCheck aria-hidden="true" />
          <p><b>What to watch:</b> individually plausible payments become suspicious when shared devices and IP addresses connect them into one ring.</p>
        </div>
      </div>

      <ol className="judge-flow" aria-label="Experiment stages">
        {STAGES.map(({ icon: Icon, title, copy }, index) => (
          <li key={title}>
            <div className="judge-flow__icon"><Icon aria-hidden="true" /></div>
            <div>
              <span>Stage {index + 1}</span>
              <h3>{title}</h3>
              <p>{copy}</p>
            </div>
            {index < STAGES.length - 1 && <ArrowRight className="judge-flow__arrow" aria-hidden="true" />}
          </li>
        ))}
      </ol>

      <div className="gate-distinction">
        <p><b>Plausibility gate</b><span>Runs live on every generated payment.</span></p>
        <p><b>Fidelity gate</b><span>Runs offline on attack batches before they can influence retraining.</span></p>
        <Link href="/evidence">See the closed-loop evidence <ArrowRight aria-hidden="true" /></Link>
      </div>
    </section>
  );
}
