import { ReplayList } from "@/components/ReplayView";

export default function ReplaysPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Recorded runs</h1>
      <p className="text-sm text-muted">
        A replay streams a recorded run at its original pace. Use it when the network or the Gemini quota fails during a demo. Replays are always
        labelled.
      </p>
      <ReplayList />
    </div>
  );
}
