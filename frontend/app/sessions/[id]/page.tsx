import { RunView } from "@/components/RunView";

export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <RunView id={id} />;
}
