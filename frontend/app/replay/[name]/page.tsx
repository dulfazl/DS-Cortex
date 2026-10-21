import { ReplayView } from "@/components/ReplayView";

export default async function ReplayPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  return <ReplayView name={name} />;
}
