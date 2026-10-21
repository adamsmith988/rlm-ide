import { OMNIROUTE_AUTO_MODEL_ID, type OmniRouteView } from "@rlm/contracts";
import { call, errText } from "./api";
import { useApp } from "./store";

/** Makes a model the agent's. Picking "OmniRoute" before its free models were tested starts the gateway and tests them. */
export async function selectAgentModel(id: string): Promise<void> {
  const { settings, updateSettings, toast } = useApp.getState();
  await updateSettings({ roles: { agent: id } });
  if (id !== OMNIROUTE_AUTO_MODEL_ID || settings?.omniroute.chain.length) return;
  try {
    const v = await call<OmniRouteView>("omniroute:state", {});
    if (v.busy) return toast(`OmniRoute: ${v.busy}. It switches to the best free model once that finishes.`);
    if (v.phase !== "running") await call("omniroute:service", { action: "start" });
    toast("Testing which free models answer; the agent uses the best one when that finishes");
    await call("omniroute:setup", { models: [] });
    toast("OmniRoute is ready: the agent starts on the best free model and switches when one fails");
  } catch (e) {
    toast(`OmniRoute: ${errText(e)}`, "error");
  }
}
