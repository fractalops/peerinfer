<script lang="ts">
  import { Button } from "@ui-sdk/svelte/button";
  import {
    Dialog,
    DialogBackdrop,
    DialogContent,
    DialogTitle,
    DialogDescription,
    DialogClose,
  } from "@ui-sdk/svelte/dialog";
  import type { Member } from "../types";
  import { shortId } from "../types";
  import { MODELS, layerLabel, teamRanges } from "../config";
  import {
    count,
    executionMode,
    relativeSeen,
    reportedDevice,
    teamDevices,
    workerStatus,
    type NetworkHistory,
  } from "../network-state";
  import Icon from "./Icon.svelte";
  interface Props {
    id: string | null;
    member?: Member;
    selfId: string;
    history: NetworkHistory;
    now: number;
    canChat: boolean;
    onclose: () => void;
    onchat: (model: string) => void;
  }
  let { id, member, selfId, history, now, canChat, onclose, onchat }: Props =
    $props();
  const model = $derived(MODELS.find((m) => m.id === member?.info.model));
  const devices = $derived(member ? teamDevices(member) : 1);
  const team = $derived(devices > 1);
  const events = $derived(
    history.events.filter((event) => event.worker === id).slice(0, 6),
  );
  const time = (at: number) =>
    new Date(at).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
</script>

<Dialog
  open={id !== null}
  onOpenChange={(open) => {
    if (!open) onclose();
  }}
>
  <DialogBackdrop class="dialog-backdrop" />
  <DialogContent class="dialog worker-details" initialFocus="content">
    <div class="dialog-top">
      <span class="eyebrow">WORKER DETAILS</span><DialogClose
        class="icon-button"
        aria-label="Close worker details"><Icon name="close" /></DialogClose
      >
    </div>
    <DialogTitle>{model?.name ?? "Worker"}</DialogTitle>
    <DialogDescription
      >{id ? shortId(id) : ""}{id === selfId
        ? " · This device"
        : ""}</DialogDescription
    >
    {#if member}
      <div class="detail-status">
        <span
          class="state-pill"
          class:busy={!member.stale && member.info.state === "busy"}
          ><i
            class="status-dot"
            class:online={workerStatus(member) === "Available"}
          ></i>{workerStatus(member)}</span
        ><span>Seen {relativeSeen(member.lastSeenMs, now).toLowerCase()}</span>
      </div>
      <dl class="worker-facts">
        <div>
          <dt>Execution</dt>
          <dd>{executionMode(member)}</dd>
        </div>
        <div>
          <dt>Model runtime</dt>
          <dd>
            {model
              ? model.runtime === "core"
                ? "peerinfer-core · WebGPU"
                : "WebLLM · WebGPU"
              : "Unknown"}
          </dd>
        </div>
        <div class="wide">
          <dt>Reported device</dt>
          <dd>
            {reportedDevice(member.info.adapter) || "Not reported"}
          </dd>
        </div>
        <div>
          <dt>Running / queued</dt>
          <dd>
            {member.stale
              ? "—"
              : `${count(member.info.active)} / ${count(member.info.waiting)}`}
          </dd>
        </div>
        <div>
          <dt>Last generation rate</dt>
          <dd>
            {member.stale || !member.info.tokensPerSec
              ? "—"
              : `${count(member.info.tokensPerSec).toFixed(1)} tokens/s`}
          </dd>
        </div>
        <div>
          <dt>Completed requests</dt>
          <dd>{count(member.info.served).toLocaleString()}</dd>
        </div>
        <div>
          <dt>Tokens generated</dt>
          <dd>{count(member.info.tokensOut).toLocaleString()}</dd>
        </div>
        <div class="wide">
          <dt>Worker ID</dt>
          <dd class="endpoint-id">{member.id}</dd>
        </div>
      </dl>
      {#if team}
        <section class="execution-detail" aria-labelledby="execution-title">
          <h3 id="execution-title">Shared inference</h3>
          <p>
            One coordinator accepts the request. {devices} devices run the model's
            layers in sequence.
          </p>
          {#if model?.shared && model.layers}
            <ol
              class="execution-chain"
              aria-label="Reported team configuration"
            >
              {#each teamRanges(model.layers, devices) as [start, end], i (start)}
                <li>
                  <span class="stage-number">{String(i + 1).padStart(2, "0")}</span>
                  <div>
                    <strong>{i === 0 ? "Coordinator" : `Device ${i + 1}`}</strong><span
                      >{layerLabel(start, end)}</span
                    >{#if i === 0}<code>{shortId(member.id)}</code>{:else}<small
                        >Peer ID not advertised</small
                      >{/if}
                  </div>
                </li>
              {/each}
            </ol>
          {/if}
          <p class="detail-note">
            Worker-reported configuration. Individual stage health is not
            advertised; the state above applies to the team.
          </p>
        </section>
      {/if}
      <section class="worker-recent" aria-labelledby="worker-recent-title">
        <h3 id="worker-recent-title">Recent activity</h3>
        {#if events.length}<ol>
            {#each events as event (event.id)}<li>
                <time datetime={new Date(event.at).toISOString()}
                  >{time(event.at)}</time
                ><span>{event.message}</span>
              </li>{/each}
          </ol>{:else}<p>No changes observed in this tab yet.</p>{/if}
      </section>
      <p class="detail-note">
        Counts are reported by this worker. Generation rate is from its last
        completed request, not current GPU utilization.
      </p>
      <div class="worker-detail-actions">
        <p>The network chooses a worker for this model.</p>
        <Button
          class="button primary"
          disabled={!canChat}
          onclick={() => onchat(member!.info.model)}
          >Chat with this model <Icon name="chat" size={16} /></Button
        >
      </div>
    {:else}
      <div class="worker-gone">
        <Icon name="network" size={28} />
        <h3>Worker left view</h3>
        <p>
          This worker is no longer in the roster. It may have closed its browser
          or lost its connection.
        </p>
        <DialogClose class="button subtle">Back to workers</DialogClose>
      </div>
    {/if}
  </DialogContent>
</Dialog>
