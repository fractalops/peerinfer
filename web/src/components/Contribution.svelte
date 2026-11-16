<script lang="ts">
  import { Button } from "@ui-sdk/svelte/button";
  import {
    Dialog,
    DialogTrigger,
    DialogBackdrop,
    DialogContent,
    DialogTitle,
    DialogDescription,
    DialogClose,
  } from "@ui-sdk/svelte/dialog";
  import { Field, FieldLabel } from "@ui-sdk/svelte/field";
  import { Radio, RadioGroup, RadioIndicator } from "@ui-sdk/svelte/radio";
  import {
    SegmentedControl,
    SegmentedControlIndicator,
    SegmentedControlItem,
  } from "@ui-sdk/svelte/segmented-control";
  import { Input } from "@ui-sdk/svelte/input";
  import {
    Progress,
    ProgressTrack,
    ProgressIndicator,
  } from "@ui-sdk/svelte/progress";
  import {
    MAX_TEAM_DEVICES,
    layerLabel,
    teamRanges,
    type ModelSpec,
  } from "../config";
  import type { TeamLayout } from "../runtime/split";
  import type { RuntimeProgress } from "../runtime/types";
  import Icon from "./Icon.svelte";
  interface Props {
    model: ModelSpec;
    shortcut?: boolean;
    progress: RuntimeProgress;
    running: boolean;
    canPause: boolean;
    paused: boolean;
    mode: string;
    invite: string;
    /** The invitation as text for `peerinfer join`. */
    inviteText: string;
    layout?: TeamLayout;
    problem: string | null;
    connected: boolean;
    contributingModel?: string;
    onstart: (mode: "whole" | "shared", devices: number) => void;
    onstop: () => Promise<void>;
    onpause: () => void;
    oncopy: (text: string) => Promise<boolean>;
  }
  let {
    model,
    shortcut = false,
    progress,
    running,
    canPause,
    paused,
    mode,
    invite,
    inviteText,
    layout,
    problem,
    connected,
    contributingModel,
    onstart,
    onstop,
    onpause,
    oncopy,
  }: Props = $props();
  let open = $state(false),
    selectedMode = $state<string | null>("whole"),
    selectedDevices = $state<string | null>("2"),
    copied = $state(false);
  const deviceChoices = $derived(
    Array.from(
      { length: Math.min(MAX_TEAM_DEVICES, model.layers ?? 2) - 1 },
      (_, i) => i + 2,
    ),
  );
  let stopping = $state(false);
  const loading = $derived(
    ["downloading", "preparing", "pairing"].includes(progress.phase),
  );
  const size = $derived(
    model.downloadMB >= 1000
      ? `${(model.downloadMB / 1000).toFixed(2)} GB`
      : `${model.downloadMB} MB`,
  );
  const blocked = $derived(
    !connected ? "Connect to the network before sharing compute." : problem,
  );
  async function stop() {
    stopping = true;
    try {
      await onstop();
    } finally {
      stopping = false;
    }
  }
</script>

<Dialog
  bind:open
  onOpenChange={(next) => {
    if (next) {
      selectedMode = "whole";
      copied = false;
    }
  }}
>
  {#if shortcut}
    <DialogTrigger class="suggestion"
      ><Icon name="device" size={18} /><span
        >{running ? "Manage sharing" : "Share compute"}</span
      ><Icon name="external" size={15} /></DialogTrigger
    >
  {:else}
    <div class="device-card">
      <div class="device-heading">
        <Icon name="device" size={18} /><span>Your device</span><span
          class="status-dot"
          class:online={running &&
            !paused &&
            (progress.phase === "ready" || progress.phase === "busy")}
        ></span>
      </div>
      <strong
        >{running
          ? paused
            ? "Sharing paused"
            : progress.phase === "ready"
              ? "Compute shared"
              : progress.phase === "busy"
                ? "Answering a request"
                : progress.phase === "error"
                  ? "Sharing interrupted"
                  : "Getting ready…"
          : "Here to chat"}</strong
      >
      <p>
        {running
          ? (contributingModel ?? model.name)
          : "No model download needed."}
      </p>
      {#if running}
        {#if loading}<Progress
            class="mini-progress"
            aria-label="Model preparation"
            value={progress.total
              ? ((progress.loaded ?? 0) / progress.total) * 100
              : null}
            ><ProgressTrack><ProgressIndicator /></ProgressTrack></Progress
          >{/if}
        <DialogTrigger class="button subtle full"
          >Manage sharing <Icon name="chevron" size={15} /></DialogTrigger
        >
      {:else}<DialogTrigger class="button subtle full"
          >Share compute <Icon name="plus" size={16} /></DialogTrigger
        >{/if}
    </div>
  {/if}
  <DialogBackdrop class="dialog-backdrop" />
  <DialogContent
    class="dialog contribution-dialog"
    keepMounted
    initialFocus="content"
  >
    <div class="dialog-top">
      <span class="eyebrow">YOUR DEVICE</span><DialogClose
        class="icon-button"
        aria-label="Close compute sharing"><Icon name="close" /></DialogClose
      >
    </div>
    <DialogTitle
      >{running
        ? "Your compute, shared."
        : "Put your browser to work."}</DialogTitle
    >
    <DialogDescription
      >{running
        ? "Keep this tab open and visible while you share. Refreshing stops sharing. You can pause or stop at any time."
        : "Run a model on this device and answer requests from the network. Chatting does not require this."}</DialogDescription
    >
    {#if running}
      <div class="sharing-status" class:error={progress.phase === "error"}>
        <span
          class="status-dot"
          class:online={!paused &&
            (progress.phase === "ready" || progress.phase === "busy")}
        ></span>
        <div>
          <strong>{contributingModel}</strong>
          <p>{progress.message}</p>
        </div>
      </div>
      {#if loading}
        <Progress
          class="download-progress"
          aria-label="Model download"
          value={progress.total
            ? ((progress.loaded ?? 0) / progress.total) * 100
            : null}
          ><ProgressTrack><ProgressIndicator /></ProgressTrack></Progress
        >
        <p class="small muted">
          {progress.total
            ? `${((progress.loaded ?? 0) / 1e6).toFixed(1)} / ${(progress.total / 1e6).toFixed(1)} MB`
            : progress.phase === "pairing"
              ? "Waiting for the other devices to join."
              : "Preparing the model…"}
        </p>
      {/if}
      {#if (mode === "shared" || mode === "helper") && layout}<div
          class="team-diagram"
        >
          {#each layout.ranges as [start, end], i (start)}{#if i > 0}<span
                class="team-connection">↔<small>iroh</small></span
              >{/if}<span
              ><Icon name="device" /><strong
                >{i === layout.mine
                  ? "This device"
                  : i === 0
                    ? "Host"
                    : `Device ${i + 1}`}</strong
              ><small>{layerLabel(start, end)}</small></span
            >{/each}
        </div>{/if}
      {#if invite && progress.phase !== "error"}
        <div class="invite-copy">
          <Field
            ><FieldLabel>Invite the other devices</FieldLabel><Input
              value={invite}
              readonly
              autocomplete="off"
              spellcheck={false}
              onclick={(event) =>
                (event.currentTarget as HTMLInputElement).select()}
            /></Field
          ><Button
            class="button subtle"
            onclick={async () => {
              copied = await oncopy(invite);
            }}
            ><Icon name={copied ? "check" : "copy"} size={16} />{copied
              ? "Copied"
              : "Copy link"}</Button
          >
        </div>
        <p class="small muted">
          Open this link on each device you trust. Every device needs a GPU
          with WebGPU. On a computer with the command-line tool, run
          <code class="invite-command">peerinfer join {inviteText}</code>
        </p>
      {/if}
      <div class="dialog-actions">
        {#if canPause}<Button class="button subtle" onclick={onpause}
            ><Icon name={paused ? "arrow" : "pause"} size={16} />{paused
              ? "Resume sharing"
              : "Pause sharing"}</Button
          >{/if}<Button class="button danger" disabled={stopping} onclick={stop}
          >{stopping ? "Stopping…" : "Stop sharing"}</Button
        >
      </div>
    {:else}
      <div class="model-summary">
        <span class="model-icon"><Icon name="grid" size={24} /></span>
        <div>
          <strong>{model.name}</strong>
          <p>{size} download · saved in this browser</p>
        </div>
      </div>
      <Field
        ><FieldLabel class="field-heading"
          >How would you like to run it?</FieldLabel
        ><RadioGroup bind:value={selectedMode} class="mode-options">
          <label class="mode-option"
            ><Radio value="whole"><RadioIndicator /></Radio><span
              ><strong>On this device</strong><small
                >One browser runs the whole model.</small
              ></span
            ></label
          >
          {#if model.shared}<label class="mode-option"
              ><Radio value="shared"><RadioIndicator /></Radio><span
                ><strong>With other devices <em>Experimental</em></strong
                ><small
                  >Split the model’s layers between browsers or computers.</small
                ></span
              ></label
            >{/if}
        </RadioGroup></Field
      >
      {#if selectedMode === "shared"}<Field
          ><FieldLabel class="field-heading">Devices in the team</FieldLabel
          ><SegmentedControl
            bind:value={selectedDevices}
            class="device-count"
            ><SegmentedControlIndicator class="device-count-indicator" />
            {#each deviceChoices as n (n)}<SegmentedControlItem
                value={String(n)}
                class="device-count-item">{n} devices</SegmentedControlItem
              >{/each}
          </SegmentedControl></Field
        ><p class="small muted">
          Layers per device: {teamRanges(
            model.layers ?? 0,
            Number(selectedDevices ?? 2),
          )
            .map(([s, e]) => `${s}–${e - 1}`)
            .join(" · ")}
        </p><p class="notice">
          This device downloads the whole model and sends each other device the
          files for its layers. Every token passes through all devices, so a
          team can be slower than one device.
        </p>{/if}
      <p class="small muted">
        Sharing uses GPU memory, power, and bandwidth. Keep this tab visible.
        You can stop whenever you need to.
      </p>
      {#if blocked}<p class="inline-error">{blocked}</p>{/if}
      <div class="dialog-actions">
        <DialogClose class="button subtle">Keep chatting</DialogClose><Button
          class="button primary"
          disabled={!!blocked}
          onclick={() => {
            open = false;
            onstart(
              selectedMode === "shared" ? "shared" : "whole",
              Number(selectedDevices ?? 2),
            );
          }}>Start sharing <Icon name="external" size={17} /></Button
        >
      </div>
    {/if}
  </DialogContent>
</Dialog>
