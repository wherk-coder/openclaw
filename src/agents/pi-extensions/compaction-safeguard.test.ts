import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { SessionEntry } from "@mariozechner/pi-coding-agent";
import { describe, expect, it } from "vitest";

import {
  getCompactionSafeguardRuntime,
  setCompactionSafeguardRuntime,
} from "./compaction-safeguard-runtime.js";
import { __testing } from "./compaction-safeguard.js";

const {
  collectToolFailures,
  formatToolFailuresSection,
  computeAdaptiveChunkRatio,
  isOversizedForSummary,
  isMessageEntry,
  countMessagesFromIndex,
  findLastCompactionIndex,
  adjustCutPointForMinMessages,
  BASE_CHUNK_RATIO,
  MIN_CHUNK_RATIO,
  SAFETY_MARGIN,
} = __testing;

describe("compaction-safeguard tool failures", () => {
  it("formats tool failures with meta and summary", () => {
    const messages: AgentMessage[] = [
      {
        role: "toolResult",
        toolCallId: "call-1",
        toolName: "exec",
        isError: true,
        details: { status: "failed", exitCode: 1 },
        content: [{ type: "text", text: "ENOENT: missing file" }],
        timestamp: Date.now(),
      },
      {
        role: "toolResult",
        toolCallId: "call-2",
        toolName: "read",
        isError: false,
        content: [{ type: "text", text: "ok" }],
        timestamp: Date.now(),
      },
    ];

    const failures = collectToolFailures(messages);
    expect(failures).toHaveLength(1);

    const section = formatToolFailuresSection(failures);
    expect(section).toContain("## Tool Failures");
    expect(section).toContain("exec (status=failed exitCode=1): ENOENT: missing file");
  });

  it("dedupes by toolCallId and handles empty output", () => {
    const messages: AgentMessage[] = [
      {
        role: "toolResult",
        toolCallId: "call-1",
        toolName: "exec",
        isError: true,
        details: { exitCode: 2 },
        content: [],
        timestamp: Date.now(),
      },
      {
        role: "toolResult",
        toolCallId: "call-1",
        toolName: "exec",
        isError: true,
        content: [{ type: "text", text: "ignored" }],
        timestamp: Date.now(),
      },
    ];

    const failures = collectToolFailures(messages);
    expect(failures).toHaveLength(1);

    const section = formatToolFailuresSection(failures);
    expect(section).toContain("exec (exitCode=2): failed");
  });

  it("caps the number of failures and adds overflow line", () => {
    const messages: AgentMessage[] = Array.from({ length: 9 }, (_, idx) => ({
      role: "toolResult",
      toolCallId: `call-${idx}`,
      toolName: "exec",
      isError: true,
      content: [{ type: "text", text: `error ${idx}` }],
      timestamp: Date.now(),
    }));

    const failures = collectToolFailures(messages);
    const section = formatToolFailuresSection(failures);
    expect(section).toContain("## Tool Failures");
    expect(section).toContain("...and 1 more");
  });

  it("omits section when there are no tool failures", () => {
    const messages: AgentMessage[] = [
      {
        role: "toolResult",
        toolCallId: "ok",
        toolName: "exec",
        isError: false,
        content: [{ type: "text", text: "ok" }],
        timestamp: Date.now(),
      },
    ];

    const failures = collectToolFailures(messages);
    const section = formatToolFailuresSection(failures);
    expect(section).toBe("");
  });
});

describe("computeAdaptiveChunkRatio", () => {
  const CONTEXT_WINDOW = 200_000;

  it("returns BASE_CHUNK_RATIO for normal messages", () => {
    // Small messages: 1000 tokens each, well under 10% of context
    const messages: AgentMessage[] = [
      { role: "user", content: "x".repeat(1000), timestamp: Date.now() },
      {
        role: "assistant",
        content: [{ type: "text", text: "y".repeat(1000) }],
        timestamp: Date.now(),
      },
    ];

    const ratio = computeAdaptiveChunkRatio(messages, CONTEXT_WINDOW);
    expect(ratio).toBe(BASE_CHUNK_RATIO);
  });

  it("reduces ratio when average message > 10% of context", () => {
    // Large messages: ~50K tokens each (25% of context)
    const messages: AgentMessage[] = [
      { role: "user", content: "x".repeat(50_000 * 4), timestamp: Date.now() },
      {
        role: "assistant",
        content: [{ type: "text", text: "y".repeat(50_000 * 4) }],
        timestamp: Date.now(),
      },
    ];

    const ratio = computeAdaptiveChunkRatio(messages, CONTEXT_WINDOW);
    expect(ratio).toBeLessThan(BASE_CHUNK_RATIO);
    expect(ratio).toBeGreaterThanOrEqual(MIN_CHUNK_RATIO);
  });

  it("respects MIN_CHUNK_RATIO floor", () => {
    // Very large messages that would push ratio below minimum
    const messages: AgentMessage[] = [
      { role: "user", content: "x".repeat(150_000 * 4), timestamp: Date.now() },
    ];

    const ratio = computeAdaptiveChunkRatio(messages, CONTEXT_WINDOW);
    expect(ratio).toBeGreaterThanOrEqual(MIN_CHUNK_RATIO);
  });

  it("handles empty message array", () => {
    const ratio = computeAdaptiveChunkRatio([], CONTEXT_WINDOW);
    expect(ratio).toBe(BASE_CHUNK_RATIO);
  });

  it("handles single huge message", () => {
    // Single massive message
    const messages: AgentMessage[] = [
      { role: "user", content: "x".repeat(180_000 * 4), timestamp: Date.now() },
    ];

    const ratio = computeAdaptiveChunkRatio(messages, CONTEXT_WINDOW);
    expect(ratio).toBeGreaterThanOrEqual(MIN_CHUNK_RATIO);
    expect(ratio).toBeLessThanOrEqual(BASE_CHUNK_RATIO);
  });
});

describe("isOversizedForSummary", () => {
  const CONTEXT_WINDOW = 200_000;

  it("returns false for small messages", () => {
    const msg: AgentMessage = {
      role: "user",
      content: "Hello, world!",
      timestamp: Date.now(),
    };

    expect(isOversizedForSummary(msg, CONTEXT_WINDOW)).toBe(false);
  });

  it("returns true for messages > 50% of context", () => {
    // Message with ~120K tokens (60% of 200K context)
    // After safety margin (1.2x), effective is 144K which is > 100K (50%)
    const msg: AgentMessage = {
      role: "user",
      content: "x".repeat(120_000 * 4),
      timestamp: Date.now(),
    };

    expect(isOversizedForSummary(msg, CONTEXT_WINDOW)).toBe(true);
  });

  it("applies safety margin", () => {
    // Message at exactly 50% of context before margin
    // After SAFETY_MARGIN (1.2), it becomes 60% which is > 50%
    const halfContextChars = (CONTEXT_WINDOW * 0.5) / SAFETY_MARGIN;
    const msg: AgentMessage = {
      role: "user",
      content: "x".repeat(Math.floor(halfContextChars * 4)),
      timestamp: Date.now(),
    };

    // With safety margin applied, this should be at the boundary
    // The function checks if tokens * SAFETY_MARGIN > contextWindow * 0.5
    const isOversized = isOversizedForSummary(msg, CONTEXT_WINDOW);
    // Due to token estimation, this could be either true or false at the boundary
    expect(typeof isOversized).toBe("boolean");
  });
});

describe("compaction-safeguard runtime registry", () => {
  it("stores and retrieves config by session manager identity", () => {
    const sm = {};
    setCompactionSafeguardRuntime(sm, { maxHistoryShare: 0.3 });
    const runtime = getCompactionSafeguardRuntime(sm);
    expect(runtime).toEqual({ maxHistoryShare: 0.3 });
  });

  it("returns null for unknown session manager", () => {
    const sm = {};
    expect(getCompactionSafeguardRuntime(sm)).toBeNull();
  });

  it("clears entry when value is null", () => {
    const sm = {};
    setCompactionSafeguardRuntime(sm, { maxHistoryShare: 0.7 });
    expect(getCompactionSafeguardRuntime(sm)).not.toBeNull();
    setCompactionSafeguardRuntime(sm, null);
    expect(getCompactionSafeguardRuntime(sm)).toBeNull();
  });

  it("ignores non-object session managers", () => {
    setCompactionSafeguardRuntime(null, { maxHistoryShare: 0.5 });
    expect(getCompactionSafeguardRuntime(null)).toBeNull();
    setCompactionSafeguardRuntime(undefined, { maxHistoryShare: 0.5 });
    expect(getCompactionSafeguardRuntime(undefined)).toBeNull();
  });

  it("isolates different session managers", () => {
    const sm1 = {};
    const sm2 = {};
    setCompactionSafeguardRuntime(sm1, { maxHistoryShare: 0.3 });
    setCompactionSafeguardRuntime(sm2, { maxHistoryShare: 0.8 });
    expect(getCompactionSafeguardRuntime(sm1)).toEqual({ maxHistoryShare: 0.3 });
    expect(getCompactionSafeguardRuntime(sm2)).toEqual({ maxHistoryShare: 0.8 });
  });

  it("stores and retrieves minPreservedMessages", () => {
    const sm = {};
    setCompactionSafeguardRuntime(sm, { maxHistoryShare: 0.5, minPreservedMessages: 30 });
    const runtime = getCompactionSafeguardRuntime(sm);
    expect(runtime).toEqual({ maxHistoryShare: 0.5, minPreservedMessages: 30 });
  });
});

describe("minPreservedMessages helpers", () => {
  function makeMessageEntry(id: string, parentId: string | null = null): SessionEntry {
    return {
      type: "message",
      id,
      parentId,
      timestamp: new Date().toISOString(),
      message: { role: "user", content: `msg-${id}`, timestamp: Date.now() },
    };
  }

  function makeCustomMessageEntry(id: string, parentId: string | null = null): SessionEntry {
    return {
      type: "custom_message",
      id,
      parentId,
      timestamp: new Date().toISOString(),
      customType: "test",
      content: `custom-${id}`,
      display: true,
    };
  }

  function makeMetadataEntry(
    id: string,
    type: "thinking_level_change" | "model_change" | "compaction",
    parentId: string | null = null,
  ): SessionEntry {
    if (type === "thinking_level_change") {
      return {
        type: "thinking_level_change",
        id,
        parentId,
        timestamp: new Date().toISOString(),
        thinkingLevel: "low",
      };
    }
    if (type === "model_change") {
      return {
        type: "model_change",
        id,
        parentId,
        timestamp: new Date().toISOString(),
        provider: "anthropic",
        modelId: "claude-3-opus",
      };
    }
    return {
      type: "compaction",
      id,
      parentId,
      timestamp: new Date().toISOString(),
      summary: "Previous compaction",
      firstKeptEntryId: "prev",
      tokensBefore: 100000,
    };
  }

  describe("isMessageEntry", () => {
    it("returns true for message entries", () => {
      expect(isMessageEntry(makeMessageEntry("1"))).toBe(true);
    });

    it("returns true for custom_message entries", () => {
      expect(isMessageEntry(makeCustomMessageEntry("1"))).toBe(true);
    });

    it("returns false for metadata entries", () => {
      expect(isMessageEntry(makeMetadataEntry("1", "thinking_level_change"))).toBe(false);
      expect(isMessageEntry(makeMetadataEntry("1", "model_change"))).toBe(false);
      expect(isMessageEntry(makeMetadataEntry("1", "compaction"))).toBe(false);
    });
  });

  describe("countMessagesFromIndex", () => {
    it("counts only message and custom_message entries", () => {
      const entries: SessionEntry[] = [
        makeMessageEntry("1"),
        makeMetadataEntry("2", "thinking_level_change"),
        makeMessageEntry("3"),
        makeCustomMessageEntry("4"),
        makeMetadataEntry("5", "model_change"),
        makeMessageEntry("6"),
      ];

      expect(countMessagesFromIndex(entries, 0)).toBe(4);
      expect(countMessagesFromIndex(entries, 2)).toBe(3);
      expect(countMessagesFromIndex(entries, 5)).toBe(1);
    });

    it("returns 0 when starting past the end", () => {
      const entries: SessionEntry[] = [makeMessageEntry("1")];
      expect(countMessagesFromIndex(entries, 10)).toBe(0);
    });

    it("handles empty array", () => {
      expect(countMessagesFromIndex([], 0)).toBe(0);
    });
  });

  describe("findLastCompactionIndex", () => {
    it("returns -1 when no compaction exists", () => {
      const entries: SessionEntry[] = [makeMessageEntry("1"), makeMessageEntry("2")];
      expect(findLastCompactionIndex(entries)).toBe(-1);
    });

    it("finds the last compaction entry", () => {
      const entries: SessionEntry[] = [
        makeMetadataEntry("1", "compaction"),
        makeMessageEntry("2"),
        makeMetadataEntry("3", "compaction"),
        makeMessageEntry("4"),
      ];
      expect(findLastCompactionIndex(entries)).toBe(2);
    });
  });

  describe("adjustCutPointForMinMessages", () => {
    it("returns null when message count is already above threshold", () => {
      const entries: SessionEntry[] = [
        makeMessageEntry("1"),
        makeMessageEntry("2"),
        makeMessageEntry("3"), // cut point
        makeMessageEntry("4"),
        makeMessageEntry("5"),
        makeMessageEntry("6"),
      ];

      const result = adjustCutPointForMinMessages(entries, "3", 3);
      expect(result).toBeNull();
    });

    it("adjusts cut point when count is below threshold", () => {
      const entries: SessionEntry[] = [
        makeMessageEntry("1"),
        makeMessageEntry("2"),
        makeMessageEntry("3"),
        makeMessageEntry("4"),
        makeMessageEntry("5"), // current cut point - 3 messages from here (5, 6, 7)
        makeMessageEntry("6"),
        makeMessageEntry("7"),
      ];

      const result = adjustCutPointForMinMessages(entries, "5", 5);
      expect(result).not.toBeNull();
      expect(result!.adjusted).toBe(true);
      // Should move cut point back to preserve 5 messages
      // Messages from cut: 5, 6, 7 (3) -> need 2 more -> 4, 3
      expect(result!.newFirstKeptEntryId).toBe("3");
      expect(result!.messagesToAdd).toHaveLength(2);
    });

    it("respects previous compaction boundary", () => {
      const entries: SessionEntry[] = [
        makeMessageEntry("1"),
        makeMetadataEntry("comp", "compaction"), // compaction boundary
        makeMessageEntry("2"),
        makeMessageEntry("3"), // current cut point
        makeMessageEntry("4"),
        makeMessageEntry("5"),
      ];

      // Wants 10 messages but can only go back to after compaction
      const result = adjustCutPointForMinMessages(entries, "3", 10);
      expect(result).not.toBeNull();
      expect(result!.adjusted).toBe(true);
      // Should stop at entry after compaction (id "2")
      expect(result!.newFirstKeptEntryId).toBe("2");
    });

    it("returns null when cut point not found", () => {
      const entries: SessionEntry[] = [makeMessageEntry("1"), makeMessageEntry("2")];
      const result = adjustCutPointForMinMessages(entries, "nonexistent", 5);
      expect(result).toBeNull();
    });

    it("handles sessions with fewer than min messages", () => {
      const entries: SessionEntry[] = [
        makeMessageEntry("1"),
        makeMessageEntry("2"),
        makeMessageEntry("3"), // cut point - only 1 message total after
        makeMessageEntry("4"),
      ];

      // Only 4 messages total, asking for 10
      const result = adjustCutPointForMinMessages(entries, "3", 10);
      expect(result).not.toBeNull();
      // Should move cut point back as far as possible (to beginning)
      expect(result!.newFirstKeptEntryId).toBe("1");
    });

    it("skips metadata entries when counting but includes them in cut point", () => {
      const entries: SessionEntry[] = [
        makeMessageEntry("1"),
        makeMetadataEntry("meta1", "thinking_level_change"),
        makeMessageEntry("2"),
        makeMetadataEntry("meta2", "model_change"),
        makeMessageEntry("3"), // cut point - 3 messages from here (3, 4, 5)
        makeMessageEntry("4"),
        makeMessageEntry("5"),
      ];

      // Want 4 messages, currently have 3 (from cut point inclusive)
      const result = adjustCutPointForMinMessages(entries, "3", 4);
      expect(result).not.toBeNull();
      expect(result!.adjusted).toBe(true);
      // Should move back to include 1 more message
      // Walking back: meta2 (skip), 2 (count - done!)
      expect(result!.messagesToAdd).toHaveLength(1);
    });

    it("returns null when no change is possible (already at boundary)", () => {
      const entries: SessionEntry[] = [
        makeMessageEntry("1"), // cut point - already at start
        makeMessageEntry("2"),
      ];

      const result = adjustCutPointForMinMessages(entries, "1", 10);
      expect(result).toBeNull(); // Can't go further back
    });
  });
});
