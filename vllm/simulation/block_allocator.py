"""
PagedAttention block allocator — simulation.

Demonstrates four things:
  1. Physical block pool with free list and ref counting
  2. Per-sequence block tables (logical -> physical mapping)
  3. Copy-on-write for beam search forks
  4. Prefix caching via shared blocks

No real KV tensors are stored — this is pure memory-management logic.
Run with: python block_allocator.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List

BLOCK_SIZE = 4   # tokens per physical block  (real vLLM default: 16)
NUM_BLOCKS = 20  # total blocks in the pool


# ------------------------------------------------------------------------------
# Physical block
# ------------------------------------------------------------------------------

@dataclass
class PhysicalBlock:
    """
    One fixed-size slot in VRAM that holds KV vectors for BLOCK_SIZE tokens.
    ref_count > 1 means the block is shared — never write without COW.
    """
    block_id: int
    ref_count: int = 0

    def __repr__(self) -> str:
        return f"B{self.block_id}(rc={self.ref_count})"


# ------------------------------------------------------------------------------
# Block allocator
# ------------------------------------------------------------------------------

class BlockAllocator:
    """
    Central manager for the physical block pool.
    Mirrors vLLM's CpuGpuBlockAllocator (GPU side).

    The free list is a stack: O(1) allocate and free.
    ref_count per block enables safe sharing without copies.
    """

    def __init__(self, num_blocks: int) -> None:
        self._blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i) for i in range(num_blocks)
        }
        self._free: List[int] = list(range(num_blocks))

    # -- core operations -------------------------------------------------------

    def allocate(self) -> PhysicalBlock:
        if not self._free:
            raise MemoryError("KV cache OOM — no free blocks")
        block = self._blocks[self._free.pop()]
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock) -> None:
        """Decrement ref_count; return to free list when it hits zero."""
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free.append(block.block_id)

    def fork(self, block: PhysicalBlock) -> PhysicalBlock:
        """
        Share a block with a new owner — just bump ref_count.
        Used for prefix caching and beam search.
        No data copy, no new allocation.
        """
        block.ref_count += 1
        return block

    def copy_on_write(self, block: PhysicalBlock) -> PhysicalBlock:
        """
        Called before writing to a block that might be shared.
        If ref_count == 1 the block is already private — return as-is.
        If ref_count > 1 allocate a fresh private copy and release our share.

        The caller must replace the old reference with the returned block.
        """
        if block.ref_count == 1:
            return block
        new_block = self.allocate()
        print(f"    [COW] B{block.block_id}(rc={block.ref_count}) "
              f"-> new B{new_block.block_id}")
        self.free(block)   # release this owner's share of the original
        return new_block

    # -- inspection ------------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free)

    def status_bar(self) -> str:
        used = len(self._blocks) - self.num_free
        bar  = "#" * used + "." * self.num_free
        return f"[{bar}] {used}/{len(self._blocks)} used"


# ------------------------------------------------------------------------------
# Sequence
# ------------------------------------------------------------------------------

class Status(Enum):
    WAITING  = auto()
    RUNNING  = auto()
    SWAPPED  = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    """
    One generation request.
    block_table maps logical block index -> physical block.
    """
    seq_id:     int
    prompt_len: int
    max_tokens: int
    block_table: List[PhysicalBlock] = field(default_factory=list)
    generated:   int = 0
    status:      Status = Status.WAITING

    @property
    def num_tokens(self) -> int:
        return self.prompt_len + self.generated

    @property
    def blocks_needed(self) -> int:
        """Logical blocks required to hold all current tokens."""
        return (self.num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    def needs_new_block(self) -> bool:
        """True when we've just filled the last slot of the current tail block."""
        return self.num_tokens > 0 and self.num_tokens % BLOCK_SIZE == 0

    def __repr__(self) -> str:
        return (f"Seq{self.seq_id}(toks={self.num_tokens}, "
                f"blocks={[b.block_id for b in self.block_table]})")


# ------------------------------------------------------------------------------
# Scheduler — simplified continuous batching
# ------------------------------------------------------------------------------

class Scheduler:
    """
    Iteration-level scheduler: after every decode step, evict finished
    sequences and admit new ones from the waiting queue.

    Preemption: if a running sequence can't get a new block, swap it out
    (free its blocks) so other sequences can continue.
    """

    def __init__(self, allocator: BlockAllocator, max_batch: int = 3) -> None:
        self.allocator  = allocator
        self.max_batch  = max_batch
        self.waiting:  List[Sequence] = []
        self.running:  List[Sequence] = []
        self.finished: List[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    # -- prefill ---------------------------------------------------------------

    def _prefill(self, seq: Sequence) -> bool:
        """Allocate blocks for the full prompt. Returns False if OOM."""
        needed = seq.blocks_needed
        if self.allocator.num_free < needed:
            return False
        for _ in range(needed):
            seq.block_table.append(self.allocator.allocate())
        seq.status = Status.RUNNING
        return True

    # -- decode step -----------------------------------------------------------

    def decode_one(self, seq: Sequence) -> None:
        """Generate one token for seq, allocating a new block if needed."""
        if seq.needs_new_block():
            if self.allocator.num_free == 0:
                self._preempt(seq)
                return
            seq.block_table.append(self.allocator.allocate())

        seq.generated += 1

        if seq.generated >= seq.max_tokens:
            self._finish(seq)

    # -- lifecycle -------------------------------------------------------------

    def _finish(self, seq: Sequence) -> None:
        seq.status = Status.FINISHED
        self.running.remove(seq)
        self.finished.append(seq)
        for b in seq.block_table:
            self.allocator.free(b)
        seq.block_table.clear()
        print(f"  [done]  Seq{seq.seq_id} done  -> freed {seq.blocks_needed} blocks")

    def _preempt(self, seq: Sequence) -> None:
        """
        OOM mid-decode: free this sequence's blocks so others can continue.
        In real vLLM two strategies exist:
          - swap: move KV cache to CPU RAM, restore later
          - recompute: drop everything, re-prefill from scratch when space frees up
        Here we just free and mark SWAPPED.
        """
        seq.status = Status.SWAPPED
        self.running.remove(seq)
        freed = len(seq.block_table)
        for b in seq.block_table:
            self.allocator.free(b)
        seq.block_table.clear()
        print(f"  [!!] PREEMPT Seq{seq.seq_id} — freed {freed} blocks")

    # -- main loop step --------------------------------------------------------

    def step(self) -> None:
        """
        One full scheduler iteration:
          1. Admit waiting sequences up to max_batch.
          2. Decode one token for every running sequence.
        """
        # Admit
        while self.waiting and len(self.running) < self.max_batch:
            candidate = self.waiting[0]
            if self._prefill(candidate):
                self.waiting.pop(0)
                self.running.append(candidate)
                print(f"  +  Admitted Seq{candidate.seq_id} "
                      f"(prompt={candidate.prompt_len}t, "
                      f"blocks={[b.block_id for b in candidate.block_table]})")
            else:
                print(f"  [!]  OOM: can't admit Seq{candidate.seq_id} "
                      f"(need {candidate.blocks_needed}, "
                      f"have {self.allocator.num_free})")
                break

        # Decode
        for seq in list(self.running):
            self.decode_one(seq)


# ------------------------------------------------------------------------------
# Demo 1 — continuous batching
# ------------------------------------------------------------------------------

def demo_continuous_batching() -> None:
    print("\n" + "=" * 64)
    print("DEMO 1  Continuous Batching + On-demand Block Allocation")
    print("=" * 64)
    print(f"Pool: {NUM_BLOCKS} blocks x {BLOCK_SIZE} tokens/block  |  max_batch=3\n")

    allocator = BlockAllocator(NUM_BLOCKS)
    scheduler = Scheduler(allocator, max_batch=3)

    # Five requests of varying prompt + output lengths
    for i, (p, o) in enumerate([(6, 8), (10, 4), (3, 12), (5, 6), (8, 10)]):
        scheduler.add(Sequence(seq_id=i, prompt_len=p, max_tokens=o))

    step = 0
    while scheduler.running or scheduler.waiting:
        step += 1
        print(f"--- step {step:02d} " + "-" * 46)
        scheduler.step()
        print(f"     pool  : {allocator.status_bar()}")
        print(f"     running: {[str(s) for s in scheduler.running]}")
        waiting_ids = [s.seq_id for s in scheduler.waiting]
        if waiting_ids:
            print(f"     waiting: {waiting_ids}")

    print(f"\nFinished: {[s.seq_id for s in scheduler.finished]}")


# ------------------------------------------------------------------------------
# Demo 2 — copy-on-write (beam search fork)
# ------------------------------------------------------------------------------

def demo_copy_on_write() -> None:
    print("\n" + "=" * 64)
    print("DEMO 2  Copy-on-Write  (Beam Search Fork)")
    print("=" * 64)

    allocator = BlockAllocator(NUM_BLOCKS)

    # A sequence has processed 8 tokens -> occupies 2 blocks
    parent = Sequence(seq_id=0, prompt_len=8, max_tokens=0)
    for _ in range(2):
        parent.block_table.append(allocator.allocate())

    print(f"\nParent:  {parent}")
    print(f"Pool:    {allocator.status_bar()}")

    # Beam search: fork into 2 children that share parent's blocks
    child_a = Sequence(seq_id=1, prompt_len=8, max_tokens=4)
    child_b = Sequence(seq_id=2, prompt_len=8, max_tokens=4)

    for blk in parent.block_table:
        child_a.block_table.append(allocator.fork(blk))
        child_b.block_table.append(allocator.fork(blk))

    # Parent is retired — release its references
    for blk in parent.block_table:
        allocator.free(blk)

    print(f"\nAfter fork (no new blocks allocated!):")
    print(f"  child_a: {child_a}")
    print(f"  child_b: {child_b}")
    print(f"  ref_counts on shared blocks: "
          f"{[b.ref_count for b in child_a.block_table]}")
    print(f"  Pool:    {allocator.status_bar()}")

    # child_a writes its first new token -> last block is shared -> COW
    print(f"\nchild_a writes token -> last block is shared, COW triggers:")
    last = child_a.block_table[-1]
    child_a.block_table[-1] = allocator.copy_on_write(last)
    child_a.generated += 1

    print(f"  child_a: {child_a}")
    print(f"  child_b: {child_b}  (untouched)")
    print(f"  Pool:    {allocator.status_bar()}")


# ------------------------------------------------------------------------------
# Demo 3 — prefix caching (shared system prompt)
# ------------------------------------------------------------------------------

def demo_prefix_caching() -> None:
    print("\n" + "=" * 64)
    print("DEMO 3  Prefix Caching  (Shared System Prompt)")
    print("=" * 64)

    allocator = BlockAllocator(NUM_BLOCKS)

    # System prompt: 8 tokens = 2 blocks, computed once and pinned
    system_blocks = [allocator.allocate() for _ in range(2)]
    print(f"\nSystem prompt cached: blocks {[b.block_id for b in system_blocks]}")
    print(f"Pool: {allocator.status_bar()}")

    # Three requests all fork the same prefix blocks — zero extra allocation
    seqs = [Sequence(seq_id=i, prompt_len=8, max_tokens=4) for i in range(3)]
    for seq in seqs:
        for blk in system_blocks:
            seq.block_table.append(allocator.fork(blk))

    print(f"\nAfter 3 requests attach to prefix (still no new blocks!):")
    print(f"  prefix ref_counts: {[b.ref_count for b in system_blocks]}")
    print(f"  Pool: {allocator.status_bar()}")

    # Each request now generates its own tokens -> private blocks
    for seq in seqs:
        private = allocator.allocate()
        seq.block_table.append(private)
        seq.generated += 1
        print(f"  Seq{seq.seq_id} appended private B{private.block_id}")

    print(f"\nPool after private blocks allocated: {allocator.status_bar()}")
    for seq in seqs:
        shared  = [b.block_id for b in seq.block_table[:-1]]
        private = seq.block_table[-1].block_id
        print(f"  Seq{seq.seq_id}: shared={shared}  private=[{private}]")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    demo_continuous_batching()
    demo_copy_on_write()
    demo_prefix_caching()
