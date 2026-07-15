# Agile & Rally

Sprint, Epic, Story, Task — and how they map onto Rally (Broadcom Agile Central), the tool my company mandates for sprint tracking.

---

## The core hierarchy

Everything in Rally hangs off one chain, from "big idea" down to "thing I do today":

```
Epic       — a goal too big to finish in one Sprint (months)
  └── Story  — a slice of the Epic small enough to finish in one Sprint (days)
        └── Task — the concrete work items a Story breaks into (hours)
```

Sprint isn't part of this chain — it's the *time box* the Stories/Tasks get scheduled into, not a container in the hierarchy itself.

## Sprint

A **Sprint** is a fixed-length work cycle — commonly 2 weeks (some teams do 1, 3, or 4).

Analogy: instead of remodeling a whole house in one uninterrupted push (and risking a redesign halfway through), you commit to a small, concrete slice of work every 2 weeks, show it to the stakeholder, and adjust before starting the next slice.

- Fixed time box (e.g. June 1–14)
- Team commits to a batch of work for that window
- At the end, whatever's done or not, the team stops, reviews, and re-plans

## Epic

An **Epic** is something too large to finish inside one Sprint.

Same house analogy: "remodel the whole house" is an Epic. It gets broken down into pieces that *can* fit in a Sprint — "tile the bathroom," "paint the living room" — and those pieces are Stories.

> Epic = what big thing we're doing (months) → Sprint = when we're doing it (2-week windows) → Story = the specific small thing we're doing (days)

## Backlog

Backlog isn't a rung in the Epic/Story/Task hierarchy — that hierarchy answers "how big is this work"; Backlog answers "is this work queued up or already scheduled."

There are usually two backlogs, at two different levels:

```
Portfolio Backlog / Epic Backlog
  └── Epics not yet started, prioritized company/business-wide
        ↓ (decide to kick off an Epic → break it into Stories)
Team Backlog / Product Backlog
  └── Stories already broken down, but not yet scheduled into a Sprint
        ↓ (Sprint Planning: pull the top Stories into the current Iteration)
Current Sprint
```

House-remodel version: "remodel the whole house" can sit in the Epic Backlog for months behind other Epics ("redo the kitchen first") until it's prioritized. Once started, it's broken into Stories ("tile the bathroom," "paint the living room") which sit in the Team Backlog until a Sprint Planning session pulls them in.

In Rally, the day-to-day Backlog view is Story-level; Epic-level sequencing lives in a separate Portfolio view.

## Who breaks Epic → Story vs. Story → Task

- **Epic → Story**: driven by the Product Owner (the "what" and priority), but done together with the Tech Lead / senior engineers in **Backlog Refinement / Grooming** — the PO knows the business need, the Tech Lead knows what can be split, parallelized, or has hard dependencies.
- **Story → Task**: done by whoever picks up the Story (usually just me), during Sprint Planning. No PO involvement needed — this is purely "what work does *this* Story break into for me."

## Story

A **Story** is a slice of an Epic that (a) fits inside one Sprint and (b) is independently valuable/demoable on its own — not just "a chunk of code that doesn't run yet."

Example, breaking the "self-serve password reset" Epic into Stories:

- Story 1: user enters email → receives a reset-link email
- Story 2: user clicks the link → can set a new password
- Story 3: password reset succeeds → confirmation email sent

Standard write-up format forces you to think user-first instead of jumping straight to implementation:

> As a [customer who forgot their password], I want [to receive an email with a reset link] so that [I can reset my password without contacting support]

## Task

A **Task** is the concrete, un-splittable work a Story breaks into — usually a day or less for one person. No user-story format needed; it's written for yourself/the team, not the business side.

Example, breaking Story 2 ("click link, set new password") into Tasks:

- Task 1: backend endpoint to validate the reset token
- Task 2: backend endpoint to set the new password (+ strength check)
- Task 3: frontend page for setting the new password
- Task 4: unit tests
- Task 5: integration + self-test

### Estimating: Task hours vs. Story Points

- **Story Point** (on the Story): relative complexity, Fibonacci-ish scale (1/2/3/5/8...), set via team estimation (Planning Poker). Not a time unit.
- **Task hours** (on each Task, "Estimate"/"To Do"): how many hours *remain* to finish that Task specifically. Update this daily — not "hours spent," but "hours I think are still left."

**Why the daily update matters:** Rally's Burndown Chart is built entirely from these remaining-hour numbers. Forget to update a Task's remaining hours after doing real work on it, and the burndown shows no progress even though work happened — the most common rookie mistake.

## Defect

A **Defect** (bug) is a separate line from the Epic/Story/Task "new feature" chain — it can stand alone or attach to a Story, but it isn't itself a rung in that hierarchy.

**When to attach to a Story vs. stand alone:**
- Found during testing, not yet shipped → attach to the Story it belongs to (it's that Story not being done right yet, not a new independent problem).
- Found in something already shipped (prod bug report, old feature breaking) → standalone Defect, not tied to any Story — it's maintenance work that gets prioritized into some future Sprint on its own.

**Two fields people conflate:**
- **Severity** — objective: how bad is the impact (data loss vs. a wrong button color). Rarely changes once set.
- **Priority** — subjective: how fast should this get fixed. A low-severity typo on a page the CEO is demoing next week can outrank a higher-severity bug in a rarely-used corner case.

**State flow:**

```
New → Open (confirmed, assigned) → Fixed (code changed) → Closed (verified)
                                                    ↘ Reopened (verification failed)
```

Example: three months after the password-reset Epic shipped, a user reports "clicking an expired reset link just shows a blank page." This becomes a standalone Defect (Severity: Medium — doesn't block the core flow; Priority: High — user complaints are piling up), not attached to any Story, competing for priority in a future Sprint like any other Backlog item.

---

*(This note is a work in progress — being filled in interactively as I learn Rally at work.)*
