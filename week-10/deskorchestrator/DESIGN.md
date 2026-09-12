# DeskOrchestrator: design memo

Replace every italic prompt with your own answer. Aim for 600 to 900 words in
total. This is four marks of Design and three of Clarity, and the last question
is the one that cannot be copied from anywhere.

---

## 1. The scope filter

*What was wrong with the shipped implementation, in your own words? Name the two
failure modes it produces and say which one is the dangerous one and why. Then
say what your fix changes about the order of operations.*

*One sentence on the thing most people get wrong here: was this a leak?*

---

## 2. The routing eval

*How many cases, and how did you choose them? Which routes are represented, and
how many of your cases sit near a boundary rather than in the middle of a route?*

*What accuracy did you measure, and what was the single worst confusion? Do not
round it up.*

*What would you add to the set if you had another hour, and why that rather than
more of what you already have?*

---

## 3. The approval policy

*Which proposals pause, and which do not? State the rule, not the code.*

*Defend the line. What is the worst case on each side of it: what happens if
something you let through should have paused, and what happens if you pause
something trivial?*

*Show that your `risky()` is a pure function of its argument. If you were tempted
to read anything else, say what it was and why you did not.*

---

## 4. What you deliberately did NOT build

*This is the week's title as a deliverable.*

*Name at least one thing you considered splitting into its own agent, or adding
as a fourth route, or building as a fifth memory layer, and did not. Say which of
the three triggers you tested it against and why it failed.*

*Then name the cost you would have paid if you had built it: which of the four
liabilities it would have added, and what would have had to be maintained
forever afterwards.*

*An engineer who can articulate what they chose not to build is demonstrating
judgement. This section is where you do that.*
