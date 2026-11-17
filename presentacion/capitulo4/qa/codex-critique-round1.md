# Codex critique, round 1 (codex exec, read-only, 7 contact sheets)

1. **Slide 1:** The validation-window graphic is too small to read and has no time scale; the title and project details compete for attention.
2. **Slide 2:** The six CRISP-DM phases and four chapter cards create two navigation systems. The active 4.1–4.4 rail is not yet explained.
3. **Slide 3:** The problem statement and two dense objective tables compete for attention; table text is too small for projection.
4. **Slide 4:** The 4.1 rail is clear, but the section opens with dataset constraints rather than an explicit real-world case motivating technique selection.
5. **Slide 5:** The complexity order is easy to follow. The progression rule and scope note are cramped below the model cards, while the supporting papers appear only on the next slide.
6. **Slide 6:** The Q1/Q2 evidence is present, but five citation rows are too small to read aloud; journal rank, contribution, and caveat have equal visual weight.
7. **Slide 7:** The window diagram conveys chronology, but its week labels and the case-count table are small. The OOT bar uses a different visual structure from P1–P3 without much explanation.
8. **Slide 8:** The four-step sequence is clear; long paragraphs inside equal-sized cards obscure which rule is most critical. The two bottom caveats compete with the procedure.
9. **Slide 9:** The metric table is dense and lacks a visual distinction between primary success criteria and supplementary measures.
10. **Slide 10:** Six equally weighted control cards make the validation safeguards hard to prioritize; several explanations are paragraph-length.
11. **Slide 11:** The headline stresses reproducibility, but the detailed experiment table dominates. Small entries for timing, variables, and status will be difficult to read on screen.
12. **Slide 12:** The table and chart repeat the same grid results. The bar chart exaggerates tiny differences between configurations without a visible scale.
13. **Slide 13:** The ablation result is buried in a table; the key finding that no group was removed should have stronger visual priority. The two explanatory boxes are text-heavy.
14. **Slide 14:** A large two-bar chart repeats values in a small comparison table. The chart has no visible vertical scale, making the size of the advantage harder to judge.
15. **Slide 15:** The weekly chart is the main evidence, but 18 tightly packed week labels and a truncated vertical range require careful reading. The three statistic cards narrow the plot.
16. **Slide 16:** The approximate 7,700 figure dominates, while the calculation connecting it to the 4.1-point FPR difference is not shown.
17. **Slide 17:** The chart lacks a vertical scale, and small percentage labels carry the comparison. The failure badge, two metrics, and explanatory text compete for the first glance.
18. **Slide 18:** The failure badge and first callout repeat the same conclusion. The decisive segment counts and AUC comparison are confined to small table text.
19. **Slide 19:** The table shows identical top-five ranks clearly, but the dense “changes between P3 and OOT” paragraph introduces a second finding without enough separation.
20. **Slide 20:** The key caveat—that the useful calibration detector was added after the run—sits below a dense table; “partial” is visually weaker than the headline claim.
21. **Slide 21:** Eight objective rows compress fulfilled, failed, and partial outcomes into small text. The most consequential failure, calibration, needs stronger hierarchy.
22. **Slide 22:** The selected model is prominent, but the two long panels dilute the decision and limitations. This slide reads more like a report page than a closing synthesis.
23. **Slide 23:** Five full-width next-step rows are easy to scan numerically, yet each has similar weight; recalibration is not distinguished as the immediate priority.
24. **Slide 24:** References are too small for projection and visually dense. The disappearance of the section rail is understandable here but breaks the established navigation pattern.
25. **Slide 25:** The closing message is clear, but the unlabeled bar motif differs from the labeled opening graphic and has no evident meaning; the three takeaways sit far apart with weak alignment.

## Decisions (round 1)
Applied: chart titles now state the axis baseline (slides 12, 14, 15, 16, 17); weekly chart axis labels redrawn as text with decimal comma (15); case-study strip added (4); rail explainer (2); motif labels (1, 25); calibration row highlighted (21); recalibration marked as immediate priority (23); reference text enlarged (24); padding fixes (5, 22).
Rejected: showing a formula for the ~7 700 figure (16) because 190 398 x 4,1 points gives ~7 800, not a clean reproduction of the source figure; removing the chart/table redundancy (12, 14), kept on purpose (chart = shape, table = exact values); adding the rail to the references slide (24); duplicate badge/callout on 18 (kept, different roles). Text-size remarks made from the 960 px sheets were treated as thumbnail artifacts (full-size slides were inspected individually).
