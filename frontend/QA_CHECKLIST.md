# Frontend QA Checklist

## Theme consistency

- Confirm there are no accidental white backgrounds on cards, inputs, selects, dropdowns, or empty states.
- Open every dropdown and confirm the expanded menu stays dark, every option remains readable, and no browser-default white menu surface appears.
- Confirm the sentiment meter, net-lean card, reconciliation note, and citation pills all keep dark-theme contrast and readable text.
- Confirm body, panels, prediction card, source cards, and model cards all use the same dark palette.
- Confirm hover, focus, active, selected, loading, warning, and error states remain readable.
- Confirm bullish, bearish, neutral, warning, and error styles are readable without relying on color alone.

## Desktop

- Verify the layout at `1366px` and `1440px`.
- Confirm the research rail stays readable and the decision panel remains the visual focus.
- Confirm model cards, source cards, and metric cards align consistently.
- Confirm the confidence meter, freshness status, and disclaimer are visible without scrolling far.
- Open the Explainability panel and confirm every citation tag jumps to the matching numbered source card.
- Confirm the reconciliation note message changes correctly when news sentiment disagrees with the model versus when it aligns.
- Confirm the current price, predicted price, likely range, volatility badge, and chart remain readable together.

## Tablet

- Verify the layout around `1024px`.
- Confirm the prediction card still appears before explanation details in the reading order.
- Confirm filters and cards do not clip or overlap.
- Confirm focus styles remain visible with keyboard navigation.

## Mobile

- Verify the layout around `390px` and `428px`.
- Confirm the "Open research setup" button reveals the controls panel.
- Confirm the prediction card appears before the setup rail.
- Confirm buttons span full width and remain easy to tap.

## Reliability states

- Confirm backend warm-up shows "System warming up. Retrying..." instead of raw `502`.
- Confirm model loading skeletons appear before setup cards.
- Confirm empty model results show a helpful reset/widen-filters message.
- Confirm prediction failure and explanation failure show retry actions.
- Confirm stale data and low-confidence states appear as warnings.
