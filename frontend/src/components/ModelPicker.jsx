import React, { useEffect, useId, useRef, useState } from "react";
import { FILTER_META, HORIZON_LABEL, HUMAN_GROUP, formatScore } from "../lib/research.js";

export default function ModelPicker({
  filters,
  onFilterChange,
  onResetFilters,
  activeChips,
  models,
  modelId,
  onModelSelect,
  loading,
  optionSets,
  availableOptions,
  selectedModel,
}) {
  const rankedModels = models;

  return (
    <section className="panel model-picker" aria-label="Signal configuration">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Signal configuration</p>
          <h2>Choose signal configuration</h2>
          <p className="muted">
            Ranked by directional accuracy, then Sharpe, then lower RMSE within the current filter set.
          </p>
        </div>
        <span className={`status-chip${loading ? " is-loading" : ""}`}>
          {loading ? "Refreshing setups" : `${rankedModels.length} setup${rankedModels.length === 1 ? "" : "s"}`}
        </span>
      </div>

      {activeChips.length ? (
        <div className="chip-row" aria-label="Active filters">
          {activeChips.map((chip) => (
            <button
              type="button"
              key={chip.key}
              className="filter-chip"
              onClick={() => onFilterChange(chip.key, "")}
              aria-label={`Remove ${chip.label} filter`}
            >
              <span>{chip.label}: {chip.value}</span>
              <span aria-hidden="true">x</span>
            </button>
          ))}
        </div>
      ) : null}

      <details className="accordion" open={activeChips.length > 0}>
        <summary>
          <span>Advanced model settings</span>
          <span className="muted small">Model type, group, family, basis, and ticker-specific setup filters</span>
        </summary>

        <div className="grid filter-grid">
          <SelectField
            label={FILTER_META.horizon.label}
            value={filters.horizon}
            onChange={(value) => onFilterChange("horizon", value)}
            options={optionSets.horizon}
            available={availableOptions.horizon}
            formatLabel={(value) => HORIZON_LABEL[value] || value}
          />
          <SelectField
            label={FILTER_META.kind.label}
            value={filters.kind}
            onChange={(value) => onFilterChange("kind", value)}
            options={optionSets.kind}
            available={availableOptions.kind}
          />
          <SelectField
            label={FILTER_META.group.label}
            value={filters.group}
            onChange={(value) => onFilterChange("group", value)}
            options={optionSets.group}
            available={availableOptions.group}
            formatLabel={(value) => HUMAN_GROUP[value] || value}
          />
          <SelectField
            label={FILTER_META.family.label}
            value={filters.family}
            onChange={(value) => onFilterChange("family", value)}
            options={optionSets.family}
            available={availableOptions.family}
          />
          <SelectField
            label={FILTER_META.basis.label}
            value={filters.basis}
            onChange={(value) => onFilterChange("basis", value)}
            options={optionSets.basis}
            available={availableOptions.basis}
            formatLabel={(value) => (value === "panel" ? "Panel / per-stock" : "Index / benchmark")}
          />
          <SelectField
            label={FILTER_META.symbol.label}
            value={filters.symbol}
            onChange={(value) => onFilterChange("symbol", value)}
            options={optionSets.symbol}
            available={availableOptions.symbol}
          />
        </div>

        <div className="inline-actions">
          <button type="button" className="button-secondary" onClick={onResetFilters}>
            Reset filters
          </button>
        </div>
      </details>

      {selectedModel ? (
        <div className="selected-setup-banner selected-setup-banner-compact" aria-label="Selected forecast setup">
          <strong>Selected:</strong> {selectedModel.model_name}
          <span className="muted">
            {selectedModel.family} / {selectedModel.kind} / {selectedModel.group}
            {selectedModel.symbol ? ` / ${selectedModel.symbol}` : " / flexible ticker"}
          </span>
        </div>
      ) : (
        <div className="state-banner state-warning" role="status">
          <div>
            <strong>No setup selected</strong>
            <p>Choose a forecast setup below or widen the filters to find an available signal.</p>
          </div>
        </div>
      )}
    </section>
  );
}

export function ModelComparisonAccordion({ models, modelId, onModelSelect, loading, open, onToggle }) {
  const rankedModels = models;

  return (
    <details
      className="accordion model-comparison-accordion"
      open={open}
      onToggle={(event) => onToggle?.(event.target.open)}
    >
      <summary>
        <span>Compare other models</span>
        <span className="muted small">
          {loading ? "Refreshing setups" : `${rankedModels.length} setup${rankedModels.length === 1 ? "" : "s"} available`}
        </span>
      </summary>

      {loading ? (
        <div className="model-comparison-scroll" aria-hidden="true">
          <div className="model-skeleton-list">
            {Array.from({ length: 5 }).map((_, index) => (
              <span key={index} className="skeleton-line wide" />
            ))}
          </div>
        </div>
      ) : rankedModels.length ? (
        <div className="model-comparison-scroll">
          <div className="model-card-list" role="list" aria-label="Available forecast setups">
            {rankedModels.map((model, index) => {
              const isSelected = model.model_id === modelId;
              return (
                <button
                  key={model.model_id}
                  type="button"
                  className={`model-card${isSelected ? " is-selected" : ""}`}
                  onClick={() => onModelSelect(model.model_id)}
                  aria-pressed={isSelected}
                  aria-label={`Rank ${index + 1}, ${model.model_name}, ${model.horizon}, ${model.kind}`}
                >
                  <div className="model-card-topline">
                    <span className="rank-badge">#{index + 1}</span>
                    <span className={`selection-indicator${isSelected ? " is-selected" : ""}`}>
                      {isSelected ? "Selected" : "Available"}
                    </span>
                  </div>
                  <strong className="model-card-title">{model.model_name}</strong>
                  <div className="model-card-meta">
                    <span>{model.group}</span>
                    <span>{HORIZON_LABEL[model.horizon] || model.horizon}</span>
                    <span>{model.symbol || "Flexible ticker"}</span>
                  </div>
                  <div className="model-card-tags">
                    <span className="pill">{model.family}</span>
                    <span className="pill">{model.kind}</span>
                    <span className="pill">{model.basis}</span>
                  </div>
                  <div className="metric-row compact">
                    <MetricMini label="DA" value={model.metrics?.directional_accuracy != null ? `${Math.round(model.metrics.directional_accuracy * 100)}%` : "--"} />
                    <MetricMini label="Sharpe" value={formatScore(model.metrics?.sharpe)} />
                    <MetricMini label="RMSE" value={formatScore(model.metrics?.rmse, 4)} />
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="model-comparison-scroll">
          <div className="empty-state-box" role="status">
            <strong>No forecast setups match these filters</strong>
            <p>Reset filters or widen the ticker/horizon combination to see available signal configurations.</p>
          </div>
        </div>
      )}
    </details>
  );
}

function SelectField({ label, value, onChange, options = [], available = new Set(), formatLabel }) {
  const normalizedKey = normalizeLabel(label);
  const optionList = [
    {
      value: "",
      label: FILTER_META[normalizedKey]?.anyLabel || "Any",
      disabled: false,
    },
    ...options.map((option) => ({
      value: option,
      label: formatLabel ? formatLabel(option) : option,
      disabled: !available.has(option) && value !== option,
    })),
  ];

  return (
    <label className="field">
      <span>{label}</span>
      <CustomSelect label={label} value={value} onChange={onChange} options={optionList} />
    </label>
  );
}

function MetricMini({ label, value }) {
  return (
    <div className="metric-mini">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function normalizeLabel(label) {
  return Object.keys(FILTER_META).find((key) => FILTER_META[key].label === label) || label;
}

function CustomSelect({ label, value, onChange, options }) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const rootRef = useRef(null);
  const triggerRef = useRef(null);
  const listboxRef = useRef(null);
  const optionRefs = useRef([]);
  const labelId = useId();
  const listboxId = useId();

  const selectedIndex = options.findIndex((option) => option.value === value);
  const fallbackIndex = options.findIndex((option) => !option.disabled);
  const resolvedSelectedIndex = selectedIndex >= 0 ? selectedIndex : fallbackIndex;
  const selectedOption = options[resolvedSelectedIndex] || options[0];

  useEffect(() => {
    if (!open) return undefined;

    function handlePointerDown(event) {
      if (!rootRef.current?.contains(event.target)) {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("touchstart", handlePointerDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("touchstart", handlePointerDown);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const nextIndex = resolvedSelectedIndex >= 0 ? resolvedSelectedIndex : fallbackIndex;
    setActiveIndex(nextIndex);
    listboxRef.current?.focus();
  }, [open, resolvedSelectedIndex, fallbackIndex]);

  useEffect(() => {
    if (!open || activeIndex < 0) return;
    optionRefs.current[activeIndex]?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, open]);

  function moveActive(direction) {
    if (!options.length) return;

    let nextIndex = activeIndex;
    let inspected = 0;

    while (inspected < options.length) {
      nextIndex = (nextIndex + direction + options.length) % options.length;
      if (!options[nextIndex].disabled) {
        setActiveIndex(nextIndex);
        return;
      }
      inspected += 1;
    }
  }

  function jumpTo(edge) {
    const indices = edge === "start"
      ? options.map((_, index) => index)
      : options.map((_, index) => index).reverse();
    const nextIndex = indices.find((index) => !options[index].disabled);
    if (nextIndex != null) {
      setActiveIndex(nextIndex);
    }
  }

  function commitIndex(index) {
    const option = options[index];
    if (!option || option.disabled) return;
    onChange(option.value);
    setOpen(false);
    triggerRef.current?.focus();
  }

  function handleTriggerKeyDown(event) {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setOpen(true);
        break;
      case "ArrowUp":
        event.preventDefault();
        setOpen(true);
        jumpTo("end");
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        setOpen((current) => !current);
        break;
      default:
        break;
    }
  }

  function handleListboxKeyDown(event) {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        moveActive(1);
        break;
      case "ArrowUp":
        event.preventDefault();
        moveActive(-1);
        break;
      case "Home":
        event.preventDefault();
        jumpTo("start");
        break;
      case "End":
        event.preventDefault();
        jumpTo("end");
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        commitIndex(activeIndex);
        break;
      case "Escape":
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
        break;
      case "Tab":
        setOpen(false);
        break;
      default:
        break;
    }
  }

  return (
    <div
      ref={rootRef}
      className={`select-shell${open ? " is-open" : ""}`}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          setOpen(false);
        }
      }}
    >
      <span id={labelId} className="sr-only">{label}</span>
      <button
        ref={triggerRef}
        type="button"
        className={`select-trigger${open ? " is-open" : ""}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-labelledby={labelId}
        aria-controls={listboxId}
        onClick={() => setOpen((current) => !current)}
        onKeyDown={handleTriggerKeyDown}
      >
        <span className={`select-trigger-text${selectedOption?.value ? "" : " is-placeholder"}`}>
          {selectedOption?.label || label}
        </span>
        <span className="select-trigger-icon" aria-hidden="true">
          v
        </span>
      </button>

      {open ? (
        <div className="select-menu" role="presentation">
          <div
            id={listboxId}
            ref={listboxRef}
            className="select-listbox"
            role="listbox"
            tabIndex={-1}
            aria-labelledby={labelId}
            aria-activedescendant={activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined}
            onKeyDown={handleListboxKeyDown}
          >
            {options.map((option, index) => {
              const isSelected = option.value === value;
              const isActive = index === activeIndex;

              return (
                <button
                  key={`${option.value || "any"}-${index}`}
                  id={`${listboxId}-option-${index}`}
                  ref={(node) => {
                    optionRefs.current[index] = node;
                  }}
                  type="button"
                  role="option"
                  className={[
                    "select-option",
                    isSelected ? "is-selected" : "",
                    isActive ? "is-active" : "",
                    option.disabled ? "is-disabled" : "",
                  ].filter(Boolean).join(" ")}
                  aria-selected={isSelected}
                  aria-disabled={option.disabled}
                  disabled={option.disabled}
                  onMouseEnter={() => !option.disabled && setActiveIndex(index)}
                  onClick={() => commitIndex(index)}
                >
                  <span className="select-option-copy">
                    <span className="select-option-label">{option.label}</span>
                    {option.disabled ? (
                      <span className="select-option-meta">Unavailable for this filter combination</span>
                    ) : null}
                  </span>
                  <span className="select-option-check" aria-hidden="true">
                    {isSelected ? "Selected" : ""}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}
