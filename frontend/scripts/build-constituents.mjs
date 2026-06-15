import { access, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(frontendRoot, "..");
const metadataDir = await resolveFirstExistingPath([
  path.join(repoRoot, "src", "data", "metadata"),
  path.join(frontendRoot, "src", "data", "metadata"),
]);
const outputDir = path.join(frontendRoot, "src", "data");
const outputPath = path.join(outputDir, "constituents.json");

const sources = [
  {
    indexId: "sp500",
    label: "S&P 500",
    file: "sp500_constituents_snapshot.csv",
    symbolKey: "Symbol",
    nameKey: "Security",
    sectorKey: "GICS Sector",
    yahooKey: "YahooSymbol",
  },
  {
    indexId: "nasdaq100",
    label: "Nasdaq-100",
    file: "nasdaq100_constituents_snapshot.csv",
    symbolKey: "Symbol",
    nameKey: "Security",
    sectorKey: "ICB Industry[15]",
    yahooKey: "YahooSymbol",
  },
];

main().catch((error) => {
  console.error("Failed to build constituents JSON.");
  console.error(error);
  process.exitCode = 1;
});

async function main() {
  const dedicatedSymbols = await readDedicatedSymbols();
  const bySymbol = new Map();

  for (const source of sources) {
    const csvPath = path.join(metadataDir, source.file);
    const rows = parseCsv(await readFile(csvPath, "utf8"));

    for (const row of rows) {
      const symbol = normalizeTicker(row[source.symbolKey]);
      if (!symbol) continue;

      const existing = bySymbol.get(symbol) || {
        symbol,
        security: "",
        yahooSymbol: "",
        sector: "",
        indexes: [],
        indexLabels: [],
        hasDedicatedModel: dedicatedSymbols.has(symbol),
      };

      const next = {
        ...existing,
        security: existing.security || cleanValue(row[source.nameKey]),
        yahooSymbol: existing.yahooSymbol || normalizeTicker(row[source.yahooKey]) || symbol,
        sector: existing.sector || cleanValue(row[source.sectorKey]),
        hasDedicatedModel: existing.hasDedicatedModel || dedicatedSymbols.has(symbol),
      };

      if (!next.indexes.includes(source.indexId)) next.indexes.push(source.indexId);
      if (!next.indexLabels.includes(source.label)) next.indexLabels.push(source.label);

      bySymbol.set(symbol, next);
    }
  }

  const constituents = [...bySymbol.values()].sort((left, right) => {
    if (left.security && right.security) {
      return left.security.localeCompare(right.security);
    }
    return left.symbol.localeCompare(right.symbol);
  });

  const output = {
    generatedAt: new Date().toISOString(),
    sourceDirectory: "src/data/metadata",
    counts: {
      total: constituents.length,
      sp500: constituents.filter((item) => item.indexes.includes("sp500")).length,
      nasdaq100: constituents.filter((item) => item.indexes.includes("nasdaq100")).length,
      dedicatedModels: constituents.filter((item) => item.hasDedicatedModel).length,
    },
    constituents,
  };

  await mkdir(outputDir, { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");

  console.log(
    `Generated ${output.counts.total} constituents to ${path.relative(repoRoot, outputPath)}`
  );
}

function parseCsv(input) {
  const rows = [];
  const records = [];
  let field = "";
  let record = [];
  let inQuotes = false;

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    const next = input[index + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        field += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      record.push(field);
      field = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") index += 1;
      record.push(field);
      field = "";
      if (record.some((value) => value !== "")) {
        records.push(record);
      }
      record = [];
      continue;
    }

    field += char;
  }

  if (field !== "" || record.length > 0) {
    record.push(field);
    if (record.some((value) => value !== "")) {
      records.push(record);
    }
  }

  const [header = [], ...body] = records;
  for (const rawRow of body) {
    const row = {};
    header.forEach((column, columnIndex) => {
      row[column] = rawRow[columnIndex] ?? "";
    });
    rows.push(row);
  }

  return rows;
}

async function readDedicatedSymbols() {
  const folder = await resolveFirstExistingPath(
    [
      path.join(
        repoRoot,
        "src",
        "artifacts",
        "nasdaq100_best_deep_per_symbol",
        "target_h1"
      ),
      path.join(
        frontendRoot,
        "src",
        "artifacts",
        "nasdaq100_best_deep_per_symbol",
        "target_h1"
      ),
    ],
    { optional: true }
  );

  if (!folder) {
    return new Set();
  }

  try {
    const entries = await readdir(folder, { withFileTypes: true });
    return new Set(
      entries
        .filter((entry) => entry.isDirectory())
        .map((entry) => normalizeTicker(entry.name))
        .filter(Boolean)
    );
  } catch {
    return new Set();
  }
}

function normalizeTicker(value) {
  return cleanValue(value).toUpperCase();
}

function cleanValue(value) {
  return String(value || "").trim();
}

async function resolveFirstExistingPath(paths, options = {}) {
  for (const candidate of paths) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next candidate path.
    }
  }

  if (options.optional) {
    return null;
  }

  throw new Error(`None of the expected paths exist: ${paths.join(", ")}`);
}
