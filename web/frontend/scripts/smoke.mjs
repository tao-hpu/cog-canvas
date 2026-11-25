const base = process.env.SMOKE_URL ?? "http://127.0.0.1:3800";
const response = await fetch(new URL("/", base));
const html = await response.text();

if (response.status !== 200) {
  console.error(`smoke failed: HTTP ${response.status}`);
  process.exit(1);
}

const required = [
  "Start a Conversation",
  "CogCanvas Enabled",
  "Type your message",
  "No objects extracted yet",
  "List",
  "Graph",
  "Guide",
];

for (const text of required) {
  if (!html.includes(text)) {
    console.error(`smoke failed: missing ${JSON.stringify(text)}`);
    process.exit(1);
  }
}

if (html.includes("window is not defined")) {
  console.error("smoke failed: force-graph SSR crash leaked into the document");
  process.exit(1);
}

console.log(`smoke ok ${response.status} ${html.length} bytes ${base}`);
