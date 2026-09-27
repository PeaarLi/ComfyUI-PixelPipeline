import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../web/video_migration.js", import.meta.url), "utf8")
    .replace('import { app } from "../../scripts/app.js";', "")
    .replace("export function", "function");
const context = vm.createContext({ app: { registerExtension() {} } });
vm.runInContext(source, context);
const migrate = context.migrateVideoNodes;
for (const mode of [
    ["Auto Detect", 8, 4, 20, 6],
    ["By Pixel Size", 3],
    ["By Grid Count", 64, 32],
]) {
    const originalValues = ["Majority Cluster", ...mode, 2, "Auto", 12, 0.25, true];
    const graph = { nodes: [
        { id: 1, type: "PixelVideoGridRestore", widgets_values: [...originalValues],
          widgets_values_named: { sampling: "Majority Cluster", refine_intensity: 0.25, fix_square: true },
          inputs: [{ name: "video", link: 10 }, { name: "refine_intensity", link: 11 }, { name: "fix_square", link: 12 }] },
        { id: 2, type: "Value", outputs: [{ links: [10, 11, 12] }] },
        { id: 3, type: "PixelGridRestore", widgets_values: [...originalValues] },
    ], links: [[10, 2, 0, 1, 0, "VIDEO"], [11, 2, 0, 1, 1, "FLOAT"], [12, 2, 0, 1, 2, "BOOLEAN"]] };
    migrate(graph);
    assert.deepEqual(graph.nodes[0].widgets_values, [...originalValues.slice(0, -2), 8]);
    assert.equal(graph.nodes[0].widgets_values_named.fallback_pixel_size, 8);
    assert.equal(graph.nodes[0].widgets_values_named.refine_intensity, undefined);
    assert.equal(graph.nodes[0].inputs.length, 1);
    assert.equal(graph.links.length, 1);
    assert.equal(graph.nodes[1].outputs[0].links.length, 1);
    assert.deepEqual(graph.nodes[2].widgets_values, originalValues);
    const once = JSON.stringify(graph);
    migrate(graph);
    assert.equal(JSON.stringify(graph), once);
    const nested = { definitions: { subgraphs: [JSON.parse(once)] } };
    migrate(nested);
    assert.equal(JSON.stringify(nested.definitions.subgraphs[0]), once);
}
console.log("PASS: 3 grid modes, positional/named values, links, image isolation, nested graphs, idempotence");
