import { app } from "../../scripts/app.js";

// v1 ended in refine_intensity / fix_square. v2 replaces them with fallback size.
// Migrate before LiteGraph assigns positional widget values, including subgraphs.
export function migrateVideoNodes(value) {
    if (!value || typeof value !== "object") return;
    if (Array.isArray(value.nodes) && Array.isArray(value.links)) {
        const removedLinks = new Set();
        for (const node of value.nodes) {
            if (node.type !== "PixelVideoGridRestore" || node.properties?.pixelVideoVersion === 2) continue;
            const kept = [];
            for (const [slot, input] of (node.inputs ?? []).entries()) {
                if (["refine_intensity", "fix_square"].includes(input.name)) {
                    if (input.link != null) removedLinks.add(input.link);
                } else {
                    for (const link of value.links) {
                        if (Array.isArray(link) && link[3] === node.id && link[4] === slot) link[4] = kept.length;
                        else if (!Array.isArray(link) && link.target_id === node.id && link.target_slot === slot) link.target_slot = kept.length;
                    }
                    kept.push(input);
                }
            }
            node.inputs = kept;
        }
        value.links = value.links.filter(link => !removedLinks.has(Array.isArray(link) ? link[0] : link.id));
        for (const node of value.nodes) {
            for (const output of node.outputs ?? []) {
                if (output.links) output.links = output.links.filter(id => !removedLinks.has(id));
            }
        }
    }
    if (value.type === "PixelVideoGridRestore") {
        value.properties ??= {};
        if (value.properties.pixelVideoVersion !== 2) {
            const values = value.widgets_values;
            if (Array.isArray(values)) {
                const nestedCount = { "Auto Detect": 4, "By Pixel Size": 1, "By Grid Count": 2 }[values[1]];
                const keptCount = 5 + nestedCount;
                if (nestedCount !== undefined &&
                    ["Auto", "OpenCV", "Lightweight"].includes(values[keptCount - 2]) &&
                    typeof values[keptCount - 1] === "number") {
                    values.splice(keptCount, values.length - keptCount, 8);
                } else if (typeof values.at(-1) === "boolean" &&
                           typeof values.at(-2) === "number" && values.at(-2) <= 0.5) {
                    values.splice(-2, 2, 8);
                }
            }
            if (value.widgets_values_named) {
                delete value.widgets_values_named.refine_intensity;
                delete value.widgets_values_named.fix_square;
                value.widgets_values_named.fallback_pixel_size ??= 8;
            }
            value.properties.pixelVideoVersion = 2;
        }
    }
    for (const child of Object.values(value)) migrateVideoNodes(child);
}

app.registerExtension({
    name: "PixelPipeline.VideoV2Migration",
    beforeConfigureGraph(graphData) {
        migrateVideoNodes(graphData);
    },
    nodeCreated(node) {
        if (node.comfyClass === "PixelVideoGridRestore") {
            node.properties ??= {};
            node.properties.pixelVideoVersion = 2;
        }
    },
});
