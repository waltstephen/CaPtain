# SO-101 twin data contract (v1.2 2026-11-11: div2 scene/mix fields, section 8)

Interfaces between the twin core, the asset pipeline, the predicates and taskgen. Schema changes are backwards
compatible (new optional fields), never renames.

Domain choice (11-09): the **cube domain is the default and the only domain with teacher budget**: rigid cubes,
side 3.0-5.0 cm (size classes 3.0/3.5/4.0/4.5/5.0, +-3 % jitter), 1-5 cubes per scene in the reachable annulus
r 0.13-0.29 m, at most one container (bowl / small box) and one mat/bin (sort tasks: 2-3 bins). Broad objects (registry,
articulated, LLM free-form families) stay built as pipelines, evaluated, but get no teacher budget until cubes SFT v1 is
evaluated.

## 1. Task ids and seeds

- `react_episode.py --task-id` is an int (harness). Every spec has a fixed integer `tid`; ids never change once data
  exists:
  - `0-15`: the 16 legacy families (`soarm.episode.FAMILY_ORDER`, sorted names; data in `runs/scale_ds`, sft_v0).
  - `100-199`: hand-written cube-domain specs (`soarm/specs/cube/*.json`).
  - `1000-9999`: taskgen specs accepted by the verifier (`soarm/specs/gen/*.json`; taskgen assigns from its range).
  - `10000+`: broad-object specs.
- Episode dir: `soarm_sim_<tid>_seed-<seed>` (unchanged).
- Seed ranges (layout seeds; `soarm.taskspec.SEED_RANGES`): `armhost` 0-999,999 (teacher data 200-8,999; eval 9,000-9,999
  as before), datahost 1,000,000-1,499,999 (data; div2 mix batches use one 100k block per model from 1,100,000)
  and 1,500,000-1,999,999 (ad-hoc eval; was 1,009,000+ before div2, unused). The fixed eval suites
  (6) use the same seeds 9000+ on every machine, so results compare across machines; data never uses an eval range. `tools/batch.py --machine <name>` refuses seeds outside that machine's range.

## 2. Task spec (`soarm/taskspec.py`, JSON, schema-validated)

One JSON object per spec. `taskspec.load(tid_or_id)`, `taskspec.validate(dict)` (jsonschema + semantic checks: roles
exist, predicates registered, templates only use declared roles), `taskspec.sample(spec, seed) -> TaskInstance`.

```json
{
  "id": "cube_stack_two", "tid": 101, "version": 1,
  "domain": "cube",                       // cube | broad | legacy
  "split": "train",                       // train | heldout
  "provenance": "hand",                   // hand | llm
  "difficulty": 2,                        // 1 (one pick/place) .. 5 (long multi-step)
  "builder": "declarative",               // or "legacy:<family>" (families.py function, the 16 legacy families)
  "vars": {},                             // optional, e.g. {"dir": {"left": "to the left of", ...}}: one key drawn per seed
  "objects": [
    {"role": "base", "kind": "cube", "count": 1, "where": {"size_class": {"min": 4.0}}},
    {"role": "top",  "kind": "cube", "count": 1, "where": {"size_class": {"max_role": "base"}, "color": {"not_role": "base"}}},
    {"role": "distractors", "kind": "cube", "count": [0, 2]}
  ],
  "layout": [],                           // constraints, see 2.3; default: everything on the table, no overlap
  "goal": [{"pred": "on_top", "args": ["@top", "@base"]}],
  "instructions": ["Stack {top} on top of {base}.", "Put {top} on {base}.", "Place {top} on top of {base}."],
  "notes": "free text"
}
```

### 2.1 Object selectors (`objects[]`)
- `role` (unique), `kind` in `cube | bowl | box | pad | bin | slot | registry` or a list of kinds (one drawn per seed;
  legacy kinds `block cylinder ball tile cup` are only used by legacy builders), `count` int or `[lo, hi]` (inclusive).
- `slot`: a tight open holder for one cube, inner side = cube side + 3 % + 8 mm, 3 cm walls; size it with
  `"where": {"size_class": {"same_role": "<cube role>"}}`.
- `where` attribute constraints. Each attribute takes a value, a list (choose one), `{"not": [...]}`,
  `{"min": x}` / `{"max": x}`, `{"same_role": r}` / `{"not_role": r}` / `{"max_role": r}` (relative to another role),
  `{"distinct": true}` (all members of this role differ in this attribute). Keys combine (`{"max_role": "a",
  "not_role": "a"}` = strictly smaller). `same_role` on the role's own first member is unconstrained ("same colour as
  items[0]").
- Cubes are registry rows (`assets/registry.jsonl`, 258 procedural cubes from soarm-assets; `objects.select_assets`,
  `objects.asset_spec`, bodies by `asset_mj`). Selector attributes = registry columns: `size_class` (cm: train 3.0-5.0;
  heldout 2.5, 5.5, 6.0 - only 2.5 passes the grasp gate), `color` (12 matte + `wood`), `texture` (`plain` | `wood`
  train; `letters` | `printed` heldout), `letter`. Density (250-1200 kg/m3, mass clipped to the row's range) and friction
  (row range) are drawn per episode, never selected. The pools come from the scene config (split, sizes, textures); a
  role constrained by `color` never draws a wood cube. Assets: `python -m assets.build_cubes` ($SOARM_ASSETS, 61 MB).
- Containers/regions: `bowl` (d 10-12 cm), `box` (10-12 x 8.5-10 cm), `pad` (mat 8.5-10 cm), `bin` (= low open box,
  sort targets); attribute `color`.
- `registry` (broad realm): `where` on registry columns (`category`, `tags`, `graspable`, `split`, `dims_m` via min/max).
- Referring expressions are generated (`objects.describe`); a sample whose goal roles have no unique description is
  rejected and resampled.

### 2.2 Goal (`goal[]`): conjunction of atoms
- `{"pred": <name in predicates.PREDICATES>, "args": [...], "params": {...}}`; arg `"@role"` = the (single) object of a
  role, `"@role[i]"` = i-th member, `"@role*"` = all members spread as varargs (in_line, grouped, stacked).
- `{"pred": ..., "for_each": "role", "args": ["$", "@bowl"]}` = one atom per member (`$` = the member).
- `{"count": {"pred": "inside", "args": ["$", "@bowl"]}, "over": "role", "eq": 2}` = exactly N members satisfy it
  (also `min` / `max`). Evaluated by `soarm.tasks.evaluate_goal`; plain atoms go to `predicates.evaluate` unchanged.
- Task instances keep atoms as `(name, args, params)` tuples (count atoms: `("count", (pred, dest_args...), {...})`), so
  task.json, driver goals and the completion gate keep their format.

### 2.3 Layout constraints (`layout[]`)
- Unconstrained objects go into the reliable top-down band r 0.15-0.26 m, |angle| <= 60 deg (`taskspec.R_MIN/R_MAX/
  ANG_MAX`; the legacy annulus 0.13-0.29 / 70 deg failed scripted picks at its edges). A spec may widen it for
  non-grasped objects with `"workspace": {"r": [0.13, 0.29], "angle_deg": 70}` and pin its cubes with `region`.
- `{"type": "on", "role": "items", "support": "@mat"}`: start on top of / inside another object.
- `{"type": "stack", "roles": ["a", "b", "c"]}`: start stacked bottom -> top.
- `{"type": "distance", "a": "@x", "b": "@y", "min": 0.12, "max": 0.15}`: centre distance (m).
- `{"type": "region", "role": "x", "r": [0.15, 0.25], "angle_deg": [-40, 40]}`: polar band in the annulus.
- Every sampled start is settled in physics and rejected if any goal atom already holds (unchanged rule).

### 2.4 Instructions
- `instructions[]`: templates with `{role}` (referring expression), `{role.color}`, `{role.size_class}`, `{role.letter}`,
  `{count}`, `{%var}` (the drawn var's text); one is chosen per seed. A multi-member role renders as "the red cube, the
  blue cube and the green cube". taskgen adds paraphrases as more entries (`paraphrase_of` index optional).
- Goal args may use `"%var"` (the drawn key, e.g. a relation direction).

## 3. Object registry (`assets/registry.jsonl`, owner soarm-assets; loader `soarm/objects.py`)
One JSON row per object:
```json
{"id": "ycb_025_mug", "name": "mug", "category": "mug", "synonyms": ["cup"], "colours": ["red"],
 "dims_m": [0.09, 0.08, 0.08], "mass_kg": 0.12, "visual": {"mesh": "ycb/025/visual.obj", "texture": "ycb/025/tex.png"},
 "collision": ["ycb/025/c0.obj"], "joints": [], "graspable": true, "scripted_grasp_success": 0.8,
 "split": "train", "source": "YCB", "licence": "CC-BY-4.0"}
```
Meshes live outside git (fetched); only the jsonl and fetch scripts are committed (no file > 5 MB).

## 4. Predicates (`soarm/predicates.py`, owner soarm-skills)
`PREDICATES` name -> `fn(world, *args, **params) -> bool`. Append-only: new names with unit tests; never change an
existing predicate's semantics. Cube specs use only the existing ones (+ the `count` goal combinator, 2.2).

## 5. Scene config (`soarm.tasks.SceneConfig`, consumed by `taskspec.sample` and `scene.Randomization.sample`)
```json
{"cube_split": "train", "cube_sizes": [3.0, 3.5, 4.0, 4.5, 5.0], "cube_textures": ["plain", "wood"], "randomize": true,
 "camera_jitter_m": 0.015, "camera_rot_jitter_deg": 0.0, "table_textures": "plain", "lights": "default"}
```
- Non-default camera/table/lights fields draw a `scene_random.Appearance` (soarm-assets: CC0 table/floor/wall
  textures, random key + fill lights, camera shift kept only if the whole workspace stays in view) after the legacy
  draws; `build_scene` applies it last.
- Defaults = training scenes; the legacy specs' scenes are bit-identical to the pre-spec ones (golden test).
- Distractors the goal or the instruction can refer to, containers, mats and bins are spec roles (`distractors` with
  a count range, `container`, `bins`); `SceneConfig.clutter` only adds objects of kinds the spec never names (section
  8). Per-cube density and friction live in `ObjSpec.attrs`.
- Episodes take the config from `SOARM_SCENE_CONFIG` (JSON; `tools/batch.py --suite` sets it); `task.json` records it.

## 6. Eval suites (`soarm/specs/suites/<id>.json`, `taskspec.suite(id)`, summary `tools/eval_summary.py`)
`{"id", "description", "select": {"domain", "split", "tids"} | [selectors] | "mix": <mix id>, "exclude": [spec ids]
(optional), "seeds": "9000-9003", "config": {SceneConfig} | {"mix": <mix id>, overrides}}`
| suite | specs | seeds | jobs |
|---|---|---|---|
| `cube_seen` | 15 hand-written train cube specs (tid 100-199) | 9000-9003 | 60 |
| `cube_heldout_families` | cube_swap, cube_gather, cube_count_place, cube_unstack | 9000-9009 | 40 |
| `cube_heldout_appearance` | the 15 hand train cube specs, held-out registry cubes only (2.5 cm; letter / printed) | 9000-9003 | 60 |
| `broad` | the 16 legacy families (blocks, cylinders, balls, tiles) + held-out broad specs (10000+) | 9000-9002 | 48 |
| `div2_seen` | every div2 mix task, div2 scenes, train views | 9000 | 79 |
| `div2_heldout_views` | every div2 mix task, div2 scenes, held-out views (rear, overhead) | 9000 | 79 |
| `div3_seen` | every div3 mix task, div3 scenes, train views | 9000 | 169 |

Run: `tools/batch.py --out runs/eval_<model>/<suite> --suite <suite> --machine armhost --model ...` per suite.

## 7. Cube specs (tid 100-118, `soarm/specs/cube/`)
Train (teacher target 60 strict each, 900 total): pick_lift, stack_two, stack_three (largest at the bottom),
ordinal_stack (named order), put_in_container, put_all_in_container, sort_by_color (2 mats), sort_by_color3 (3 mats),
line_up, line_up_ordered, push_to_region, clear_region, relational_place, slot_insert, multi_step.
Held out: swap, gather, count_place (exactly two of three), unstack. Robot intro v3 (collection) adds the fixed-jaw
offset rule: centred grasps of cubes wider than ~3.5 cm land the fixed jaw on the cube.

## 8. div2 training distribution (v1.2, 2026-11-11, branch soarm-div2)
- SceneConfig fields (all default = pre-div2 draws, golden test unchanged): `table_mount` center|edge (edge: base
  clamped 6-10 cm in front of the table's back edge, table 0.55-0.85 m deep, 0.75-1.4 m wide, base +-0.2 m along the
  edge, legs, floor 0.65-0.80 m down), `camera` jitter|train|heldout|<style> (`scene_random.VIEW_STYLES`: train =
  front 30 % / front_high 15 % / oblique 25 % / side 15 % / low 15 %; held-out = rear (|az| 130-180) and overhead
  (el 75-88), eval only; fovy fitted so the whole workspace is in view), `layout` band|mixed (`taskspec.LAYOUT_STYLES`
  uniform 30 / cluster 25 / side 15 / row 15 / spread 15 over the band + a r 0.17-0.24 wing to 70 deg), `clutter`
  [lo, hi] non-goal objects of `clutter_kinds` the spec does not use (placed last in the r 0.13-0.29 annulus, dropped
  when there is no room), `platforms` (probability of one 1-4 cm static box; `TaskInstance.platforms`).
- Directions are table-fixed: front = +x, right = +y as seen from in front of the robot facing it (predicates
  unchanged). Robot intro v4 says so and replaces TASK.md's camera-relative relation sentence.
- Specs: `specs/cube_var` (tid 200+) and `specs/broad` (tid 10000+, split train, primitives) from
  `tools/make_variants.py` (append-only list; tid = position); taskgen specs `specs/gen` (tid 1000+).
- Mixes (`specs/mixes/<id>.json`, `soarm/mix.py`, written by `tools/make_mix.py` from scripted-solver results): scene
  config + intro + per-task weights (shares of strict successes) + difficulty floor. A mix holds only `train_safe`
  specs (no grouped / at_initial_of / count goals, no stacked start + resting_on_table goal) and train cubes/views.
  `tools/batch.py --mix div2 --target-total N --seed-start <block>`.
- Suites `div2_seen` / `div2_heldout_views`: every mix task on seed 9000, train vs held-out views. Suites may list
  several selectors (`select: [...]`, keys domain/split/tids), name a `mix`, and take `config: {"mix": id, ...}`.

## 9. div3 training distribution (v1.3, 2026-11-11, branch soarm-div3)
- Scene `configs/div3_scene.json` (= the div3 mix config). New SceneConfig values/fields, all default = the div2
  draws (div2 task samples and appearances are bit-identical): `table_mount` edge3 (`scene_random.TABLE_RANGES`:
  back edge 5-11 cm behind the base, 0.50-0.95 m deep, 0.70-1.50 m wide, base +-0.25 m along the edge, top 2-5 cm,
  floor 0.58-0.92 m down; covers |y| <= 0.33 and x <= 0.39), `lights` random3 (`LIGHT_RANGES`: key elevation
  25-88 deg, intensity 0.25-1.0, colour shift +-18 %, ambient 0.08-0.5, 0-3 fill lights), `layout` div3
  (`taskspec.LAYOUT_STYLES3`: uniform 16 / cluster 12 / side 10 / row 10 / spread 10 / near 10 / far 10 / wide 8 /
  grid 7 / two_clusters 7; with p 0.5 the containers/mats/trays go first into one sub-band near/far/left/right/centre;
  yaw mode random 60 / table-aligned 25 / one shared yaw 15), `platforms_max` (1..n static boxes when present),
  `distractor_stacks` (p that two "distractors" cubes start stacked, larger below; no goal names them),
  `cube_size_weights` (relative draw weight per `cube_sizes` entry; div3 0.5/0.7/1.2/1.3/1.3 -> ~76 % 4-5 cm).
  div3 also uses clutter 0-3 of cylinder/block/cup and p 0.3 of 1-2 platforms. Camera views: train set only.
- Size-variant kinds (`taskspec.PSEUDO_KINDS`, scene kind + noun): tray (box 14-18 x 11-14 cm, 2.2-3 cm rim, noun
  "tray"), big_bowl (13-15 cm), big_mat (12-15 cm), big_cylinder (3.6-4.2 x 5-7 cm), big_block (6-7 x 3-3.5 cm).
- Specs: div3 variants appended to `tools/make_variants.py` (cube tids 249+, broad tids 10016+, notes "div3
  variant"): MoGe-2-tolerant goals (trays, big bowls/mats, several objects to one region, clearing, sorting into
  regions, two-step chains of easy steps); GSO/YCB mesh specs `specs/broad_mesh` (tid 11000+, kind registry); the
  taskgen specs in `specs/gen` (tid 1100+ round 1, 1200+ div3 round with the "tolerant" axis).
- Mix div3 (`tools/make_mix.py --shares div3 --precise-factor 0.35`): groups cube_base 14 / cube_var 22 / cube_div3 26 /
  cube_llm 10 / noncube 16 / mesh 12 %; depth-precise specs (`make_mix.precise`: on_top / stacked / in_line goals,
  slots, relations to a cube, the single-step tasks that collapsed under MoGe-2) keep their place at 0.35x weight;
  per-task cap 2.5 %, retire rule as div2.
