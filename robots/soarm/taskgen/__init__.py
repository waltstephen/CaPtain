"""soarm-taskgen: LLM-proposed task families as JSON specs + a verifier pipeline (schema -> layouts -> reachability /
no initial success -> scripted solvability in physics -> difficulty -> dedupe) + paraphrases checked by a resolver.

Specs first, no free-form generated code (GenSim, Wang et al. 2023 / RoboGen, Wang et al. 2023 generate simulation code;
here the LLM only fills a schema whose every field is interpreted by `taskgen.spec`).
"""
