# Fix8 mixed-task source backup

This branch archives the earlier laptop backup explicitly selected by the owner.
It does not reconstruct a new model or merge later Press-only experiments.

## Provenance and limits

- Input backup: C:/Users/ondec/Documents/A800/H800_xcx/source/memoryWAM_softk.
- The main model file also matches the separately retained
  C:/Users/ondec/Documents/A800/reference_memorywam_current/imagewam.py.
- The backup contains the ImageWAM/FLUX.2 memory implementation, EATB dataset and
  distributed sampler, trainer, configurations, launch scripts, and tests.
- Fix8 supports Initial2 Full, Recent4 Full (including current), and eight learned
  Gist tokens per historical frame. EATB window length is configured separately.
- The core model does not contain the later EventVLA raw-keyframe, Reward Writer,
  or RTM-ImageWAM branches.
- This is an earlier backup, NOT a proven byte-for-byte snapshot of the final
  August 8 mixed-task step_005920 training run. Later balanced-LPT sampler
  changes and final effective training overrides are not certified by this archive.
- Do not interpret the generic task YAML defaults as the final step5920 recipe.
  Do not claim reproduction of its results solely from this archive.

## Preservation

Source, configuration, script, test, dependency, license, and original README
contents are preserved byte-for-byte from the selected backup. Python bytecode
caches are excluded. No model weights, datasets, rollout videos, or credentials
are included. FIX8_SOURCE_MANIFEST.json records the original archived files.

Original CRLF line endings are retained where present. Before executing shell
scripts on Linux, convert line endings to LF in a separate working copy.

No training, rollout, or GPU allocation was performed for this publication.
Static Python parsing was checked; GPU/runtime equivalence was not tested.
The original laptop backup and existing remote experiment directories are unchanged.

External FLUX.2, simulator dependencies, weights, and datasets must be obtained
separately under their respective licenses. Consult the original README and
launch scripts for their expected locations. No external dependency is silently
substituted here.
