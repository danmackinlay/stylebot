<!-- Wave-2b work file for ship-shape-2026-09.md slice X4: every stylebot test classified K/D/P/L/T/W with a reason, per-file verdicts, the segmenter six-to-three merge, and "would the suite have caught it" for 14 QA defects (0 of 14). Part A covers the twelve largest files, part B the rest. DELETED, not stubbed, in the X4 commit — git holds it. -->

# Part A

# Test-quality audit, stylebot set A (2026-09-04, read-only)

Scope: 12 test modules, 8,959 lines, 449 test functions (parametrised tests counted once), plus `conftest.py` and `synth_helpers.py`. Every test function was read. Classes: **K** keep, **D** decorative, **P** performative, **L** low-signal duplicate, **T** tautological; **W** flags a real behaviour reached only through `CliRunner` when a library call would be sharper. Line counts are from `ast` (decorators included). "Twin" names the test that already asserts the same thing.

Fixtures (`conftest.py`): `make_blog` and `write_pairs_corpus` are good — one factory each, the two meta shapes documented at the fixture rather than re-explained in every test. `synth_helpers._fake/_targets` are fine. No fixture bloat.

---

## tests/test_synth.py (1016 lines, 47 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_pairs_validate_and_carry_synthetic_meta | K | write contract: validates, `synthetic` meta, user≠assistant |
| test_idempotent_resume | K | resume writes 0 and the file does not grow |
| test_per_generator_mode_doubles_pairs | K | `per_generator` semantics: 2× records |
| test_heading_context_prepended_both_sides | K | context prefix on both sides; generator sees body only |
| test_context_changes_synth_key | L | twin test_synth_keys:test_synth_key_golden_all_axes (axes pinned by golden) |
| test_strategy_changes_synth_key | L | same twin; merge 5/6/7 into one axis-parametrised test |
| test_reasoning_and_prompt_change_synth_key | L | same twin; `inspect.signature` line is D (signature pin) |
| test_meta_records_slop_strategy | L | twin test_strategies_coexist_no_collision (reads `slop_strategy` set) |
| test_strategies_coexist_no_collision | K | resume does not collide across strategies |
| test_resolve_strategy | K | registry resolution tuple + custom override + unknown raises |
| test_prompt_id_stable_and_distinct | L | stability is test_synth_keys:test_prompt_id_golden; fold distinctness into test_resolve_strategy |
| test_dry_run_writes_nothing | K | planned count, nothing on disk |
| test_gen_output_meta_recorded | K | `meta.gen` recorded verbatim |
| test_bare_string_generator_has_no_gen_meta | L | back-compat inverse of the above; one extra assert there |
| test_model_stats_fold_run_economics | K | per-model cost/token accumulation, live via `on_progress` |
| test_replicate_mints_new_cells | K | replicate label mints a new cell, base unlabelled |
| test_skip_covered_prevents_cross_epoch_target_doubling | K | coverage is context-agnostic; without flag the run doubles |
| test_coverage_key_folds_cosmetic_drift (×4) | K | the fold contract, one row per mechanical edit |
| test_skip_covered_survives_cosmetic_drift (×4) | L | composition of coverage_key test + skip_covered test; keep one param at most |
| test_coverage_key_keeps_genuine_edits_distinct | K | the fold's narrowness; a false "covered" loses a pair |
| test_coverage_key_leaves_unbalanced_link_alone | K | regression: malformed link not guessed |
| test_skip_covered_still_generates_a_genuinely_edited_passage | L | composition of the two above; already proven |
| test_coverage_normalisation_does_not_reach_synth_key | L | twin test_synth_keys goldens (its own comment says so) |
| test_prompts_sidecar_written_and_deduped | K | sidecar schema and dedupe on resume |
| test_transform_similarity_bounds | K | public similarity bounds, whitespace-insensitive |
| test_transform_sim_recorded_on_pairs | K | covariate recorded; classify_train gates on it |
| test_measured_strategy_registered | D | asserts a registry entry exists and a phrase absent from a prompt |
| test_reasoning_text_routed_to_sidecar_not_corpus | K | trace leaves corpus, lands joinable in sidecar |
| test_no_sidecar_without_capture | L | one `not exists()` assert; fold into the test above |
| test_max_transform_sim_gate_drops_identity_outputs | K | gate counts, never writes, dropped cell re-attempted |
| test_report_result_echoes_degenerate_count | L | twin test_synth_cli:test_hygiene_drop_lines_are_yellow_on_stdout; wrong file |
| test_max_length_ratio_gate_drops_inflated_outputs | K | inflation gate (qwen3-32b failure named) |
| test_invented_anchor_gate_drops_fabrication_lessons | K | on by default; opt-out reproduces old corpus |
| test_invented_anchor_gate_keeps_a_faithful_generator | K | false-positive side of the gate |
| test_gate_retries_redraw_and_recover | K | redraw counts and paid-draw count |
| test_gate_drops_sidecar_recovers_rate_and_wasted_spend | K | `lost_draws.jsonl` schema, cost joinable after the run |
| test_transient_provider_error_is_redrawn | K | 502 redrawn once, no error recorded |
| test_permanent_error_is_not_retried | K | 400 tried once, one error |
| test_error_lands_in_the_lost_draws_sidecar | K | error rows have kind/gate/reason |
| test_gate_redraw_failure_does_not_cost_the_session | K | regression; NB it enshrines `errors == []` (defect 4 below) |
| test_gate_retries_give_up_and_still_drop | K | bounded redraws, recovered=0 |
| test_gate_retries_are_off_unless_asked | D | asserts the default `gate_retries=0` is the default |
| test_is_anchor_faithful_selects_over_an_existing_corpus | K | corpus-level selector; the `__name__` assert is D |
| test_written_records_carry_the_five_provenance_keys | K | W8 record contract, policy digest, automation level |
| test_provenance_records_a_higher_automation_level_it_was_handed | K | pre-selected files record `automation: 2` |
| test_a_cell_claimed_mid_run_by_another_writer_is_not_duplicated | K | deterministic plan→write race; first writer wins |
| test_two_concurrent_synth_runs_write_one_record_per_cell | L | same defect via threads; twin above + test_corpus:test_two_writers_on_different_shards_still_refuse_the_same_cell |

**Verdict.** Delete: test_context_changes_synth_key, test_strategy_changes_synth_key, test_reasoning_and_prompt_change_synth_key (replace with one parametrised axis test or lean on test_synth_keys), test_meta_records_slop_strategy, test_prompt_id_stable_and_distinct, test_bare_string_generator_has_no_gen_meta, test_skip_covered_survives_cosmetic_drift, test_skip_covered_still_generates_a_genuinely_edited_passage, test_coverage_normalisation_does_not_reach_synth_key, test_measured_strategy_registered, test_no_sidecar_without_capture, test_report_result_echoes_degenerate_count, test_gate_retries_are_off_unless_asked, test_two_concurrent_synth_runs_write_one_record_per_cell. Merge: (gen_output_meta_recorded ← bare_string_generator), (reasoning_text_routed ← no_sidecar_without_capture). Estimated deletable: **~170 lines**. Good at: the gates and the sidecars — every drop path has a test that reads the artifact back and names the failure it was built for.

---

## tests/test_eval.py (1092 lines, 49 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_score_candidate_without_judge_runs_keyless | K | keyless record schema (vale/detector keys present, judge None) |
| test_score_candidate_with_fake_judge_and_eyeball | P | asserts the fake judge's own return and an argument echoed back |
| test_null_detector_is_unconfigured | D | a null object's constant shape |
| test_vale_score_degrades_gracefully_when_absent | L | env-branching; absent case is twin test_vale_score_many_short_circuits_when_vale_is_absent |
| test_vale_score_applies_glob_scoped_styles | K | regression: stdin filename matched no glob (skips without vale) |
| test_vale_score_many_empty_input_is_empty_output | K | empty batch must not spawn; 2 lines |
| test_vale_score_many_short_circuits_when_vale_is_absent | K | no spawn, full-length list, distinct dicts |
| test_vale_score_many_degrades_whole_batch_with_the_right_length (×3) | K | whole-run failures produce full-length unavailable lists |
| test_vale_score_many_degrades_when_vale_is_unexecutable | L | fourth degrade path; make it a param of the row above |
| test_vale_score_many_preserves_order_across_a_mixed_batch | K | filename-keyed join lands on the right text (skips without vale) |
| test_vale_score_many_chunks_across_invocations_without_losing_order | K | multi-invocation chunking keeps order |
| test_vale_score_agrees_with_a_batch_of_one | K | wrapper parity |
| test_vale_score_many_writes_md_files_so_glob_scoped_styles_still_fire | K | only guard for the suffix when vale is absent on CI |
| test_judge_reply_parser_extracts_score_from_surrounding_prose | K | JSON amid prose |
| test_judge_reply_parser_clamps_and_falls_back | K | clamp; NB pins the bare-digit fallback as desired (defect 2) |
| test_openrouter_judge_records_what_the_call_cost | K | `extra_body` wire contract; the usage half is mock-echo |
| test_score_pairs_file_emits_id_keyed_scores | K | scores.jsonl schema |
| test_score_pairs_file_strips_heading_context | K | judge sees body only |
| test_record_id_precedence | L | twin golden test below + test_record_id_synthetic_pairs_ignore_recut |
| test_record_id_golden_shapes_for_non_recut_records | K | golden with a stated reason (committed scores join on it) |
| test_record_id_recut_differs_from_the_original_cut | K | the collision that was the bug |
| test_record_id_is_stable_for_the_same_recut_policy | K | canonical-JSON digest, key order irrelevant |
| test_record_id_differs_across_recut_policies | L | twin test_capture:test_recut_ids_are_stable_across_reruns_at_the_same_policy (last assert) |
| test_record_id_synthetic_pairs_ignore_recut | K | synth_key precedence over recut |
| test_score_pairs_file_resumable | K | resume skips, file does not grow |
| test_score_pairs_file_keyless | L | twin test_score_candidate_without_judge_runs_keyless through the file path |
| test_score_pairs_file_concurrency_same_ids | K | id set independent of worker count |
| test_score_candidate_accepts_a_precomputed_vale_result | P | `rec["vale"] is precomputed`: identity of a passed-in argument |
| test_score_pairs_file_batches_vale_into_one_process_per_slice | K | the D57 win: one process per slice |
| test_score_pairs_file_still_streams_across_batches | K | crash-safety: file grows mid-run, resumable |
| test_score_pairs_file_survives_a_vale_batch_failure | K | batch failure degrades, other signals intact |
| test_score_pairs_file_prunes_orphans_and_reports_the_count | K | orphan prune, survivor byte-identical, consumers agree |
| test_score_pairs_file_leaves_an_orphan_free_file_alone | K | no rewrite churn on a clean file |
| test_score_pairs_file_prune_orphans_off_is_the_wrong_corpus_escape_hatch | K | the valve, no heuristic |
| test_prune_orphan_scores_round_trips_undecodable_lines | K | corrupt line survives the rewrite |
| test_prune_orphan_scores_on_a_missing_scores_file_is_a_noop | T | builds a corpus to assert a missing file returns 0 |
| test_summarize_scores_by_facet | K | facet means |
| test_carried_gen_meta_flattened_and_faceted | K | `meta.gen` flattens onto score rows |
| test_real_pair_without_gen_buckets_null | K | null bucket, no error (could use `_score_rec`) |
| test_score_candidate_accepts_a_precomputed_detector_result | P | same identity-passthrough shape as the vale one |
| test_score_pairs_file_batches_the_detector_into_one_call_per_slice | K | one batch call carrying every (record, field) |
| test_a_failing_detector_batch_falls_back_to_per_text_scoring | K | batch failure → per-text on the calling task (D59) |
| test_a_detector_that_fails_per_text_too_records_the_error_in_the_field | K | last resort: error in the field, record survives |
| test_an_async_judge_is_awaited_with_max_workers_calls_in_flight | K | semaphore peak == max_workers |
| test_a_judge_error_fails_its_own_record_and_no_other | K | per-call exception carried back |
| test_a_sync_judge_with_max_workers_above_one_is_refused | K | F19c: refusal names the async judge; serial still allowed |
| test_score_pairs_file_inside_a_running_loop_says_which_entry_point_to_await | K | F19c: no bare loop death |
| test_the_async_judges_client_is_built_on_first_await_and_closed_by_the_pass | K | F19c leak: one client, closed |
| test_chunked_detector_batch_path_agrees_with_the_per_passage_path | K | batching moves no number |

**Verdict.** Delete: test_score_candidate_with_fake_judge_and_eyeball, test_null_detector_is_unconfigured, test_vale_score_degrades_gracefully_when_absent, test_record_id_precedence, test_record_id_differs_across_recut_policies, test_score_pairs_file_keyless, test_score_candidate_accepts_a_precomputed_vale_result, test_score_candidate_accepts_a_precomputed_detector_result, test_prune_orphan_scores_on_a_missing_scores_file_is_a_noop. Merge: test_vale_score_many_degrades_when_vale_is_unexecutable → a fourth param of test_vale_score_many_degrades_whole_batch_with_the_right_length. The six `record_id` tests belong in test_pairs.py. Estimated deletable: **~90 lines**. Good at: the D57/D59 concurrency story — every "one process / one call / N in flight" claim has a counting test, and every degradation path is exercised with the record still written.

---

## tests/test_synth_cli.py (942 lines, 37 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_synth_options_repins_defaults | L | twin test_cli_options:test_defaults_repin_only_what_differs (module docstring concedes this) |
| test_synth_options_exclude_removes_option | L | twin test_cli_options:test_exclude_drops_the_flag_and_the_callback_param; half is `--help` text |
| test_synth_options_rejects_unknown_names | L | twin test_cli_options:test_defaults_reject_a_typo… + test_exclude_rejects_an_unknown_name |
| test_chunk_policy_translates_the_uncapped_cli_spelling | K | `0 → None` on the type, not the caller |
| test_chunk_policy_has_no_ordering_knob | D | asserts a deleted attribute stays deleted |
| test_iter_targets_refuses_policy_and_loose_kwargs_together | K | two spellings refused together |
| test_pop_bundles_folds_a_callback_kwargs | L | twin test_cli_options:test_bundles_from_kwargs_round_trips_a_callback |
| test_bundle_field_names_are_unambiguous | L | runs at import time already; twin test_cli_options:test_field_owners_detects_collisions |
| test_every_bundle_field_has_a_consumer | K | AST drift guard: a bundle field nobody reads fails |
| test_run_synth_takes_bundles_only | D | asserts a signature (TypeError on loose kwargs) and `is None` |
| test_synth_command_surface_is_exactly_the_bundles_plus_its_own | D | option-count ratchet (49) and name set; one line would do |
| test_synth_help_is_sectioned | D | help text layout |
| test_dry_run_stub_names_match_real_generator_names | K | dry-run resume counts only honest if stub keys equal real keys |
| test_ai_style_synth_dry_run | K, W | CLI smoke; the counts are asserted again via the sink test |
| test_ai_style_synth_sample_needs_no_data_dir | L | exit-0 only; twin test_run_synth_needs_no_data_dir_for_inspection |
| test_inspection_mode_banner_and_generation_flag_pointer | D | banner and pointer strings |
| test_ai_style_synth_requires_a_generator | K | exit 2 mapping is the CLI's one job |
| test_run_synth_raises_on_generation_errors | K | library raises with the result attached; attribution in message |
| test_ai_style_synth_still_exits_1_on_generation_errors | K | exit 1 not 2, no traceback |
| test_timeout_reaches_generator_factories | P | asserts `timeout`/`provider_sort` kwargs arrived at a patched factory |
| test_run_synth_needs_no_data_dir_for_inspection | K | inspection legal without data_dir; generating run refused |
| test_role_without_splits_is_a_config_error | K | regression: guard before `os.fspath(None)` |
| test_run_synth_without_a_sink_is_silent | K | library caller gets clean streams |
| test_run_synth_emits_every_line_through_the_sink | K | nothing bypasses the sink; declared levels only |
| test_click_emitter_routes_streams_and_colour | K | stream routing; the `click.secho` kwarg half (12 lines) is P |
| test_hygiene_drop_lines_are_yellow_on_stdout | K | regression crash + report-vs-alarm routing |
| test_max_workers_auto_rule | K | auto-16 / explicit / injected-1 policy; captured via spy but value is production's |
| test_run_synth_sessions_end_to_end | K | session covariates and window_fill through the kit |
| test_strategy_rotation_cross_product | K, W | rotation not multiplication; 60 lines of CLI argv |
| test_custom_system_file_rejects_multiple_strategies | K | exit 2 usage error |
| test_plan_sessions_separates_same_model_strategies | K | grouping by generator identity; belongs in test_synth_sessions |
| test_missing_openrouter_key_aborts_before_any_generation | K | keyless run dies before writing |
| test_reasoning_effort_rotation_cross_product | L | twin test_strategy_rotation_cross_product with `effort` swapped in; parametrise |
| test_synth_refuses_eval_posts_from_the_corpus_splits | K | the W0 leak guard |
| test_synth_eval_guard_follows_a_rename | K | aliases honoured by the guard |
| test_synth_role_eval_is_the_deliberate_opt_in | K | `--role eval` bypasses the guard |
| test_synth_without_a_splits_file_is_unguarded_but_unchanged | L | every other splits-less `run_synth` test already writes |

**Verdict.** Delete: test_synth_options_repins_defaults, test_synth_options_exclude_removes_option, test_synth_options_rejects_unknown_names, test_chunk_policy_has_no_ordering_knob, test_pop_bundles_folds_a_callback_kwargs, test_bundle_field_names_are_unambiguous, test_run_synth_takes_bundles_only, test_synth_help_is_sectioned, test_ai_style_synth_sample_needs_no_data_dir, test_inspection_mode_banner_and_generation_flag_pointer, test_timeout_reaches_generator_factories, test_synth_without_a_splits_file_is_unguarded_but_unchanged; shrink test_synth_command_surface_is_exactly_the_bundles_plus_its_own to the count ratchet; drop the `click.secho` half of test_click_emitter_routes_streams_and_colour. Merge: test_reasoning_effort_rotation_cross_product into test_strategy_rotation_cross_product (parametrise on flag/axis). Estimated deletable: **~270 lines**. Good at: the emit seam and the eval guard — the silence contract and the W0 leak are pinned at library level with a recording sink.

---

## tests/test_infer.py (1031 lines, 72 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_rewrite_identity_preserves_text_exactly | K | round-trip; LF-only, no trailing spaces (see defect 7) |
| test_rewrite_frontmatter_never_sent_to_styler | K | frontmatter never reaches the model |
| test_rewrite_transforms_and_reassembles | K | transform + separator reassembly |
| test_empty_styler_answer_keeps_input | K | blank answer keeps chunk |
| test_best_of_picks_best_candidate_and_reports | K | best-of via scorer, decision line |
| test_do_no_harm_guard_keeps_better_input | K | input competes and wins |
| test_rewrite_pairs_file_writes_output_and_resumes | K | output rows + resume; extractor asserts belong in test_pairs |
| test_headings_and_fences_protected_and_context_flows | K | headings/fences protected; heading as context prefix |
| test_anchor_guard_rejects_information_loss | K | lossy candidate disqualified before voice scoring |
| test_anchor_guard_reverts_when_all_samples_lose_content | K | all-lossy → input kept, decision names anchor |
| test_anchor_guard_rejects_invention | K | stylebot#1: fabricated URL disqualified |
| test_anchor_guard_reverts_and_names_invention | L | twin of the two above + test_anchor_guard_counts_both_causes (names "invented") |
| test_repeating_a_source_anchor_is_not_invention | K | count increase on a source key is legitimate |
| test_anchor_guard_counts_both_causes_for_one_sample | K | overlap not partition; no double count |
| test_anchor_audit_records_every_candidate | K | audit schema, truncated gets no verdict, no text in sidecar |
| test_on_decision_reports_candidates_scores_and_choice | K | preference seam payload |
| test_on_decision_kept_input_has_no_chosen_index | L | same fixture as test_do_no_harm_guard_keeps_better_input; add `on_decision` there |
| test_on_decision_anchor_rejected_candidate_is_unscored | K | rejected candidates reported with score None |
| test_on_decision_receives_heading_context_and_fires_per_prose_chunk | K | hook per prose chunk with context |
| test_inline_headings_records_no_context | K | defect A4 |
| test_protected_block_does_not_reset_heading_context | K | protected block does not break a section |
| test_content_anchors_counting | K | anchor multiset and delta |
| test_inline_anchors_are_off_by_default | L | default-is-default; fold into an inline-anchor param table |
| test_inline_math_ignores_dollar_amounts | K | regression: prices were anchors |
| test_inline_math_and_code_spans_do_not_confuse_each_other | K | code span dollar vs math |
| test_inline_code_anchor_does_not_swallow_a_fenced_block | K | single-line by construction |
| test_inline_anchors_skip_escaped_dollars_and_display_math | K | `\$` and `$$` excluded |
| test_inline_anchors_skip_double_backtick_spans | K | conservative on ``…`` |
| test_rewrite_text_inline_anchors_disqualify_a_lost_code_span | K | opt-in guard rejects dropped code span |
| test_rewrite_candidates_always_flags_inline_anchors | K | editor seam always flags inline |
| test_record_anchor_delta_reads_both_sides_of_a_pair | L | exercised by the four is_anchor_faithful tests below |
| test_anchor_faithful_rejects_a_fabrication_lesson | K | corpus selector, gain direction |
| test_anchor_faithful_rejects_a_deletion_lesson | K | lose direction the synth gate lacks |
| test_anchor_faithful_keeps_a_pair_that_carries_its_anchors_through | K | positive case, 3 lines |
| test_anchor_faithful_inline_is_opt_in | K | inline asymmetry mirrored |
| test_rewrite_candidates_scores_and_flags_anchors | K | editor result dict contract |
| test_rewrite_candidates_flags_invention_without_disqualifying | K | flags both directions, rejects neither |
| test_rewrite_candidates_dedupes_and_flags_identity | K | dedupe, identity flag, honest n_sampled |
| test_rewrite_candidates_strips_heading_context_from_samples | K | context stripped from samples |
| test_rewrite_candidates_drops_context_under_the_cross_heading_diet | K | W2: editor seam gated like rewrite_text |
| test_the_inference_and_training_chunk_budgets_are_the_same_number | K | cross-module constant contract with a stated reason |
| test_rewrite_candidates_enforces_the_single_prose_chunk_contract | K | four refusals in one |
| test_rewrite_candidates_names_an_empty_selection_as_empty | K | message regression |
| test_rewrite_candidates_still_reports_a_real_overrun_as_over_budget | L | twin of the last assert in …enforces_the_single_prose_chunk_contract |
| test_rewrite_candidates_rejects_an_unclosed_div_opener | K | F19a |
| test_rewrite_candidates_rejects_a_lone_div_closer | L | parametrise into the F19a test above |
| test_rewrite_candidates_rejects_an_opener_whose_closer_is_unselected | L | parametrise into the F19a test above |
| test_segment_default_protects_headings | K | headings break grouping; lossless |
| test_segment_inline_headings_groups_across_sections | K | cross-heading grouping |
| test_segment_inline_headings_is_lossless | K | budget-forced splits still byte-exact |
| test_segment_sees_an_indented_heading | K | defect A3 |
| test_segment_rejects_a_tab_indented_heading | K | CommonMark tab rule (merge with A3 test) |
| test_segment_inline_headings_still_protects_fences | K | fence never groupable |
| test_rewrite_candidates_rejects_heading_by_default | L | twin of `match="heading"` in …enforces_the_single_prose_chunk_contract |
| test_rewrite_candidates_accepts_section_under_inline_headings | K | editor section mode |
| test_rewrite_candidates_still_rejects_fences_under_inline_headings | K | fence refusal survives inline_headings |
| test_max_tokens_scales_with_chunk_budget | K | floor and growth |
| test_run_cli_threads_inline_headings_into_rewrite_text (×3) | P | asserts a kwarg arrived at a patched `rewrite_text`; test_smoke has the same shape |
| test_truncation_is_reported_as_truncation_not_anchor_loss | K | truncation filed as truncation |
| test_truncation_is_caught_even_with_no_anchors_to_lose | K | anchor guard blind spot |
| test_legitimate_compression_is_not_truncation | K | floor does not eat real compression |
| test_expansion_is_never_truncation | K | no ceiling, deliberately |
| test_is_truncated_boundary | K | 0.3 boundary |
| test_rewrite_candidates_flags_truncation_without_filtering | K | editor flags, never filters |
| test_prose_chunks_is_exactly_what_rewrite_text_sends | K | parity that makes chunked_detector honest |
| test_prose_chunks_drops_frontmatter_headings_and_protected_blocks | L | implied by parity test + identity round-trip |
| test_prose_chunks_honours_the_diet_it_is_handed | K | budget groups |
| test_prose_chunks_of_a_document_with_no_prose_is_empty | L | twin test_eval_integrity:test_chunked_detector_reports_no_signal_for_a_document_with_no_prose (same doc) |
| test_message_text_passes_a_plain_string_through | T | str in, same str out |
| test_message_text_rewraps_a_thinking_part_rather_than_dropping_it | K | D75: `<think>` re-rendered |
| test_message_text_survives_an_empty_or_unknown_part | K | None/unknown part → "" |
| test_a_literal_think_block_round_trips_through_the_renderer_parser | K | live defect 43f0e688; importorskip |

**Verdict.** Delete: test_anchor_guard_reverts_and_names_invention, test_on_decision_kept_input_has_no_chosen_index, test_inline_anchors_are_off_by_default, test_record_anchor_delta_reads_both_sides_of_a_pair, test_rewrite_candidates_still_reports_a_real_overrun_as_over_budget, test_rewrite_candidates_rejects_heading_by_default, test_run_cli_threads_inline_headings_into_rewrite_text, test_prose_chunks_drops_frontmatter_headings_and_protected_blocks, test_prose_chunks_of_a_document_with_no_prose_is_empty, test_message_text_passes_a_plain_string_through. Merge: the three F19a div tests into one parametrised test; the six inline-anchor tests into one `(text, expected)` table; tab-heading into the A3 test. Estimated deletable: **~100 lines**. Good at: the anchor guard and truncation — both directions, overlap, audit and the editor-seam asymmetry are all pinned with named defects.

---

## tests/test_eval_integrity.py (862 lines, 35 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_detector_stamps_its_artifact_identity_on_every_score | K | identity on every score, answerable without an embed |
| test_null_detector_has_an_identity_too | D | null object's constant shape (twin-ish test_eval:test_null_detector_is_unconfigured) |
| test_score_rows_carry_the_detector_identity | K | identity in rows and `detector_identities` |
| test_resuming_with_a_refit_detector_is_refused | K | D39 guard and its escape hatch |
| test_an_unconfigured_detector_never_trips_the_guard | T | "no raise" only; add `len(identities)==1` or fold into the guard test |
| test_compare_refuses_across_detector_artifacts | K | cross-artifact compare refused unless forced |
| test_cluster_bootstrap_is_wider_than_the_iid_one_on_a_concentrated_stratum | K | the whole point of clustering |
| test_paired_delta_declines_to_interval_a_single_cluster | K | single cluster → no interval |
| test_compare_clusters_by_post_by_default | K | n_clusters counts posts |
| test_summary_facets_by_provenance_and_names_the_headline | K | headline is the real facet |
| test_compare_scores_reports_the_real_facet_by_default | K | default provenance filter |
| test_document_scale_rewrite_emits_a_scorable_output_jsonl | K | document rows schema; the `max_chunk_chars` kwarg line is P |
| test_pangram_detector_scores_fraction_ai_through_an_injected_transport | K | wire payload/header contract |
| test_pangram_polls_the_async_task_until_a_terminal_stage | K | regression: 404 URL never caught |
| test_pangram_raises_rather_than_hanging_on_a_failed_task | K | STAGE_FAILED raises |
| test_pangram_says_none_rather_than_zero_when_the_field_is_absent | K | None not 0 |
| test_extend_eval_is_add_only_and_versions_the_stratum | K | add-only, versioned, input untouched |
| test_extend_eval_counts_a_renamed_post_as_already_pinned | L | twin test_splits:test_extending_with_a_renamed_post_pins_both_spellings (last two asserts) |
| test_check_splits_flags_a_stratum_dominated_by_one_post | K | CONCENTRATED warning |
| test_check_splits_flags_a_synthetic_majority_eval_stratum | K | EVAL PURITY warning |
| test_splits_sha256_moves_when_the_partition_does | T | a file hash changes when the file changes |
| test_extend_eval_dry_run_reports_the_move_and_writes_nothing | K | preview report, from_role, no write |
| test_extend_eval_apply_writes_the_bumped_stratum | K | apply writes v2 with note |
| test_extend_eval_refuses_a_post_that_is_not_on_disk | K | refusal, version unchanged |
| test_extend_eval_needs_posts_and_leaves_a_no_op_unversioned | K | no posts → ConfigError; no-op unversioned |
| test_extend_eval_reads_a_posts_file | K | comments/whitespace in the listing |
| test_extend_eval_cli_is_a_dry_run_by_default | K, W | dry-run default at the CLI boundary |
| test_chunked_detector_scores_chunks_not_the_document | K | chunks seen, aggregate, sd |
| test_chunked_detector_reports_no_signal_for_a_document_with_no_prose | K | None not 0.0 |
| test_chunked_detector_does_not_align_chunks_across_fields | T | asserts two different docs cut to different counts |
| test_chunked_detector_carries_the_head_identity_with_the_scale_appended | K | scale in the instrument name |
| test_a_chunk_wise_pass_refuses_to_append_to_a_whole_document_file | K | D39 across scales |
| test_a_div_policy_is_a_different_instrument_at_the_same_scale | K | policy qualname in identity |
| test_chunk_wise_document_scores_keep_the_ordinary_scores_jsonl_shape | K | aggregate lands in the usual slot |
| test_the_callout_policy_crosses_the_document_scale_seam | K | F4: rewrite and score on the same diet; 74 lines, earns it |

**Verdict.** Delete: test_null_detector_has_an_identity_too, test_splits_sha256_moves_when_the_partition_does, test_chunked_detector_does_not_align_chunks_across_fields, test_extend_eval_counts_a_renamed_post_as_already_pinned; fix or fold test_an_unconfigured_detector_never_trips_the_guard. Estimated deletable: **~40 lines**. Good at: instrument identity — every "which head, which scale, which div policy" question has a test that makes the D39 guard fire and a test that shows the fresh-file fix.

---

## tests/test_session.py (674 lines, 48 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_store_resolves_through_the_config_ladder | K | env read per call; library twin of the CLI test |
| test_closed_snapshots_are_a_sibling_tree_not_a_suffix | K | rglob invariant |
| test_snapshot_round_trips_as_ordinary_files | K | files not JSON strings; `.md` line is D |
| test_load_snapshots_carries_an_unreadable_one_as_data | K | broken snapshot does not abort a listing |
| test_snapshot_dirs_is_empty_for_a_missing_tree | T | missing dir → [] |
| test_canonical_source_keys_a_symlink_by_its_target | K | `_focus/` harness case |
| test_canonical_source_leaves_an_ordinary_path_alone | K | no existing key can move |
| test_canonical_source_falls_back_to_absolute_outside_the_tree | K | outside-tree fallback |
| test_open_is_sticky | K | refusal preserves the worst state |
| test_save_archives_the_snapshot_instead_of_deleting_it | K | archive holds the original base |
| test_reopen_restores_and_consumes_the_archive | K | stamp preserved; archive consumed |
| test_reopen_refuses_when_a_session_is_open | K | |
| test_reopen_without_an_archive_errors | K | |
| test_drop_is_recoverable_via_reopen | K | |
| test_drop_without_a_session_errors | K | |
| test_save_without_a_session_names_the_redo_route_once_one_exists | K | message names `--redo` only once an archive exists |
| test_save_with_nothing_changed_errors | K | |
| test_redo_without_an_archive_errors | K | |
| test_redo_is_a_noop_when_a_session_is_already_open | K | |
| test_reopen_then_save_matches_redo | K | redo ≡ reopen + save |
| test_redo_only_replaces_its_own_sessions_pairs | K | earlier session's pair survives |
| test_redo_replaces_this_sessions_pairs_against_the_same_base | L | twin test_reopen_then_save_matches_redo; move notice/re-archive asserts there |
| test_smart_disposition_keeps_earlier_sessions | K | |
| test_replace_all_drops_earlier_sessions_too | K | |
| test_append_keeps_even_this_sessions_interim_pairs | K | |
| test_keep_open_interim_pairs_are_replaced_by_default | K | |
| test_prior_pairs_separates_sessionless_from_earlier_sessions | K | regression: None/str sort TypeError |
| test_save_tags_and_whole_mode_reach_the_record | K | |
| test_unknown_disposition_is_an_argument_error | K | |
| test_capture_pair_appends_by_default | L | fold the first capture into the replace test below |
| test_capture_pair_replace_drops_prior_pairs_for_the_source | K | replace + notice |
| test_capture_pair_identical_texts_are_an_unready_world | K | RunError → SessionError class mapping |
| test_capture_pair_replace_without_a_source_is_an_argument_error | K | ConfigError, not SessionError |
| test_capture_pair_with_no_changed_chunk_is_an_unready_world | L | same class mapping as the identical-texts test; twin test_ai_style_log:test_diff_chunks_skips_pure_insert_and_delete |
| test_summarize_state_groups_by_source_and_counts_saves | K | |
| test_tidy_keeps_only_the_latest_save_per_source | K | |
| test_tidy_can_be_scoped_to_one_source | K | |
| test_tidy_dry_run_reports_without_writing | K | |
| test_tidy_keeps_sourceless_and_undecodable_records | K | |
| test_tidy_on_an_empty_corpus_says_so | K | 3 lines |
| test_tidy_with_nothing_to_remove_does_not_rewrite | K | no churn |
| test_recut_refuses_legacy_archives_as_a_session_error | K | class translation (third layer of the same message) |
| test_recut_with_no_archives_reports_nothing | T | empty → zeros |
| test_recut_dry_run_leaves_the_corpus_alone | K | adds stats over the library twin |
| test_corpus_summary_counts_heading_context | K | |
| test_read_pairs_of_a_missing_corpus_is_empty | T | missing file → [] |
| test_humanize_age_buckets (×4) | K | user-visible formatting table |
| test_humanize_age_passes_through_an_unparseable_stamp | K | |

**Verdict.** Delete: test_snapshot_dirs_is_empty_for_a_missing_tree, test_capture_pair_with_no_changed_chunk_is_an_unready_world, test_recut_with_no_archives_reports_nothing, test_read_pairs_of_a_missing_corpus_is_empty. Merge: test_redo_replaces_this_sessions_pairs_against_the_same_base → test_reopen_then_save_matches_redo; test_capture_pair_appends_by_default → test_capture_pair_replace_drops_prior_pairs_for_the_source. Estimated deletable: **~45 lines**. Good at: being the library twin the CLI file should defer to — small, direct, one rule per test, error classes asserted by type.

---

## tests/test_ai_style_log.py (599 lines, 30 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_diff_chunks_resolves_nearest_preceding_heading | K | nearest heading; belongs in test_capture |
| test_diff_chunks_preamble_region_has_empty_context | K | |
| test_diff_chunks_off_yields_no_context | K | |
| test_diff_chunks_picks_immediate_not_breadcrumb | K | immediate depth, D14 |
| test_diff_chunks_coalesces_contiguous_changed_paragraphs | K | contiguous coalesce rule |
| test_diff_chunks_gap_breaks_into_two_pairs | L | twin test_capture:test_diff_chunks_default_does_not_bridge_gaps |
| test_diff_chunks_skips_pure_insert_and_delete | L | twins test_smoke:test_diff_chunks_skips_pure_insert, test_capture:test_coalescing_preserves_pure_insert_delete_skip |
| test_captured_pair_has_heading_prefix_and_validates | K | D14 contract incl. body-only char counts |
| test_build_pair_record_context_matches_build_pair_content | K | capture ↔ pairs contract |
| test_no_heading_context_leaves_content_bare | K | opt-out |
| test_whole_mode_has_no_context_even_when_on | K | whole mode overrides |
| test_preamble_change_is_valid_with_no_context | L | preamble test + bare-content test already cover it |
| test_save_archives_snapshot_instead_of_deleting | L | twin test_session:test_save_archives_the_snapshot_instead_of_deleting_it |
| test_redo_replaces_prior_pair_against_same_base | L | twin test_session:test_redo_replaces_this_sessions_pairs_against_the_same_base |
| test_reopen_then_save_matches_redo | L | twin test_session:test_reopen_then_save_matches_redo |
| test_redo_preserves_cross_session_pairs | L | twin test_session:test_redo_only_replaces_its_own_sessions_pairs |
| test_reopen_refuses_when_session_already_open | L | twin test_session:test_reopen_refuses_when_a_session_is_open; exit 1 covered by unready-world test |
| test_reopen_without_archive_errors | L | twin test_session:test_reopen_without_an_archive_errors |
| test_redo_without_archive_errors | L | twin test_session:test_redo_without_an_archive_errors |
| test_redo_is_noop_when_session_already_open | L | twin test_session:test_redo_is_a_noop_when_a_session_is_already_open |
| test_drop_is_recoverable_via_reopen | L | twin test_session:test_drop_is_recoverable_via_reopen |
| test_closed_sessions_are_not_reported_as_open | L | twin test_session:test_summarize_state_groups_by_source_and_counts_saves |
| test_save_reports_sessionless_prior_pairs_without_crashing | L | twin test_session:test_prior_pairs_separates_sessionless_from_earlier_sessions |
| test_env_supplied_data_dir_is_honoured | K | only test asserting no stray `_training_pairs/` under cwd |
| test_contradictory_flags_are_usage_errors | K | exit 2 |
| test_pair_flag_misuse_is_a_usage_error | K | exit 2, both flag contradictions |
| test_unready_world_is_a_click_exception | K | exit 1 class, four cases |
| test_recut_reports_legacy_archives_as_an_error_not_a_traceback | K | exit 1, no traceback; share the legacy-blob fixture with test_session |
| test_symlinked_source_is_keyed_by_its_target | L | twin test_session:test_canonical_source_keys_a_symlink_by_its_target |
| test_ordinary_source_path_is_not_rewritten | L | twin test_session:test_canonical_source_leaves_an_ordinary_path_alone |

**Verdict.** The module docstring of test_session.py says these CLI twins "stay, as the check that the shell is thin" — but a thin shell is proven by the four exit-code tests, not by re-running every session rule through `CliRunner`. Delete all 16 L rows and the `session_repo`/`_run`/`_sources_and_stamps` scaffolding they alone use. Move the seven `diff_chunks`/capture tests to test_capture.py. Estimated deletable: **~220 lines**. Good at: the exit-code classes (2 vs 1) — that is the one thing only this file can assert.

---

## tests/test_corpus.py (582 lines, 33 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_shard_of_splits_by_producer_then_month | K | the routing rule |
| test_a_synthetic_record_with_no_timestamp_is_not_guessed_into_a_month | K | `undated`, never this month |
| test_a_shard_directory_and_a_file_have_the_same_data_dir | D | asserts `Path("/d/dd/pairs").parent.name == "dd"` |
| test_sharding_round_trips_every_record | K | byte-equal round trip, id set |
| test_the_shards_are_the_split_the_rule_names | L | shard_of test + round-trip test already prove it |
| test_config_prefers_the_shard_directory_once_it_exists | K | config seam |
| test_config_keeps_the_single_file_when_nothing_migrated | K | opt-in by existing |
| test_a_rewrite_touches_only_the_shards_whose_content_changed | K | done-criterion on mtime and bytes |
| test_rewriting_a_single_file_corpus_is_unchanged | K | back-compat |
| test_an_unparseable_line_survives_a_round_trip | K | `unparsed.jsonl` |
| test_a_duplicate_that_straddles_two_shards_is_caught | K | corpus-level uniqueness |
| test_dedupe_across_shards_keeps_the_first_and_backs_up | K | dedupe + `.bak` dir |
| test_a_single_file_corpus_still_gets_a_dot_bak | L | twin test_pairs:test_dedupe_keeps_the_first_draw; only the `.bak` name is new |
| test_synth_appends_only_to_its_own_month_shard | K | writer routing |
| test_two_writers_on_different_shards_still_refuse_the_same_cell | K | one lock per corpus + catch-up |
| test_rewrite_holds_the_corpus_lock_while_it_replaces_shards | K | real flock probe from a child process |
| test_a_rewrite_between_two_appends_does_not_swallow_the_second | K | regression: unlinked inode |
| test_an_append_between_a_read_and_its_rewrite_survives_only_under_rewrite_with | K | D60 window, deterministic |
| test_capture_save_appends_to_the_captures_shard | K | capture routing and archive location |
| test_log_save_replace_rule_rewrites_only_the_captures_shard | K | |
| test_validate_pairs_accepts_a_directory | K | a number, not exit 0 |
| test_make_splits_accepts_a_directory | K | totals over every shard |
| test_check_splits_accepts_a_directory | K | |
| test_train_dry_run_assembles_the_whole_sharded_corpus | K | |
| test_the_manifest_corpus_hash_covers_every_shard | K | |
| test_content_sha_of_a_single_file_is_still_the_file_sha | K | manifests already keyed on it |
| test_shard_pairs_migration_is_dry_run_by_default | K | D38 |
| test_shard_pairs_migration_removes_the_source_file | K | |
| test_shard_pairs_refuses_a_corpus_that_is_already_sharded | K | |
| test_eval_sees_every_shards_record_ids | K | |
| test_train_clf_assembles_from_every_shard | K | |
| test_run_pairs_mode_rewrites_every_shards_records | K | |
| test_prune_stale_accepts_a_directory | K | |

**Verdict.** Delete: test_a_shard_directory_and_a_file_have_the_same_data_dir, test_the_shards_are_the_split_the_rule_names, test_a_single_file_corpus_still_gets_a_dot_bak. Estimated deletable: **~20 lines**. Good at: exactly what its docstring promises — every consumer is checked on a total, and the locking tests are deterministic interleavings rather than hopeful races. The strongest file in the set.

---

## tests/test_report.py (624 lines, 38 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_render_writes_self_contained_html | K | no network assets |
| test_report_escapes_target_text | L | twin test_hostile_target_cannot_inject_tags_or_attributes (DOM-level) |
| test_report_max_rows_caps_table_not_stats | K | cap table, not stats |
| test_summary_stats | T | count==len(input); max>=median is arithmetic |
| test_sample_targets_seeded_deterministic | K | seeded sampling |
| test_sample_targets_caps_at_population | L | one assert; fold into the seeded test |
| test_scores_report_self_contained | L | twin test_render_writes_self_contained_html + shared-assets test |
| test_scores_report_escapes_text | L | twin test_scores_report_escapes_hostile_facet_values |
| test_scores_report_joins_and_headlines | K | join, heading stripped, delta |
| test_scores_report_sort_attributes | D | attribute names present in markup |
| test_scores_report_keyless_no_crash | K | keyless placeholder |
| test_scores_report_max_rows_caps | L | twin test_cap_note_states_what_the_stats_still_cover; 21 lines rebuild a corpus |
| test_format_scores_sample_seeded | K | |
| test_scores_report_shows_gen_params | D | `<td class=facet>` and `<th>` names present |
| test_scores_report_prompt_column_is_human_readable | K | label v1 visible, hash in tooltip |
| test_scores_report_facet_by_covariate | K | |
| test_scores_report_row_facet_attrs_and_dropdowns | K | dropdown only when a covariate discriminates |
| test_scores_report_vale_chip_keyless | K | |
| test_both_reports_share_one_css_and_one_js | K | the consolidation pin |
| test_controls_bind_to_the_data_table_not_the_headline | K | regression: wrong tbody |
| test_sort_handlers_match_the_shared_js_signature | K | markup/JS join |
| test_reports_carry_a_dark_palette | D | a CSS string is present |
| test_empty_histogram_is_an_empty_svg | P | private function's exact return literal |
| test_hostile_target_cannot_inject_tags_or_attributes | K | security, parsed DOM |
| test_scores_report_escapes_hostile_facet_values | K | four escape sites, parsed DOM |
| test_empty_targets_report_renders | K | |
| test_empty_scores_report_renders | K | no headline when empty |
| test_cap_note_states_what_the_stats_still_cover | K | both reports' cap note |
| test_long_target_is_clipped_with_the_full_text_in_the_tooltip | K | |
| test_signal_reader_degrades_instead_of_raising | K | one accessor's tolerance is all three signals' |
| test_scores_report_tolerates_malformed_signal_payloads | L | end-to-end twin of the accessor test; move `vale 0` assert to the chip test |
| test_detector_cards_appear_only_when_a_detector_ran | K | |
| test_middle_field_gets_the_neutral_badge | K | three-field rule |
| test_comparison_report_puts_every_arm_in_one_table | K | |
| test_comparison_report_drops_the_empty_covariate_columns | K | |
| test_comparison_report_does_not_truncate_the_text | K | |
| test_comparison_report_needs_two_arms | K | |
| test_comparison_columns_are_equal_width_and_not_a_flex_row | K | two named regressions |

**Verdict.** Delete: test_report_escapes_target_text, test_summary_stats, test_sample_targets_caps_at_population, test_scores_report_self_contained, test_scores_report_escapes_text, test_scores_report_sort_attributes, test_scores_report_max_rows_caps, test_scores_report_shows_gen_params, test_reports_carry_a_dark_palette, test_empty_histogram_is_an_empty_svg, test_scores_report_tolerates_malformed_signal_payloads. Estimated deletable: **~105 lines**. Good at: escaping — the DOM-skeleton comparison against a benign twin is the right way to test it and makes the substring tests redundant.

---

## tests/test_capture.py (504 lines, 24 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_capture_pairs_writes_valid_records_to_explicit_path | K | explicit path, validates, on-disk == returned |
| test_capture_pairs_archives_full_texts_for_rechunking | K | archive layout, `.md` not `.qmd`, reader shape |
| test_capture_pairs_raises_when_nothing_changed | K | |
| test_remove_pairs_for_source_roundtrips_corrupt_lines | K | |
| test_diff_chunks_default_does_not_bridge_gaps | K | legacy contract |
| test_diff_chunks_bridges_a_short_gap | K | bridged paragraph on both sides |
| test_diff_chunks_respects_gap_tolerances | K | both tolerances block independently |
| test_diff_chunks_oversized_merge_falls_back_to_parts | K | |
| test_coalescing_preserves_pure_insert_delete_skip | K | |
| test_recut_rederives_pairs_from_the_archive | K | old pairs replaced, `meta.recut` stamped |
| test_recut_is_idempotent | L | twin test_recut_ids_are_stable_across_reruns_at_the_same_policy |
| test_recut_leaves_unarchived_captures_alone | K | |
| test_recut_dry_run_does_not_write | K | |
| test_recut_output_validates | L | one assert; add to test_recut_rederives_pairs_from_the_archive |
| test_recut_preserves_a_corrupt_line | K | |
| test_archive_write_is_atomic | K | stale `.tmp` invisible and cleared |
| test_iter_archives_refuses_legacy_json | K | raises, names the migration, even beside good archives |
| test_hand_edited_archive_changes_what_recut_emits | K | the capability the layout exists for |
| test_recut_does_not_reuse_the_previous_cuts_record_ids | K | regression with the consequence spelled out |
| test_recut_ids_are_stable_across_reruns_at_the_same_policy | K | idempotent keys; different policy disjoint |
| test_recut_drops_null_edits_that_min_chars_cannot_catch | K | `max_similarity` floor |
| test_captured_records_carry_provenance | K | W8 on the capture side |
| test_capture_policy_digest_tracks_the_cut_knobs | K | digest differs across coalesce policies |
| test_recut_records_carry_provenance | K | |

**Verdict.** Delete: test_recut_is_idempotent, test_recut_output_validates (fold its assert into the rederive test). Estimated deletable: **~15 lines**. Good at: the archive as the re-cuttable source of truth — atomicity, legacy refusal, hand-edit-then-recut, and the record-id regression all read the artifact back.

---

## tests/test_classify_train.py (464 lines, 17 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_assemble_dataset_labels_and_groups | K | twins, labels, groups, provenance counts |
| test_assemble_dataset_records_synthetic_provenance | K | |
| test_assemble_dataset_extra_positives_are_injected_and_flagged | K | |
| test_evaluate_separates_classes | K | cross-val, nested C recorded, fixed C recorded |
| test_select_C_nested_and_fallback | K | |
| test_select_C_scores_only_the_validation_rows | K | perf regression pinned by a spy on the fold's own rows |
| test_subset_dataset_remaps_pairs_and_aligns_rows | K | |
| test_evaluate_facets_by_provenance | K | anti-oriented stratum proves the facet wiring |
| test_fit_head_polarity_and_artifact_roundtrip | K | polarity, artifact meta, dep-free runtime |
| test_partition_posts_deterministic_and_explicit | K | |
| test_holdout_eval_and_subset_fit_are_leakage_safe | K | name over-claims: it checks holdout mode and polarity, not leakage |
| test_train_with_splits_contract | K | eval posts never embedded; 71 lines earn it |
| test_train_splits_mutually_exclusive_with_holdout | K | |
| test_holdout_eval_facets_by_provenance | K | holdout path of the facet |
| test_assemble_drops_near_identity_pairs | K | transform_sim gate, None kept |
| test_assemble_dataset_pair_ok_is_the_callers_policy | K | D43-style policy hook |
| test_train_refuses_a_splits_file_whose_pins_left_the_tree | K | D34 check before embedding |

**Verdict.** Nothing to delete. Estimated deletable: **0 lines**. Good at: making the metric honest with a deliberately anti-oriented synthetic stratum — the facet cannot pass by accident.

---

## tests/test_train.py (409 lines, 19 tests)

| test | class | reason (≤ 15 words) |
|---|---|---|
| test_assemble_keeps_all_without_splits | K | |
| test_assemble_keeps_near_copy_pairs | K | pinned data policy (opposite of the classifier's) |
| test_assemble_filters_to_styler_role | K | |
| test_selector_and_per_passage_hooks | K | |
| test_passage_groups_fold_nested_cuts_of_one_stretch_of_prose | K | W9 |
| test_per_passage_cap_counts_a_nested_target_once | K | W9 done-criterion |
| test_chunk_scale_reports_the_per_passage_weight_histogram | K | |
| test_pair_weight_replicates_rows_in_the_train_file_only | K | replication, val unreplicated, estimate follows |
| test_pair_weight_of_is_defensive_about_junk | K | |
| test_val_split_deterministic_and_by_post | K | |
| test_validation_gate_refuses_malformed_corpus | K | |
| test_dry_run_writes_manifest_without_spending | K | runner never called |
| test_run_training_records_result_via_runner_seam | K | hp bundle handed == hp bundle recorded |
| test_file_sha256_still_reachable_as_train_attribute | D | a re-exported symbol exists |
| test_presplit_builder_seed_has_no_silent_default | P | pins a private function's signature, not the shuffle |
| test_estimate_tokens_scales_with_text | D | pins the chars-per-token literal (400 → 100) |
| test_manifest_pins_the_partition_the_selection_and_the_stack | K | W10 item 5 |
| test_blog_root_refuses_a_run_whose_eval_pins_left_the_tree | K | D34 in front of the spend |
| test_train_cli_exits_2_on_a_missing_pin | K, W | exit 2 at the CLI |

**Verdict.** Delete: test_file_sha256_still_reachable_as_train_attribute, test_presplit_builder_seed_has_no_silent_default, test_estimate_tokens_scales_with_text. Estimated deletable: **~20 lines**. Good at: the manifest as the reproducibility record — what the runner was handed is what was written.

---

## Would the suite have caught these?

1. **`capture.capture_pairs` appends outside `corpus_lock`** (`src/stylebot/capture.py:365`, bare `append_jsonl`). Path covered by test_capture:test_capture_pairs_writes_valid_records_to_explicit_path and test_corpus:test_capture_save_appends_to_the_captures_shard. **No** — both read the file afterwards; nothing probes the lock during a capture. The only lock probe (test_corpus:test_rewrite_holds_the_corpus_lock_while_it_replaces_shards) is on `rewrite`, and the interleaving tests use `PairsWriter`, which does lock.
2. **`parse_judge_reply` returns 3 with no JSON and no digit, and takes any bare digit**. Path covered by test_eval:test_judge_reply_parser_clamps_and_falls_back. **No — it enshrines the defect**: it asserts `"I would rate this a 3 overall."` → 3 as desired behaviour, and never exercises the no-digit fallback. A fix would have to change this test.
3. **`rewrite_pairs_file` rows carry no reranker identity** (`{**rec, "output": output}`), so `compare_scores` cannot see that reranker == verdict detector. Path covered by test_infer:test_rewrite_pairs_file_writes_output_and_resumes and test_corpus:test_run_pairs_mode_rewrites_every_shards_records. **No** — both assert `output` and the id set only; test_eval_integrity:test_compare_refuses_across_detector_artifacts compares two score files' detector identity, which is a different key.
4. **`_draw_past_gates` redraw exception is recorded twice and never reaches `result.errors`**. Path covered by test_synth:test_gate_redraw_failure_does_not_cost_the_session. **No — it enshrines half of it**: `assert result.errors == []` pins the exception staying out of `errors`, and the test never reads `lost_draws.jsonl`, so the double `_record_gate_drop`/lost-draw row is unobserved. test_gate_drops_sidecar_recovers_rate_and_wasted_spend counts rows only on the no-exception path.
5. **`import asyncio` before the `_draw`/`_draw_past_gates` docstrings** (fixed at HEAD, de10fe7). **No test in this set** asserts `__doc__`. tests/test_budget.py:140 does `if ast.get_docstring(node) is None: continue`, so a shadowed docstring silently undercounts and the budget test passes — the wrong direction.
6. **Bare-URL regex keeps trailing punctuation** (`infer.py:299`, `[^\s)\]]+`). Path covered by test_infer:test_content_anchors_counting. **No** — its bare URL is followed by a space; every other URL in the file sits inside `](…)`. No test has `https://x.com.` or `https://x.com,`.
7. **`rewrite_text` whitespace "verbatim" claim; CRLF and trailing-space hard breaks lost** (candidates go through `.strip()` at `rewrite_text`; `_segment` splits on `"\n"`). Path covered by test_infer:test_rewrite_identity_preserves_text_exactly and test_segment_inline_headings_is_lossless. **No** — both fixtures are LF-only with no trailing spaces; the `_segment` lossless tests cannot see the `.strip()` that happens after segmentation.
8. **`_score_record` passes an async judge into `score_candidate`'s sync fallback** (a coroutine lands in the record if a `judge_by_field` slot is missing). Path covered by test_eval:test_an_async_judge_is_awaited_with_max_workers_calls_in_flight and test_a_judge_error_fails_its_own_record_and_no_other. **No** — every test's pre-pass fills every slot; the missing-slot branch is only reachable when an extractor raised, and `_score_record` re-raises on extraction before it gets there, so the fallback is latent and unexercised.

Score: 0 of 8 would have failed; two (2, 4) actively pin the defective behaviour.

---

## Totals

| class | tests |
|---|---|
| K keep | 353 |
| L low-signal duplicate | 65 |
| D decorative | 15 |
| T tautological | 9 |
| P performative | 7 |
| **total** | **449** |

Estimated deletable lines: **~1,060 of 8,959 (≈12%)**, by file: test_synth_cli ~270, test_ai_style_log ~220, test_synth ~170, test_report ~105, test_infer ~100, test_eval ~90, test_session ~45, test_eval_integrity ~40, test_train ~20, test_corpus ~20, test_capture ~15, test_classify_train 0.

Worst signal per line (deletable ÷ file lines):
1. **tests/test_ai_style_log.py** — ~37%: sixteen `CliRunner` re-runs of rules test_session.py already pins directly.
2. **tests/test_synth_cli.py** — ~29%: option-machinery twins of test_cli_options.py, help-text pins, kwarg-capture on patched factories, and one 63-line rotation test that is the other one with a flag renamed.
3. **tests/test_report.py** — ~17%: substring escaping tests made redundant by the DOM-skeleton tests, plus markup-presence pins (tied with test_synth at ~17%, whose fat is `_synth_key`-axis tests already covered by the goldens).

Output file: /private/tmp/claude-501/-Users-dan-Source-stylebot/15060575-5b17-490e-b502-d12f96d37c52/scratchpad/tests-audit-stylebot-A.md


# Part B

# stylebot test-quality audit — part B (29 files, conftest, fixtures)

Read-only. 387 test functions over 6,770 lines (conftest excluded). Classes: **K** keep, **D** decorative, **P** performative, **L** low-signal duplicate, **T** tautological; **W** = wrong-place (CLI where a library call is sharper).

Two facts that shape everything below:

- `lib.segment_for_edit` is `return markdown.segment_spans(...)` (`src/stylebot/lib.py:94`) and `infer._segment` reads `markdown.line_kinds` (`infer.py:172`). Since 2026-09-03 there is **one** segmenter, so five of the six segmenter files are testing one function through wrappers. The "parity" and "conformance" framings both predate that collapse.
- The lossless round-trip (`"".join(spans) == doc`) is asserted **seven** times across five files; the unclosed-`:::` case **four** times; the unclosed-fence case **four** times; the mid-line `$$d$$` case **three** times.

`conftest.py`: two fixtures (`make_blog`, `write_pairs_corpus`), both used; `write_pairs_corpus` is used by test_corpus/test_train (other reviewer), none of mine. Fine.
`fixtures/`: `protection_parity.qmd` and `segmenter_conformance.qmd` are ~90% the same document (same fences, headings, math, quotes, refs, figure, callout, nested figure, unclosed div, runaway fence); conformance adds `<script>` and three HTML comments, parity adds `callout-warning` with an inner heading and the packing-parity paragraph pair. Two JSON goldens (343 + 607 lines) and `hard/` (11 excerpts + manifest).

---

## tests/test_splits.py — 18 tests, 351 lines

| test | class | reason |
|---|---|---|
| test_make_splits_deterministic_and_pinned | K | seed determinism, seed sensitivity, eval_candidates restriction: the make contract |
| test_role_of_covers_and_is_growth_stable | K | newcomer posts get a stable non-eval role; existing roles untouched |
| test_role_of_is_a_frozen_contract | K | golden hash-rule pin with the reason (whole-corpus reassignment) and bracketing |
| test_role_of_memo_does_not_leak_between_splits_docs | T | docstring admits it "passes trivially today"; guards a reverted mechanism |
| test_save_load_roundtrip_and_validation | K | file schema contract; bad version / bad rule rejected at load |
| test_summarize_roles_counts_posts_pairs_and_provenance | K | tallies check_splits consumes |
| test_check_splits_danger_report | K | four named warnings fire on a bad partition |
| test_check_splits_healthy_partition_is_quiet | K | no false positives |
| test_role_of_survives_a_rename_of_a_pinned_eval_post | K | W0 regression pin, reproduces the bug without the alias |
| test_aliases_need_no_migration_of_existing_pairs | K | alias direction invariant: hash posts keep roles |
| test_alias_chains_resolve_and_cycles_raise | K | chain resolution; cycle raises |
| test_load_splits_rejects_a_malformed_alias_map | K | load-time rejection of non-string alias and cycle |
| test_eval_stratum_is_counted_by_identity_not_by_spelling | K | two spellings, one eval post, no STALE |
| test_missing_eval_posts_is_an_error_not_a_warning | K | ERROR line when neither spelling on disk; renamed spelling suffices |
| test_check_splits_without_blog_root_checks_no_paths | L | twin test_check_splits_danger_report (add one `not any(is_error)` there) |
| test_role_of_reads_the_pinned_list_through_the_alias_map | K | F3/D34: pin spelled with the new path |
| test_extending_with_a_renamed_post_pins_both_spellings | K | extend_eval stores identity; re-add is identity no-op |
| test_require_pinned_posts_is_the_gate_check_splits_only_reported | L | docstring: "same fact as missing_eval_posts, raised"; twin test_missing_eval_posts… |

**Verdict.** Delete: memo test, without_blog_root. Merge: require_pinned_posts → missing_eval_posts. ~30 lines. Good at: the rename/alias story is pinned from every angle a user hits it (role, count, load, extend, on-disk).

## tests/test_synth_sessions.py — 13 tests, 296 lines

Overlap with `test_synth.py` (by name): `test_idempotent_resume` there is the second half of `test_parallel_stateless_run_writes_each_pair_once` here (second run → written 0, skipped n).

| test | class | reason |
|---|---|---|
| test_parallel_stateless_run_writes_each_pair_once | K | max_workers=4 + two arms + unique keys; resume half twins test_synth:test_idempotent_resume |
| test_async_injected_generator_works | K | async generate accepted (asyncio-first, D56) |
| test_keys_are_content_only_in_all_modes | K | 2026-07-21 post-mortem: keys never fold session; via privates but the invariant is public |
| test_identical_cells_collapse_within_a_run | K | duplicate text = one cell |
| test_session_history_accumulates_and_covariates_recorded | K | history shape per turn; GEN_SESSION_KEYS forward/backward |
| test_session_resume_packs_only_missing_cells | K | retry runs in a fresh session |
| test_turn_error_reflows_remainder | K | reflow + attributable error; 20-line `_assign` derivation preamble is bloat |
| test_estimate_next_prompt | P | pins private arithmetic (`chars//4`, `+100`) exactly |
| test_session_token_cap | P | pins private `0.8 × window` and precedence; precedence is covered end-to-end below |
| test_budget_bind_reflows_instead_of_dropping | K | post-mortem: budget ends sessions, sheds nothing |
| test_generator_session_budget_beats_global | K | per-generator budget precedence end-to-end |
| test_window_cap_stops_session | K | context_windows path binds; near-twin of budget_bind but a different cap |
| test_session_loop_uses_begin_session_per_session | K | one begin_session per session; none stateless |

**Verdict.** Delete: `test_estimate_next_prompt`, `test_session_token_cap`. Trim the derivation in `test_turn_error_reflows_remainder` to three lines. ~30 lines. Good at: session reflow/budget invariants are real regressions with the post-mortem named.

## tests/test_synth_generator.py — 18 tests, 372 lines

All mock `openai.AsyncOpenAI` and read the kwargs production code sent. The wire *body* (`extra_body`) is this module's whole output, so asserting it is K; asserting a plain kwarg arrived is P.

| test | class | reason |
|---|---|---|
| test_openrouter_generator_name_and_strategy | D | name-string formatting and a pass-through attribute |
| test_openrouter_reasoning_effort_enum | K | effort → `{"effort": …}` mapping plus provider sort |
| test_gen_meta_keys_match_schema_constants | K | writer == GEN_BASE_KEYS ∪ GEN_EXTRA_KEYS, both directions; facet contract |
| test_gen_meta_base_keys_are_exactly_the_plain_writer | L | twin of the previous; pins GEN_BASE_KEYS a second way |
| test_openrouter_reasoning_budget_family | K | budget-family → `max_tokens` |
| test_openrouter_reasoning_off | K | default effort "off" → `enabled: False` |
| test_openrouter_provider_sort_none_omits_field | K | no provider field, no covariate |
| test_reasoning_effort_recorded_regardless_of_shape | K | covariate is the request string, not the wire int |
| test_max_tokens_threaded | P | kwarg name arrived at patched callee |
| test_temperature_top_p_passed_and_recorded | K | sampling recorded in meta, tokens recorded; kwarg half is P |
| test_temperature_omitted_when_none | K | unset params not sent (provider defaults) |
| test_empty_choices_raises_clear_error | K | `choices=None` → named RuntimeError |
| test_truncated_slop_raises | K | finish_reason=length → actionable message |
| test_openrouter_context_windows_registry | K | registry parse; entries without window dropped |
| test_sticky_provider_pins_session_after_first_turn | K | provider pinned from turn 2; stateless never pins |
| test_no_sticky_provider_disables_begin_session | D | asserts an attribute is None |
| test_anthropic_cache_breakpoint_marks_last_history_message | K | cache_control on last history message, Anthropic only, off-switch |
| test_wire_capture_reasoning_and_truncation_tail | K | reasoning trace captured; truncation error carries tail |

**Verdict.** Delete: name_and_strategy, max_tokens_threaded, no_sticky. Merge: base_keys → gen_meta_keys. ~25 lines. Good at: the OpenRouter request contract is pinned branch by branch; the two fake-client helpers could be one.

## tests/test_merge.py — 20 tests, 297 lines

| test | class | reason |
|---|---|---|
| test_split_sections_excludes_headers_and_carries_heading | K | section carries heading, header excluded; the `or any(...)` assert is sloppy |
| test_pack_respects_budget | L | twin test_merge_respects_budget_within_section (end-to-end) |
| test_pack_keeps_oversized_paragraph_whole | L | twin test_merge_oversized_single_paragraph_kept_whole |
| test_is_list_item | P | private predicate, trivial inputs; end-to-end is test_drop_list_items_both_modes |
| test_is_link_list_prose_residual | K | link-dense prose kept: the density-metric trap, named |
| test_merge_packs_short_paragraphs | K | packing lifts short paras over the floor; merge_max_chars facet rides |
| test_merge_does_not_cross_section_headers | K | section wall |
| test_merge_respects_budget_within_section | K | budget breaks a section |
| test_merge_oversized_single_paragraph_kept_whole | K | never split mid-paragraph |
| test_merge_max_chars_hard_drop | L | twin test_synth_targets:test_max_chars_caps_giant_chunk (same 15k giant) |
| test_merge_drops_packed_link_list | K | bare-link run dropped after packing |
| test_merge_chunk_index_sequential | K | index/total contract downstream keys on |
| test_heading_context_attached_immediate | K | `heading_context="immediate"` semantics |
| test_drop_list_items_both_modes | K | flag drops bullets |
| test_cross_heading_packs_across_sections_with_headings_inline | K | v2 mode: headings inline, document order, chunk_mode stamped |
| test_cross_heading_forces_context_off | K | verbatim-prefix invariant protected |
| test_cross_heading_requires_merge | K | flag inert without merge |
| test_cross_heading_is_what_buys_length_not_the_budget | L | a finding as a test; section wall already pinned by not_cross + cross_packs |
| test_each_post_is_parsed_once | P | counts calls to a monkeypatched parser; a perf number belongs in findings.md |
| test_unparseable_frontmatter_warns_once_per_post | K | degrade to `({}, text)`, warn once |

**Verdict.** Delete: pack_respects_budget, pack_keeps_oversized, is_list_item, merge_max_chars_hard_drop, cross_heading_is_what_buys_length, each_post_is_parsed_once. ~65 lines. Good at: chunk-mode semantics (section-bounded vs cross-heading, context) are pinned as the corpus contract they are.

## tests/test_migrations.py — 15 tests, 399 lines

Docstring claims "Every command in the group is covered here"; `shard-pairs` is covered in `test_corpus.py`, not here.

| test | class | reason |
|---|---|---|
| test_migrate_dry_run_writes_nothing | K | dry-run contract |
| test_migrate_apply_converts_and_unlinks | K | on-disk shape after apply; legacy unlinked |
| test_a_legacy_archive_is_refused_and_the_named_command_exists | L | twin test_capture:test_iter_archives_refuses_legacy_json; `--help` half is D |
| test_data_group_is_hidden_from_the_help_tree | D | `group.hidden is True` and a name set |
| test_orphans_are_cells_the_current_cut_no_longer_produces | K | orphan = cell the cut no longer yields |
| test_a_real_capture_is_unchecked_rather_than_clean | K | captures are unchecked, never pruned |
| test_a_missing_source_is_reported_apart_from_a_real_orphan | K | rename-without-alias surfaces separately |
| test_an_alias_re_cuts_a_renamed_post_instead_of_orphaning_it | K | alias map shared with the re-cut |
| test_eval_contamination_drops_synthetic_and_spares_captures | K | synthetic on eval dropped, real spared |
| test_eval_contamination_follows_the_alias_map | K | W0 leak counted by identity |
| test_plan_accounts_for_the_overlap_between_the_two_sets | K | union not double-count; absent variant is absent |
| test_plan_refuses_rather_than_guessing_when_an_input_is_missing | K | MigrationError on no input |
| test_styler_passage_scale_counts_prose_not_pairs | K | containment grouping → passages; W9's number |
| test_prune_stale_dry_run_writes_nothing | K | CLI dry-run; no `.bak` |
| test_prune_stale_apply_writes_exactly_what_the_dry_run_reported | K | apply == dry-run plan; backup kept |

**Verdict.** Delete: data_group_hidden; trim legacy-archive test to the "command resolves" assert. ~22 lines. Good at: a destructive migration is pinned on exactly the safety properties (dry-run, spare captures, backup) that matter.

## tests/test_pairs.py — 25 tests, 329 lines

| test | class | reason |
|---|---|---|
| test_good_record_validates | K | baseline |
| test_wrong_system_prompt_flagged | K | frozen STYLE_SYSTEM enforced |
| test_bad_roles_flagged | K | weak (asserts truthy), but a contract |
| test_empty_assistant_flagged | K | blank target rejected |
| test_missing_meta_key_flagged | K | REQUIRED_META_KEYS |
| test_file_validation_reports_line_numbers | K | blank skipped; bad lines numbered |
| test_file_all_valid_returns_empty | L | file-level twin of test_good_record_validates; dedupe tests assert it too |
| test_build_pair_content_prepends_heading | K | shared Phase 1/2 context contract |
| test_build_pair_content_empty_context_passthrough | K | mergeable with the previous |
| test_context_pair_validates | K | context-prefixed record passes |
| test_context_must_prefix_both_sides | K | the guarded failure |
| test_absent_context_unconstrained | L | byte-identical to test_good_record_validates |
| test_cross_heading_record_validates_with_inline_headings | K | v2 representable without meta.context |
| test_gen_facet_keys_are_a_subset_of_recorded_keys | K | facet keys ⊆ recorded keys; silent-empty-column guard |
| test_pre_provenance_record_still_validates | T | `"schema_version" not in meta` is the fixture; the rest is test_good_record_validates |
| test_meta_type_violations_flagged | K | type table |
| test_meta_domain_violations_flagged | K | domain table |
| test_chunk_index_past_chunk_total_flagged | K | cross-field |
| test_bool_does_not_satisfy_an_int_field | K | the `bool ⊂ int` trap, named |
| test_system_sha_must_match_the_frozen_prompt | K | present-and-wrong flagged |
| test_provenance_shape_is_checked | K | provenance keys and types |
| test_duplicate_record_id_is_reported_against_the_later_line | K | later line blamed; `check_duplicates=False` for shards |
| test_dedupe_keeps_the_first_draw | K | first draw survives; dry vs apply; backup |
| test_dedupe_is_a_no_op_on_a_clean_file | K | no rewrite, no `.bak` |
| test_validate_pairs_cli_reports_and_fixes | K | exit 1 / `--fix --no-backup`; message half is W |

**Verdict.** Delete: file_all_valid, absent_context, pre_provenance. Merge: the two build_pair_content tests. ~18 lines. Good at: the schema is a real cross-phase contract and it is pinned by table, with the traps (bool/int, later-line blame) named.

## tests/test_serve.py — 22 tests, 350 lines

| test | class | reason |
|---|---|---|
| test_score_roundtrip_preserves_order_and_id | K | NDJSON id/order contract |
| test_info_returns_meta_as_ready_handshake | K | handshake shape |
| test_empty_texts_is_valid_and_cheap | K | empty list is a response, not an error |
| test_malformed_lines_error_without_killing_the_loop | K | loop survives bad JSON, bad types, unknown op, blank |
| test_rewrite_and_prefer_unarmed_error_politely | K | unarmed ops answer, not crash |
| test_rewrite_delegates_and_flattens_result | P | asserts stub's kwargs arrived and stub's dict came back plus id |
| test_rewrite_rejects_blank_text_and_survives_rewriter_errors | K | validation + survives ValueError |
| test_prefer_hands_decision_to_sink | K | decision validated as object |
| test_detector_exception_becomes_error_response | K | exception → error field, id kept |
| test_segment_op_matches_stylebot_markdown_on_the_parity_fixture | K | op == markdown.protected_spans on a fixture; 30-line helper is a second impl |
| test_segment_offsets_slice_back_to_the_paragraph | K | offsets slice to stripped text |
| test_segment_frees_callout_prose_only_under_the_policy | L | third surface for div_is_prose; twin test_markdown:test_div_is_prose_frees_contents… |
| test_segment_protects_an_unclosed_fence_to_end_of_document | L | fourth copy; twin test_markdown:test_unclosed_fence_runs_to_end_of_document |
| test_segment_splits_a_heading_glued_to_prose | K | the port's 28-file defect, named |
| test_segment_reports_the_nearest_heading_as_context | L | context already asserted in the glued-heading test |
| test_segment_drops_frontmatter_but_keeps_document_offsets | K | document offsets survive frontmatter |
| test_segment_min_chars_drops_short_paragraphs_only | K | min_chars |
| test_segment_offsets_are_utf16_code_units | K | VS Code indexing; names the astral-char defect |
| test_segment_rejects_a_non_string_and_survives | L | same "survives" shape as malformed_lines; fold in |
| test_score_op_uses_the_detector_batch_path_when_there_is_one | K | D57 batch path |
| test_score_op_still_works_for_a_detector_with_no_batch_path | L | twin test_score_roundtrip (stub_detector has no batch path) |
| test_serve_answers_a_failed_styler_build_and_lives | K | D47: RunError reaches client by name, loop lives (d30b2c3) |

**Verdict.** Delete: rewrite_delegates, frees_callout, unclosed_fence, no_batch_path. Merge: nearest_heading → glued_heading; rejects_non_string → malformed_lines. ~45 lines. Good at: the editor wire protocol (ids, survival, offsets, UTF-16) is exactly what the marker relies on.

## tests/test_backends.py — 19 tests, 294 lines

Note: tests 1–2 assert `ValueError`/`RuntimeError`. `ConfigError`/`RunError` subclass those, so the tests pass before and after D47 and cannot detect a regression to bare builtins (see defect 4).

| test | class | reason |
|---|---|---|
| test_build_styler_raises_plain_exceptions_not_click | K | three usage errors; should assert ConfigError, not ValueError |
| test_missing_key_is_a_runtime_error_not_a_usage_one | K | unready world is the other class; should assert RunError |
| test_openai_backend_gets_the_scaled_token_budget | P | kwarg arrived at patched `infer.openai_backend`; value pinned by opts_from_manifest test |
| test_serving_diet_falls_back_when_a_manifest_predates_chunk_scale | K | fallback and caller default |
| test_serving_diet_reads_a_mixed_scale_manifest | K | majority budget + cross-heading |
| test_serving_diet_ignores_a_thinly_supported_scale | K | thin tail doesn't drag the diet |
| test_serving_diet_needs_a_cross_heading_majority_to_inline | K | majority rule |
| test_opts_from_manifest_fills_the_checkpoint_and_the_diet | K | the 2000-token-floor bug, pinned |
| test_explicit_flags_beat_the_manifest | K | precedence; tri-state |
| test_renderer_falls_back_to_the_hyperparameter_when_the_run_left_none | K | dry-run manifests' renderer location |
| test_no_mismatch_when_nothing_disagrees | K | baseline |
| test_style_system_pin_is_checked_only_when_the_manifest_carries_one | K | stale sha flagged; absent = cannot say |
| test_renderer_and_base_model_overrides_are_reported | K | mismatch notes |
| test_opts_from_manifest_logs_the_mismatch_it_did_not_override | K | override wins, warns |
| test_run_manifest_threads_the_diet_and_the_token_budget | P W | both callees patched; asserts kwargs arrived through `run` |
| test_both_backends_return_the_shape_the_protocol_declares | D | parameter names via inspect; return is the fake's content |
| test_the_protocol_cannot_see_the_chat_template | T | `isinstance(fn, Protocol)` is the type system |
| test_an_openai_backend_warns_that_the_renderer_pin_is_uncheckable | K | "unverifiable" note, once, not on tinker |
| test_the_warning_is_skipped_when_the_manifest_pins_no_renderer | K | silence when nothing to check |

**Verdict.** Delete: scaled_token_budget, run_manifest_threads, both_backends_shape, protocol_cannot_see. ~70 lines. Good at: `serving_diet`/`opts_from_manifest`/`manifest_mismatches` are pure functions tested as such, with the truncation bug named.

## tests/test_status.py — 12 tests, 299 lines

| test | class | reason |
|---|---|---|
| test_promoted_manifest_takes_the_newest_completed_non_nixed_run | K | the promoted rule — and it enshrines defect 2's "absent = promoted" |
| test_promoted_manifest_is_none_rather_than_an_error_when_nothing_qualifies | K | None, not raise |
| test_collect_on_an_empty_world_reports_absence_and_does_not_raise | K | half-built pipeline is safe |
| test_collect_counts_a_sharded_corpus_across_every_shard | K | shard counts; flat reader same |
| test_collect_flags_a_detector_fit_under_a_stale_eval_stratum | K | the cross-artifact check |
| test_collect_says_when_a_promoted_run_predates_the_splits_pin | K | predates warning on/off |
| test_ai_style_status_prints_the_promoted_run_id_and_lists_the_rest | K W | output text; `format_status(collect(...))` would do |
| test_ai_style_status_json_round_trips | K W | `--json` serialises; library-level would do |
| test_a_verdict_manifest_is_listed_with_its_pins | K | verdict pins listed, state "not promoted" |
| test_a_verdict_manifest_does_not_become_the_promoted_one | L | promoted=False and result=None both already in the first test |
| test_manifests_are_ordered_by_their_own_clock_not_their_filename | K | D74 |
| test_an_offset_stamp_is_parsed_rather_than_compared_as_a_string | K | tz-aware ordering |

**Verdict.** Delete: verdict_not_promoted. ~12 lines. Good at: `collect` on an absent world and the cross-artifact stale-stratum check.

## tests/test_markdown.py — 34 tests, 258 lines (the canonical segmenter file)

| test | class | reason |
|---|---|---|
| test_atx_heading_accepts_up_to_three_spaces | K | CommonMark indent rule |
| test_atx_heading_rejects_a_leading_tab | K | parametrise with the other heading tests |
| test_atx_heading_rejects_hashtags_and_over_deep_levels | K | same |
| test_bare_hashes_are_an_empty_heading | K | same |
| test_heading_title_strips_hashes_and_trailing_space | K | same |
| test_heading_line_tolerates_a_trailing_newline | K | same |
| test_protected_spans_label_each_construct | K | kinds in order |
| test_unclosed_fence_runs_to_end_of_document | K | canonical home; three other copies are L |
| test_a_closed_fence_still_stops_at_its_closer | K | lazy closer |
| test_a_four_backtick_fence_can_close_itself | K | names the live 999-char overrun; comment describes pre-D69 rule |
| test_unclosed_div_yields_no_span | K | canonical (D76); three other copies are L |
| test_block_only_keeps_display_math | K | multi-line `$$` with attrs |
| test_block_only_drops_midline_math | K | canonical A5; two other copies are L |
| test_block_only_keeps_indented_display_math | K | 3 spaces yes, 4 no |
| test_segment_spans_roundtrip_is_lossless | K | canonical lossless; six other copies are L |
| test_raw_html_is_off_by_default | K | opt-in |
| test_raw_html_protects_script_blocks_when_enabled | K | blank line inside doesn't end it |
| test_raw_html_protects_style_blocks_and_ignores_case | K | STYLE/Style |
| test_raw_html_leaves_other_elements_alone | K | only raw-text elements |
| test_unclosed_script_yields_no_span | K | consistent with div rule |
| test_raw_html_requires_the_opener_to_start_a_line | K | inline `<script>` is prose |
| test_raw_html_does_not_span_two_separate_blocks | K | lazy closer |
| test_protected_kinds_gains_raw_html_only_when_enabled | D | constant equals constant ∪ {RAW_HTML} |
| test_raw_html_stays_lossless_and_reaches_line_kinds | K | line_kinds half; lossless half is a twin |
| test_div_default_protects_everything | K | default wholesale |
| test_div_is_prose_frees_contents_but_not_fences_or_nested_figures | K | canonical policy semantics |
| test_div_is_prose_stays_lossless | L | lossless twin |
| test_line_kinds_labels_one_per_line | K | one kind per line, trailing "" |
| test_line_kinds_do_not_bleed_past_a_span | K | span end respected |
| test_line_kinds_mark_blank_lines_inside_a_fence | K | blank inside fence is fence |
| test_line_kinds_ignore_midline_math | L | line_kinds is block_only; twin test_block_only_drops_midline_math |
| test_line_kinds_can_skip_headings | K | `headings=False` |
| test_line_kinds_see_callout_prose_under_policy | K | policy reaches line_kinds |
| test_line_kinds_pin_a_heading_inside_a_callout | K | canonical; parity's copy is L |

Missing here: `html_comments=True` has **no** unit test in this file (only in conformance); the one-line `$$x$$ {#eq}` case and "markers inside a fence open no spans" live only in test_segment/test_lib.

**Verdict.** Delete: protected_kinds, div_is_prose_stays_lossless, line_kinds_ignore_midline_math. Parametrise the six heading tests into one table. Absorb: one-line math+attrs (test_segment), fence-swallows-inner-markers (test_lib), html_comments (conformance). ~25 lines net. Good at: this is the segmenter's unit spec and reads like one.

## tests/test_export.py — 19 tests, 254 lines

`torch`/`safetensors` are not in the dev group, so tests 10–16 (7 of 19) `importorskip` on a default checkout: a third of the file is green by skipping.

| test | class | reason |
|---|---|---|
| test_language_model_prefix_is_stripped | K | key inversion (parametrise with next three) |
| test_lm_head_becomes_unembed_tokens | K | same |
| test_raw_keys_pass_through_unchanged | K | same |
| test_inversion_is_idempotent | K | double-strip guard, named |
| test_is_serving_format_detects_either_prefix | K | detection; empty |
| test_equal_alpha_and_r_is_accepted | T | "no raise" only |
| test_unequal_scaling_is_refused_and_says_why | K | message names both numbers |
| test_missing_scaling_fields_are_refused | K | unverifiable is refused |
| test_base_model_of_manifest | K | manifest read + missing key |
| test_normalize_writes_an_inverted_copy | K | (skips) source untouched, config rides |
| test_normalize_leaves_an_already_raw_adapter_alone | K | (skips) in-place |
| test_normalize_refuses_a_misscaled_adapter | L | (skips) twin test_unequal_scaling_is_refused_and_says_why |
| test_normalize_names_the_missing_file | K | (skips) |
| test_merge_inverts_keys_then_renames_into_place | K | (skips) raw keys seen; partial/scratch cleaned |
| test_merge_is_idempotent | K | (skips) existing merge not rebuilt |
| test_merge_refuses_a_merge_that_produced_nothing | K | (skips) no config.json → refuse, no rename |
| test_cli_needs_a_base_model_from_somewhere | K | usage error exit 2 with both flags named |
| test_cli_reads_the_base_model_from_a_manifest | P W | merge_adapter patched; asserts kwarg arrived |
| test_cli_reports_a_manifest_with_no_base_model | L W | twin test_base_model_of_manifest + test_errors' exit-1 translation |

**Verdict.** Delete: equal_alpha, normalize_misscaled, cli_reads_manifest, cli_reports_no_base_model. ~45 lines. Good at: the merge seam (`merge_fn`) lets the rename/idempotence/partial-dir contract run without a model — when the extra is installed.

## tests/test_protection_parity.py — 15 tests, 343 lines

Framing is stale: "training" = `markdown.segment_spans`, "inference" = `markdown.line_kinds`. What survives as real: (a) `segment_spans` and `line_kinds` agree line-for-line on a fixture (two derivations inside markdown.py), (b) `infer._segment`'s own lossless reassembly and packing, (c) `targets._post_targets` vs `infer._segment` chunk boundaries.

| test | class | reason |
|---|---|---|
| test_training_and_inference_protect_the_same_lines | K | segment_spans ↔ line_kinds agreement on a full fixture |
| test_headings_are_protected_at_inference | K | infer._segment's heading kind never sent as prose |
| test_segment_reassembly_is_byte_exact | K | infer._segment lossless, both heading modes |
| test_segment_reassembly_survives_a_tight_budget | K | lossless at max_chars=40 — infer's packing, not markdown's |
| test_segment_for_edit_is_lossless | L | lossless twin (test_markdown) |
| test_fixture_exercises_every_protected_kind | K | fixture self-check (merges with conformance's if fixtures merge) |
| test_fixture_contains_every_named_case | L | hand-written mirror of conformance CASES on a near-identical fixture |
| test_an_unclosed_div_protects_nothing | L | fourth copy (test_markdown, conformance, test_lib) |
| test_an_unclosed_fence_runs_to_eof_on_both_sides | L | docstring admits the line-set test "asserts it generally"; fourth copy |
| test_packing_parity_holds_when_the_policy_asks_for_it | K | targets vs infer boundaries under pack_across_protected=False |
| test_the_default_policy_packs_across_a_deleted_block | K | the declared exception; move the numbers to findings.md (P4) |
| test_callout_policy_frees_the_same_contents_on_both_sides | L | parametrise same_lines over `div_is_prose` instead |
| test_callout_policy_admits_a_heading_inside_a_callout | L | line_kinds half twins test_markdown; infer half folds into headings test |
| test_callout_policy_keeps_fences_and_nested_figures_protected | L | twin test_markdown:test_div_is_prose_frees_contents… and test_div_policy |
| test_callout_policy_stays_lossless | L | lossless twin |

**Verdict.** Delete the 9 L tests; parametrise `same_lines` over policy ∈ {None, callout}. Rename to `test_infer_segment.py` (what remains is infer's segmentation vs markdown and vs targets). ~130 lines. Good at: the packing-parity pair is the one cross-module contract nothing else covers.

## tests/test_smoke.py — 12 tests (+parametrisations), 260 lines

| test | class | reason |
|---|---|---|
| test_cli_help_runs | D | help text |
| test_pair_writes_contract_schema | K W | `log pair` writes a valid record; env precedence is test_config's |
| test_diff_chunks_skips_pure_insert | L | twin test_capture:test_coalescing_preserves_pure_insert_delete_skip |
| test_backend_flags_reach_both_commands (×2) | D | option presence in `--help` |
| test_backend_defaults_differ_where_they_are_meant_to | D | defaults are the defaults; one exit-2 sliver |
| test_cli_default_is_the_library_constant (×11) | D | `param.default is CONSTANT` — a default is the default |
| test_no_hand_written_default_strings (×3) | D | help-string grep |
| test_computed_defaults_stay_none_and_say_so | D | default is None; help mentions a number |
| test_run_backend_errors_keep_their_exit_codes (×3) | L W | twin test_errors:test_every_subcommand_exits_2/1 + test_backends raise classes |
| test_run_strict_anchors_reaches_rewrite_text | T | asserts exit 0 only; docstring says "flip when A merges" — stale |
| test_run_anchor_log_writes_a_joinable_sidecar | K W | sidecar schema and "1 lost, 1 invented" |
| test_config_require_key_is_actionable | L | twin test_config:test_require_key_missing_raises_actionable_error |

**Verdict.** Delete everything except pair_writes and anchor_log. ~155 of 260 lines. Good at: nothing the rest of the suite is not better at; the "defaults are the constants" block is the purest decorative test in the repo (11 parametrisations asserting identity).

## tests/test_errors.py — 14 tests (+2×~17 parametrisations), 277 lines

The exit-code boundary is on the **group** (`ErrorTranslatingGroup`), so the ×17 leaf parametrisations exercise three code paths (main, log, data) thirty-four times.

| test | class | reason |
|---|---|---|
| test_each_class_is_also_the_builtin_it_replaces (×3) | T | `isinstance` of a declared base |
| test_missing_extra_is_a_run_error_not_a_config_error | T | `issubclass` of a declared base |
| test_synth_errors_are_the_same_vocabulary | T | same |
| test_session_error_is_the_same_vocabulary | T | same |
| test_translate_errors_maps_the_two_classes | K | ConfigError→2, RunError→1 |
| test_a_param_hint_names_the_flag | K | BadParameter hint |
| test_a_quiet_run_error_exits_1_without_re_rendering | K | quiet=True |
| test_a_bare_builtin_is_not_translated | K | bugs get tracebacks |
| test_the_tree_is_not_empty_and_covers_both_groups | T | guards the parametrise list is non-empty; a module-level assert would do |
| test_every_subcommand_exits_2_on_a_config_error (×17) | K | the contract over the live tree; parametrise over groups, not leaves |
| test_every_subcommand_exits_1_on_a_run_error (×17) | K | same |
| test_both_groups_translate_so_a_direct_invoke_is_covered_too | D | `isinstance(group, ErrorTranslatingGroup)` |
| test_no_pairs_left_after_filtering_now_exits_2 | K | the W11.c symptom end to end |
| test_a_malformed_corpus_still_exits_1 | K | neighbour stays 1 |

**Verdict.** Delete: the four hierarchy tests, tree_not_empty, both_groups. ~50 lines; another ~30 runs saved by parametrising over groups. Good at: the two end-to-end `train` cases are the symptom, pinned.

## tests/test_classify.py — 10 tests, 170 lines

| test | class | reason |
|---|---|---|
| test_logreg_p_slop_matches_sigmoid | P | private `_logreg_p_slop` vs the same formula typed twice |
| test_sigmoid_is_stable_at_extremes | K | overflow guard |
| test_detector_returns_p_slop_polarity_and_p_dan | K | score = P(slop) polarity; record shape |
| test_detector_dim_mismatch_raises | K | RunError on dim mismatch |
| test_load_linear_head_roundtrip_and_validation | K | head.json contract |
| test_detector_composes_into_score_candidate | K | cross-module seam |
| test_runtime_import_pulls_no_ml_deps | K | the dep-free guarantee, in a subprocess |
| test_embedding_classifier_detector_exposes_a_batch_path_that_agrees | K | D57 batch == per-item |
| test_detect_many_falls_back_when_the_detector_has_no_batch_path | K | fallback loop |
| test_a_detector_with_no_batch_embedder_simply_has_no_batch_path | D | `not hasattr` |

**Verdict.** Delete: logreg, no_batch_embedder. ~15 lines. Good at: the seam's dependency-freedom and polarity are the two things a consumer relies on.

## tests/test_segmenter_conformance.py — 12 tests, 326 lines (module docstring 45 lines)

D28 is superseded; the file now pins the *livingthing configuration* (`block_only=False, raw_html=True, html_comments=True`), which nothing else in this repo exercises. That is the value. The invariant tests duplicate `test_segment_hard.violations()`.

| test | class | reason |
|---|---|---|
| test_stylebot_matches_the_conformance_fixture | K | golden with regen flag and reason |
| test_the_configurations_disagree_only_where_intended | K | the exact diff set of the two policies |
| test_html_comments_are_protected_for_livingthing_only | K | only html_comments test in the suite |
| test_midline_math_is_prose_for_stylebot_only | L | in disagree_only_where_intended and test_markdown |
| test_the_script_block_is_protected_for_livingthing_only | L | in disagree_only_where_intended and test_markdown |
| test_spans_are_sorted_disjoint_and_in_bounds | L | twin test_segment_hard:_span_shape |
| test_both_configurations_stay_lossless | L | lossless twin; _span_shape checks it |
| test_every_named_case_occurs_exactly_once | K | fixture self-check |
| test_the_fixture_exercises_every_protected_kind | L | twin test_segment_hard:test_every_protected_kind_appears_somewhere |
| test_the_headings_are_not_swallowed_by_any_span | L | twin test_segment_hard:_headings_survive |
| test_the_unclosed_div_protects_nothing | L | fourth copy |
| test_the_unclosed_fence_runs_to_end_of_document | L | fourth copy; _fence_stops_at_its_closer covers |

**Verdict.** Keep 4, delete 8; fold `segmenter_conformance.qmd` into `fixtures/hard/` as the synthetic 12th document so the hard invariants run over it (D68 already says it keeps the zero-corpus constructs). Cut the docstring to 12 lines. ~110 lines. Good at: the livingthing policy row and the diff set.

## tests/test_synth_assignment.py — 8 tests, 154 lines

| test | class | reason |
|---|---|---|
| test_rotation_uses_at_least_two_generators | L | twin test_synth_sessions:test_parallel_stateless_run… (same two asserts) |
| test_hash_assignment_is_set_stable_and_position_free | K | dropping a *target* moves nothing; salt half twins assign_seed test |
| test_hash_assignment_roughly_balances | L | its own comment: "same assertion as test_no_arm_starves…" |
| test_assignment_is_independent_of_generator_order | K | roster is a set (2026-08-10) |
| test_dropping_one_arm_churns_about_one_nth | K | HRW property; only the departed arm moves |
| test_same_name_arms_each_get_work | L | subsumed by test_no_arm_starves (36 arms, 6 names) |
| test_no_arm_starves_across_a_realistic_rotation | K | the live shape |
| test_assign_seed_rerandomizes_and_reproduces | K | replicate design; order-free |

**Verdict.** Delete: rotation, roughly_balances, same_name. ~42 lines. Good at: the HRW properties are stated as properties, with the modulo failure named.

## tests/test_cli_options.py — 11 tests, 182 lines

`test_synth_cli.py` (other reviewer) re-tests repin/exclude/unknown-name on the real bundles (`test_synth_options_repins_defaults`, `_exclude_removes_option`, `_rejects_unknown_names`). One of the two sets can go; this one is the generic mechanism and should stay.

| test | class | reason |
|---|---|---|
| test_opt_generates_flags_from_field_defaults | K | defaults, parsing, types |
| test_opt_refuses_a_duplicate_default | K | the one-declaration rule |
| test_library_only_field_gets_no_flag | K | library-only mechanism |
| test_defaults_repin_only_what_differs | K | repin; still overridable |
| test_defaults_reject_a_typo_and_a_foreign_bundle | K | foreign bundle rejected; typo half is a dataclass guarantee |
| test_defaults_reject_a_library_only_field | K | cannot repin a non-flag |
| test_exclude_drops_the_flag_and_the_callback_param | K | exclude semantics |
| test_exclude_rejects_an_unknown_name | K | strict exclude |
| test_field_owners_detects_collisions | K | two bundles, reserved names |
| test_bundles_from_kwargs_round_trips_a_callback | K | kwargs → bundles, leftovers, defaults |
| test_sections_group_help_and_fall_back_to_the_class_name | D | help layout ordering |

**Verdict.** Delete: sections. ~20 lines. Good at: a toy bundle makes the kit's rules readable in one screen.

## tests/test_lib.py — 21 tests, 197 lines

| test | class | reason |
|---|---|---|
| test_frontmatter_parses_meta_and_body | K | parametrise the seven frontmatter tests |
| test_no_frontmatter_returns_whole_text | K | same |
| test_empty_text | K | same |
| test_frontmatter_dots_terminator | K | same |
| test_frontmatter_crlf_line_endings | K | same |
| test_empty_frontmatter_block_gives_empty_meta | K | same |
| test_malformed_frontmatter_raises | K | YAMLError propagates; callers guard |
| test_split_on_blank_lines | K | parametrise the five |
| test_multiline_paragraph_stays_together | K | same |
| test_runs_of_blank_lines_collapse | K | same |
| test_no_trailing_blank_line | K | same |
| test_empty_input_gives_no_paragraphs | K | same |
| test_segment_roundtrip_is_lossless (×6) | L | lossless twin; move its six inputs into test_markdown's one lossless test |
| test_fence_swallows_inner_markers | K | not in test_markdown — move it there |
| test_unclosed_div_stays_editable | L | fourth copy |
| test_is_valid_qmd_file | K | predicate table |
| test_is_valid_qmd_file_rejects_underscore_directories (×4) | K | 2026-08-12 defect named |
| test_is_valid_qmd_file_base_makes_the_judgement_project_relative | K | `base` semantics |
| test_gather_qmd_files_walk_skips_build_dirs | L | same predicate via the walk; twin is_valid + underscore |
| test_gather_qmd_files_honours_a_non_qmd_glob | K | glob is the caller's |
| test_gather_qmd_files_explicit_list_filters_missing_and_invalid | K | explicit-list mode |

**Verdict.** Parametrise the twelve one-liners; move the two segment tests to test_markdown; delete walk_skips. ~55 lines. Good at: frontmatter edge cases (`...`, CRLF, empty block) that would corrupt the corpus silently.

## tests/test_budget.py — 6 tests, 416 lines

| test | class | reason |
|---|---|---|
| test_no_module_is_over_budget_or_over_its_allowance | K | P2 |
| test_a_renamed_over_budget_function_is_flagged | K | M1a; tests the checker's own rule |
| test_allowlist_names_real_modules | K | absent modules are never visited otherwise |
| test_prose_does_not_outgrow_the_code_it_explains | K | P2 ratio |
| test_decisions_stay_indexed_and_short | K | P3 |
| test_claude_md_stays_an_operating_manual | K | P6 |

**Verdict.** Nothing to delete; it is a linter in test clothing, and the length is the checker. Good at: exactly what CLAUDE.md says it is for.

## tests/test_provenance.py — 9 tests, 121 lines

| test | class | reason |
|---|---|---|
| test_policy_digest_is_stable_and_value_keyed | K | value-keyed, None → None |
| test_policy_digest_sees_the_div_is_prose_predicate | K | A1-class asymmetry |
| test_policy_digest_survives_a_process_boundary | K | subprocess with PYTHONHASHSEED |
| test_policy_digest_takes_a_mapping_too | K | order-free mapping |
| test_pair_provenance_always_carries_all_five_keys | K | schema; thin |
| test_repo_provenance_on_a_real_checkout | K | needs a git checkout — fails from a tarball |
| test_repo_provenance_outside_a_checkout_is_all_none | K | None outside |
| test_git_repo_root_walks_up_from_a_file | K | walk-up |
| test_git_dirty_reports_none_outside_a_repo | L | twin outside_a_checkout (blog_dirty None) |

**Verdict.** Delete git_dirty. ~3 lines. Good at: the cross-process stability requirement is tested the only honest way.

## tests/test_synth_targets.py — 8 tests, 150 lines

| test | class | reason |
|---|---|---|
| test_selector_keeps_only_human_authored | K | default selector excludes AI post |
| test_pre_selected_files_skip_selector | K | files= mode is unselected |
| test_prose_only_drops_divs | L | twin test_segment:test_div_block_is_protected / test_div_policy:test_default_protects_every_div |
| test_ignore_markers_drops_stub_paragraphs | K | marker filter |
| test_link_dump_dropped | K | bulleted link list dropped |
| test_max_chars_caps_giant_chunk | K | canonical giant-chunk drop |
| test_backward_compat_defaults_unchanged | L | identical `len(...) == 2` to test_selector_keeps_only_human_authored |
| test_stop_at_headers_truncates_trailing_section | K | cut header, case/level agnostic |

**Verdict.** Delete: prose_only_drops_divs, backward_compat. ~13 lines. Good at: hygiene knobs each get one clear case. Absorb test_div_policy's two K tests here.

## tests/test_segment_hard.py — 8 tests, 280 lines

| test | class | reason |
|---|---|---|
| test_each_hard_post_matches_its_recorded_spans | K | golden, regen flag, D68 |
| test_the_hard_corpus_breaks_exactly_where_the_corpus_does | K | invariants over fixtures; KNOWN_DEFECTS empty |
| test_the_two_paths_agree_on_the_fixture_corpus | T | `sweep(HARD)` and the RHS are the same comprehension over the same files |
| test_the_manifest_matches_the_fixtures_on_disk | L | sha of each fixture already asserted against spans.json in the golden test |
| test_every_landmark_construct_survives_in_the_corpus | K | fixture self-check |
| test_the_corpus_keeps_its_structural_edge_cases | K | nested div, mid-line math |
| test_every_protected_kind_appears_somewhere | K | fixture self-check |
| test_the_whole_blog_corpus_breaks_only_where_it_is_known_to | K | opt-in sweep (skipped by default) |

**Verdict.** Delete: two_paths_agree, manifest_matches. ~12 lines. Good at: `violations()` is the single invariant runner the other segmenter files should call instead of restating.

## tests/test_segment.py — 9 tests, 83 lines

Module docstring says "a fork of livingthing's qmd_core.segment_for_edit so the two copies don't drift" — there is no fork; `segment_for_edit` is one line calling `markdown.segment_spans`.

| test | class | reason |
|---|---|---|
| test_roundtrip_is_lossless | L | lossless twin |
| test_code_fence_is_protected | L | twin test_markdown:test_protected_spans_label_each_construct |
| test_math_block_is_protected | L | same |
| test_one_line_display_math_with_attrs_is_protected | K | `$$x = 1$$ {#eq-a}` on one line — nowhere else; move to test_markdown |
| test_midline_math_stays_in_the_prose | L | twin test_block_only_drops_midline_math; move the "high- behaviour" naming there |
| test_div_block_is_protected | L | twin label_each_construct / test_synth_targets |
| test_nested_div_blocks | L | test_markdown CALLOUT already nests |
| test_blockquote_is_protected | L | twin label_each_construct |
| test_plain_prose_is_all_editable | L | trivial |

**Verdict.** Delete the file; move one case. ~75 of 83 lines. Good at: nothing test_markdown is not.

## tests/test_synth_keys.py — 6 tests, 65 lines

Versus `test_synth.py`: `test_context_changes_synth_key`, `test_strategy_changes_synth_key`, `test_reasoning_and_prompt_change_synth_key`, `test_prompt_id_stable_and_distinct` there are relational; the goldens here are absolute. Complementary, not duplicate — but the relational four could be one parametrised test.

| test | class | reason |
|---|---|---|
| test_synth_key_golden_stateless_defaults | K | golden, reason and re-key note in the docstring |
| test_synth_key_golden_all_axes | K | golden |
| test_synth_key_golden_replicate | K | golden; replicate axis |
| test_capture_id_golden | K | golden |
| test_prompt_id_golden | K | golden |
| test_default_key_axes_pinned | D | constant equals literal; the stateless golden already fails if either moves |

**Verdict.** Delete default_key_axes. ~5 lines. Good at: exactly what its docstring says — the one place the hash recipe is pinned absolutely.

## tests/test_selection.py — 10 tests, 68 lines

All K: automation 0 / higher / missing / unparseable / string-digit / retargetable; modified_after ISO / date object / malformed / retargetable. Reasons: each is one branch of the conservative-exclude policy the corpus depends on. **Verdict.** Parametrise the six `is_human_authored` one-liners; ~15 lines. Good at: dense, one branch per line.

## tests/test_config.py — 5 tests, 42 lines

| test | class | reason |
|---|---|---|
| test_data_dir_flag_beats_env | K | precedence |
| test_data_dir_env_beats_default | K | precedence |
| test_data_dir_default | D | default is the default (fold into one precedence test) |
| test_get_key_blank_is_none | K | blank → None |
| test_require_key_missing_raises_actionable_error | K | canonical; test_smoke's copy is L |

**Verdict.** Merge the three data_dir tests. ~3 lines. Good at: the D45/D50 ladder in one screen.

## tests/test_jsonl.py — 4 tests, 36 lines

All K: skip blank/undecodable, skip non-object, `keep_undecodable` wraps `_raw` (the rewrite-path contract), missing file yields nothing. **Verdict.** Nothing. Good at: 36 lines, four behaviours, no prose.

## tests/test_div_policy.py — 4 tests, 101 lines

| test | class | reason |
|---|---|---|
| test_default_protects_every_div | K | end-to-end through iter_targets; merge into test_synth_targets |
| test_policy_admits_callout_prose_only | K | the only end-to-end of the hook |
| test_chunk_policy_carries_the_hook_without_a_flag | D | `policy.div_is_prose is _is_callout` is a dataclass field |
| test_div_is_prose_is_not_a_cli_flag | D | help text |

**Verdict.** Move the two K tests to test_synth_targets, delete the file. ~35 lines. Good at: the blog/generic boundary argument in the docstring — which belongs in D-something, not here.

---

## Would the suite have caught these?

| # | defect | covering test(s) | fails? | why |
|---|---|---|---|---|
| 1 | `check_splits` guards only pinned eval posts; a renamed hash-pool post changes role silently | test_splits:test_missing_eval_posts_is_an_error_not_a_warning, test_check_splits_danger_report (STALE), test_aliases_need_no_migration_of_existing_pairs; test_migrations:test_a_missing_source_is_reported_apart_from_a_real_orphan | **no** | `missing_eval_posts` and the STALE set iterate `eval_posts` only (`splits.py:403,448`); test_aliases_need_no_migration asserts the hash pool's silence as *the* invariant; the migrations test surfaces a missing source only for synthetic records and only under `prune-stale` |
| 2 | `promoted_manifest` serves the newest whose `promoted` is not False; `train` never writes the key; dry runs land in `runs/` | test_status:test_promoted_manifest_takes_the_newest_completed_non_nixed_run, …is_none_rather_than_an_error; test_errors:test_no_pairs_left_after_filtering_now_exits_2 | **no** | the status test asserts "absent key ⇒ promoted" as correct; no test asserts `train` writes `promoted` (grep: no writer in `src/`); dry-run manifests carry no `result` (`train.py:770`) so the `result` gate hides them from serving, but nothing tests where `train`'s default `--manifest-out` lands |
| 3 | `serve` is one blocking loop; a `rewrite` stalls every `score` behind it | test_serve:test_score_roundtrip_preserves_order_and_id, test_rewrite_delegates_and_flattens_result, test_rewrite_rejects_blank_text_and_survives_rewriter_errors | **no** | every test feeds a StringIO and reads responses in order; `test_score_roundtrip` asserts in-order delivery, which is the blocking behaviour; no test issues a slow rewrite and a score concurrently or matches responses by id |
| 4 | `build_styler` docstring names `ValueError`/`RuntimeError` after D47 | test_backends:test_build_styler_raises_plain_exceptions_not_click, test_missing_key_is_a_runtime_error_not_a_usage_one | **no** | (source now reads ConfigError/RunError at `backends.py:159`, so the defect is already fixed) no test reads a docstring; the two tests assert the builtins, which ConfigError/RunError subclass, so they pass in both states and could not even detect a regression to bare builtins |
| 5 | `PairsWriter`'s initial full read and its offset snapshot are not under one lock | test_synth:test_a_cell_claimed_mid_run_by_another_writer_is_not_duplicated, test_two_concurrent_synth_runs_write_one_record_per_cell; test_corpus:test_two_writers_on_different_shards_still_refuse_the_same_cell | **no** | `seen` is `st.disk_keys` read at plan time and `__enter__` snapshots `st_size` later (`synth.py:557,1104`); every rival append in the tests happens inside `generate` — after `__enter__` — where the tail catch-up sees it; the barrier in the thread test sits in the same window; nothing appends between the full read and the snapshot |
| 6 | `meta.system_sha` on 9 of 8,157 records; validator checks only when present | test_pairs:test_system_sha_must_match_the_frozen_prompt, test_pre_provenance_record_still_validates; test_capture:test_captured_records_carry_provenance, test_synth:test_written_records_carry_the_five_provenance_keys | **no** | absence is pinned as valid ("the version-0 tell"); new writers are asserted to stamp it, but nothing asserts `schema_version ≥ 1 ⇒ system_sha present`, and nothing measures a corpus-level rate — which is the only way 9/8,157 becomes visible |

Six for six: no. Three of the six (1, 2, 6) are pinned by a test as the intended behaviour.

---

## Totals

| class | tests | share |
|---|---|---|
| K keep | 290 | 75% |
| L low-signal duplicate | 58 | 15% |
| D decorative | 18 | 5% |
| T tautological | 11 | 3% |
| P performative | 10 | 3% |
| **total** | **387** | |

W flags (real behaviour tested through CliRunner where a call is sharper): test_backends:run_manifest_threads, test_export:cli_reads/cli_reports, test_status:ai_style_status ×2, test_smoke:pair_writes/anchor_log/run_backend_errors.

**Deletable lines: ~1,125 of 6,770 (17%)**, by file: smoke 155, parity 130, conformance 110, segment 75, backends 70, merge 65, lib 55, errors 50, serve 45, export 45, assignment 42, div_policy 35, splits 30, sessions 30, markdown 25, generator 25, migrations 22, cli_options 20, pairs 18, selection 15, classify 15, targets 13, status 12, hard 12, keys 5, provenance 3, config 3, budget 0, jsonl 0.

**Worst signal per line:**
1. `test_segment.py` — 75/83 lines (90%) restate test_markdown through a one-line wrapper, under a docstring about a fork that no longer exists.
2. `test_smoke.py` — 155/260 (60%); six of twelve tests assert help text or that a default is the default; the eleven-way parametrise asserts `x is x`.
3. `test_protection_parity.py` — 130/343 (38%); nine of fifteen tests are the fourth copy of something, and the "two sides" are two functions in one module.

(Honourable mention: `test_segmenter_conformance.py`, 110/326 — 34% deletable and a 45-line module docstring that is a decision entry, not a test.)

## Segmenter files: merge proposal

| file | lines | unique behaviour | fate |
|---|---|---|---|
| test_markdown.py | 258 | headings, spans, fences, math, raw_html, div policy, line_kinds | **keep** — canonical unit spec; absorb 3 cases (one-line `$$…$$ {#eq}`, fence-swallows-markers, html_comments); parametrise headings |
| test_segment.py | 83 | one-line display math with attrs | **delete**; move that case to test_markdown |
| test_lib.py (segment block, ~25) | — | fence swallows inner markers; six nasty lossless inputs | **move** both into test_markdown; keep test_lib for frontmatter/gather |
| test_segmenter_conformance.py | 326 | the *livingthing* configuration; html_comments; the exact policy diff set | **keep 4 tests** (golden, disagree_only_where_intended, html_comments, named_case_once); add `segmenter_conformance.qmd` to `fixtures/hard/` so `violations()` runs on it; delete 8 |
| test_segment_hard.py | 280 | span golden over 11 excerpts; the invariant runner; opt-in sweep | **keep** as the single invariant runner (`violations()`); delete 2 |
| test_protection_parity.py | 343 | segment_spans ↔ line_kinds agreement; infer._segment lossless/packing; targets↔infer packing parity | **keep 6**, rename `test_infer_segment.py`; delete 9; retire `protection_parity.qmd` in favour of `segmenter_conformance.qmd` plus the two packing paragraphs |
| test_div_policy.py | 101 | div_is_prose reaches iter_targets | **delete**; its 2 K tests go to test_synth_targets |
| test_serve.py (segment op, ~120) | — | UTF-16 offsets, min_chars, frontmatter offsets, heading context, glued heading | **keep 6**, delete 3 twins |

Net: six files → three (`test_markdown.py`, `test_segment_hard.py` + conformance rows, `test_infer_segment.py`), one fixture document instead of two near-identical ones, and the lossless / unclosed-div / unclosed-fence / mid-line-math cases asserted once each.
