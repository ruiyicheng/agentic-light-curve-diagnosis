import json
import math
from pathlib import Path

import tools.BLS_diagnosis as BLS_diagnosis
import tools.BLS_plot as BLS_plot
import tools.GLS_diagnosis as GLS_diagnosis
import tools.GLS_plot as GLS_plot
import tools.metadata_diagnose as metadata_diagnose
import tools.parse_light_curve_df as parse_light_curve_df
import tools.plot_whole_light_curve as plot_whole_light_curve
import tools.summary_all as summary_all
import tools.use_vlm as use_vlm
import tools.view_full_plot as view_full_plot
import tools.zoom_in_diagnosis as zoom_in_diagnosis
import tools.zoom_in_plot as zoom_in_plot

class LightcurveAagent:
    def __init__(self,MAX_ZOOM_IN_DIAGNOSIS_ITERATIONS = 8,
                 MAX_GLS_PREWHITENING_ITERATIONS = 3,
                 ALL_VARIABLE_TYPES = set(['EA', 'EB', 'EW', 'EP','MICROLENSING','ELL','SXA','ACV','FKCOM','BY','UV','RS','RCB','FU','GCAS','WR','LBV','SN','ZAND','N','UG','RPHS','PVTEL','GWVIR','ZZ','Hot OB Supergiants','ACYG','BE','BCEP','SPB','SXPHE','DSCT','PMSDSCT','roAP','GDOR','RRAB','RRC','RRD','DCEP','CW','RV','MIRA','SR','L','SARV', 'CST'])):
        
        self.ALL_VARIABLE_TYPES =ALL_VARIABLE_TYPES


        self.MAX_ZOOM_IN_DIAGNOSIS_ITERATIONS = MAX_ZOOM_IN_DIAGNOSIS_ITERATIONS
        self.MAX_GLS_PREWHITENING_ITERATIONS = MAX_GLS_PREWHITENING_ITERATIONS
        self.metadata_diagnosis = None
        self.gls_period_records = []


        self.vlm = use_vlm.VLM()

    def _finite_positive_float(self, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0:
            return None
        return number

    def _selected_gls_period_day(self, plot_result, diagnosis_result):
        best_period = self._finite_positive_float(plot_result.get("best_period"))
        if best_period is None or not diagnosis_result:
            return None

        period_label = diagnosis_result.get("best_period")
        if period_label == "P":
            return best_period
        if period_label == "P/2":
            return best_period / 2.0
        if period_label == "2P":
            return best_period * 2.0
        return None

    def _gls_folded_periods(self, plot_result):
        best_period = self._finite_positive_float(plot_result.get("best_period"))
        if best_period is None:
            return []
        return [
            {"label": "P", "period_day": best_period},
            {"label": "P/2", "period_day": best_period / 2.0},
            {"label": "2P", "period_day": best_period * 2.0},
        ]

    def _gls_removed_signal_periods(self, plot_result):
        removed_signals = (
            plot_result.get("prewhitening", {}).get("removed_signals", [])
            if isinstance(plot_result.get("prewhitening"), dict)
            else []
        )
        periods = []
        for signal in removed_signals:
            period = self._finite_positive_float(signal.get("period_day"))
            if period is None:
                continue
            periods.append(
                {
                    "signal_number": signal.get("signal_number"),
                    "period_day": period,
                    "power": signal.get("power"),
                }
            )
        return periods

    def _record_gls_period_result(self, step_name, plot_result, diagnosis_result):
        selected_period = self._selected_gls_period_day(plot_result, diagnosis_result)
        record = {
            "step": step_name,
            "analysis_type": plot_result.get("analysis_type", "GLS"),
            "best_period_day": self._finite_positive_float(plot_result.get("best_period")),
            "diagnosed_best_period_label": diagnosis_result.get("best_period"),
            "diagnosed_selected_period_day": selected_period,
            "folded_periods_day": self._gls_folded_periods(plot_result),
            "removed_signal_periods_day": self._gls_removed_signal_periods(plot_result),
        }
        self.gls_period_records.append(record)

    def _period_entries_for_ratios(self, records, current_plot_result=None):
        entries = []
        for record in records:
            selected_period = self._finite_positive_float(
                record.get("diagnosed_selected_period_day")
            )
            if selected_period is not None:
                entries.append(
                    {
                        "id": f"{record['step']}:diagnosed_{record.get('diagnosed_best_period_label')}",
                        "period_day": selected_period,
                    }
                )
            best_period = self._finite_positive_float(record.get("best_period_day"))
            if best_period is not None:
                entries.append(
                    {
                        "id": f"{record['step']}:best_P",
                        "period_day": best_period,
                    }
                )
            for folded_period in record.get("folded_periods_day", []):
                period = self._finite_positive_float(folded_period.get("period_day"))
                if period is None:
                    continue
                entries.append(
                    {
                        "id": f"{record['step']}:folded_{folded_period.get('label')}",
                        "period_day": period,
                    }
                )
            for signal in record.get("removed_signal_periods_day", []):
                period = self._finite_positive_float(signal.get("period_day"))
                if period is None:
                    continue
                entries.append(
                    {
                        "id": f"{record['step']}:removed_signal_{signal.get('signal_number')}",
                        "period_day": period,
                    }
                )

        if current_plot_result is not None:
            current_best_period = self._finite_positive_float(
                current_plot_result.get("best_period")
            )
            if current_best_period is not None:
                entries.append(
                    {"id": "current_plot:best_P", "period_day": current_best_period}
                )
            for folded_period in self._gls_folded_periods(current_plot_result):
                entries.append(
                    {
                        "id": f"current_plot:folded_{folded_period.get('label')}",
                        "period_day": folded_period["period_day"],
                    }
                )
            for signal in self._gls_removed_signal_periods(current_plot_result):
                entries.append(
                    {
                        "id": f"current_plot:removed_signal_{signal.get('signal_number')}",
                        "period_day": signal["period_day"],
                    }
                )
        return entries

    def _build_gls_period_context(self, current_plot_result=None):
        period_entries = self._period_entries_for_ratios(
            self.gls_period_records,
            current_plot_result=current_plot_result,
        )
        pairwise_ratios = []
        for index, numerator in enumerate(period_entries):
            for denominator in period_entries[:index]:
                numerator_period = numerator["period_day"]
                denominator_period = denominator["period_day"]
                pairwise_ratios.append(
                    {
                        "numerator_id": numerator["id"],
                        "denominator_id": denominator["id"],
                        "period_ratio": numerator_period / denominator_period,
                        "inverse_period_ratio": denominator_period / numerator_period,
                    }
                )

        context = {
            "previous_gls_period_records": self.gls_period_records,
            "period_ratio_reference": (
                "Ratios compare diagnosed physical periods, best P periods, and "
                "removed prewhitening periods from earlier GLS steps. Ratios near "
                "1, 2, 0.5, 3, 0.333, 1.5, or 0.667 can indicate repeated, "
                "harmonic, or alias-related signals."
            ),
            "pairwise_period_ratios": pairwise_ratios,
        }
        if current_plot_result is not None:
            context["current_gls_plot_periods"] = {
                "analysis_type": current_plot_result.get("analysis_type"),
                "best_period_day": self._finite_positive_float(
                    current_plot_result.get("best_period")
                ),
                "folded_periods_day": self._gls_folded_periods(current_plot_result),
                "removed_signal_periods_day": self._gls_removed_signal_periods(
                    current_plot_result
                ),
            }
        return context
    def _log(self,path, head,json_this):
        if path:
            log_output_path = Path(path)
            log_output_path.parent.mkdir(parents=True, exist_ok=True)
            with log_output_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"{head}\n{json.dumps(json_this, ensure_ascii=False, indent=2)}\n\n"
                )

    def update_state(self,state, diagnosis_step,diagnosis_result):
        # Append current done step
        state["diagnosis_done"].append(diagnosis_step)
        # Update the diagnosis_dependency according to the last step result, it can be new, add or remove
        for step, dependencies in diagnosis_result.get("new_diagnosis_dependency", {}).items():
            state["diagnosis_dependency"][step] = dependencies
        for step, dependencies  in diagnosis_result.get("removed_diagnosis_dependency", {}).items():
            if step in state["diagnosis_dependency"]:
                state["diagnosis_dependency"][step] = list(set(state["diagnosis_dependency"][step]) - set(dependencies))
        for step, dependencies in diagnosis_result.get("added_diagnosis_dependency", {}).items():
            if step in state["diagnosis_dependency"]:
                state["diagnosis_dependency"][step] = list(set(state["diagnosis_dependency"][step]) | set(dependencies))
            else:
                state["diagnosis_dependency"][step] = dependencies

        # remove the candidates according to the diagnosis result
        if "strong_excluded_variable_type" in diagnosis_result:
            state["possible_candidates"] = state["possible_candidates"] - set(diagnosis_result["strong_excluded_variable_type"])

        # Push ready diagnosis steps into diagnosis_to_do
        for step, dependencies in state["diagnosis_dependency"].items():
            if step in state["diagnosis_done"] or step in state["diagnosis_to_do"]:
                continue
            if all(dep in state["diagnosis_done"] for dep in dependencies):
                state["diagnosis_to_do"].append(step)
      
    def can_do_diagnosis_step(self, diagnosis_step, state) -> tuple[bool, str]:
        # This is for steps that have special requirements beyond the dependency, for example, BLS diagnosis requires exoplanet or EA candidate to be still possible after the previous diagnosis steps. We can add more rules here in the future to control the diagnosis process.
        if "BLS" in diagnosis_step and ( not "EP" in state["possible_candidates"] and not "EA" in state["possible_candidates"]):
            return False, f"Skip {diagnosis_step} because no exoplanet or detached eclipsing binary candidate remains."
        return True, "" # For now we can always do the diagnosis step, but in the future we can add some rules here to control the diagnosis process.

    def _add_prewhitening_dependency(
        self,
        diagnosis_result: dict,
        state: dict,
        dependency_step: str,
        signal_count: int,
    ) -> None:
        if state["GLS_prewhitening_count"] >= self.MAX_GLS_PREWHITENING_ITERATIONS:
            diagnosis_result["require_prewhitening"] = False
            diagnosis_result["prewhitening_limit_reached"] = True
            return

        prewhitening_index = state["GLS_prewhitening_count"]
        prewhitening_process_name = f"GLS_prewhitening_diagnosis_{prewhitening_index}"
        diagnosis_result.setdefault("new_diagnosis_dependency", {})[
            prewhitening_process_name
        ] = [dependency_step]
        diagnosis_result.setdefault("added_diagnosis_dependency", {})[
            "final_report"
        ] = [prewhitening_process_name]
        state["pipeline_parameters"]["GLS-prewhitening"][prewhitening_index] = {
            "signal_count": max(1, int(signal_count)),
        }
        state["GLS_prewhitening_count"] += 1
    
    def run_diagnosis_step(self, diagnosis_step, state, csv_path, ra_deg, dec_deg, scientific_target, log_output_path) -> dict:
        # The executer that runs the diagnosis step, and return the results for the next iteration.
        if diagnosis_step == "query_and_diagnosis_metadata":
            if not ra_deg is None and not dec_deg is None:
                print("querying metadata")
                self.metadata = metadata_diagnose.lookup_source_metadata(
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                )
                self._log(log_output_path, "Gaia astrometry result:", self.metadata)
                print("diagnosing metadata")
                self.metadata_diagnosis = metadata_diagnose.diagnose_metadata(self.metadata,state["possible_candidates"] ,self.vlm)
                self._log(log_output_path, "Metadata diagnosis result:", self.metadata_diagnosis)
            else:
                self.metadata_diagnosis = {"status": "skipped", "reason": "RA or Dec is missing, cannot query metadata."}
            return self.metadata_diagnosis
        
        if diagnosis_step == "overall_diagnosis":
            print("plotting")
            self.light_curve_csv_path = Path(csv_path)
            self.light_curve_schema = parse_light_curve_df.parse_light_curve_schema(
                self.light_curve_csv_path,
                vlm=self.vlm,
            )
            self.light_curve_png_path = plot_whole_light_curve.plot_whole_light_curve(
                self.light_curve_csv_path,
                self.light_curve_schema,
            )
            print("diagnosing full plot")
            self.full_plot_report = view_full_plot.view_full_plot(
                self.light_curve_png_path,
                self.metadata_diagnosis,
                state["possible_candidates"],
                self.vlm,
            )
            self._log(log_output_path, "Full light-curve plot diagnosis:", self.full_plot_report)
            final_report_dep = ["overall_diagnosis"]
            self.full_plot_report['new_diagnosis_dependency'] = {}
            if self.full_plot_report.get("require_GLS"):
                GLS_process_name = f"GLS_diagnosis_{state['GLS_count']}"
                self.full_plot_report['new_diagnosis_dependency'][GLS_process_name] = ["overall_diagnosis"]
                final_report_dep.append(GLS_process_name)
                state["GLS_count"] += 1
            if self.full_plot_report.get("require_BLS"):
                BLS_process_name = f"BLS_diagnosis_{state['BLS_count']}"
                self.full_plot_report['new_diagnosis_dependency'][BLS_process_name] = ["overall_diagnosis"]
                final_report_dep.append(f"BLS_diagnosis_{state['BLS_count']}")
                state["BLS_count"] += 1
            if self.full_plot_report.get("require_zoom_in"):
                zoom_in_process_name = f"zoom_in_diagnosis_{state['zoom_in_count']}"
                self.full_plot_report['new_diagnosis_dependency'][zoom_in_process_name] = ["overall_diagnosis"]
                state["pipeline_parameters"]["zoom-in"][state["zoom_in_count"]] = self.full_plot_report.get("require_zoom_in")
                final_report_dep.append(f"zoom_in_diagnosis_{state['zoom_in_count']}")
                state["zoom_in_count"] += 1
            
            self.full_plot_report['new_diagnosis_dependency']["final_report"] = final_report_dep
            return self.full_plot_report
        
        if "GLS_diagnosis_" in diagnosis_step:
            print("Calculating GLS")
            gls_plot_info = GLS_plot.plot_GLS(
                self.light_curve_csv_path,
                self.light_curve_schema,
            )

            print(
                "GLS plot result:",
                json.dumps(gls_plot_info, ensure_ascii=False, indent=2),
            )
            self._log(log_output_path, "GLS plot result:", gls_plot_info)
            print("Diagnosing GLS")
            gls_diagnosis_input = {
                "metadata_diagnosis": self.metadata_diagnosis,
                "full_light_curve_diagnosis": self.full_plot_report,
                "GLS_plot_result": gls_plot_info,
            }
            self.gls_diagnosis_report = GLS_diagnosis.diagnose_GLS(
                gls_plot_info["result_png_path"],
                gls_diagnosis_input,
                state["possible_candidates"],
                self.vlm,
                gls_plot_report=gls_plot_info,
            )
            self._log(log_output_path, "GLS plot diagnosis:", self.gls_diagnosis_report)
            print(
                "GLS plot diagnosis:",
                json.dumps(self.gls_diagnosis_report, ensure_ascii=False, indent=2),
            )
            self._record_gls_period_result(
                diagnosis_step,
                gls_plot_info,
                self.gls_diagnosis_report,
            )
            if self.gls_diagnosis_report.get("require_prewhitening"):
                self._add_prewhitening_dependency(
                    self.gls_diagnosis_report,
                    state,
                    diagnosis_step,
                    self.gls_diagnosis_report.get("prewhitening_signal_count", 1),
                )
            return self.gls_diagnosis_report
        if "GLS_prewhitening_diagnosis_" in diagnosis_step:
            task_index = int(diagnosis_step.split("_")[-1])
            prewhitening_parameters = state["pipeline_parameters"]["GLS-prewhitening"][
                task_index
            ]
            signal_count = prewhitening_parameters.get("signal_count", 1)
            print(f"Calculating GLS prewhitening with {signal_count} removed signal(s)")
            gls_prewhitening_plot_info = GLS_plot.plot_GLS_prewhitening(
                self.light_curve_csv_path,
                self.light_curve_schema,
                signal_count=signal_count,
            )

            print(
                "GLS prewhitening plot result:",
                json.dumps(gls_prewhitening_plot_info, ensure_ascii=False, indent=2),
            )
            self._log(
                log_output_path,
                f"GLS prewhitening plot result {task_index}:",
                gls_prewhitening_plot_info,
            )

            gls_history = []
            for key, value in state["diagnosis_results"].items():
                if "GLS" in key:
                    gls_history.append({key: value})

            gls_prewhitening_diagnosis_input = {
                "metadata_diagnosis": self.metadata_diagnosis,
                "full_light_curve_diagnosis": self.full_plot_report,
                "GLS_history": gls_history,
                "GLS_period_context": self._build_gls_period_context(
                    current_plot_result=gls_prewhitening_plot_info,
                ),
                "GLS_prewhitening_plot_result": gls_prewhitening_plot_info,
            }
            gls_prewhitening_diagnosis_report = GLS_diagnosis.diagnose_GLS(
                gls_prewhitening_plot_info["result_png_path"],
                gls_prewhitening_diagnosis_input,
                state["possible_candidates"],
                self.vlm,
                gls_plot_report=gls_prewhitening_plot_info,
            )
            print(
                "GLS prewhitening plot diagnosis:",
                json.dumps(
                    gls_prewhitening_diagnosis_report,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            self._log(
                log_output_path,
                f"GLS prewhitening plot diagnosis {task_index}:",
                gls_prewhitening_diagnosis_report,
            )
            self._record_gls_period_result(
                diagnosis_step,
                gls_prewhitening_plot_info,
                gls_prewhitening_diagnosis_report,
            )
            if gls_prewhitening_diagnosis_report.get("require_prewhitening"):
                requested_signal_count = gls_prewhitening_diagnosis_report.get(
                    "prewhitening_signal_count",
                    signal_count + 1,
                )
                self._add_prewhitening_dependency(
                    gls_prewhitening_diagnosis_report,
                    state,
                    diagnosis_step,
                    max(signal_count + 1, requested_signal_count),
                )
            return gls_prewhitening_diagnosis_report
        if "BLS_diagnosis_" in diagnosis_step:
            print("Calculating BLS")
            bls_plot_info = BLS_plot.plot_BLS(
                self.light_curve_csv_path,
                self.light_curve_schema,
            )
            print(
                "BLS plot place:",
                json.dumps(bls_plot_info, ensure_ascii=False, indent=2),
            )
            bls_diagnosis_input = {
                "metadata_diagnosis": self.metadata_diagnosis,
                "full_light_curve_diagnosis": self.full_plot_report,
                "BLS_plot_result": bls_plot_info,
            }
            bls_diagnosis_report = BLS_diagnosis.diagnose_BLS(
                bls_plot_info["result_png_path"],
                bls_diagnosis_input,
                state["possible_candidates"],
                self.vlm,
                bls_plot_report=bls_plot_info,
            )
            self._log(log_output_path, "BLS plot diagnosis:", bls_diagnosis_report)

            return bls_diagnosis_report
        if "zoom_in_diagnosis_" in diagnosis_step:
            task_index = int(diagnosis_step.split("_")[-1])
            zoom_ranges = state['pipeline_parameters']['zoom-in'][task_index]

            zoom_plot_meta = zoom_in_plot.plot_zoom_in_light_curve(
                self.light_curve_csv_path,
                self.light_curve_schema,
                self.full_plot_report,
                zoom_ranges=zoom_ranges,
                output_path=None,
            )

            print(
                "Zoom-in plot result:",
                json.dumps(zoom_plot_meta, ensure_ascii=False, indent=2),
            )
            # self._log(log_output_path, f"Zoom-in plot result {zoom_iteration}:", zoom_plot_report)
            zoom_in_history = []
            

            for k,v in state['diagnosis_results'].items():
                if "zoom_in" in k:
                    zoom_in_history.append(v)

            zoom_diagnosis_input = {
                "metadata_diagnosis": self.metadata_diagnosis,
                "full_light_curve_diagnosis": self.full_plot_report,
                "zoom_in_history": zoom_in_history,
            }
            zoom_in_diagnosis_report = zoom_in_diagnosis.diagnose_zoom_in(
                zoom_plot_meta["result_png_path"],
                zoom_diagnosis_input,
                state["possible_candidates"],
                self.vlm,
                zoom_plot_report=zoom_plot_meta,
            )
            print(
                "Zoom-in plot diagnosis:",
                json.dumps(zoom_in_diagnosis_report, ensure_ascii=False, indent=2),
            )
            self._log(log_output_path, f"Zoom-in plot diagnosis {task_index}:", zoom_in_diagnosis_report)
            if zoom_in_diagnosis_report["need_further_zoom_in"]:
                zoom_in_process_name = f"zoom_in_diagnosis_{state['zoom_in_count']}"
                self.full_plot_report.setdefault("new_diagnosis_dependency", {})[zoom_in_process_name] = ["overall_diagnosis"]
                state["pipeline_parameters"]["zoom-in"][state["zoom_in_count"]] = zoom_in_diagnosis_report.get("need_further_zoom_in")
                self.full_plot_report.setdefault("added_diagnosis_dependency", {})["final_report"] = [zoom_in_process_name]
                state["zoom_in_count"] += 1

            return zoom_in_diagnosis_report
        if diagnosis_step == "final_report":
            # final_summary_input = {
            #     "scientific_target":scientific_target,
            #     "metadata_diagnosis": self.metadata_diagnosis,
            #     "light_curve_diagnosis_results": state,
            #     "remaining_possible_variable_types": sorted(state["possible_candidates"]),
            # }
            self.final_summary_report = summary_all.summarize_results(
                state,
                sorted(state["possible_candidates"]),
                scientific_target,
                self.vlm,
            )
            self._log(log_output_path, "Final summary diagnosis:", self.final_summary_report)
            print(
                "Final summary diagnosis:",
                json.dumps(self.final_summary_report, ensure_ascii=False, indent=2),
            )
            return self.final_summary_report

    def pipeline(self, csv_path: str, ra_deg, dec_deg, scientific_target: str | None, log_output_path: str | None = None) -> dict[str, any]:
        self.gls_period_records = []

        state = {
            "possible_candidates": self.ALL_VARIABLE_TYPES,
            "diagnosis_done":[],
            "diagnosis_to_do":["query_and_diagnosis_metadata"], # This record the diagnosis steps that can be done now. Agent would wait until all steps are done, then start the next agentic loop.
            "diagnosis_dependency":{"overall_diagnosis":["query_and_diagnosis_metadata"]}, # "A":["B","C"] means A's diagnosis requires B and C to be done first
            "diagnosis_results":{}, # "diagnosis_step": diagnosis_result
            "pipeline_parameters":{ # This is to record the paramters that is required by pipeline that is called multiple times.
                "GLS":{}, # CT:{args}
                "BLS":{},
                "zoom-in":{},
                "GLS-prewhitening":{},
            },
            "GLS_count": 0,
            "BLS_count": 0,
            "zoom_in_count": 0,
            "GLS_prewhitening_count": 0,
        }

        while True: # The Agentic loop for diagnosis
            # do all diagnosis steps to be done
            if not state["diagnosis_to_do"]: # If all steps are none, break the loop
                break
            diagnosis_step = state["diagnosis_to_do"].pop(0) #FIFO
            can_do, reason = self.can_do_diagnosis_step(diagnosis_step, state)
            if not can_do:
                diagnosis_result = {"status": "skipped", "reason": reason}
                state["diagnosis_results"][diagnosis_step] = diagnosis_result
            else:
                diagnosis_result = self.run_diagnosis_step(diagnosis_step, state, csv_path, ra_deg, dec_deg, scientific_target, log_output_path)
                state["diagnosis_results"][diagnosis_step] = {"status": "done", "result": diagnosis_result}
            # Add new states according to the diagnosis results, clear all done diagnosis_to_do, and update diagnosis_waiting_for according to the new states.
            self.update_state(state, diagnosis_step, diagnosis_result)



if __name__ == "__main__":
    log_output_path = Path(__file__).resolve().parent / "output" / "diagnosis_log_example.txt"
    agent = LightcurveAagent()
    agent.pipeline(
        csv_path="input/test/example_lc.csv",
        ra_deg=13.99739779,
        dec_deg=-79.92633949,
        # ra_deg = None,
        # dec_deg= None,
        scientific_target="Find the variable stars with extreme parameters (e.g., period, amplitude, evolutionary stage) that challenge or expand current astrophysical theories of stellar variability.",
        log_output_path=log_output_path,
    )
