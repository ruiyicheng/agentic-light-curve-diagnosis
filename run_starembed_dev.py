import pandas as pd
from diagnose import LightcurveAagent
from pathlib import Path
import glob

FINAL_DONE_MARKER = "Final summary diagnosis:"


def is_final_result_done(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return FINAL_DONE_MARKER in text


if __name__ == "__main__":
    dev_input_dir = "/home/rui/code/project/pipeline_agentic_light_curve/input/starembed_sample/val"
    dev_csv_path = "/home/rui/code/project/pipeline_agentic_light_curve/input/starembed_sample/val/manifest.csv"
    dev_df = pd.read_csv(dev_csv_path)
    log_output_root = Path(__file__).resolve().parent / "output" / "starembed_dev" 
    CLASS_ID_TO_NAME = {
    "1": "EW",
    "2": "EA",
    "4": "RRab",
    "5": "RRc",
    "6": "RRd",
    "8": "RS CVn",
    "13": "LPV",
    }

    classes = set([v for k,v in CLASS_ID_TO_NAME.items()])
    agent = LightcurveAagent(ALL_VARIABLE_TYPES = classes)

    for _, row in dev_df.iterrows():
        csv_path = row["light_curve_path"]
        ra_deg = row["ra"]
        dec_deg = row["dec"]
        
        log_output_path = log_output_root / f"{csv_path.split('/')[-1].split('.csv')[0]}_diagnosis.txt"
        if is_final_result_done(log_output_path):
            print(f"Skip completed target: {csv_path}")
            continue
        if log_output_path.exists():
            print(f"Delete incomplete previous report: {log_output_path}")
            log_output_path.unlink()

        if not "unknown" in csv_path:
            agent.pipeline(
                csv_path=csv_path,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                scientific_target="Find the variable stars with extreme parameters (e.g., period, amplitude, evolutionary stage) that challenge or expand current astrophysical theories of stellar variability.",
                log_output_path=log_output_path,
            )

        else:

            agent.pipeline(
                csv_path=csv_path,
                ra_deg=None,
                dec_deg=None,
                scientific_target="Find the variable stars with extreme parameters (e.g., period, amplitude, evolutionary stage) that challenge or expand current astrophysical theories of stellar variability.",
                log_output_path=log_output_path,
            )
