import itertools as it
import subprocess
import time
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterator, List, Literal, Tuple

import numpy as np


class Dimension(ABC):
    @cached_property
    @abstractmethod
    def options(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class KeyValueDimension(Dimension):
    def __init__(self, key: str, values: List[str]):
        self._key = key
        self._values = values

    @cached_property
    def options(self):
        return list(map(lambda x: f"--{self._key} {x}", self._values))
    
    def __len__(self):
        return len(self._values)


class FlagDimension(Dimension):
    def __init__(self, flags: List[str], iterate: Literal["single", "permutations"] = "single", empty:bool=True):
        """
        Args
        ----
        flags: List[str]
            List of flags to iterate over
        iterate: Literal["single", "permutations"]
            Whether to iterate over all flags setting one at a time, or over all possible permutations.
        empty: bool
            Whether to include the empty option
        """
        self._flags = flags
        self._iterate = iterate
        self._empty = empty

    @cached_property
    def options(self):
        if self._iterate == "single":
            res = list(map(lambda x: f"--{x}", self._flags))
            
        else:
            perms:list[tuple[str, ...]] = []
            for i in range(1, len(self._flags) + 1):
                perms.extend(it.permutations(self._flags, i))
            res = list(map(lambda x: " ".join(map(lambda y: f"--{y}", x)), perms))

        if self._empty:
            return [""] + res
        else:
            return res
        
    def __len__(self):
        if self._iterate == "single":
            return len(self._flags)
        else:
            return len(self.options)


class GridSearch:
    """
    Perform a grid search over a set of dimensions. Dimensions will be iterated over in the revere order they were added.
    """
    def __init__(self, base_command:str, dimensions: List[Dimension] = []):
        self._base_command = base_command
        self._dimensions = dimensions

    def add_dimension(self, dimension: Dimension):
        self._dimensions.append(dimension)

    def __len__(self):
        return np.prod(list(map(lambda x: len(x), self._dimensions)))
    
    def __iter__(self) -> Iterator[str]:
        for prod in it.product(*map(lambda x: x.options, self._dimensions)):
            yield f"{self._base_command} {' '.join(prod)}"


class RunningAverage:
    def __init__(self):
        self.running_average = 0.0
        self.count = 0

    def update(self, new_time: float):
        self.running_average = (self.running_average * self.count + new_time) / (self.count + 1)
        self.count += 1

def call_process(command: str) -> Tuple[bool, str, str, float]:
    print(f"Running command: {command}")
    start = time.time()
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    end = time.time()
    return process.returncode == 0, process.stdout, process.stderr, end - start

def fmt_output(txt:str):
    return "\t# "+ txt.replace("\n", "\n\t# ") if txt else ""


if __name__ == "__main__":
    import argparse
    import sys
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Grid search over langdetect.py")
    parser.add_argument("--max-workers", help="Maximum number of workers to use", type=int, default=1)
    parser.add_argument("--show-success", help="Show stdout of successful commands", action="store_true")
    parser.add_argument("--show-error", help="Show communications of failed commands", action="store_true")
    parser.add_argument("--report-file", help="CSV file to report the results to", type=Path, default=None)
    args = parser.parse_args()

    python = Path(sys.executable)
    #TODO: Windows shit maybe? marc decide if this needs removal or not 
    #python = Path(sys.executable).relative_to(Path(".").resolve())
    langdetect = Path(__file__).parent.relative_to(Path(".").resolve()) / "src" / "langdetect.py"

    
    if args.report_file:
        with open(args.report_file, "w") as f:
            f.write("dataset,max_voc_size,tokenizer,vectorizer,classifier,remove_urls,remove_symbols,split_sentences,lower,"
                    "remove_stopwords,lemmatize,stemmatize,train_size,test_size,voc_size,train_coverage,test_coverage,"
                    "f1_micro,f1_macro,f1_weighted,pca_explained_variance_ratio,duration\n")
        cmd = f'"{python}" "{langdetect}" --report-results "{args.report_file}" --hide-plots'
    else:
        cmd = f'"{python}" "{langdetect}" --hide-plots'

    grid = GridSearch(cmd)
    grid.add_dimension(FlagDimension(["remove-urls", "remove-symbols", "split-sentences",
                                      "lower"], iterate="permutations"))
    #grid.add_dimension(FlagDimension(["remove-urls", "remove-symbols", "split-sentences",
    #                                 "lower", "remove-stopwords", "lemmatize", "stemmatize"], iterate="permutations"))
    grid.add_dimension(KeyValueDimension("tokenizer", ["word", "char", "bigram"]))
    grid.add_dimension(KeyValueDimension("vectorizer", ["token-count"]))
    grid.add_dimension(KeyValueDimension("classifier", ["dt", "knn", "lda", "lr", "mlp", "nb", "rf", "svm"]))


    print("Total number of executions to run:", len(grid))
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        print("Submitting jobs...")
        jobs = executor.map(call_process, grid)
        print("All jobs submitted. Waiting for jobs to finish...")

        failed: List[str] = []
        job_duration = RunningAverage()
        total_jobs = len(grid)
        for j, (job, cmd) in enumerate(zip(jobs, grid), 1):
            success, out, err, duration = job
            job_duration.update(duration)
            eta = job_duration.running_average * (total_jobs - j) / args.max_workers

            if not success:
                failed.append(cmd)
                # Do it in a single print to avoid interleaving
                if args.show_error:
                    print(f"========\nCommand {cmd} failed ({duration:.1f}s) with the following communications. ETA: {eta:.1f}s:\n------- stdout -------\n"
                        f"{fmt_output(out)}\n------- stderr -------\n{fmt_output(err)}\n========")
                else:
                    print(f"* Command {cmd} failed ({duration:.1f}s). ETA: {eta:.1f}s")
            else:
                if args.show_success:
                    print(f"========\nCommand {cmd} succeeded ({duration:.1f}s). ETA: {eta:.1f}s:\n------- stdout -------\n"
                        f"{fmt_output(out)}\n========")
                else:
                    print(f"* Command {cmd} succeeded ({duration:.1f}s). ETA: {eta:.1f}s")

        if failed:
            print("The following commands failed:")
            for cmd in failed:
                print(f" - {cmd}")

