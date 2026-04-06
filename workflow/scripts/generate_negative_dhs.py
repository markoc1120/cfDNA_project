import bisect
import glob
import logging
import random

import numpy as np
import pandas as pd
import py2bit

logger = logging.getLogger(__name__)


# TODO: move logic to src/cfdna/
def merge_intervals(accessible_sites):
    if not accessible_sites:
        return []
    accessible_sites = sorted(accessible_sites)
    merged = [list(accessible_sites[0])]
    for s, e in accessible_sites[1:]:
        last = merged[-1]
        if s <= last[1]:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def subtract_interval(all_sites, accessible_site):
    s, e = accessible_site
    out = []
    for a, b in all_sites:
        if b <= s or a >= e:
            out.append((a, b))
        else:
            if a < s:
                out.append((a, s))
            if e < b:
                out.append((e, b))
    return out


def sample_inaccessible_site(sites):
    total = sum(b - a for a, b in sites)
    if total <= 0:
        return None

    r = random.randrange(total)
    acc = 0
    for a, b in sites:
        if acc + (b - a) > r:
            return a + (r - acc)
        acc += b - a
    return None


def gc_for_mid(chrom: str, mid: int, hg38_genome, chrom_sizes, gc_bias_window) -> float:
    half_gc_bias_window = gc_bias_window / 2
    lower, upper = mid - half_gc_bias_window, mid + half_gc_bias_window
    lower, upper = (
        int(np.clip(lower, 0, chrom_sizes[chrom])),
        int(np.clip(upper, 0, chrom_sizes[chrom])),
    )
    # hard requirement py2bit expects python int types for start and end
    base_distr = hg38_genome.bases(chrom, lower, upper, False)
    return (base_distr['G'] + base_distr['C']) / gc_bias_window


def get_bin_id(edges: list[float], query: float) -> int | None:
    bin_id = bisect.bisect_right(edges, query) - 1  # ranges from -1 to 20 (-1 and 20 are outside)
    if 0 <= bin_id <= len(edges) - 2:
        return bin_id
    return None


def generate_negative_dhs_df(
    df,
    window_size: int,
    hg38_genome,
    chrom_sizes,
    gc_bias_window,
    n_quantile_bins=20,
    max_tries_multiplier=50,
):
    window_half = window_size // 2

    # binning gc content into bins for the whole genome
    df['gc_bin'], edges = pd.qcut(
        df['gc_content'], q=n_quantile_bins, retbins=True, duplicates='drop'
    )  # bins
    df['gc_bin_id'] = df['gc_bin'].cat.codes  # bins -> bin_id ranging from 0 to 19

    # (chr1, 1): 999 -> (chr, bin_id): number of fragments
    global_sites_needed_per_chrom = df.groupby(['chr', 'gc_bin_id']).size().to_dict()

    neg_rows = []
    for chromosome, sites in df.groupby('chr'):
        if chromosome not in chrom_sizes:
            continue

        chr_len = chrom_sizes[chromosome]
        mids = sites['mid'].to_numpy()

        accesible_sites = []
        for m in mids:
            s, e = max(0, m - window_half), min(chr_len, m + window_half)
            accesible_sites.append((s, e))
        accesible_sites = merge_intervals(accesible_sites)

        all_sites = [(0, chr_len)]
        for s, e in accesible_sites:
            all_sites = subtract_interval(all_sites, (s, e))

        chrom_bin_ids = sorted(sites['gc_bin_id'].unique().tolist())
        n_sites_needed_per_bin = {
            _bin: global_sites_needed_per_chrom[(chromosome, _bin)] for _bin in chrom_bin_ids
        }
        n_sites_needed = sum(n_sites_needed_per_bin.values())
        if n_sites_needed == 0:
            continue

        curr = {_bin: 0 for _bin in chrom_bin_ids}
        tries = 0
        while sum(curr.values()) < n_sites_needed and all_sites:
            if tries % 5000 == 0:
                logger.info(
                    f'{chromosome}: {tries}/{n_sites_needed * max_tries_multiplier} tries, {sum(curr.values())}/{n_sites_needed} sites generated'
                )

            tries += 1
            if tries > n_sites_needed * max_tries_multiplier:
                break

            m = sample_inaccessible_site(all_sites)
            if m is None:
                break

            gc = gc_for_mid(chromosome, int(m), hg38_genome, chrom_sizes, gc_bias_window)
            bin_id = get_bin_id(edges, gc)
            if (
                bin_id is None or bin_id not in curr
            ):  # outside of our bins -> either lower or higher GC content
                continue

            if curr[bin_id] >= n_sites_needed_per_bin[bin_id]:  # already depleted this bin
                continue

            # save as a tiny DHS interval (BED-like)
            neg_rows.append((chromosome, m - 1, m + 1))
            curr[bin_id] += 1

            # remove this window so negatives don't overlap each other
            used_inaccessible_site = (max(0, m - window_half), min(chr_len, m + window_half))
            all_sites = subtract_interval(all_sites, used_inaccessible_site)

    neg_df = pd.DataFrame(neg_rows, columns=['chr', 'start', 'end'])
    neg_df = neg_df.sort_values(['chr', 'start']).reset_index(drop=True)
    return neg_df


if 'snakemake' in globals():
    training_dhs_dir = snakemake.params.training_dhs_dir
    hg38_2bit_file = snakemake.params.hg38_2bit_file
    gc_bias_window = snakemake.params.gc_bias_window
    matrix_columns = snakemake.params.matrix_columns
    n_quantile_bins = snakemake.params.n_quantile_bins
    max_tries_multiplier = snakemake.params.max_tries_multiplier

    hg38_genome = py2bit.open(hg38_2bit_file)
    chrom_sizes = hg38_genome.chroms()

    dhs_fnames = glob.glob(f'{training_dhs_dir}*.bed')
    # Filter out files that already have '_negative' in the name
    dhs_fnames = [f for f in dhs_fnames if '_negative' not in f]

    for fname in dhs_fnames:
        dhs_df = pd.read_csv(fname, sep='\t', names=['chr', 'start', 'end'])
        dhs_df['mid'] = (dhs_df['start'] + dhs_df['end']) // 2

        gc = np.empty(len(dhs_df))
        for i, (chrom, mid) in enumerate(zip(dhs_df['chr'].to_numpy(), dhs_df['mid'].to_numpy())):
            gc[i] = gc_for_mid(chrom, int(mid), hg38_genome, chrom_sizes, gc_bias_window)
        dhs_df['gc_content'] = gc

        new_path = fname.rsplit('.', 1)
        new_path[0] = f'{new_path[0]}_negative'
        output_file = '.'.join(new_path)

        negative_dhs_df = generate_negative_dhs_df(
            dhs_df,
            matrix_columns,
            hg38_genome,
            chrom_sizes,
            gc_bias_window,
            n_quantile_bins=n_quantile_bins,
            max_tries_multiplier=max_tries_multiplier,
        )
        negative_dhs_df.to_csv(output_file, sep='\t', header=False, index=False)
        logger.info(f'Saved: {output_file} ({len(negative_dhs_df)} negative sites)')
