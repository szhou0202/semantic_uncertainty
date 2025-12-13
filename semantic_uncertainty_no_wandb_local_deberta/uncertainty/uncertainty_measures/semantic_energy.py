# szhou: for the semantic energy metric 
# for this, we require tracking the logits of the model 
# so we need to additionally store that in in generate_answers.py
import math
from math import prod
from statistics import mean



def cal_cluster_ce(logits, clusters):
    """
    Calculate summed logits.

    Args:
        logits (List[float]): List of logits for each item.
        clusters (List[List[int]]): Index clusters over which to aggregate values.

    Returns:
        Tuple:
            - logits_se: Negative summed logits per cluster.
    """
    logits_se = []
    for cluster in clusters:
        # Sum logits (negated) within the cluster
        cluster_logit_sum = -sum(logits[i] for i in cluster)
        logits_se.append(cluster_logit_sum)#/len(cluster))

    return logits_se



def fermi_dirac(E, mu, kT):
    """
    Apply the Fermi-Dirac function.

    Args:
        E (float): Energy level (logit value).
        mu (float): Chemical potential.
        kT (float): Thermal energy (default = 1.0).

    Returns:
        float: Fermi-Dirac value.
    """
    return E / (math.exp((E - mu) / kT) + 1)



def cal_boltzmann_logits(logits_list):
    """
    Apply Boltzmann transformation (negated mean of logits).

    Args:
        logits_list (List[List[float]]): List of token-level logits per item.

    Returns:
        List[float]: Transformed logits per item.
    """
    return [-mean(sublist) for sublist in logits_list]


def cal_flow(logits_list, clusters, fermi_mu=None):
    """
    Compute cluster-wise flow using probabilities and logits.

    Args:
        question_group (List[Dict]): Group of model responses for a question.
        clusters (List[List[int]]): Cluster indices.
        fermi_mu (float, optional): If provided, applies Fermi-Dirac transformation.

    Returns:
        Tuple[List[float], List[float]]:
            - Cluster-level weighted logits (Fermi-Dirac or Boltzmann)
    """

    # Apply appropriate logit transformation
    if fermi_mu is not None:
        logits = cal_fermi_dirac_logits(logits_list, mu=fermi_mu)
    else:
        logits = cal_boltzmann_logits(logits_list)

    # Return logit scores
    return cal_cluster_ce(logits, clusters)

