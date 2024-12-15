---
layout: page
title: research
permalink: /research/
description: 
nav: true
nav_order: 4
display_categories: [limited, multimodal, subgroup]
---

I develop statistical machine learning and deep learning methodologies for modeling complex datasets with high dimensionality, multi-modality, and limited supervision. Most of these methods are motivated by biomedical applications but are generalizable to other application domains. 

My research has three main directions:

1. **Machine Learning with limited supervision and knowledge integration**
2. **Disentanglement and fusion of multi-modal/high-dimensional datasets**
3. **Data mining and subgroup identification**

<!-- Applications:
* Precision Medicine of Brain Cancer
* Biomarker Discovery for Persistent Post-traumatic Headache
* Organ and Lesion Segmentation from Medical Images
* Early Prediction of Alzheimer’s Disease
* Analyzing Influencers on Social Media
* Predicting of Unplanned Hospitalization and Readmission of Medicare Patients -->


<br>

#### 1. Machine Learning with Limited Supervision and Knowledge Integration

Labeled data is often scarce in biomedical applications, leading to the challenge of how to learn with **limited supervision**. One common strategy is **weakly supervised learning**, where models are trained with incomplete or noisy labels. Another approach involves **integrating domain knowledge**, enhancing model performance by incorporating expert insights or external data sources into the learning process.

<br>

<div class="projects">
{% if site.enable_project_categories and page.display_categories %}
  {% assign selected_category = "limited" %}
  {% assign categorized_projects = site.projects | where: "category", selected_category %}
  {% assign sorted_projects = categorized_projects | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}

{% else %}

<!-- Display projects without categories -->

{% assign sorted_projects = site.projects | sort: "importance" %}

  <!-- Generate cards for each project -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>


<br>
<br>

#### 2. Disentanglement and Fusion of Multi-modal/High-dimensional Datasets

With advancements in technology, high-dimensional and multi-source data are increasingly being collected for biomedical applications, including imaging, genomics, clinical questionnaires, and Molecular Dynamics (MD) simulations. Learning from multi-modal datasets can leverage complementary information and lead to improved performance for prediction tasks. 

Analyzing these datasets presents interesting challenges, primarily due to the limited availability of precise labels in biomedical contexts. Additionally, some datasets may have missing modalities for a portion of the samples; for instance, not all patients may have all imaging modalities collected due to accessibility or financial constraints. Moreover, complex datasets often contain a mix of signals influenced by environment constraints, obscuring the true patterns of interest to researchers. Models must effectively disentangle various sources of signals and fuse information from these datasets for predictive modeling and knowledge discovery.

<br>

<div class="projects">
{% if site.enable_project_categories and page.display_categories %}
  {% assign selected_category = "multimodal" %}
  {% assign categorized_projects = site.projects | where: "category", selected_category %}
  {% assign sorted_projects = categorized_projects | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}

{% else %}

<!-- Display projects without categories -->

{% assign sorted_projects = site.projects | sort: "importance" %}

  <!-- Generate cards for each project -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>

<br>
<br>

#### 3. Data Mining and Subgroup Identification

Data mining is a powerful technique for knowledge-discovery and information analysis from large, complex datasets. For example, we analyzed millions of tweets from social media to identify influencers in a social network. In healthcare, we mined large-scale medical claims to discover patients in similar risk groups for hospital readmission. 

<br>

<div class="projects">
{% if site.enable_project_categories and page.display_categories %}
  {% assign selected_category = "subgroup" %}
  {% assign categorized_projects = site.projects | where: "category", selected_category %}
  {% assign sorted_projects = categorized_projects | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}

{% else %}

<!-- Display projects without categories -->

{% assign sorted_projects = site.projects | sort: "importance" %}

  <!-- Generate cards for each project -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>