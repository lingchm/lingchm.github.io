---
layout: page
title: projects
permalink: /projects/
description: 
nav: true
nav_order: 4
display_categories: []
---

### Research 

My research fall under three main categories:

1. **Machine Learning with limited supervision**
- A review of knowledge-informed machine learning for cancer applications ([website](https://lingchm.github.io/kinformed-machine-learning-cancer/))
- Weakly Supervised Transfer Learning for precision medicine of brain cancer (*Mayo Clinic*) ([paper](https://ieeexplore.ieee.org/abstract/document/10292790)) (*🏆 Best Student Paper, IISE DAIS 2022*)
- A weakly supervised approach for liver tumor segmentation ([website](https://lingchmao-medassist-liver-cancer.hf.space/))(*🏆 Winner, IISE DAIS 2024*)
- Self-supervised learning for the reconstruction of accelerated MRI 
- Segmentation of retinal layers from OCT Images with uncertainty quantification (*Feola Lab*)
  
2. **Machine Learning with multi-source/multi-modal data**
- Supervised Multi-modal Fission Learning (paper under review)
- A cross-modal Mutual Knowledge Distillation framework for Alzheimer's Disease: addressing incomplete modalities (paper under review)
- Prediction of post-traumatic headache recovery using clinical and imaging data (*Mayo Clinic*) ([paper](https://headachejournal.onlinelibrary.wiley.com/doi/abs/10.1111/head.14450)) ([paper](https://journals.sagepub.com/doi/full/10.1177/03331024231172736)) 

3. **Data mining and subgroup identification**
- Identifying subgroups of patients with post-traumatic headache
- Hierarchical clustering and predictive modeling for of unplanned hospitalizations of Medicare patients ([*🏆 CMS AI Challenge*](https://www.cms.gov/priorities/innovation/innovation-models/artificial-intelligence-health-outcomes-challenge))([paper](https://link.springer.com/chapter/10.1007/978-3-030-75166-1_34))
- Finding social media influencers for sodium reduction on Twitter in the U.S. ([paper](https://www.jmir.org/2023/1/e45897/)) ([interactive dashboard](https://us-sodium-policies.shinyapps.io/Rshiny/))


### Personal Projects 

Outside research, I like to create AI/ML solutions to automatically analyze data. Here are some favorites:

- [MMTrip](https://youtu.be/g0p3DScMEJs), Your Personal Multi-Modal Planner (website under development)
- [AskMendel](https://askmendel.ai/), a LLM chatbot for automatic bioinformatics data analysis and visualization 




<!-- pages/projects.md -->
<!--
<div class="projects">
{%- if site.enable_project_categories and page.display_categories %}
  {%- for category in page.display_categories %}
  <h2 class="category">{{ category }}</h2>
  {%- assign categorized_projects = site.projects | where: "category", category -%}
  {%- assign sorted_projects = categorized_projects | sort: "importance" %}
  
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for project in sorted_projects -%}
      {% include projects_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for project in sorted_projects -%}
      {% include projects.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
  {% endfor %}

{%- else -%}
  {%- assign sorted_projects = site.projects | sort: "importance" -%}
  
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for project in sorted_projects -%}
      {% include projects_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for project in sorted_projects -%}
      {% include projects.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
{%- endif -%}
</div>

-->