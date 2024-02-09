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

My research fall under two main categories:

1. **Machine Learning with limited supervision**
- A review of knowledge-informed machine learning for cancer applications ([website](https://lingchm.github.io/kinformed-machine-learning-cancer/))
- Predicting Tumor Cell Density maps for brain cancer using MRI (*Mayo Clinic*) ([paper](https://ieeexplore.ieee.org/abstract/document/10292790))
- Reconstruction of accelerated MRI using self-supervised learning 
- Automated segmentation and classification of dental lesion from 3D CBCT (*Upenn Dental*)
- Segmentation of retinal layers from OCT Images with uncertainty quantification (*Feola Lab*)
  
2. **Multi-modal learning**
- Prediction of recovery from post-traumatic headache using clinical and imaging data (*Mayo Clinic*) ([paper](https://headachejournal.onlinelibrary.wiley.com/doi/abs/10.1111/head.14450)) ([paper](https://journals.sagepub.com/doi/full/10.1177/03331024231172736)) 
- Early prognosis of Alzheimer’s Disease using incomplete multi-modal neuroimaging and genetics data ([abstract](https://alz-journals.onlinelibrary.wiley.com/doi/abs/10.1002/alz.078606)) 

I also analyzed population-level data for public health applications: 

3. **Large-scale data mining and predictive modeling**
- Prediction of unplanned hospitalizations for Medicare patients ([*CMS AI Challenge*](https://www.cms.gov/priorities/innovation/innovation-models/artificial-intelligence-health-outcomes-challenge))([paper](https://link.springer.com/chapter/10.1007/978-3-030-75166-1_34))
- Analyzing the public influence of health organizations on Twitter ([paper](https://www.jmir.org/2023/1/e45897/))
- Analyzing public health policies for sodium reduction in the U.S. ([interactive dashboard](https://us-sodium-policies.shinyapps.io/Rshiny/))


### Fun 

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