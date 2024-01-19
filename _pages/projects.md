---
layout: page
title: 
permalink: /projects/
description: 
nav: true
nav_order: 4
display_categories: []
horizontal: false
---

### Research 

My research fall under two main categories:

1. **Machine Learning with limited supervision**
- A [live review of](https://lingchm.github.io/kinformed-machine-learning-cancer/) Knowledge-informed machine learning for cancer applications 
- Predicting Tumor Cell Density maps for brain cancer using MRI (*Mayo Clinic*)
- Reconstruction of accelerated MRI using self-supervised learning 
- Automated segmentation and classification of dental lesion from 3D CBCT (*Upenn Dental*)
  
2. **Multi-modal learning**
- Prediction of recovery from post-traumatic headache using clinical and imaging data 
- Early prognosis of Alzheimer’s Disease using incomplete multi-modal neuroimaging and genetics data (*Mayo Clinic*)

Other earlier research projects included:  
- Prediction of unplanned hospitalizations for Medicare patients (*CMS AI Challenge*)
- Segmentation of retinal layers from OCT Images 


### Fun 

Outside research, I like to create AI/ML solutions to automatically analyze data. Here are some favorites:

- MMTrip, Your Personal Multi-Modal Planner (website under development)
- An [interactive dahsboard](https://us-sodium-policies.shinyapps.io/Rshiny/) to analyze sodium-related US policies 
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