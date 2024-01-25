---
layout: page
title: projects
permalink: /projects/
description: 
nav: true
nav_order: 4
---

### Research 

My research fall under two main categories:

1. **Machine Learning with limited supervision**
- A [live review of knowledge-informed machine learning](https://lingchm.github.io/kinformed-machine-learning-cancer/) for cancer applications 
- Predicting Tumor Cell Density maps for brain cancer using MRI (*Mayo Clinic*)
- Reconstruction of accelerated MRI using self-supervised learning 
- Automated segmentation and classification of dental lesion from 3D CBCT (*Upenn Dental*)
- Segmentation of retinal layers from OCT Images with uncertainty quantification (*Feola Lab*)
  
2. **Multi-modal learning**
- Prediction of recovery from post-traumatic headache using clinical and imaging data 
- Early prognosis of Alzheimer’s Disease using incomplete multi-modal neuroimaging and genetics data (*Mayo Clinic*)

In addition to clinical applications, I also analyzed population-level data for public health applications: 

3. **Large-scale data mining and predictive modeling**
- Prediction of unplanned hospitalizations for Medicare patients (*CMS AI Challenge*)
- Analyzing the public influence of health organizations on Twitter 
- An [interactive dahsboard](https://us-sodium-policies.shinyapps.io/Rshiny/) to visualize trends about dietary sodium-related US policies 


### Fun 

Outside research, I like to create AI/ML solutions to automatically analyze data. Here are some favorites:

- MMTrip, Your Personal Multi-Modal Planner (website under development)
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