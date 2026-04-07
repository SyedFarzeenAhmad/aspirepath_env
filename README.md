---
title: AspirePath Career Counselor
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
---

# AspirePath-v1: Grade 10 Career Counseling Environment

## Overview
This is a real-world OpenEnv simulation designed to evaluate AI agents on their ability to provide structured academic and career guidance to 10th-grade students in India.

## Task Descriptions
1. **Easy (STEM Alignment):** Analyze a high-analytical profile to recommend PCM.
2. **Medium (Arts/Humanities):** Identify creative strengths for Humanities.
3. **Hard (Commerce/Business):** Navigate a conflict between high verbal skills and parental pressure toward Science.

## Action & Observation Space
- **Observation:** Includes analytical_score, creative_score, verbal_score, and a list of interests.
- **Action:** A JSON response containing `recommended_stream`, `career_cluster`, and `justification`.

## Setup
Built using the OpenEnv framework. Run via `inference.py`.