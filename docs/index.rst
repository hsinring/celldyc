.. CellDyc documentation master file, created by
   sphinx-quickstart on Fri Jan  9 21:14:42 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CellDyc
=====================

**CellDyc** is a novel semi-supervised learning framework designed to reconstruct transcriptomic velocities and recover intrinsic "gene-embedded time" by leveraging experimental time-point supervision.

.. raw:: html

   <div style="display: flex; align-items: center; margin: 25px 0;">
     <!-- 左图 -->
     <div style="width: 30%; text-align: center;">
       <img src="_static/index_1.png" style="width: 100%; max-width: 200px;">
       <p style="margin-top: 8px; font-size: 14px;">Asynchronous temporal signals</p>
     </div>
     
     <!-- SVG箭头 -->
     <div style="width: 20%; text-align: center; padding: 0 10px;">
       <svg width="80" height="40" style="display: block; margin: 0 auto;">
         <defs>
           <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                   refX="9" refY="3.5" orient="auto">
             <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
           </marker>
         </defs>
         <line x1="15" y1="20" x2="70" y2="20" 
               stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
       </svg>
       <p style="margin-top: 5px; font-size: 13px; color: #666;">CellDyc</p>
     </div>
     
     <!-- 右图 -->
     <div style="width: 60; text-align: center;">
       <img src="_static/index_2.png" style="width: 100%; max-width: 300px;">
       <p style="margin-top: 8px; font-size: 14px;">Projection of transcriptomic velocities</p>
     </div>
   </div>




**Key Highlights**:
~~~~~~~~~~~~~~
• **Data-Driven Velocity Inference:** Provides a robust, data-centric solution for calculating transcriptomic velocities.
• **Intrinsic Time Recovery:** Accurately recovers the intrinsic "gene-embedded time" from the transcriptome.
• **Flexible Supervision:** Capable of handling sparse and noisy supervision signals effectively.  
• **Seamless Integration:** Fully compatible with existing computational biology infrastructure and workflows.

.. toctree::
   :maxdepth: 2
   :caption: Main
   :titlesonly:
   :hidden:
   
   installation
   apis
   datasets

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :titlesonly:
   :hidden:

   Quick Start<quick_start>
   Recover Transcriptomic Velocity<recover_transcriptomic_velocity>
   Recover Masked Time Points<recover_masked_time_points>
   Gene-Specific Velocity Dynamics<gene_specific_velocity_dynamics>
   Handling Zman-seq Timestamps<handling_timestamps>

