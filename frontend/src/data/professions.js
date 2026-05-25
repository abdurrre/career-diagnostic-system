export const professions = [
  "Data Engineer",
  "Data Analyst",
  "Backend Developer",
  "AI / Machine Learning Engineer",
  "Data Scientist",
  "Fullstack Developer",
  "Frontend Developer"
];

// Rich custom dynamic profile data for pixel-perfect realism matching the selected target role
export const professionsData = {
  "Data Scientist": {
    matchScore: 78,
    skills: ["Python", "SQL", "Data Visualization", "Machine Learning Basics", "Communication", "Project Management", "Agile Methodologies", "Pandas"],
    gaps: [
      {
        title: "Deep Learning",
        tier: "CRITICAL",
        description: "Essential for Senior roles. Your profile lacks experience with frameworks like TensorFlow or PyTorch."
      },
      {
        title: "MLOps Pipeline",
        tier: "IMPORTANT",
        description: "Crucial for deploying models to production. Familiarize yourself with Docker, Kubernetes, and CI/CD for ML."
      },
      {
        title: "Cloud Architecture",
        tier: "SUPPLEMENTARY",
        description: "While not always mandatory, AWS or GCP certifications significantly boost a Senior profile."
      }
    ]
  },
  "Backend Developer": {
    matchScore: 84,
    skills: ["Node.js", "Express.js", "SQL", "PostgreSQL", "REST APIs", "Git", "Docker", "Database Design", "Communication"],
    gaps: [
      {
        title: "Redis Caching",
        tier: "CRITICAL",
        description: "Your profile lacks experience with in-memory caching databases. Essential for high-concurrency systems."
      },
      {
        title: "Kubernetes",
        tier: "IMPORTANT",
        description: "Crucial for microservices orchestration. Familiarize yourself with container deployment pipelines."
      },
      {
        title: "GraphQL",
        tier: "SUPPLEMENTARY",
        description: "While REST is strong, GraphQL experience significantly boosts your profile for modern API design."
      }
    ]
  },
  "Frontend Developer": {
    matchScore: 81,
    skills: ["React", "HTML/CSS", "JavaScript", "Tailwind CSS", "Vite", "TypeScript", "Responsive Design", "Git"],
    gaps: [
      {
        title: "Next.js & SSR",
        tier: "CRITICAL",
        description: "Your profile focuses on standard SPAs. Knowledge of server-side rendering is critical for modern frontend roles."
      },
      {
        title: "Cypress Testing",
        tier: "IMPORTANT",
        description: "Crucial for robust application deployment. Focus on learning end-to-end and component testing."
      },
      {
        title: "Web Accessibility",
        tier: "SUPPLEMENTARY",
        description: "Understanding WCAG compliance and semantic markup is a great supplementary advantage for Senior roles."
      }
    ]
  },
  "Data Engineer": {
    matchScore: 76,
    skills: ["Python", "SQL", "ETL Pipelines", "Apache Spark", "Data Modeling", "BigQuery", "Data Warehouse", "Git"],
    gaps: [
      {
        title: "Apache Airflow",
        tier: "CRITICAL",
        description: "Essential for workflow orchestration. Your profile lacks experience in scheduling complex data pipelines."
      },
      {
        title: "dbt (data build tool)",
        tier: "IMPORTANT",
        description: "Crucial for modern analytical engineering. Focus on learning SQL transformations in warehouse pipelines."
      },
      {
        title: "Snowflake Cloud",
        tier: "SUPPLEMENTARY",
        description: "Understanding Snowflake or Redshift data warehousing architectures is a valuable cloud extension."
      }
    ]
  },
  "Data Analyst": {
    matchScore: 82,
    skills: ["SQL", "Tableau", "Power BI", "Excel", "Data Analysis", "Statistics", "Communication", "Pandas"],
    gaps: [
      {
        title: "Advanced SQL",
        tier: "CRITICAL",
        description: "Your profile lacks complex window functions and performance tuning skills. Essential for handling massive datasets."
      },
      {
        title: "Statistical Modeling",
        tier: "IMPORTANT",
        description: "Crucial for advanced business insights. Focus on regression analysis and A/B testing methodologies."
      },
      {
        title: "Python Scripting",
        tier: "SUPPLEMENTARY",
        description: "While GUI tools are strong, Python scripting (Pandas) helps automate routine reports."
      }
    ]
  },
  "AI / Machine Learning Engineer": {
    matchScore: 75,
    skills: ["Python", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning Basics", "Git", "Scikit-Learn"],
    gaps: [
      {
        title: "Large Language Models",
        tier: "CRITICAL",
        description: "Essential for modern AI roles. Your profile lacks experience with prompt engineering, RAG, and LLM fine-tuning."
      },
      {
        title: "Model Quantization",
        tier: "IMPORTANT",
        description: "Crucial for edge deployments. Learn to optimize models using ONNX or TensorRT."
      },
      {
        title: "GCP AI Services",
        tier: "SUPPLEMENTARY",
        description: "Knowledge of Vertex AI or AWS SageMaker significantly boosts enterprise AI engineering roles."
      }
    ]
  },
  "Fullstack Developer": {
    matchScore: 85,
    skills: ["React", "Node.js", "Express.js", "JavaScript", "SQL", "Git", "Tailwind CSS", "REST APIs", "Docker"],
    gaps: [
      {
        title: "CI/CD Pipelines",
        tier: "CRITICAL",
        description: "Essential for modern agile engineering. Your profile lacks experience with automated Github Actions or GitLab CI/CD."
      },
      {
        title: "System Design",
        tier: "IMPORTANT",
        description: "Crucial for scaling applications. Focus on microservices architectures and database replication techniques."
      },
      {
        title: "TypeScript",
        tier: "SUPPLEMENTARY",
        description: "Strongly typed Javascript is a valuable supplementary skill for codebase scaling and code safety."
      }
    ]
  }
};

export const defaultProfessionData = {
  matchScore: 78,
  skills: ["Python", "SQL", "Git", "Communication", "Problem Solving", "Teamwork", "Project Management"],
  gaps: [
    {
      title: "Advanced Specialization",
      tier: "CRITICAL",
      description: "Essential for advanced roles. Your profile lacks deep experience with specialized target role frameworks."
    },
    {
      title: "Systems Integration",
      tier: "IMPORTANT",
      description: "Crucial for production environments. Familiarize yourself with cloud deployment and workflow automation."
    },
    {
      title: "Cloud Operations",
      tier: "SUPPLEMENTARY",
      description: "While not always mandatory, public cloud certification (AWS, Azure, or GCP) significantly boosts profiles."
    }
  ]
};
