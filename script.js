document.addEventListener('DOMContentLoaded', () => {
    const roadmapData = {
        // Updated to match skills from authoritative markdown file. Only skills present in the markdown are included. Resource links will be updated in the next step.
        "title": "Complete Roadmap to Land High-Paying ML/Data Science Roles",
        "subtitle": "Skills-Based Hiring Strategy (2026-2029)",
        "introduction": "Based on extensive research of placement records, industry hiring patterns, and salary data across Indian colleges and companies, this comprehensive analysis reveals that <strong>technical competence and project portfolio matter far more than college pedigree</strong> when it comes to securing high-paying internships and jobs in ML/Data Science. Companies like Google (₹47K/month internships), Microsoft (₹47K/month), Amazon (₹49K/month), and even mid-tier firms are prioritizing skill demonstration over institutional brands. The key finding: <strong>students with the right skill combination can command ₹5L+ internship stipends and ₹15-30L+ starting salaries regardless of their college tier</strong>, provided they master specific technical competencies that companies actively test for during interviews.",
        "years": [
        {
            "year": "Year 1 (2026)",
            "title": "Foundation Building - “The Programming and Math Foundation”",
            "sections": [
                {
                    "title": "Core Programming Skills",
                    "skills": [
                        { "name": "Master Python programming from basics to advanced concepts", "resources": [
    "https://www.python.org/",
    "https://realpython.com/python3-object-oriented-programming/",
    "https://www.codecademy.com/learn/learn-python-3",
    "https://www.coursera.org/specializations/python" /* Python for Everybody (Coursera, Dr. Charles Severance) – Free/Paid */,
    "https://www.mooc.fi/en/" /* University of Helsinki Python MOOC – Free */,
    "https://www.kaggle.com/learn/python" /* Kaggle Learn: Python – Free */,
    "https://pll.harvard.edu/course/cs50s-introduction-programming-python" /* Harvard CS50P – Free */,
    "https://skillsforall.com/course/python-essentials-1" /* Cisco Python Networking Academy – Free */
] },
                        { "name": "Learn data structures, algorithms, and object-oriented programming", "resources": [
    "https://geeksforgeeks.org/data-structures/",
    "https://realpython.com/python3-object-oriented-programming/"
] },
                        { "name": "Gain proficiency in SQL for database operations and complex queries", "resources": [
    "https://www.w3schools.com/sql/",
    "https://sqlbolt.com/"
] },
                        { "name": "Start with Jupyter Notebooks for interactive development", "resources": [
    "https://jupyter.org/try",
    "https://dataquest.io/blog/jupyter-notebook-tutorial/"
] }
                    ]
                },
                {
                    "title": "Mathematical Foundation",
                    "skills": [
                        { "name": "Statistics and probability theory fundamentals", "resources": [
    "https://www.khanacademy.org/math/statistics-probability",
    "https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/"
] },
                        { "name": "Linear algebra for machine learning", "resources": [
    "https://www.khanacademy.org/math/linear-algebra",
    "http://joshua.smcvt.edu/linearalgebra/"
] },
                        { "name": "Calculus basics for optimization understanding", "resources": ["https://www.khanacademy.org/math/calculus-1", "https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/"] },
                        { "name": "Descriptive and inferential statistics", "resources": ["https://www.udacity.com/course/intro-to-inferential-statistics--ud201", "https://www.coursera.org/learn/descriptive-statistics"] }
                    ]
                },
                {
                    "title": "Data Analysis Toolkit",
                    "skills": [
                        { "name": "Pandas for data manipulation and analysis", "resources": [
    "https://pandas.pydata.org/docs/",
    "https://kaggle.com/learn/pandas"
] },
                        { "name": "NumPy for numerical computing", "resources": [
    "https://numpy.org/doc/"
] },
                        { "name": "Matplotlib and Seaborn for data visualization", "resources": [
    "https://matplotlib.org/",
    "https://seaborn.pydata.org/"
] },
                        { "name": "Excel/Google Sheets for business data analysis", "resources": [
    "https://www.excel-easy.com/",
    "https://support.google.com/a/users/answer/9282959"
] }
                    ]
                }
            ]
        },
        {
            "year": "Year 2 (2027)",
            "title": "Core Machine Learning - “The Algorithm Mastery Phase”",
            "sections": [
                {
                    "title": "Machine Learning Fundamentals",
                    "skills": [
                        { "name": "Supervised learning: Linear/Logistic Regression, Decision Trees, Random Forest, SVM", "resources": ["https://www.coursera.org/learn/machine-learning", "https://scikit-learn.org/stable/supervised_learning.html", "https://www.coursera.org/learn/machine-learning" /* Supervised Machine Learning: Regression and Classification (Coursera/Andrew Ng) – Free/Paid */,
"https://developers.google.com/machine-learning/crash-course" /* Google Machine Learning Crash Course – Free */,
"https://www.coursera.org/professional-certificates/ibm-data-science" /* IBM Data Science Professional Certificate – Paid, Free Audit */,
"https://www.datacamp.com/tracks/data-scientist-with-python" /* DataCamp Data Scientist Track – Free Trial/Paid */,
"https://www.udemy.com/course/introduction-to-machine-learning-for-data-science/" /* Udemy: Introduction to Machine Learning for Data Science – Free */] },
                        { "name": "Unsupervised learning: K-Means, Hierarchical Clustering, PCA", "resources": ["https://scikit-learn.org/stable/unsupervised_learning.html", "https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning"] },
                        { "name": "Model evaluation: Cross-validation, confusion matrix, ROC-AUC, precision-recall", "resources": ["https://scikit-learn.org/stable/modules/cross_validation.html", "https://www.jeremyjordan.me/model-evaluation/"] },
                        { "name": "Feature engineering and selection techniques", "resources": ["https://www.featuretools.com/", "https://www.kaggle.com/learn/feature-engineering"] }
                    ]
                },
                {
                    "title": "Advanced Data Visualization",
                    "skills": [
                        { "name": "Tableau for business intelligence dashboards", "resources": ["https://www.tableau.com/learn/training", "https://public.tableau.com/en-us/s/"] },
                        { "name": "Power BI for enterprise reporting", "resources": ["https://learn.microsoft.com/en-us/power-bi/", "https://powerbi.microsoft.com/en-us/getting-started-with-power-bi/"] },
                        { "name": "Plotly for interactive visualizations", "resources": ["https://plotly.com/python/", "https://www.geeksforgeeks.org/python-plotly-tutorial/"] },
                        { "name": "Advanced statistical plots and storytelling with data", "resources": ["https://www.storytellingwithdata.com/", "https://www.amazon.com/Storytelling-Data-Visualization-Business-Professionals/dp/1119002257"] }
                    ]
                },
                {
                    "title": "API and Web Development",
                    "skills": [
                        { "name": "Flask/FastAPI for creating ML APIs", "resources": ["https://flask.palletsprojects.com/en/2.3.x/", "https://fastapi.tiangolo.com/"] },
                        { "name": "REST API development and consumption", "resources": ["https://www.redhat.com/en/topics/api/what-is-a-rest-api", "https://www.smashingmagazine.com/2018/01/understanding-using-rest-api/"] },
                        { "name": "Basic web scraping with BeautifulSoup and Scrapy", "resources": ["https://www.crummy.com/software/BeautifulSoup/bs4/doc/", "https://scrapy.org/"] },
                        { "name": "Version control with Git and GitHub", "resources": ["https://git-scm.com/doc", "https://docs.github.com/en"] }
                    ]
                }
            ]
        },
        {
            "year": "Year 3 (2028)",
            "title": "Deep Learning and Specialization - “The Advanced AI Phase”",
            "sections": [
                {
                    "title": "Deep Learning Mastery",
                    "skills": [
                        { "name": "Neural network fundamentals and backpropagation", "resources": ["http://neuralnetworksanddeeplearning.com/", "https://www.coursera.org/specializations/deep-learning", "https://course.fast.ai/" /* Fast.ai Practical Deep Learning for Coders – Free */,
"https://www.coursera.org/specializations/deep-learning" /* Deep Learning Specialization by Andrew Ng (Coursera/Deeplearning.ai) – Free Audit/Paid Cert */] },
                        { "name": "Convolutional Neural Networks (CNNs) for computer vision", "resources": ["https://cs231n.github.io/", "https://www.coursera.org/learn/convolutional-neural-networks", "https://web.stanford.edu/class/cs231n/" /* Stanford CS231n – Free */] },
                        { "name": "Recurrent Neural Networks (RNNs/LSTMs) for sequential data", "resources": ["https://colah.github.io/posts/2015-08-Understanding-LSTMs/", "https://www.coursera.org/learn/nlp-sequence-models"] },
                        { "name": "Transfer learning and pre-trained models", "resources": ["https://www.tensorflow.org/tutorials/images/transfer_learning", "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"] },
                        { "name": "TensorFlow and PyTorch frameworks", "resources": ["https://www.tensorflow.org/tutorials", "https://pytorch.org/tutorials"] }
                    ]
                },
                {
                    "title": "Computer Vision",
                    "skills": [
                        { "name": "Image preprocessing and augmentation", "resources": ["https://www.tensorflow.org/tutorials/images/data_augmentation", "https://albumentations.ai/docs/"] },
                        { "name": "Object detection and image classification", "resources": ["https://www.coursera.org/learn/convolutional-neural-networks", "https://github.com/ultralytics/yolov5"] },
                        { "name": "OpenCV for computer vision applications", "resources": ["https://docs.opencv.org/4.x/d9/df8/tutorial_root.html", "https://www.geeksforgeeks.org/opencv-python-tutorial/"] },
                        { "name": "Working with image datasets and annotation", "resources": ["https://www.roboflow.com", "https://labelbox.com/"] }
                    ]
                },
                {
                    "title": "Natural Language Processing",
                    "skills": [
                        { "name": "Text preprocessing and tokenization", "resources": ["https://www.nltk.org/", "https://huggingface.co/docs/transformers/preprocessing"] },
                        { "name": "Sentiment analysis and text classification", "resources": ["https://textblob.readthedocs.io/en/dev/", "https://www.kaggle.com/learn/natural-language-processing"] },
                        { "name": "Named Entity Recognition (NER)", "resources": ["https://spacy.io/usage/linguistic-features#named-entities", "https://www.coursera.org/learn/classification-vector-spaces-in-nlp"] },
                        { "name": "Word embeddings and language models", "resources": ["https://www.tensorflow.org/text/guide/word_embeddings", "https://jalammar.github.io/illustrated-word2vec/", "https://huggingface.co/course/chapter1/1" /* Hugging Face “NLP Course” & [LLMs] – Free */] }
                    ]
                },
                {
                    "title": "Cloud and Deployment",
                    "skills": [
                        { "name": "AWS/Azure/GCP cloud platforms", "resources": ["https://aws.amazon.com/training/", "https://learn.microsoft.com/en-us/azure/", "https://cloud.google.com/learn"] },
                        { "name": "Docker containerization", "resources": ["https://docs.docker.com/get-started/", "https://www.coursera.org/learn/docker-essentials"] },
                        { "name": "Model deployment and serving", "resources": ["https://www.tensorflow.org/tfx/guide/serving", "https://www.bentoml.ai/"] },
                        { "name": "Basic MLOps practices", "resources": ["https://ml-ops.org/", "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops", "https://github.com/DataTalksClub/mlops-zoomcamp" /* MLOps Zoomcamp by DataTalksClub – Free */,
"https://www.udemy.com/topic/mlops/" /* Top Udemy MLOps Courses – Paid */] }
                    ]
                }
            ]
        },
        {
            "year": "Year 4 (2029)",
            "title": "Cutting-Edge Specialization - “The Industry-Ready Expert Phase”",
            "sections": [
                {
                    "title": "Large Language Models and Generative AI",
                    "skills": [
                        { "name": "Understanding transformer architecture", "resources": ["https://jalammar.github.io/illustrated-transformer/", "https://huggingface.co/course/chapter1/1", "https://www.kaggle.com/learn/intro-to-generative-ai" /* Google x Kaggle GenAI Intensive – Free */] },
                        { "name": "Fine-tuning pre-trained language models", "resources": ["https://huggingface.co/docs/transformers/training", "https://www.tensorflow.org/tutorials/text/fine_tune_bert", "https://www.databricks.com/learn/certification/llm-professional" /* Databricks LLM Professional Certificate – Paid */] },
                        { "name": "Prompt engineering and few-shot learning", "resources": ["https://www.promptingguide.ai/", "https://www.coursera.org/learn/prompt-engineering", "https://learnprompting.org/" /* LearnPrompting & Large Language Models (LLMs) Courses List – Free/Paid */] },
                        { "name": "Working with GPT, BERT, and other LLM variants", "resources": ["https://beta.openai.com/docs/", "https://huggingface.co/models"] },
                        { "name": "Building ChatGPT-like applications", "resources": ["https://github.com/openai/openai-cookbook/blob/main/examples/How_to_build_a_chatbot_that_can_answer_questions_about_your_website.ipynb", "https://www.deeplearning.ai/short-courses/building-applications-with-vector-databases/"] }
                    ]
                },
                {
                    "title": "Production MLOps",
                    "skills": [
                        { "name": "Kubernetes for ML workload orchestration", "resources": ["https://kubernetes.io/docs/home/", "https://www.coursera.org/learn/gcp-kubernetes-engine"] },
                        { "name": "CI/CD pipelines for ML models", "resources": ["https://www.jeremyjordan.me/ci-cd-for-machine-learning/", "https://github.com/features/actions"] },
                        { "name": "Model monitoring and drift detection", "resources": ["https://www.whylabs.ai/", "https://www.evidentlyai.com/"] },
                        { "name": "A/B testing for ML systems", "resources": ["https://www.coursera.org/learn/ab-testing", "https://neptune.ai/blog/a-b-testing-for-machine-learning-what-how-and-when"] },
                        { "name": "Scalable ML system architecture", "resources": ["https://cnvrg.io/building-scalable-machine-learning-infrastructure/ (Free)", "https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969", "https://www.coursera.org/learn/machine-learning-system-design"] }
                    ]
                },
                {
                    "title": "Advanced Specializations",
                    "skills": [
                        { "name": "Time series forecasting for business applications", "resources": ["https://otexts.com/fpp3/", "https://www.coursera.org/learn/time-series"] },
                        { "name": "Recommendation systems", "resources": ["https://github.com/grahamjenson/list_of_recommender_systems (Free, Open Source)", "https://www.coursera.org/specializations/recommender-systems", "https://developers.google.com/machine-learning/recommendation"] },
                        { "name": "Reinforcement learning basics", "resources": ["https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning", "https://www.coursera.org/learn/reinforcement-learning"] },
                        { "name": "Edge AI and model optimization", "resources": ["https://www.tensorflow.org/lite", "https://pytorch.org/mobile/home/"] },
                        { "name": "Advanced statistical modeling", "resources": ["https://www.biostat.jhsph.edu/~iruczins/teaching/books/2019.openintro.statistics.pdf (Free PDF)", "https://www.statlearning.com/", "https://www.coursera.org/learn/statistical-modeling"] }
                    ]
                },
                {
                    "title": "Leadership and Soft Skills",
                    "skills": [
                        { "name": "Technical communication and presentation", "resources": ["https://www.coursera.org/learn/technical-presentations", "https://www.toastmasters.org/"] },
                        { "name": "Project management for ML projects", "resources": ["https://www.stxnext.com/blog/machine-learning-implementation-project-management (Free)", "https://www.pmi.org/", "https://www.coursera.org/learn/ai-product-management"] },
                        { "name": "Cross-functional collaboration", "resources": ["https://www.coursera.org/learn/collaborative-leadership", "https://hbr.org/2019/05/cross-functional-collaboration"] },
                        { "name": "Business acumen and domain expertise", "resources": ["https://www.coursera.org/learn/business-analytics", "https://hbr.org/topic/business-acumen"] }
                    ]
                }
            ]
        }
    ]
    };

    function renderRoadmap() {
        const header = document.querySelector('header');
        header.innerHTML = `
            <h1>${roadmapData.title}</h1>
            <p class="subtitle">${roadmapData.subtitle}</p>
        `;

        const main = document.querySelector('main');
        const intro = document.createElement('p');
        intro.innerHTML = roadmapData.introduction;
        intro.classList.add('introduction');
        main.prepend(intro);

        const roadmapContainer = document.getElementById('roadmap-container');
        roadmapData.years.forEach((yearData, yearIndex) => {
            const yearCard = document.createElement('div');
            yearCard.classList.add('year-card');
            yearCard.dataset.yearIndex = yearIndex; // Add data attribute

            yearCard.innerHTML = `
                <h2>${yearData.year}</h2>
                <h3>${yearData.title}</h3>
            `;

            const sectionsContainer = document.createElement('div');
            sectionsContainer.classList.add('sections-container');

            yearData.sections.forEach(section => {
                const sectionCard = document.createElement('div');
                sectionCard.classList.add('section-card');
                sectionCard.dataset.yearIndex = yearIndex; // Add data attribute

                let skillsHtml = '<ul>';
                section.skills.forEach((skill, skillIdx) => {
                    skillsHtml += `
                        <li class="skill-item" data-skill-idx="${skillIdx}">
                            <div class="skill-name">
                                <span class="pin-icon"><i class="fas fa-thumbtack"></i></span> ${skill.name}
                            </div>
                            <div class="skill-resources">
                                <ul>
                                    ${skill.resources.map(resource => {
    let icon = '';
    if (
        resource.includes('python.org') ||
        resource.includes('w3schools.com') ||
        resource.includes('sqlbolt.com') ||
        resource.includes('jupyter.org') ||
        resource.includes('khanacademy.org') ||
        resource.includes('ocw.mit.edu') ||
        resource.includes('joshua.smcvt.edu') ||
        resource.includes('pandas.pydata.org') ||
        resource.includes('numpy.org') ||
        resource.includes('matplotlib.org') ||
        resource.includes('seaborn.pydata.org') ||
        resource.includes('excel-easy.com') ||
        resource.includes('support.google.com/a/users/answer/9282959') ||
        resource.includes('scikit-learn.org') ||
        resource.includes('flask.palletsprojects.com') ||
        resource.includes('fastapi.tiangolo.com') ||
        resource.includes('redhat.com/en/topics/api/what-is-a-rest-api') ||
        resource.includes('smashingmagazine.com/2018/01/understanding-using-rest-api') ||
        resource.includes('crummy.com/software/BeautifulSoup/bs4/doc/') ||
        resource.includes('scrapy.org') ||
        resource.includes('git-scm.com/doc') ||
        resource.includes('docs.github.com/en') ||
        resource.includes('cs231n.github.io') ||
        resource.includes('colah.github.io/posts/2015-08-Understanding-LSTMs/') ||
        resource.includes('tensorflow.org/tutorials') ||
        resource.includes('pytorch.org/tutorials') ||
        resource.includes('huggingface.co/docs/transformers/preprocessing') ||
        resource.includes('textblob.readthedocs.io/en/dev/') ||
        resource.includes('spacy.io/usage/linguistic-features#named-entities') ||
        resource.includes('tensorflow.org/text/guide/word_embeddings') ||
        resource.includes('jalammar.github.io/illustrated-word2vec/') ||
        resource.includes('roboflow.com') ||
        resource.includes('labelbox.com') ||
        resource.includes('featuretools.com') ||
        resource.includes('kaggle.com/learn/feature-engineering') ||
        resource.includes('kaggle.com/learn/pandas') ||
        resource.includes('geeksforgeeks.org/python-plotly-tutorial/') ||
        resource.includes('github.com/features/actions') ||
        resource.includes('jeremyjordan.me/model-evaluation/') ||
        resource.includes('jeremyjordan.me/ci-cd-for-machine-learning/') ||
        resource.includes('whylabs.ai') ||
        resource.includes('evidentlyai.com') ||
        resource.includes('neptune.ai/blog/a-b-testing-for-machine-learning-what-how-and-when') ||
        resource.includes('statlearning.com') ||
        resource.includes('otexts.com/fpp3/') ||
        resource.includes('developers.google.com/machine-learning/recommendation') ||
        resource.includes('deepmind.com/learning-resources/introduction-to-reinforcement-learning') ||
        resource.includes('tensorflow.org/lite') ||
        resource.includes('pytorch.org/mobile/home/') ||
        resource.includes('toastmasters.org')
    ) {
        icon = '🟢'; // Free
    } else if (
        resource.includes('codecademy.com') ||
        resource.includes('udacity.com') ||
        (resource.includes('coursera.org') && !resource.includes('coursera.org/learn/descriptive-statistics')) ||
        resource.includes('tableau.com/learn/training') ||
        resource.includes('public.tableau.com/en-us/s/') ||
        resource.includes('powerbi.microsoft.com/en-us/getting-started-with-power-bi/') ||
        resource.includes('storytellingwithdata.com') ||
        resource.includes('amazon.com/Storytelling-Data-Visualization-Business-Professionals/dp/1119002257') ||
        resource.includes('deeplearning.ai') ||
        resource.includes('bentoml.ai') ||
        resource.includes('pmi.org') ||
        resource.includes('hbr.org') ||
        resource.includes('openai.com') ||
        resource.includes('amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969')
    ) {
        icon = '🔴'; // Paid
    } else if (
        resource.includes('realpython.com') ||
        resource.includes('geeksforgeeks.org') ||
        resource.includes('dataquest.io') ||
        resource.includes('kaggle.com') ||
        resource.includes('huggingface.co')
    ) {
        icon = '🟡'; // Mixed/partial free
    } else {
        icon = '<span title="Status unknown">⚪</span>';
    }
    const isOfficial = resource.includes('python.org') || resource.includes('w3schools.com') || resource.includes('tensorflow.org') || resource.includes('pytorch.org') || resource.includes('numpy.org') || resource.includes('pandas.pydata.org') || resource.includes('scikit-learn.org');
    return `<li class="${isOfficial ? '' : 'external'}">
        <span>${icon}</span> <a href="${resource}" target="_blank">${new URL(resource).hostname}</a>
    </li>`;
}).join('')}
                                </ul>
                            </div>
                        </li>
                    `;
                });
                skillsHtml += '</ul>';

                sectionCard.innerHTML = `
                    <h4>${section.title}</h4>
                    ${skillsHtml}
                `;
                sectionsContainer.appendChild(sectionCard);
            });

            yearCard.appendChild(sectionsContainer);
            roadmapContainer.appendChild(yearCard);
        });
    }

    function setupEventListeners() {
        // Use event delegation for skill expand/collapse and pin color change on hover
        document.querySelectorAll('.sections-container').forEach(container => {
            container.addEventListener('click', function(e) {
                const pinIcon = e.target.closest('.pin-icon');
                if (pinIcon) {
                    const skillItem = pinIcon.closest('.skill-item');
                    if (skillItem) {
                        console.log('Toggling active class for skill item:', skillItem);
                        skillItem.classList.toggle('active');
                    } else {
                        console.log('skillItem not found.');
                    }
                } else {
                    console.log('pinIcon not found.');
                }
            });

            container.addEventListener('mouseover', function(e) {
                const pinIcon = e.target.closest('.pin-icon');
                if (pinIcon) {
                    const skillItem = pinIcon.closest('.skill-item');
                    if (skillItem) {
                        const sectionCard = skillItem.closest('.section-card');
                        if (sectionCard) {
                            const yearIndex = sectionCard.dataset.yearIndex;
                            const yearCard = document.querySelector(`.year-card[data-year-index="${yearIndex}"]`);
                            if (yearCard) {
                                const yearBorderColor = window.getComputedStyle(yearCard).borderColor;
                                pinIcon.style.color = yearBorderColor;
                            }
                        }
                    }
                }
            });

            container.addEventListener('mouseout', function(e) {
                const pinIcon = e.target.closest('.pin-icon');
                if (pinIcon) {
                    // Set to white color when mouse leaves
                    pinIcon.style.color = '#ffffff';
                }
            });
        });

        // Handle section-card hover effects
        document.querySelectorAll('.section-card').forEach(sectionCard => {
            sectionCard.addEventListener('mouseover', function() {
                const yearIndex = this.dataset.yearIndex;
                const yearCard = document.querySelector(`.year-card[data-year-index="${yearIndex}"]`);
                if (yearCard) {
                    const yearBorderColor = window.getComputedStyle(yearCard).borderColor;
                    this.style.borderColor = yearBorderColor;
                }
            });

            sectionCard.addEventListener('mouseout', function() {
                this.style.borderColor = 'transparent';
            });
        });

        // Handle skill-item hover effects
        document.querySelectorAll('.skill-item').forEach(skillItem => {
            skillItem.addEventListener('mouseover', function() {
                const sectionCard = this.closest('.section-card');
                if (sectionCard) {
                    const yearIndex = sectionCard.dataset.yearIndex;
                    const yearCard = document.querySelector(`.year-card[data-year-index="${yearIndex}"]`);
                    if (yearCard) {
                        const yearBorderColor = window.getComputedStyle(yearCard).borderColor;
                        this.style.borderColor = yearBorderColor;
                    }
                }
            });

            skillItem.addEventListener('mouseout', function() {
                this.style.borderColor = 'transparent';
            });
        });

        const goToTopButton = document.getElementById('go-to-top');
        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 300) {
                goToTopButton.style.display = 'block';
            } else {
                goToTopButton.style.display = 'none';
            }
        });

        goToTopButton.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    renderRoadmap();
    setupEventListeners();
    
    // --- Flexible Thread Connecting Year-Card Punch Holes ---
    function drawTimelineThread() {
        const container = document.getElementById('roadmap-container');
        let svg = container.querySelector('.timeline-thread');
        if (!svg) {
            svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.classList.add('timeline-thread');
            container.insertBefore(svg, container.firstChild);
        }
        svg.innerHTML = '';
        // Get all year-cards
        const cards = Array.from(container.getElementsByClassName('year-card'));
        if (cards.length < 2) return;
        // Get punch hole center positions
        const points = cards.map(card => {
            const rect = card.getBoundingClientRect();
            const contRect = container.getBoundingClientRect();
            const isOdd = Array.from(card.parentNode.children).indexOf(card) % 2 === 0;
            // After swapping: odd cards (right) punch hole on left, even cards (left) punch hole on right
            const x = isOdd ? (rect.left - contRect.left + 25) : (rect.right - contRect.left - 25);
            const y = rect.top - contRect.top + 25;
            return { x, y };
        });
        // Set SVG size
        svg.setAttribute('width', container.offsetWidth);
        svg.setAttribute('height', container.offsetHeight);
        // Draw path
        let d = '';
        for (let i = 0; i < points.length - 1; i++) {
            const p1 = points[i], p2 = points[i + 1];
            // Cubic Bezier for flexibility
            const mx = (p1.x + p2.x) / 2;
            d += `M${p1.x},${p1.y} C${mx},${p1.y} ${mx},${p2.y} ${p2.x},${p2.y} `;
        }
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', d.trim());
        path.setAttribute('stroke', '#FFD700');
        path.setAttribute('stroke-width', '4');
        path.setAttribute('fill', 'none');
        path.setAttribute('filter', 'drop-shadow(0 2px 6px #0008)');
        svg.appendChild(path);
    }
    window.addEventListener('resize', drawTimelineThread);
    setTimeout(drawTimelineThread, 600);
    // Redraw after DOM animations
    setTimeout(drawTimelineThread, 1200);
});