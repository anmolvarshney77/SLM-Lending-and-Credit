## Hackathon Problem Statement 

## Fine-Tune a Small Language Model for Lending Intelligence & Credit Risk Assessment 

## Theme / Summary 

Lending AI – Fine-tune a Small Language Model (SLM) with fewer than 3 billion parameters on a real-world lending dataset to build a domain-aware Lending Intelligence Assistant. 

Participants will prepare training data, apply LoRA-based fine-tuning, and demonstrate measurable improvement over the base model on lending-specific tasks such as: 

- **Loan Summary Generation** 

- **Credit Risk Classification** 

- **Loan Approval Recommendation** 

## Business Scenario 

ABC Finance Ltd. is a leading retail lending organization that processes thousands of loan applications every month across Personal Loans, Vehicle Loans, MSME Loans, Consumer Durable Loans, and Home Loans. 

Loan officers, underwriters, and collections teams face several challenges: 

- Assessing borrower creditworthiness consistently 

- Identifying high-risk customers early 

- Predicting future delinquencies 

- Prioritizing collection activities 

- Explaining lending decisions transparently 

The organization recently piloted a generic Large Language Model chatbot to assist credit officers. However, the results have been disappointing. 

The model: 

- Does not understand lending concepts such as DPD, FOIR, Bureau Score, and Collection Buckets. 

- Confuses sanctioned amount with outstanding balance. 

- Generates inconsistent underwriting recommendations. 

- Fails to identify early warning signals of delinquency. 

- Produces unreliable explanations for loan approval and rejection decisions. 

The Data Science team has curated a lending dataset containing customer demographics, income details, loan characteristics, bureau scores, repayment behavior, delinquency history, and loan outcomes. 

Your mission is to fine-tune a Small Language Model (SLM) using LoRA/QLoRA to transform a generic model into a domain-aware Lending Intelligence Assistant capable of helping underwriters, credit analysts, portfolio managers, and collection teams make informed decisions. 

## Dataset 

Participants will receive a lending dataset containing approximately 1,000 loan records. Example fields include: 

## Loan Portfolio Dataset – Data Dictionary 

|**Field Name**|**Data**<br>**Type**|**Description**|**Example**|**Business**<br>**Signifcance**|
|---|---|---|---|---|
|**LOAN_ID**|String|Unique identifer<br>assigned to a loan<br>account.|<br>LN00012345|Primary key used to<br>identify a loan.|
|**CUSTOMER_ID**|String|Unique identifer<br>assigned to a<br>customer. A<br>customer may<br>have multiple<br>loans.|CUST10025|Used to link<br>multiple loans<br>belonging to the<br>same customer.|
|**AGE**|Integer|Age of the<br>borrower in years.|35|Important factor in<br>credit underwriting<br>and risk<br>assessment.|
|**GENDER**|String|Gender of the<br>borrower.|Male, Female|Demographic<br>attribute used for<br>portfolio analysis.|
|**STATE**|String|State where the<br>borrower resides.|Maharashtra|Helps analyze<br>regional lending<br>and riskpatterns.|
|**OCCUPATION**|String|Borrower's<br>profession or|Salaried,<br>Business,|Key indicator of<br>income stability|



|**Field Name**|**Data**<br>**Type**|**Description**|**Example**|**Business**<br>**Signifcance**|
|---|---|---|---|---|
|||employment<br>type.|Self-<br>Employed|and repayment<br>capacity.|
|**MONTHLY_INCOME**|Decimal|Monthly income<br>declared by the<br>borrower.|75000|Used to determine<br>afordability and<br>repayment ability.|
|**LOAN_PRODUCT**|String|Type of loan<br>availed by the<br>customer.|Personal<br>Loan, Vehicle<br>Loan, Home<br>Loan|Diferent products<br>have diferent risk<br>profles.|
|**LOAN_AMOUNT**|Decimal|Amount<br>requested by the<br>customer during<br>application.|500000|Represents<br>customer demand.|
|**SANCTION_AMOUNT**|Decimal|Amount approved<br>by the lender after<br>underwriting.|<br> <br>450000|May difer from<br>requested amount<br>based on risk<br>evaluation.|
|**LOAN_TENURE**|Integer|Duration of the<br>loan in months.|36|Longer tenures<br>typically reduce<br>EMI but increase<br>interestpaid.|
|**INTEREST_RATE**|Decimal|Annual interest<br>rate charged on<br>the loan.|14.5|Major contributor to<br>EMI and<br>proftability.|
|**EMI_AMOUNT**|Decimal|Equated Monthly<br>Installment<br>payable by the<br>borrower.|15600|Monthly repayment<br>obligation.|
|**BUREAU_SCORE**|Integer|Credit score<br>obtained from a<br>credit bureau.<br>Usually ranges<br>from 300-900.|785|One of the<br>strongest<br>indicators of<br>creditworthiness.|
|**VINTAGE_MONTHS**|Integer|Number of<br>months since<br>loan<br>disbursement.|18|Indicates loan age.<br>Older loans often<br>have richer<br>repayment history.|
|**CURRENT_DPD**|Integer|Current Days Past<br>Due. Number of<br>days the borrower|<br> <br>30|Critical indicator of<br>repayment<br>delinquency.|



|**Field Name**|**Data**<br>**Type**|**Description**|**Example**|**Business**<br>**Signifcance**|
|---|---|---|---|---|
|||is overdue on the<br>latestpayment.|||
|**MAX_DPD**|Integer|Maximum DPD<br>observed during<br>the loan lifecycle.|90|Refects the worst<br>repayment<br>behavior seen so<br>far.|
|**OUTSTANDING_BALANCE**|Decimal|Remaining unpaid<br>loan principal.|<br>325000|Amount yet to be<br>recovered by the<br>lender.|
|**COLLECTION_BUCKET**|String|Delinquency<br>category assigned<br>based on DPD.|<br>Current, 1-30,<br>31-60, 61-90,<br>90+|<br>Used by collection<br>teams to prioritize<br>recoveryactions.|
|**LOAN_STATUS**|String|Current status of<br>the loan account.|Active,<br>Closed,<br>Delinquent|Indicates whether<br>the loan is healthy<br>orproblematic.|
|**DEFAULT_FLAG**|Binary<br>(0/1)|Indicates whether<br>the borrower has<br>defaulted on the<br>loan.|<br>0 = No, 1 =<br>Yes|Common target<br>variable for<br>machine learning<br>models.|
|**WRITE_OFF_FLAG**|Binary<br>(0/1)|Indicates whether<br>the lender has<br>written of the<br>loan as<br>unrecoverable.|<br>0 = No, 1 =<br>Yes|Represents severe<br>credit loss.|
|**DISBURSAL_DATE**|Date|Date on which the<br>loan amount was<br>released to the<br>borrower.|<br>2024-01-15|Starting point of the<br>loan lifecycle.|
|**LAST_PAYMENT_DATE**|Date|Date of the most<br>recent EMI<br>payment<br>received.|2025-05-10|Used to calculate<br>payment recency<br>and delinquency<br>metrics.|
|Important Lending Concepts<br>Bureau Score|||||



A numerical score generated by credit bureaus based on a borrower's repayment history. 

|**Score Range **|**Interpretation**|
|---|---|
|800+|Excellent|
|750-799|VeryGood|
|700-749|Good|
|650-699|Moderate Risk|
|Below 650|High Risk|



## DPD (Days Past Due) 

Measures how many days a borrower is late in making an EMI payment. 

|**DPD**|**Meaning**|
|---|---|
|0|Payment made on time|
|1-30|Slight delay|
|31-60|Moderate delinquency|
|61-90|Serious delinquency|
|>90|Potential default|



## Collection Bucket 

Collection teams classify customers into buckets based on DPD. 

|**Bucket**|**DPD Range**|
|---|---|
|Current|0|
|1-30|1-30 days|
|31-60|31-60 days|
|61-90|61-90 days|
|90+|More than 90 days|



## Default vs Write-Off 

## **Default** 

- Borrower stops making payments. 

- Loan is still recoverable. 

## **Write-Off** 

- Lender treats the loan as a loss for accounting purposes. 

- Collection efforts may continue, but probability of recovery is low. 

## What To Build 

## A) Data Preparation & Prompt Engineering Pipeline 

Recommended Derived Features 

Participants are encouraged to engineer additional features that improve the model's understanding of borrower behaviour and credit risk. The following are some commonly used features in lending analytics. 

|**Feature**|**Description**|**Typical Use**|
|---|---|---|
|**FOIR (Fixed**<br>**Obligation to**<br>**Income Ratio)**|Measures the proportion of a borrower's monthly<br>income that is committed towards fxed fnancial<br>obligations, including the loan EMI. It is widely used<br>to assess repayment capacity and afordability.<br>**Formula**: FOIR = EMI Amount ÷ Monthly Income<br>**Business Logic**: Measures the proportion of borrower's monthly<br>income committed to fxed fnancial obligations. FOIR > 0.6 indicates<br>fnancial stress; values are capped at 1.0 (100%) as higher ratios are<br>unsustainable.|Loan Approval<br>Recommendation,<br>Credit Risk Classifcation|
|**EMI-to-Income**<br>**Ratio**|Indicates how much of the borrower's monthly<br>income is allocated specifcally towards the current<br>loan EMI. Higher values generally indicate greater<br>repayment burden.<br>**Formula:**EMI_Income_Ratio = (EMI Amount ÷ Monthly Income) × 100<br>**Business Logic:**Percentage form of FOIR showing how much of the<br>borrower's monthly income is allocated to current EMI. Higher<br>percentages indicate greater repayment burden.|Loan Summary<br>Generation, Loan<br>Approval<br>Recommendation|
|**Credit Utilization**|Represents the extent to which the borrower is<br>utilizing available credit relative to sanctioned or<br>available limits. High utilization may indicate<br>increased fnancial stress and higher credit risk.<br>Participants may derive this feature using<br>appropriate assumptions based on the available<br>dataset.<br>**Formula**: Credit_Utilization = Outstanding Balance ÷ Sanction Amount<br>**Business Logic**: Represents the extent to which the borrower is<br>utilizing available credit limits. High utilization (>80%) may indicate<br>increased fnancial stress and higher credit risk.|Credit Risk<br>Classifcation, Loan<br>Approval<br>Recommendation|



|**Feature**|**Description**|**Typical Use**|
|---|---|---|
|**Delinquency**<br>**Frequency**|Captures how frequently the borrower has<br>experienced payment delays or delinquent behaviour<br>over the loan lifecycle. It provides an indication of<br>historical repayment discipline.<br>**Current Implementation**: IS_DELINQUENT (binary: 1 if Current_DPD<br>> 0, else 0)<br>**Ideal Formula**: Count of delinquent periods ÷ Total payment periods<br>**Business Logic**: Captures how frequently the borrower has<br>experienced payment delinquency over the loan lifecycle, providing an<br>indication of repayment discipline.|<br>Credit Risk<br>Classifcation, Loan<br>Summary Generation|



## Instruction-Tuning Dataset Generation 

Generate prompt-completion pairs from the structured lending data. 

Example task types include: 

## Loan Summary Generation 

Given borrower profile and loan details, generate a natural language loan summary. 

Input: 

`Age: 35 Income:` ₹ `60,000 Loan Amount:` ₹ `5,00,000 Bureau Score: 785 Current DPD: 0` 

Output: 

```
Customer is a low-risk salaried borrower with strong repayment behavior...
```

## Credit Risk Classification 

Given customer demographics, bureau score, DPD history, and income details, classify borrower risk. 

Input: 

```
Borrower profile
```

Output: 

```
Low Risk
Medium Risk
High Risk
```

## Loan Approval Recommendation 

Given borrower profile and requested loan details, recommend: 

- Approve 

- Approve with Conditions 

- Reject 

## Input: 

```
Applicant Profile
```

Output: 

```
Approve
Approve with Conditions
Reject
```

## B) SLM Fine-Tuning Implementation 

Participants must: 

Model Selection 

Select an SLM not more than 3 billion parameters. 

Fine-Tuning Requirements 

- Base model weights must remain frozen 

- Use an open-source fine-tuning framework that supports parameter-efficient adaptation of Small Language Models. Participants should justify the choice of framework and tooling based on their selected architecture and hardware constraints. 

## Training Metrics 

Provide detailed training metrics 

## Evaluation & Demo 

Demonstrate before-vs-after comparison using at least 3 representative lending scenarios. 

Example questions: 

- Summarize this borrower profile. 

- Should this loan be approved? 

- What is the risk category of this customer? 

Participants should design and implement an appropriate evaluation methodology that objectively measures the effectiveness of their solution. The evaluation should include both quantitative and qualitative assessment of the model's performance, demonstrate measurable improvements over the base model, and justify the chosen evaluation metrics based on the task being performed. 

The submission should clearly explain the evaluation approach, present comparative results, and discuss the strengths, limitations, and business relevance of the fine-tuned model. 

## Constraints & Non-Functional Requirements 

## Hardware Constraints 

## Solution must run on: 

- 8–16 GB GPU VRAM 

- Consumer GPUs 

- CPU Offloading 

Training should complete within 1 hours. 

## Software Requirements 

## Mandatory: 

- Python 3.9+ 

- Transformers 

- PEFT 

- TRL 

- BitsAndBytes 

## Optional: 

- LangChain 

- LlamaIndex 

## Required Deliverables 

Participants must submit: 

- Final Dataset 

- JSONL Prompt-Completion Dataset 

- Fine-Tuning Notebook / Scripts 

- Evaluation Report 

## Judging Rubric 

## 1. Data Preparation & Prompt Engineering (30%) 

Evaluates how effectively the participant transformed the raw lending dataset into a highquality instruction-tuning dataset. 

## Data Understanding & Preparation (30%) 

- Demonstrates a clear understanding of the lending dataset. 

- Identifies and appropriately addresses data quality issues. 

- Applies meaningful preprocessing and feature engineering where necessary. 

- Clearly documents assumptions and design decisions. 

Instruction Dataset Design (45%) 

- Generates high-quality prompt-completion pairs suitable for instruction tuning. 

- Covers multiple lending use cases where appropriate. 

- Produces consistent, well-structured training examples. 

- Demonstrates an understanding of how prompt design influences model behaviour. 

Dataset Design Decisions (25%) 

- Justifies preprocessing choices. 

- Explains prompt generation strategy. 

- Discusses trade-offs and limitations. 

- Demonstrates thoughtful engineering rather than mechanical data transformation. 

## 2. SLM Fine-Tuning Implementation (30%) 

Evaluates the participant's ability to effectively adapt a Small Language Model for the lending domain. 

Model Selection & Technical Justification (25%) 

- Selects an appropriate Small Language Model for the given constraints. 

- Justifies architectural and tooling choices. 

- Demonstrates awareness of hardware limitations and trade-offs. 

## Fine-Tuning Strategy (45%) 

- Configures and executes an effective parameter-efficient fine-tuning approach. 

- Demonstrates stable and reproducible training. 

- Makes appropriate hyperparameter choices and explains the rationale behind them. 

## Engineering Execution (30%) 

- Implements a well-organized and maintainable solution. 

- Demonstrates good experimentation practices. 

- Clearly communicates the implementation approach and key decisions. 

## 3. Evaluation & Business Impact (35%) 

Evaluates the quality of the evaluation methodology and the participant's ability to demonstrate measurable improvement. 

## Evaluation Methodology (30%) 

- Designs an evaluation strategy appropriate for the tasks being solved. 

- Selects meaningful evaluation techniques and justifies why they are suitable. 

- Demonstrates an understanding of the strengths and limitations of the chosen approach. 

## Demonstrated Improvement (40%) 

- Clearly compares the fine-tuned model with the baseline model. 

- Discusses both successful outcomes and observed limitations. 

Lending Domain Understanding (30%) 

- Produces technically sound lending recommendations. 

- Minimizes hallucinations and unsupported conclusions. 

- Explains business relevance and practical applicability of the solution. 

## 4. Innovation & Technical Depth (5%) 

Recognizes participants who go beyond the baseline implementation. 

Examples may include: 

- Creative prompt engineering strategies 

- Novel evaluation techniques 

- Improved data preparation approaches 

- Efficient fine-tuning workflows 

- Additional lending capabilities beyond the minimum requirements 

## Overall Judging Philosophy 

Participants are evaluated not only on the final model performance, but also on the quality of their engineering decisions, technical reasoning, experimental methodology, and ability to justify the choices made throughout the solution. 

## Solution Submission Guidelines 

Participants must submit their solution as a GitHub repository. 

## Repository Requirements 

The repository should contain all artifacts required to reproduce the solution, including: 

- Source code and notebooks 

- Training and evaluation scripts 

- Prompt generation pipeline 

- Configuration files 

- Documentation 

- Dependency definitions (for example, requirements.txt or equivalent) 

- Any additional assets required to run the solution 

Participants should organize the repository in a logical and maintainable structure that makes it easy for reviewers to understand and execute the solution. 

## Repository Access 

The completed solution must be pushed to the participant's personal GitHub account. 

Participants should grant access to the repository by adding the following GitHub user as a collaborator: 

**GitHub User:** azentio-talent-Aquisition 

Additionally, participants must share the repository URL with: 

**Email:** tateam@azentio.com 

## Submission Deadline 

The GitHub repository must be shared before the end of the hackathon. Only the latest commit available at the submission deadline will be considered for evaluation. 

## Important Notes 

- The submitted repository should be self-contained and reproducible. 

- Ensure that all required files are committed before submission. 

- Avoid committing unnecessary files such as virtual environments, temporary files, caches, or large datasets that can be regenerated. 

- If external models or datasets are required, provide clear instructions for downloading them. 

