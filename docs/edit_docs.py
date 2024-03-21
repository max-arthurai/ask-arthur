import os

bench_string_to_remove = """Contents
Menu
Expand
Light mode
Dark mode
Auto light/dark mode
Hide navigation sidebar
Hide table of contents sidebar
Toggle site navigation sidebar
bench
documentation
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
Bench Documentation Home
Setup
Quickstart
Scoring
GuidesToggle navigation of Guides
Concepts
Creating test suites
Compare LLM Providers
Compare Prompts
Compare Generation Settings
Add Scorer Configurations
Custom Scoring
Code Evaluation
Python API ReferenceToggle navigation of Python API Reference
arthur_bench.clientToggle navigation of arthur_bench.client
arthur_bench.client.auth
arthur_bench.client.http
arthur_bench.client.local
arthur_bench.client.restToggle navigation of arthur_bench.client.rest
arthur_bench.client.rest.admin
arthur_bench.client.rest.bench
arthur_bench.client.auth
arthur_bench.client.http
arthur_bench.client.local
arthur_bench.client.restToggle navigation of arthur_bench.client.rest
arthur_bench.client.rest.admin
arthur_bench.client.rest.bench
arthur_bench.client.rest.admin
arthur_bench.client.rest.bench
arthur_bench.exceptions
arthur_bench.models
arthur_bench.run
arthur_bench.scoring
arthur_bench.server
arthur_bench.telemetry
arthur_bench.utils
Contributing
Usage Data Collection
v: latest
Versions
latest
stable
Downloads
On Read the Docs
Project Home
Builds
Back to top
Edit this page
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
"""

blog_string_to_remove = """Solutions
EvaluationFirewallObservabilityProducts
The Most Robust Way to Evaluate LLMs
The First Firewall for LLMs
The Complete AI Performance Solution
Fast, Safe, Custom AI for BusinessModels
LLMNLPCVTabularResources
BlogGAPDocsCompany
R&DTeamCareersNews / PressSolutions
EvaluationFirewallObservabilityProducts
The Most Robust Way to Evaluate LLMs
The First Firewall for LLMs
The Complete AI Performance Solution
Fast, Safe, Custom AI for BusinessModels
LLMNLPCVTabularResources
BlogGAPDocsCompany
R&DTeamCareersNews / PressRequest a demoSign InSign InGet Started"""


scope_string_to_remove = """Jump to ContentProduct DocumentationAPI and Python SDK ReferenceRelease Notesv3.6.0v3.7.0v3.8.0v3.9.0v3.10.0v3.11.0v3.12.0Schedule a DemoSchedule a DemoMoon (Dark Mode)Sun (Light Mode)v3.12.0Product DocumentationAPI and Python SDK ReferenceRelease NotesSearchLoading…Welcome to ArthurWelcome to Arthur Scope!Pages in the Arthur Scope PlatformExamplesArthur SDKArthur APIModel TypesModel Input / Output TypesTabularBinary ClassificationMulticlass ClassificationRegressionTextBinary ClassificationMulticlass ClassificationRegressionToken Sequence (LLM)ImageBinary ClassificationMulticlass ClassificationRegressionObject DetectionRanked List (Recommender Systems)Time SeriesCore ConceptsMetricsPerformance MetricsData Drift MetricsFairness MetricsUser-Defined MetricsAlertingManaging AlertsAlert Summary ReportsEnrichmentsAnomaly DetectionBias MitigationExplainabilityHot SpotsToken LikelihoodVersioningModel OnboardingQuickstartCV OnboardingNLP OnboardingGenerative TextRanked List Outputs OnboardingTime Series OnboardingRegistering A Model with the APIData Preparation for ArthurCreating Arthur Model ObjectRegistering Model Attributes ManuallyEnabling EnrichmentsAssets Required For ExplainabilityTroubleshooting ExplainabilitySending InferencesSending Historical DataSending Ground TruthIntegrations and ExamplesAlerting ServicesEmailPagerDutyServiceNowData PipelinesSageMakerML PlatformsLangchainSingle Sign On (SSO)OIDCSAMLSpark MLArthur Query GuideOverviewCreating QueriesFundamentalsCommon Queries QuickstartQuerying FunctionsDefault Evaluation FunctionsAggregation FunctionsTransformation FunctionsComposing Advanced FunctionsEnrichments + Data DriftQuerying Data DriftQuerying ExplainabilityAdvanced Walk ThroughsGrouped Inference QueriesResourcesArthur Scope FAQGlossaryModel Metric DefinitionsArthur AlgorithmsPlatform AdministrationWelcome to Platform AdministrationInstallation OverviewOn-Prem Deployment RequirementsPlatform Readiness for Existing Cluster InstallsVirtual Machine InstallationConfiguring for High AvailabilityExternalizing the Relational DatabaseInstalling KubernetesInstalling Arthur Pre-requisitesDeploying on Amazon AWS EKSKubernetes Cluster (K8s) Install with Namespace Scope PrivilegesOnline Kubernetes Cluster (K8s) InstallAirgap Kubernetes Cluster (K8s) InstallAirgap Kubernetes Cluster (K8s) Install with CLIPlatform Access ControlAccess ControlDefault Access ControlCustom RBACIntegrationsOngoing Platform MaintenanceWhat does ongoing maintenance look like?Audit LogAdministrationOrganizations and UsersUpgradingMonitoring Best PracticesPlatform ResourcesConfig TemplateExporting Platform ConfigurationsFull Directory of Arthur PermissionsArthur Permissions by Standard RolesArthur Permissions by EndpointBackup and RestorePre-RequisitesBacking Up the Arthur PlatformRestoring the Arthur PlatformAppendixPowered by """

for file in os.listdir("docs/txt_files/"):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        print(filename)
        with open(f"docs/txt_files/{filename}", 'r') as f:
            s = f.read()
        s = s.replace(scope_string_to_remove, '')
        with open(f"docs/txt_files/{filename}", 'w') as f:
            f.write(s)
        print('done')

