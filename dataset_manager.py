import pandas as pd
datasets = {
        'client_contracts': pd.read_csv('./Company/Clients/Client_Contracts.csv'),
        'client_feedback': pd.read_csv('./Company/Clients/Client_Feedback.csv'),
        'client_list': pd.read_csv('./Company/Clients/Client_List.csv'),
        'employee_list': pd.read_csv('./Company/Employees/Employee_List.csv'),
        'employee_performance': pd.read_csv('./Company/Employees/Employee_Performance.csv'),
        'employee_training': pd.read_csv('./Company/Employees/Employee_Training.csv'),
        'expense_reports': pd.read_csv('./Company/Financial/Expense_Reports.csv'),
        'project_budgets': pd.read_csv('./Company/Financial/Project_Budgets.csv'),
        'revenue_reports': pd.read_csv('./Company/Financial/Revenue_Reports.csv'),
        'project_list': pd.read_csv('./Company/Projects/Project_List.csv'),
        'project_milestones': pd.read_csv('./Company/Projects/Project_Milestones.csv'),
        'project_team_members': pd.read_csv('./Company/Projects/Project_Team_Members.csv'),
    }
metadata = {
    "client_contracts": {
        "key_fields": ["Contract ID", "Client ID", "Project Name","Start Date","End Date","Contract Value (USD)","Status"],
        "common_analyses": [
            "Count of active contracts by client",
            "Contract value distribution",
            "Trend of contracts over time"
        ]
    },
    "client_feedback": {
        "key_fields": ["Client ID", "Project ID", "Feedback Score", "Comments", "Date"],
        "common_analyses": [
            "Average feedback score by client",
            "Correlation between feedback scores and project performance",
            "Distribution of feedback scores"
        ]
    },
    "client_list": {
        "key_fields": ["Client ID", "Client Name", "Industry" ,"Contact Person","Contact Email","Phone Number","Country","Start Date","End Date","Contract Status"],
        "common_analyses": [
            "Count of clients by industry",
            "Geographic distribution of clients",
            "Client tenure analysis (start/end dates)"
        ]
    },
    "employee_list": {
        "key_fields": ["Employee ID", "Name","Designation","Department","Joining Date","Salary (USD)","Status","Email","Phone"],
        "common_analyses": [
            "Headcount by department",
            "Employee attrition trends",
            "Average tenure and salary by department"
        ]
    },
    "employee_performance": {
        "key_fields": ["Employee ID", "Review Period", "Performance Score" ,"Manager Comments"],
        "common_analyses": [
            "Average performance score by department",
            "Trends in employee performance over time",
            "Correlation between training and performance"
        ]
    },
    "employee_training": {
        "key_fields": ["Employee ID", "Training Program", "Completion Date","Trainer Name","Duration (Hours)","Outcome"],
        "common_analyses": [
            "Most popular training programs",
            "Training completion rate by department",
            "Impact of training on employee performance"
        ]
    },
    "expense_reports": {
        "key_fields": ["Month", "Year", "Department", "Expense Type","Amount (USD)","Description"],
        "common_analyses": [
            "Departmental expense distribution",
            "Trends in monthly/annual expenses",
            "Comparison of actual vs. budgeted expenses"
        ]
    },
    "project_budgets": {
        "key_fields": ["Project ID", "Client ID", "Estimated Budget (USD)","Actual Cost (USD)","Profit/Loss (USD)"],
        "common_analyses": [
            "Variance between estimated and actual budgets",
            "Budget allocation by client",
            "Profitability analysis by project"
        ]
    },
    "revenue_reports": {
        "key_fields": ["Month", "Year","Total Revenue (USD)","Client Contributions","New Deals Closed"],
        "common_analyses": [
            "Monthly revenue trends",
            "Revenue contributions by client",
            "Year-over-year revenue growth"
        ]
    },
    "project_list": {
        "key_fields": ["Project ID", "Client ID","Project Name","Start Date","End Date","Project Manager","Status","Budget (USD)"],
        "common_analyses": [
            "Count of projects by client",
            "Status distribution of projects (Ongoing/Completed)",
            "Average project duration and budget"
        ]
    },
    "project_milestones": {
        "key_fields": ["Project ID", "Milestone Name", "Due Date", "Completion Status","Comments"],
        "common_analyses": [
            "Milestone completion rates",
            "Projects with overdue milestones",
            "Average time to milestone completion"
        ]
    },
    "project_team_members": {
        "key_fields": ["Project ID", "Employee ID", "Role", "Contribution (%)"],
        "common_analyses": [
            "Team size distribution across projects",
            "Roles distribution by project",
            "Employee contribution percentages across projects"
        ]
    }
}



relationships = {
    "client_contracts": {
        "related_to": {
            "client_list": {"shared_fields": ["Client ID"]},
            "project_list": {"shared_fields": ["Project Name"]},
            "project_budgets": {"shared_fields": ["Client ID"]},
            "client_feedback": {"shared_fields": ["Client ID"]}
        }
    },
    "client_feedback": {
        "related_to": {
            "client_list": {"shared_fields": ["Client ID"]},
            "project_list": {"shared_fields": ["Project ID"]},
            "employee_performance": {"shared_fields": ["Project ID"]}
        }
    },
    "client_list": {
        "related_to": {
            "client_contracts": {"shared_fields": ["Client ID"]},
            "client_feedback": {"shared_fields": ["Client ID"]},
            "project_list": {"shared_fields": ["Client ID"]},
            "project_budgets": {"shared_fields": ["Client ID"]}
        }
    },
    "employee_list": {
        "related_to": {
            "employee_performance": {"shared_fields": ["Employee ID"]},
            "employee_training": {"shared_fields": ["Employee ID"]},
            "project_team_members": {"shared_fields": ["Employee ID"]}
        }
    },
    "employee_performance": {
        "related_to": {
            "employee_list": {"shared_fields": ["Employee ID"]},
            "project_list": {"shared_fields": ["Project ID"]},
            "client_feedback": {"shared_fields": ["Project ID"]}
        }
    },
    "employee_training": {
        "related_to": {
            "employee_list": {"shared_fields": ["Employee ID"]}
        }
    },
    "expense_reports": {
        "related_to": {
            "project_budgets": {"shared_fields": ["Department"]},
            "revenue_reports": {"shared_fields": ["Month", "Year"]}
        }
    },
    "project_budgets": {
        "related_to": {
            "project_list": {"shared_fields": ["Project ID"]},
            "client_contracts": {"shared_fields": ["Client ID"]},
            "expense_reports": {"shared_fields": ["Department"]}
        }
    },
    "revenue_reports": {
        "related_to": {
            "expense_reports": {"shared_fields": ["Month", "Year"]},
            "project_budgets": {"shared_fields": ["Project ID"]}
        }
    },
    "project_list": {
        "related_to": {
            "client_contracts": {"shared_fields": ["Project Name"]},
            "project_budgets": {"shared_fields": ["Project ID"]},
            "project_milestones": {"shared_fields": ["Project ID"]},
            "project_team_members": {"shared_fields": ["Project ID"]}
        }
    },
    "project_milestones": {
        "related_to": {
            "project_list": {"shared_fields": ["Project ID"]}
        }
    },
    "project_team_members": {
        "related_to": {
            "project_list": {"shared_fields": ["Project ID"]},
            "employee_list": {"shared_fields": ["Employee ID"]}
        }
    }
}


class DatasetManager:
    """Manages dataset loading, metadata, and documentation for the RAG system"""
    
    def __init__(self):
        self.datasets = datasets
        self.metadata = metadata
        self.relationships = relationships

    def get_analysis_suggestions(self, question):
        """
        Suggest relevant datasets and fields based on a user question.
        
        Args:
            question (str): A natural language question.
        
        Returns:
            list[dict]: Relevant datasets, metadata, and matched fields.
        """
        relevant_datasets = []
        for dataset, meta in self.metadata.items():
            fields_match = any(field.lower() in question.lower() for field in meta['key_fields'])
            analysis_match = any(analysis.lower() in question.lower() for analysis in meta['common_analyses'])
            if fields_match or analysis_match:
                relevant_datasets.append({
                    'dataset': dataset,
                    'metadata': meta,
                    'relevant_fields': [field for field in meta['key_fields'] if field.lower() in question.lower()]
                })
        return relevant_datasets

    def get_relationships(self, dataset_name):
        """
        Retrieve relationships for a specific dataset.
        
        Args:
            dataset_name (str): The name of the dataset.
        
        Returns:
            dict: Datasets related to the specified dataset and shared fields.
        """
        return self.relationships.get(dataset_name, {}).get('related_to', {})

    def generate_code_context(self, question):
        """
        Generate a context for code generation based on a user question.
        
        Args:
            question (str): A natural language question.
        
        Returns:
            dict: Context containing relevant datasets, relationships, and schemas.
        """
        relevant_datasets = self.get_analysis_suggestions(question)
        relevant_dataset_names = [ds['dataset'] for ds in relevant_datasets]
        available_relationships = {
            dataset: self.get_relationships(dataset) for dataset in relevant_dataset_names
        }

        return {
            'relevant_datasets': relevant_datasets,
            'available_relationships': available_relationships,
            'dataset_schemas': {name: list(df.columns) for name, df in self.datasets.items()},
            'metadata': {name: self.metadata[name] for name in relevant_dataset_names}
        }

    def explore_dataset(self, dataset_name):
        """
        Provide an overview of a dataset, including its schema and sample data.
        
        Args:
            dataset_name (str): The name of the dataset to explore.
        
        Returns:
            dict: Schema and a sample of the dataset.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' does not exist.")
        dataset = self.datasets[dataset_name]
        return {
            'schema': list(dataset.columns),
            'sample_data': dataset.head(5).to_dict(orient='records')
        }

    def join_datasets(self, dataset1, dataset2):
        """
        Automatically join two datasets based on their relationships.

        Args:
            dataset1 (str): The name of the first dataset.
            dataset2 (str): The name of the second dataset.

        Returns:
            pd.DataFrame: The joined dataset, if a relationship exists.
        """
        relationships = self.get_relationships(dataset1)
        if dataset2 not in relationships:
            raise ValueError(f"No relationship exists between '{dataset1}' and '{dataset2}'.")
        
        shared_fields = relationships[dataset2]['shared_fields']
        df1 = self.datasets[dataset1]
        df2 = self.datasets[dataset2]
        
        return pd.merge(df1, df2, on=shared_fields, how='inner')

    def run_analysis(self, dataset_name, analysis_type):
        """
        Run a predefined analysis for a specific dataset.

        Args:
            dataset_name (str): The name of the dataset.
            analysis_type (str): The type of analysis to run.

        Returns:
            Any: The result of the analysis.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' does not exist.")
        if dataset_name not in self.metadata:
            raise ValueError(f"Metadata for '{dataset_name}' does not exist.")

        dataset = self.datasets[dataset_name]
        analyses = self.metadata[dataset_name]['common_analyses']
        
        if analysis_type not in analyses:
            raise ValueError(f"Analysis type '{analysis_type}' not found for dataset '{dataset_name}'.")
        
        # Example: Run predefined analysis (extend as needed)
        if analysis_type == "Count of active contracts by client":
            return dataset[dataset['Status'] == 'Active'].groupby('Client ID').size()
        
        # Add more analysis logic as required
        raise NotImplementedError(f"Analysis type '{analysis_type}' is not implemented.")
