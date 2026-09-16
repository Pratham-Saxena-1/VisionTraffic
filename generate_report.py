from src.report_generator import ReportGenerator

if __name__ == "__main__":
    import os
    config_path = 'config/config.yaml'
    metrics_path = 'outputs/metrics/vehicle_statistics.json'
    
    if not os.path.exists(metrics_path):
        print(f"[!] Error: {metrics_path} not found. Run the pipeline first.")
    else:
        gen = ReportGenerator(config_path, metrics_path)
        gen.generate()
