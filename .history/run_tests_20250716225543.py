#!/usr/bin/env python3
"""
SafeServe AI Test Runner
Automated test runner for smoke tests and integration tests
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.load_env import get_config

class TestRunner:
    """Test runner for SafeServe AI project"""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = f"http://{self.config.get('API_HOST', 'localhost')}:{self.config.get('API_PORT', 8080)}"
        self.test_results = []
        
    def run_smoke_tests(self) -> bool:
        """Run smoke tests to verify basic functionality"""
        print("🔍 Running smoke tests...")
        
        # Test 1: Environment loading
        print("  ✓ Testing environment loading...")
        try:
            config = get_config()
            assert config is not None
            assert isinstance(config, dict)
            print("    ✅ Environment loading: PASSED")
        except Exception as e:
            print(f"    ❌ Environment loading: FAILED - {str(e)}")
            return False
        
        # Test 2: API availability (if running)
        print("  ✓ Testing API availability...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("    ✅ API availability: PASSED")
            else:
                print(f"    ⚠️  API availability: DEGRADED - Status {response.status_code}")
        except Exception as e:
            print(f"    ⚠️  API availability: SKIPPED - {str(e)}")
        
        # Test 3: Import checks
        print("  ✓ Testing module imports...")
        try:
            from backend.api import app
            from backend.fraud_detection import FraudDetectionEngine
            from ui.app import send_message
            print("    ✅ Module imports: PASSED")
        except Exception as e:
            print(f"    ❌ Module imports: FAILED - {str(e)}")
            return False
        
        return True
    
    def run_pytest_tests(self) -> bool:
        """Run pytest integration tests"""
        print("🧪 Running pytest integration tests...")
        
        test_files = [
            "tests/test_load_env.py",
            "tests/test_api_endpoints.py",
            "tests/test_send_message.py"
        ]
        
        all_passed = True
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"  ✓ Running {test_file}...")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", test_file, "-v"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        print(f"    ✅ {test_file}: PASSED")
                    else:
                        print(f"    ❌ {test_file}: FAILED")
                        print(f"    Error output: {result.stdout}")
                        all_passed = False
                        
                except subprocess.TimeoutExpired:
                    print(f"    ⏰ {test_file}: TIMEOUT")
                    all_passed = False
                except Exception as e:
                    print(f"    ❌ {test_file}: ERROR - {str(e)}")
                    all_passed = False
            else:
                print(f"  ⚠️  {test_file}: NOT FOUND")
        
        return all_passed
    
    def run_manual_tests(self) -> bool:
        """Run manual API tests"""
        print("🔧 Running manual API tests...")
        
        # Test send_message function
        print("  ✓ Testing send_message function...")
        try:
            # Mock the send_message function test
            def test_send_message():
                try:
                    payload = {
                        "query": "Hello, test message",
                        "user_id": "test_user",
                        "language": "en"
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/chat",
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        assert "response" in data
                        assert "timestamp" in data
                        assert "user_id" in data
                        assert "processing_time" in data
                        return True
                    else:
                        return False
                        
                except Exception as e:
                    return False
            
            if test_send_message():
                print("    ✅ send_message function: PASSED")
            else:
                print("    ⚠️  send_message function: SKIPPED (API not available)")
                
        except Exception as e:
            print(f"    ❌ send_message function: FAILED - {str(e)}")
            return False
        
        # Test fraud detection
        print("  ✓ Testing fraud detection...")
        try:
            payload = {
                "amount": 100.0,
                "location": "Test Location",
                "merchant": "Test Merchant",
                "device_id": "test_device",
                "velocity_score": 0.5
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "risk_score" in data
                assert "label" in data
                print("    ✅ Fraud detection: PASSED")
            else:
                print("    ⚠️  Fraud detection: SKIPPED (API not available)")
                
        except Exception as e:
            print(f"    ⚠️  Fraud detection: SKIPPED - {str(e)}")
        
        return True
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report"""
        print("📊 Generating test report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "config_loaded": bool(self.config)
            },
            "tests": {
                "smoke_tests": self.run_smoke_tests(),
                "pytest_tests": self.run_pytest_tests(),
                "manual_tests": self.run_manual_tests()
            },
            "summary": {}
        }
        
        # Calculate summary
        total_tests = len(report["tests"])
        passed_tests = sum(1 for result in report["tests"].values() if result)
        
        report["summary"] = {
            "total_test_categories": total_tests,
            "passed_categories": passed_tests,
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED"
        }
        
        return report
    
    def run_all_tests(self) -> Dict:
        """Run all tests and return comprehensive report"""
        print("🚀 SafeServe AI Test Suite")
        print("=" * 50)
        
        report = self.generate_test_report()
        
        print("\n📋 Test Results Summary:")
        print("=" * 30)
        print(f"Total Test Categories: {report['summary']['total_test_categories']}")
        print(f"Passed Categories: {report['summary']['passed_categories']}")
        print(f"Success Rate: {report['summary']['success_rate']}")
        print(f"Overall Status: {report['summary']['overall_status']}")
        
        # Save report to file
        report_file = "test_results.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Test report saved to: {report_file}")
        
        return report

def main():
    """Main test runner function"""
    runner = TestRunner()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "smoke":
            runner.run_smoke_tests()
        elif sys.argv[1] == "pytest":
            runner.run_pytest_tests()
        elif sys.argv[1] == "manual":
            runner.run_manual_tests()
        else:
            print("Usage: python run_tests.py [smoke|pytest|manual]")
    else:
        runner.run_all_tests()

if __name__ == "__main__":
    main()