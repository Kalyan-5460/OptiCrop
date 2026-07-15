import unittest
from app import app

class TestOptiCrop(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_page(self):
        """Test that the homepage loads successfully and contains key hero texts."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('Precision Agriculture For Maximum Yields', html)
        self.assertIn('OptiCrop is an advanced machine learning-powered decision engine', html)
        print("[OK] Homepage verification passed successfully.")

    def test_about_page(self):
        """Test that the about page loads successfully and contains the project info and team members."""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('About OptiCrop Engine', html)
        # Check team members
        self.assertIn('Nehan Chowdary Muvva', html)
        self.assertIn('Jaswanth Malapareddy', html)
        self.assertIn('Madasu Rushi Venkata Ganesh', html)
        self.assertIn('Rajani Meesala', html)
        self.assertIn('Mounika Sonti', html)
        print("[OK] About page verification passed successfully (team members verified).")

    def test_find_your_crop_page(self):
        """Test that the prediction page loads successfully and contains form controls."""
        response = self.client.get('/findyourcrop')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('Crop Recommendation Engine', html)
        self.assertIn('name="nitrogen"', html)
        self.assertIn('name="phosphorous"', html)
        self.assertIn('name="potassium"', html)
        print("[OK] Prediction page structure verification passed successfully.")

    def test_prediction_post(self):
        """Test form submission with test inputs yields the correct recommendation (Coffee)."""
        test_data = {
            'nitrogen': 105,
            'phosphorous': 35,
            'potassium': 40,
            'temperature': 25.0,
            'humidity': 64.0,
            'ph': 7.0,
            'rainfall': 160.0
        }
        response = self.client.post('/predict', data=test_data)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        
        # Verify the custom formatted recommendation
        self.assertIn('Recommended Crop:', html)
        self.assertIn('Coffee', html)
        print("[OK] Crop recommendation model prediction test passed (Result: Coffee).")

if __name__ == '__main__':
    print("Starting OptiCrop automated app testing...")
    unittest.main()
