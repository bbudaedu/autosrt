import unittest
import logging
from unittest.mock import patch, MagicMock

# Mock the Google Colab and API related libraries as they are not available in the test environment
# and would raise ImportError or other issues if not mocked.
# Also, these modules interact with external services/filesystem which we want to avoid in unit tests.
mock_drive = MagicMock()
mock_auth = MagicMock()
mock_userdata = MagicMock()
mock_default_auth = MagicMock(return_value=(None, None))
mock_html = MagicMock()
mock_gspread = MagicMock()
mock_genai = MagicMock()
mock_pypdf = MagicMock()
mock_files = MagicMock()
mock_google = MagicMock()
mock_google_colab = MagicMock()
mock_google_auth = MagicMock()
mock_ipython = MagicMock()
mock_ipython_display = MagicMock()

# Assign attributes to the mocked parent modules
mock_google_colab.drive = mock_drive
mock_google_colab.auth = mock_auth
mock_google_colab.files = mock_files # Added files to google.colab mock
mock_google_colab.userdata = mock_userdata
mock_google_auth.default = mock_default_auth
mock_ipython_display.HTML = mock_html

modules_to_mock = {
    'google': mock_google,
    'google.colab': mock_google_colab,
    # Individual submodule mocks below are no longer strictly needed here if parent is correctly attributed
    # but keeping them for now to be safe during sys.modules patching.
    'google.colab.drive': mock_drive,
    'google.colab.auth': mock_auth,
    'google.colab.files': mock_files,
    'google.colab.userdata': mock_userdata,
    'google.auth': mock_google_auth,
    'google.auth.default': mock_default_auth,
    'IPython': mock_ipython,
    'IPython.display': mock_ipython_display,
    'IPython.display.HTML': mock_html,
    'gspread': mock_gspread,
    'google.generativeai': mock_genai,
    'pypdf': mock_pypdf
}

# Apply the mocks
patcher = patch.dict('sys.modules', modules_to_mock)
# patcher.start() will be called in setUpClass

# Module will be imported in setUp

class TestSheetsGeminiProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        patcher.start() # Start patching before any tests run or module is imported
        # Import the module here to make it available for patch targets if needed by decorators
        # This is a bit of a chicken-and-egg, but decorators are evaluated at class definition time.
        # Alternatively, use string paths for patching that are resolved at runtime.
        global sheets_gemini_processor
        import sheets_gemini_processor
        cls.module_under_test_for_patching = sheets_gemini_processor


    @classmethod
    def tearDownClass(cls):
        patcher.stop()

    def setUp(self):
        # Module is already imported in setUpClass for patch targeting,
        # but we can re-assign to self.module_under_test for instance access.
        self.module_under_test = self.module_under_test_for_patching

        # Create a logger for tests to avoid issues with unconfigured logger
        self.logger = logging.getLogger('TestLogger')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler()) # Add a null handler to prevent "No handlers could be found"

        # Reset mocks before each test if necessary
        mock_drive.reset_mock()
        mock_auth.reset_mock()
        mock_userdata.reset_mock()
        mock_default_auth.reset_mock()
        mock_html.reset_mock()
        mock_gspread.reset_mock()
        mock_genai.reset_mock()
        mock_pypdf.reset_mock()
        mock_files.reset_mock()


    # Patch targets should be strings referring to the module name as it will be in sys.modules
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('glob.glob')
    @patch('sheets_gemini_processor.extract_text_from_pdf_dir')
    # No need to patch 'drive' and 'files' here if sys.modules mocking is effective
    def test_initial_setup_runs(self, mock_extract_text, mock_glob, mock_makedirs, mock_os_path_exists):
        """Test that initial_setup can be called without errors (smoke test)."""
        mock_os_path_exists.return_value = True
        mock_glob.return_value = []
        mock_extract_text.return_value = "pdf text"
        mock_userdata.get.return_value = "fake_api_key" # Make sure mock_userdata is from the global scope

        # Ensure google.colab.files.upload is properly mocked
        # The mock_files object is already in modules_to_mock['google.colab.files']
        # and mock_google_colab.files is set to mock_files.
        # So, sheets_gemini_processor should see it.
        mock_files.upload.return_value = {}

        mock_gc_client = MagicMock()
        mock_gspread.authorize.return_value = mock_gc_client # mock_gspread is global

        try:
            gc_returned, pdf_text, instruction, rules = self.module_under_test.initial_setup(self.logger)
            self.assertIsNotNone(gc_returned)
            self.assertEqual(pdf_text, "pdf text")
            self.assertIn("# ROLE", instruction)
            self.assertIn("# RULES", rules)
            mock_files.upload.assert_called_once()
        except Exception as e:
            self.fail(f"initial_setup raised an exception: {e}")

    @patch('sheets_gemini_processor.load_gemini_processed_state')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="line1\nline2")
    @patch('sheets_gemini_processor.get_gemini_correction')
    @patch('sheets_gemini_processor.save_gemini_processed_state')
    @patch('sheets_gemini_processor.display', new_callable=MagicMock, create=True) # Mock 'display'
    # HTML mock is via sys.modules
    def test_process_transcriptions_and_apply_gemini_runs(self,
                                                         mock_display_func, # Added mock_display_func
                                                         mock_save_state,
                                                         mock_get_gemini_correction,
                                                         mock_open_file,
                                                         mock_isdir,
                                                         mock_listdir,
                                                         mock_os_path_exists,
                                                         mock_load_state):
        """Test that process_transcriptions_and_apply_gemini can be called (smoke test)."""
        mock_load_state.return_value = set()
        mock_os_path_exists.return_value = True
        mock_listdir.return_value = ['item1']
        mock_isdir.return_value = True

        mock_gc_client = MagicMock()
        mock_spreadsheet = MagicMock()
        mock_worksheet = MagicMock()
        mock_spreadsheet.url = "http://fake.url"

        self.module_under_test.gc = mock_gc_client
        mock_gc_client.open.return_value = mock_spreadsheet
        mock_gc_client.create.return_value = mock_spreadsheet
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_spreadsheet.add_worksheet.return_value = mock_worksheet

        self.module_under_test.execute_gspread_write = MagicMock(return_value=mock_worksheet)
        mock_get_gemini_correction.return_value = "corrected line1\ncorrected line2"
        self.module_under_test.pdf_context_text = "some pdf context"

        try:
            self.module_under_test.process_transcriptions_and_apply_gemini(self.logger, "instr", "rules")
            mock_load_state.assert_called_once()
            mock_listdir.assert_called_once()
            mock_open_file.assert_called()
            mock_get_gemini_correction.assert_called_once()
            mock_save_state.assert_called_once()

            # Assert that our global mock_html (representing IPython.display.HTML) was called
            mock_html.assert_called_once()
            # Assert that the mocked display function was called with the result of HTML(...)
            mock_display_func.assert_called_once_with(mock_html.return_value)

            self.assertTrue(self.module_under_test.execute_gspread_write.call_count >= 2)
        except Exception as e:
            self.fail(f"process_transcriptions_and_apply_gemini raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
