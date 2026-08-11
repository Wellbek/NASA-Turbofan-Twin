"""Dashboard smoke tests.

Every page gets rendered headlessly through Streamlit's AppTest and checked for
exceptions. This exists because two dashboard bugs shipped that no unit test
could have caught: a page whose navigation entry did not match its branch
condition, so it was unreachable for months, and a `Styler.applymap` call that
pandas 3 removed.

Skipped when the artifacts are absent, which is the case in CI.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = REPO_ROOT / 'webapp' / 'dashboard.py'
PIPELINE = REPO_ROOT / 'data' / 'models' / 'cmapss' / 'feature_pipeline.joblib'

PAGES = ["Overview", "New Prediction", "Engine Analysis", "Model Comparison",
         "Fleet Management", "Performance Metrics", "Workflow"]

pytestmark = [
    pytest.mark.skipif(not PIPELINE.exists(),
                       reason='no trained artifacts; run `python -m src.train`'),
    pytest.mark.slow,
]


@pytest.fixture(scope='module')
def app_factory():
    streamlit_testing = pytest.importorskip('streamlit.testing.v1')

    def build():
        app = streamlit_testing.AppTest.from_file(str(DASHBOARD), default_timeout=300)
        app.run()
        return app

    return build


def test_app_starts_without_error(app_factory):
    """Catches the whole app failing to execute, e.g. a truncated file."""
    app = app_factory()
    assert not app.exception, app.exception[0].message if app.exception else ''
    assert app.selectbox, 'navigation selectbox never rendered'


def test_navigation_offers_every_page(app_factory):
    app = app_factory()
    assert list(app.selectbox[0].options) == PAGES


@pytest.mark.parametrize('page', PAGES)
def test_page_renders(app_factory, page):
    """Each page must render without raising.

    The Workflow page is the reason this is parametrized over all of them: its
    branch checked for a string the navigation list never produced, so it was
    dead code that no amount of loading the app would have exposed.
    """
    app = app_factory()
    app.selectbox[0].set_value(page).run()

    assert not app.exception, (
        f'{page} raised: '
        f'{app.exception[0].message if app.exception else ""}')

    errors = [e.value for e in app.error]
    assert not errors, f'{page} rendered st.error: {errors}'
