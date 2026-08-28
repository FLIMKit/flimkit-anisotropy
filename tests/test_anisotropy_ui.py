from queue import Queue
from unittest.mock import patch

import pytest

from flimkit.UI.gui import _UIBuilder


def _tk_root_or_skip():
    import tkinter as tk

    root = None
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display is not available')
    assert root is not None
    root.withdraw()
    return root


def test_plugin_registers_a_tools_menu_entry():
    from flimkit import plugins
    import flimkit_anisotropy
    found = plugins.get_tool('anisotropy')
    assert found is not None
    assert found.label == 'Time-Resolved Anisotropy...'
    assert found.menu_path == ('Tools',)


def test_plugin_entry_opens_the_tool_with_the_app_root():
    from flimkit import plugins
    import flimkit_anisotropy
    builder = _UIBuilder.__new__(_UIBuilder)
    builder.root = object()

    with patch('flimkit_anisotropy.tool.show_anisotropy_tool') as show:
        plugins.get_tool('anisotropy').callback(builder)

    show.assert_called_once_with(builder.root)


def test_anisotropy_dialog_exposes_explicit_file_roles():
    import tkinter as tk
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        labels = []
        pending = list(dialog.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            try:
                labels.append(widget.cget('text'))
            except tk.TclError:
                pass
        assert 'Parallel PTU' in labels
        assert 'Perpendicular PTU' in labels
        assert 'Parallel exposure (relative)' in labels
        assert 'Perpendicular exposure (relative)' in labels
        assert 'Shared scale value' in labels
        assert 'Late-window start (ns)' in labels
        assert 'Shared scale source' in labels
        assert tuple(dialog.scale_source_combo['values']) == (
            'Assumed scale', 'Calibrated G', 'Effective late-window scale')
        assert 'Parallel photon channel' in labels
        assert 'Perpendicular photon channel' in labels
        assert 'Calculate' in labels
        assert 'Scale diagnostic...' in labels
        assert any('G=1 is an assumption' in label for label in labels)
    finally:
        root.destroy()


def test_anisotropy_settings_record_shared_scale_source(tmp_path):
    from flimkit_anisotropy.tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.g_mode.set('late_window')
        dialog.late_window_start_ns.set(9.6)

        settings = dialog._settings()

        assert settings['g_mode'] == 'late_window'
        assert settings['late_window_start_ns'] == 9.6
        assert settings['g_factor'] == 1.0
    finally:
        root.destroy()


def test_anisotropy_dialog_can_collapse_and_restore_inputs():
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        root.update_idletasks()

        assert dialog.input_panel.winfo_manager() == 'pack'
        assert dialog.toggle_inputs_button.cget('text') == 'Hide inputs'

        dialog._toggle_inputs()
        root.update_idletasks()

        assert dialog.input_panel.winfo_manager() == ''
        assert dialog.toggle_inputs_button.cget('text') == 'Show inputs'

        dialog._toggle_inputs()
        root.update_idletasks()

        assert dialog.input_panel.winfo_manager() == 'pack'
        assert dialog.toggle_inputs_button.cget('text') == 'Hide inputs'
    finally:
        root.destroy()


def test_successful_analysis_collapses_inputs_to_show_results():
    from types import SimpleNamespace
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        result = SimpleNamespace(
            perpendicular_shift=(0.0, 0.0), late_window_stability=object(),
            g_factor=2.6,
            metadata={'shared_scale_source': 'late_window'})

        with patch.object(dialog, '_draw_result'):
            dialog._analysis_finished(result, 0)
        root.update_idletasks()

        assert dialog.input_panel.winfo_manager() == ''
        assert dialog.toggle_inputs_button.cget('text') == 'Show inputs'
        assert 'disabled' not in dialog.calculate_button.state()
        assert 'disabled' not in dialog.scale_diagnostic_button.state()
        assert 'disabled' not in dialog.save_npz_button.state()
        assert 'disabled' not in dialog.save_csv_button.state()
        assert 'not calibrated physical G' in dialog.status.get()
    finally:
        root.destroy()


def test_success_status_warns_for_uncalibrated_g_one():
    from types import SimpleNamespace
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        result = SimpleNamespace(
            perpendicular_shift=(0.0, 0.0),
            g_factor=1.0,
            metadata={'shared_scale_source': 'assumed'})

        with patch.object(dialog, '_draw_result'):
            dialog._analysis_finished(result, 0)

        assert 'WARNING' in dialog.status.get()
        assert 'G=1 is assumed, not calibrated' in dialog.status.get()
        assert 'disabled' in dialog.scale_diagnostic_button.state()
    finally:
        root.destroy()


def test_success_status_marks_unresolved_advanced_fit():
    from types import SimpleNamespace
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        selected = SimpleNamespace(
            identifiable=False,
            identifiability_warnings=('component time reached a bound',))
        result = SimpleNamespace(
            perpendicular_shift=(0.0, 0.0), g_factor=1.0,
            late_window_stability=None,
            multicomponent_fit=SimpleNamespace(
                selected_component_count=2, selected_fit=selected),
            metadata={'shared_scale_source': 'assumed'})

        with patch.object(dialog, '_draw_result'):
            dialog._analysis_finished(result, 0)

        assert '2-component model' in dialog.status.get()
        assert 'NOT RESOLVED' in dialog.status.get()
        assert 'disabled' not in dialog.fit_details_button.state()
    finally:
        root.destroy()


def test_all_failed_advanced_candidates_show_fail_closed_result():
    from types import SimpleNamespace
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        comparison = SimpleNamespace(
            selected_component_count=0, selected_fit=None, candidates=())
        result = SimpleNamespace(
            multicomponent_fit=comparison, polarized_fit=None,
            late_window_stability=None, perpendicular_shift=(0.0, 0.0),
            metadata={'shared_scale_source': 'assumed'}, g_factor=1.0)

        dialog._analysis_finished(result, 0)

        assert 'No optimizer candidate converged' in dialog.status.get()
        assert 'NOT RESOLVED' in dialog.status.get()
        assert 'disabled' in dialog.fit_details_button.state()
        assert 'No optimizer candidate converged' in (
            dialog.axes[0, 0].texts[0].get_text())
    finally:
        root.destroy()


def test_replacement_run_clears_stale_result_actions_on_start_and_failure():
    from unittest.mock import MagicMock, patch
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    dialog = None
    try:
        dialog = show_anisotropy_tool(root)
        dialog.result = object()
        for button in (
                dialog.scale_diagnostic_button, dialog.fit_details_button,
                dialog.save_npz_button, dialog.save_csv_button):
            button.configure(state='normal')
        thread = MagicMock()
        with (patch.object(dialog, '_settings', return_value={}),
              patch('flimkit_anisotropy.tool.threading.Thread',
                    return_value=thread),
              patch.object(dialog, 'after', return_value='poll-id')):
            dialog._start_analysis()

        assert dialog.result is None
        assert 'disabled' in dialog.calculate_button.state()
        for button in (
                dialog.scale_diagnostic_button, dialog.fit_details_button,
                dialog.save_npz_button, dialog.save_csv_button):
            assert 'disabled' in button.state()
        thread.start.assert_called_once_with()

        with patch('flimkit_anisotropy.tool.messagebox.showerror'):
            dialog._analysis_failed(RuntimeError('replacement failed'))
        assert dialog.result is None
        assert 'disabled' not in dialog.calculate_button.state()
        for button in (
                dialog.scale_diagnostic_button, dialog.fit_details_button,
                dialog.save_npz_button, dialog.save_csv_button):
            assert 'disabled' in button.state()
    finally:
        if dialog is not None:
            dialog._poll_after_id = None
        root.destroy()


def test_load_irf_curve_reads_ptu_histogram_and_pads_trailing_bins(tmp_path):
    import numpy as np
    from flimkit_anisotropy.tool import load_irf_curve

    path = tmp_path / 'measured_irf.ptu'
    path.touch()
    calls = {}

    class FakePTUFile:
        def __init__(self, source, verbose=False):
            calls['source'] = source
            calls['verbose'] = verbose
            self.tcspc_res = 0.1e-9
            self.period_ns = 0.4

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            calls['closed'] = True

        def summed_decay(self, channel=None):
            calls['channel'] = channel
            return np.array([1.0, 4.0, 2.0])

    curve = load_irf_curve(
        path, n_bins=4, tcspc_res=0.1e-9,
        expected_period_ns=0.4, ptu_channel=2,
        ptu_reader_class=FakePTUFile)

    np.testing.assert_allclose(curve, [1.0, 4.0, 2.0, 0.0])
    assert calls == {
        'source': path,
        'verbose': False,
        'channel': 2,
        'closed': True,
    }


def test_load_irf_curve_crops_trailing_ptu_period_bin(tmp_path):
    import numpy as np
    from flimkit_anisotropy.tool import load_irf_curve

    path = tmp_path / 'measured_irf.ptu'
    path.touch()

    class FakePTUFile:
        tcspc_res = 0.1e-9
        period_ns = 13.2

        def __init__(self, source, verbose=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def summed_decay(self, channel=None):
            return np.arange(133, dtype=float) + 1.0

    curve = load_irf_curve(
        path, n_bins=132, tcspc_res=0.1e-9,
        expected_period_ns=13.2, ptu_reader_class=FakePTUFile)

    np.testing.assert_allclose(curve, np.arange(132, dtype=float) + 1.0)


def test_load_irf_curve_preserves_supported_export_path(tmp_path):
    from unittest.mock import patch
    from flimkit_anisotropy.tool import load_irf_curve

    path = tmp_path / 'measured_irf.csv'
    path.touch()
    exported = object()
    converted = object()

    with (patch('flimkit.utils.xlsx_tools.load_irf_export',
                return_value=exported) as load_export,
          patch('flimkit.FLIM.irf_tools.irf_from_xlsx',
                return_value=converted) as convert):
        result = load_irf_curve(path, n_bins=132, tcspc_res=0.1e-9)

    assert result is converted
    load_export.assert_called_once_with(path, debug=False)
    convert.assert_called_once_with(exported, 132, 0.1e-9)


def test_load_irf_curve_validates_discarded_trailing_ptu_bin(tmp_path):
    import numpy as np
    from flimkit_anisotropy.tool import load_irf_curve

    path = tmp_path / 'invalid_trailing_bin.ptu'
    path.touch()

    class FakePTUFile:
        tcspc_res = 0.1e-9
        period_ns = 0.3

        def __init__(self, source, verbose=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def summed_decay(self, channel=None):
            return np.array([1.0, 4.0, 2.0, float('nan')])

    with pytest.raises(ValueError, match='finite, non-negative, and non-zero'):
        load_irf_curve(
            path, n_bins=3, tcspc_res=0.1e-9,
            expected_period_ns=0.3, ptu_reader_class=FakePTUFile)


@pytest.mark.parametrize('decay_size', [2, 6])
def test_load_irf_curve_rejects_large_ptu_bin_count_mismatch(
        tmp_path, decay_size):
    import numpy as np
    from flimkit_anisotropy.tool import load_irf_curve

    path = tmp_path / 'wrong_bin_count.ptu'
    path.touch()

    class FakePTUFile:
        tcspc_res = 0.1e-9
        period_ns = 0.4

        def __init__(self, source, verbose=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def summed_decay(self, channel=None):
            return np.ones(decay_size)

    with pytest.raises(ValueError, match='differs by more than one bin'):
        load_irf_curve(
            path, n_bins=4, tcspc_res=0.1e-9,
            expected_period_ns=0.4, ptu_reader_class=FakePTUFile)


def test_load_irf_curve_rejects_ptu_timing_resolution_mismatch(tmp_path):
    import numpy as np
    from flimkit_anisotropy.tool import load_irf_curve

    path = tmp_path / 'wrong_timing.ptu'
    path.touch()

    class FakePTUFile:
        def __init__(self, source, verbose=False):
            self.tcspc_res = 0.2e-9
            self.period_ns = 0.4

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def summed_decay(self, channel=None):
            return np.ones(4)

    with pytest.raises(ValueError, match='timing resolution'):
        load_irf_curve(
            path, n_bins=4, tcspc_res=0.1e-9,
            expected_period_ns=0.4, ptu_reader_class=FakePTUFile)


def test_load_irf_curve_rejects_ptu_laser_period_mismatch(tmp_path):
    import numpy as np
    from flimkit_anisotropy.tool import load_irf_curve

    path = tmp_path / 'wrong_period.ptu'
    path.touch()

    class FakePTUFile:
        def __init__(self, source, verbose=False):
            self.tcspc_res = 0.1e-9
            self.period_ns = 0.8

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def summed_decay(self, channel=None):
            return np.ones(8)

    with pytest.raises(ValueError, match='laser period'):
        load_irf_curve(
            path, n_bins=4, tcspc_res=0.1e-9,
            expected_period_ns=0.4, ptu_reader_class=FakePTUFile)


@pytest.mark.parametrize('decay', [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, 0.0],
    [1.0, float('nan'), 0.0, 0.0],
])
def test_load_irf_curve_rejects_invalid_ptu_histogram(tmp_path, decay):
    import numpy as np
    from flimkit_anisotropy.tool import load_irf_curve

    path = tmp_path / 'invalid_irf.ptu'
    path.touch()

    class FakePTUFile:
        def __init__(self, source, verbose=False):
            self.tcspc_res = 0.1e-9
            self.period_ns = 0.4

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def summed_decay(self, channel=None):
            return np.asarray(decay)

    with pytest.raises(ValueError, match='finite, non-negative, and non-zero'):
        load_irf_curve(
            path, n_bins=4, tcspc_res=0.1e-9,
            expected_period_ns=0.4, ptu_reader_class=FakePTUFile)


def test_scale_diagnostic_plot_shows_nested_ratio_and_caveat():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    diagnostic = None
    try:
        dialog = show_anisotropy_tool(root)
        stability = SimpleNamespace(
            time_ns=np.array([8.0, 9.0, 10.0, 11.0]),
            rolling_scale=np.array([3.7, 3.6, 3.5, np.nan]),
            nested_scale=np.array([3.65, 3.58, 3.50, 3.45]),
            nested_standard_error=np.array([0.08, 0.09, 0.12, 0.20]),
            selected_start_bin=2,
            selected_scale=3.50,
            selected_parallel_photons=700.0,
            selected_perpendicular_photons=200.0,
            rolling_bins=3)
        dialog.result = SimpleNamespace(
            late_window_stability=stability,
            g_factor=3.50,
            metadata={
                'shared_scale_source': 'late_window',
                'shared_scale_interpretation': 'effective_late_window_scale',
            })

        diagnostic = dialog._show_scale_diagnostic()
        root.update_idletasks()

        axis = diagnostic.scale_axes[0]
        assert axis.get_title() == 'Late-window stability'
        labels = [line.get_label() for line in axis.lines]
        assert '3-bin rolling ratio' in labels
        assert 'Nested ratio to period end' in labels
        assert 'Approx. 95% Poisson band' in (
            collection.get_label() for collection in axis.collections)
        summary = diagnostic.scale_summary.cget('text')
        assert 'Effective scale: 3.5' in summary
        assert '700 parallel, 200 perpendicular photons' in summary
        assert 'not a calibrated physical G' in summary
        assert 'approximate delta-method Poisson 95% interval' in summary
        assert 'unreliable at low counts' in summary
        assert 'Nested windows share photons' in summary
    finally:
        if diagnostic is not None:
            diagnostic.destroy()
        root.destroy()


def test_anisotropy_irf_browser_lists_ptu_and_supported_exports():
    from flimkit_anisotropy.tool import AnisotropyTool

    tool = AnisotropyTool.__new__(AnisotropyTool)
    variable = type('Variable', (), {'set': lambda self, value: setattr(self, 'value', value)})()
    with patch(
            'flimkit_anisotropy.tool.filedialog.askopenfilename',
            return_value='/tmp/parallel.csv') as browse:
        tool._browse(variable, file_kind='irf')

    options = browse.call_args.kwargs
    assert options['title'] == 'Select measured IRF'
    patterns = options['filetypes'][0][1]
    for extension in (
            '*.ptu', '*.xlsx', '*.csv', '*.tsv', '*.txt', '*.dat',
            '*.ascii', '*.asc'):
        assert extension in patterns
    assert options['filetypes'][-1] == ('All files', '*.*')
    assert variable.value == '/tmp/parallel.csv'


def test_anisotropy_dialog_offers_direct_and_preferred_modes():
    import tkinter as tk
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        labels = []
        pending = list(dialog.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            try:
                labels.append(widget.cget('text'))
            except tk.TclError:
                pass
        assert 'Direct r(t) diagnostic (no IRF)' in labels
        assert 'Preferred global fit (Lakowicz Section 11.2.2)' in labels
        assert 'Parallel IRF (PTU or export)' in labels
        assert 'Perpendicular IRF (PTU or export)' in labels
        assert 'Method info...' in labels
    finally:
        root.destroy()


def test_advanced_multicomponent_controls_default_safely_and_hint_at_scale_estimator():
    import tkinter as tk
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        labels = []
        stack = [dialog]
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            try:
                labels.append(widget.cget('text'))
            except tk.TclError:
                pass

        assert dialog.g_factor.get() == 1.0
        assert dialog.max_components.get() == 1
        assert dialog.component_range_mode.get() == 'Auto'
        assert 'Advanced 1/2/3-component fit' in labels
        hint = next(text for text in labels if 'better G' in str(text))
        assert 'late-window' in hint
        assert 'not calibrated' in hint
    finally:
        root.destroy()


def test_automatic_component_ranges_disable_manual_range_fields():
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.analysis_mode.set('advanced')
        root.update_idletasks()
        assert all('disabled' in entry.state()
                   for entry in dialog.component_bound_entries)

        dialog.component_range_mode.set('Manual')
        root.update_idletasks()
        assert all('disabled' not in entry.state()
                   for entry in dialog.component_bound_entries)
    finally:
        root.destroy()


def test_advanced_multicomponent_settings_include_user_g_and_model_controls(tmp_path):
    from flimkit_anisotropy.tool import show_anisotropy_tool

    paths = [tmp_path / name for name in (
        'parallel.ptu', 'perpendicular.ptu', 'parallel_irf.ptu',
        'perpendicular_irf.ptu')]
    for path in paths:
        path.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(paths[0])
        dialog.perpendicular_path.set(paths[1])
        dialog.parallel_irf_path.set(paths[2])
        dialog.perpendicular_irf_path.set(paths[3])
        dialog.analysis_mode.set('advanced')
        dialog.max_components.set(3)
        dialog.multistart.set(7)
        dialog.g_factor.set(2.75)

        settings = dialog._settings()

        assert settings['analysis_mode'] == 'advanced'
        assert settings['max_components'] == 3
        assert settings['multistart'] == 7
        assert settings['g_factor'] == 2.75
        assert settings['component_bounds_ns'] is None

        dialog.multistart.set(33)
        with pytest.raises(ValueError, match='at most 32'):
            dialog._settings()
    finally:
        root.destroy()


def test_advanced_manual_shared_range_is_positive_and_increasing(tmp_path):
    from flimkit_anisotropy.tool import show_anisotropy_tool

    paths = [tmp_path / name for name in (
        'parallel.ptu', 'perpendicular.ptu', 'parallel_irf.ptu',
        'perpendicular_irf.ptu')]
    for path in paths:
        path.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(paths[0])
        dialog.perpendicular_path.set(paths[1])
        dialog.parallel_irf_path.set(paths[2])
        dialog.perpendicular_irf_path.set(paths[3])
        dialog.analysis_mode.set('advanced')
        dialog.max_components.set(3)
        dialog.component_range_mode.set('Manual')
        dialog.component_lower_ns[0].set(0.2)
        dialog.component_upper_ns[0].set(6.0)

        assert dialog._settings()['component_bounds_ns'] == (0.2, 6.0)

        dialog.component_lower_ns[0].set(7.0)
        with pytest.raises(ValueError, match='positive increasing'):
            dialog._settings()
    finally:
        root.destroy()


def test_method_info_states_global_fit_requirements():
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        with patch('flimkit_anisotropy.tool.messagebox.showinfo') as showinfo:
            dialog._show_method_info()
        message = showinfo.call_args.args[1]
        assert 'known fluorescence lifetime' in message
        assert 'separate IRFs' in message
        assert 'PTU' in message
        assert 'separate fitted backgrounds' in message
        assert 'time-zero model parameter' in message
        assert 'assumed, calibrated, or effective late-window scale' in message
        assert 'not a calibrated physical G' in message
        assert 'Advanced 1/2/3-component fit' in message
        assert 'ordered correlation times' in message
        assert 'BIC' in message
        assert 'No advanced candidate is labelled' in message
    finally:
        root.destroy()


def test_preferred_mode_requires_two_irf_files(tmp_path):
    from flimkit_anisotropy.tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.analysis_mode.set('global')

        with pytest.raises(ValueError, match='IRF'):
            dialog._settings()
    finally:
        root.destroy()


def test_preferred_mode_requires_positive_known_lifetime(tmp_path):
    from flimkit_anisotropy.tool import show_anisotropy_tool

    paths = [tmp_path / name for name in (
        'parallel.ptu', 'perpendicular.ptu', 'parallel.csv', 'perpendicular.csv')]
    for path in paths:
        path.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(paths[0])
        dialog.perpendicular_path.set(paths[1])
        dialog.parallel_irf_path.set(paths[2])
        dialog.perpendicular_irf_path.set(paths[3])
        dialog.analysis_mode.set('global')
        dialog.fixed_lifetime_ns.set(0.0)

        with pytest.raises(ValueError, match='lifetime'):
            dialog._settings()
    finally:
        root.destroy()


def test_anisotropy_dialog_rejects_nonfinite_exposure(tmp_path):
    import tkinter as tk
    from flimkit_anisotropy.tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.parallel_exposure.set(float('nan'))

        with pytest.raises(ValueError, match='Exposure values'):
            dialog._settings()
    finally:
        root.destroy()


def test_anisotropy_dialog_rejects_nonfinite_photon_threshold(tmp_path):
    import tkinter as tk
    from flimkit_anisotropy.tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.min_map_photons.set(float('nan'))

        with pytest.raises(ValueError, match='Photon thresholds'):
            dialog._settings()
    finally:
        root.destroy()


def test_anisotropy_dialog_rejects_nonfinite_analysis_time(tmp_path):
    import tkinter as tk
    from flimkit_anisotropy.tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.analysis_stop_ns.set(float('nan'))

        with pytest.raises(ValueError, match='Post-peak times'):
            dialog._settings()
    finally:
        root.destroy()


def test_anisotropy_dialog_rejects_negative_photon_channel(tmp_path):
    import tkinter as tk
    from flimkit_anisotropy.tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.parallel_channel.set(-1)

        with pytest.raises(ValueError, match='Photon channels'):
            dialog._settings()
    finally:
        root.destroy()


def test_worker_returns_result_through_queue_without_touching_tk():
    from flimkit_anisotropy.tool import AnisotropyTool

    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool._result_queue = Queue()
    expected = (object(), 7)

    with patch('flimkit_anisotropy.tool.run_analysis', return_value=expected):
        tool._analysis_worker({})

    assert tool._result_queue.get_nowait() == ('success', *expected)


def test_closed_dialog_does_not_reschedule_worker_poll():
    from flimkit_anisotropy.tool import AnisotropyTool

    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool._closed = True
    tool._result_queue = Queue()
    scheduled = []
    tool.after = lambda *args: scheduled.append(args)

    tool._poll_worker()

    assert scheduled == []


def test_redrawing_result_replaces_existing_colorbar():

    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.result = SimpleNamespace(
            parallel_intensity=np.ones((3, 3)),
            perpendicular_intensity=np.ones((3, 3)),
            time_ns=np.arange(4, dtype=float),
            anisotropy_decay=np.zeros(4),
            anisotropy_map=np.array([
                [-0.35, -0.25, -0.20],
                [-0.10, 0.00, 0.10],
                [-0.30, -0.20, 0.05],
            ]),
            spatial_window=1,
            stride=1,
            metadata={'analysis_start_ns': 0.0, 'analysis_stop_ns': 2.0},
        )
        dialog.peak_bin = 1

        dialog._draw_result()
        axes_after_first_draw = len(dialog.figure.axes)
        color_limits = dialog.axes[1, 1].images[0].get_clim()
        plotted_time = dialog.axes[1, 0].lines[0].get_xdata()
        dialog._draw_result()

        np.testing.assert_array_equal(plotted_time, [0.0, 1.0])
        assert color_limits[0] <= -0.34
        assert color_limits[1] >= 0.4
        assert dialog._colorbar.ax.yaxis.get_ticks_position() == 'left'
        assert axes_after_first_draw == 5
        assert len(dialog.figure.axes) == axes_after_first_draw
    finally:
        root.destroy()


def test_global_fit_mode_draws_polarized_models_and_residuals():
    from types import SimpleNamespace
    import matplotlib as mpl
    import numpy as np
    import warnings
    from flimkit_anisotropy.tool import show_anisotropy_tool

    rc_overrides = {
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'axes.titlecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'font.size': 18.0,
        'axes.titlesize': 18.0,
        'axes.labelsize': 16.0,
        'xtick.labelsize': 14.0,
        'ytick.labelsize': 14.0,
    }
    original_colors = {key: mpl.rcParams[key] for key in rc_overrides}
    mpl.rcParams.update(rc_overrides)
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.geometry('1180x820')
        dialog.update_idletasks()
        fit = SimpleNamespace(
            parallel_model=np.array([11.0, 8.0, 5.0]),
            perpendicular_model=np.array([7.0, 6.0, 4.0]),
            parallel_residual=np.array([0.0, 1.0, -1.0]),
            perpendicular_residual=np.array([1.0, 0.0, -1.0]),
            intensity_lifetime_ns=3.2,
            rotational_correlation_ns=1.4,
            initial_anisotropy=0.32,
            poisson_deviance=5.0,
            success=False,
            message='maximum evaluations reached',
            parameters_at_bounds=('initial_anisotropy',),
            common_irf_shift_bins=0.7,
            parallel_background=2.5,
            perpendicular_background=7.0,
        )
        dialog.result = SimpleNamespace(
            parallel_decay=np.array([9.0, 7.0, 3.0, 1.0]),
            perpendicular_decay=np.array([5.0, 4.0, 2.0, 1.0]),
            parallel_background=2.0,
            perpendicular_background=2.0,
            polarized_fit=fit,
            time_ns=np.arange(4, dtype=float),
            g_factor=3.5,
            metadata={
                'shared_scale_source': 'late_window',
                'shared_scale_interpretation': 'effective_late_window_scale',
            },
        )
        dialog.peak_bin = 1

        dialog._draw_result()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'error', message='constrained_layout not applied.*')
            dialog.canvas.draw()

        plot_axes = (dialog.axes[0, 0], dialog.axes[0, 1], dialog.axes[1, 0])
        assert all(axis.get_position().height >= 0.18 for axis in plot_axes)
        renderer = dialog.canvas.get_renderer()
        assert not dialog.axes[0, 0].xaxis.label.get_window_extent(renderer).overlaps(
            dialog.axes[1, 0].title.get_window_extent(renderer))
        assert dialog.axes[1, 1].texts[0].get_window_extent(renderer).y0 >= 0
        assert dialog.axes[0, 0].get_title() == 'Parallel global fit'
        assert dialog.axes[0, 1].get_title() == 'Perpendicular global fit'
        assert dialog.axes[1, 0].get_title() == 'Residuals'
        summary = dialog.axes[1, 1].texts[0].get_text()
        assert 'Rotational correlation' in summary
        assert 'Fixed fluorescence lifetime' in summary
        assert 'Common IRF shift: 0.7 bins' in summary
        assert 'Fitted backgrounds: 2.5, 7' in summary
        assert 'Effective late-window scale: 3.5' in summary
        assert 'not calibrated physical G' in summary
        assert 'WARNING' in summary
        assert 'did not converge' in summary
        assert 'maximum evaluations reached' in summary
        assert 'initial_anisotropy' in summary
        assert len(dialog.axes[0, 0].lines) == 2
        assert len(dialog.axes[0, 1].lines) == 2
        assert len(dialog.axes[0, 0].lines[0].get_xdata()) == 3
        assert len(dialog.axes[1, 0].lines[0].get_xdata()) == 3
        assert dialog.axes[0, 0].title.get_color() == '#222222'
        assert dialog.axes[0, 0].xaxis.label.get_color() == '#222222'
        assert dialog.axes[0, 0].get_legend().get_texts()[0].get_color() == '#222222'
        assert dialog.axes[1, 1].texts[0].get_color() == '#222222'
    finally:
        mpl.rcParams.update(original_colors)
        root.destroy()


def test_advanced_fit_draws_selected_model_and_fail_closed_warnings():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        candidate_one = SimpleNamespace(
            component_count=1, bic=120.0, identifiable=True,
            identifiability_warnings=())
        candidate_two = SimpleNamespace(
            component_count=2, bic=90.0, identifiable=False,
            identifiability_warnings=(
                'fastest component is below the IRF resolution',
                'slowest component reached the observation bound'),
            rotational_correlation_times_ns=np.array([0.2, 6.4]),
            component_weights=np.array([0.3, 0.7]),
            parallel_model=np.array([11.0, 8.0, 5.0]),
            perpendicular_model=np.array([7.0, 6.0, 4.0]),
            parallel_residual=np.array([0.0, 1.0, -1.0]),
            perpendicular_residual=np.array([1.0, 0.0, -1.0]),
            intensity_lifetime_ns=3.2, initial_anisotropy=0.09,
            poisson_deviance=80.0, aicc=100.0, success=True,
            message='converged', parameters_at_bounds=(),
            common_irf_shift_bins=0.7, parallel_background=2.5,
            perpendicular_background=7.0)
        advanced_fit = SimpleNamespace(
            selected_component_count=2,
            selected_fit=candidate_two,
            candidates=(candidate_one, candidate_two))
        dialog.result = SimpleNamespace(
            parallel_decay=np.array([9.0, 7.0, 3.0, 1.0]),
            perpendicular_decay=np.array([5.0, 4.0, 2.0, 1.0]),
            parallel_background=2.0, perpendicular_background=2.0,
            polarized_fit=None, multicomponent_fit=advanced_fit,
            time_ns=np.arange(4, dtype=float), g_factor=1.0,
            metadata={'shared_scale_source': 'assumed'})
        dialog.peak_bin = 1

        dialog._draw_result()

        assert dialog.axes[0, 0].get_title() == 'Parallel advanced fit'
        assert dialog.axes[0, 1].get_title() == 'Perpendicular advanced fit'
        assert dialog.axes[1, 0].get_ylabel() == 'Signed Poisson deviance'
        summary = dialog.axes[1, 1].texts[0].get_text()
        assert 'BIC-selected model: 2 components' in summary
        assert 'NOT RESOLVED' in summary
        assert 'No resolved rotational-component model' in summary
        assert '0.2 ns' not in summary
        assert 'weight 0.3' not in summary
        assert 'Resolved r(0)' not in summary
        assert 'below the IRF resolution' in summary
        assert 'observation bound' in summary
        assert 'G=1 is assumed' in summary
    finally:
        root.destroy()


def test_advanced_details_window_has_fit_component_and_identifiability_tabs():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        selected = SimpleNamespace(
            component_count=2, bic=90.0, aicc=85.0,
            rotational_correlation_times_ns=np.array([0.2, 6.4]),
            component_weights=np.array([0.3, 0.7]),
            component_bounds_ns=np.array([[0.19, 6.4], [0.19, 6.4]]),
            parallel_model=np.array([9.0, 6.0]),
            perpendicular_model=np.array([4.0, 3.0]),
            parallel_residual=np.array([1.0, 0.0]),
            perpendicular_residual=np.array([0.0, 1.0]),
            identifiable=False,
            identifiability_warnings=(
                'fastest component is below the IRF resolution',),
            success=True)
        comparison = SimpleNamespace(
            selected_component_count=2, selected_fit=selected,
            candidates=(selected,))
        dialog.result = SimpleNamespace(
            multicomponent_fit=comparison,
            time_ns=np.array([0.0, 1.0]),
            parallel_decay=np.array([8.0, 5.0]),
            perpendicular_decay=np.array([3.0, 2.0]),
            parallel_background=2.0, perpendicular_background=3.0)
        dialog.peak_bin = 0

        details = dialog._show_advanced_details()

        assert details is not None
        tabs = [details.details_notebook.tab(tab, 'text')
                for tab in details.details_notebook.tabs()]
        assert tabs == ['Fit curves', 'Components', 'Identifiability']
        assert '0.2 ns' in details.components_text.cget('text')
        assert 'weight 0.3' in details.components_text.cget('text')
        assert 'Optimizer values only' in details.components_heading.cget('text')
        warning_text = details.identifiability_text.cget('text')
        assert 'NOT RESOLVED' in warning_text
        assert 'below the IRF resolution' in warning_text
        assert 'No resolved rotational-component model' in (
            details.identifiability_heading.cget('text'))
        assert str(details.identifiability_heading.cget('foreground')) == '#b00020'
        details.destroy()
    finally:
        root.destroy()


def test_run_analysis_preferred_mode_fits_both_decays_with_separate_irfs():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import run_analysis

    stack = np.ones((2, 2, 16), dtype=float)

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.time_ns = np.arange(16, dtype=float) * 0.1
            self.tcspc_res = 0.1e-9
            self.period_ns = 1.51

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stack

    preferred_fit = object()
    expected = SimpleNamespace(
        metadata={}, parallel_background=2.0,
        perpendicular_background=3.0, polarized_fit=None)
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'parallel_irf_path': 'parallel.csv',
        'perpendicular_irf_path': 'perpendicular.csv',
        'analysis_mode': 'global',
        'fixed_lifetime_ns': 3.0,
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 1.0,
        'auto_register': False,
        'background_bins': slice(0, 2),
        'g_factor': 1.2,
        'g_mode': 'late_window',
        'late_window_start_ns': 0.0,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 0.0,
        'min_map_photons': 0.0,
        'parallel_exposure': 1.5,
        'perpendicular_exposure': 0.8,
    }
    parallel_irf = np.eye(1, 15, 2).ravel()
    perpendicular_irf = np.eye(1, 15, 4).ravel()

    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit_anisotropy.anisotropy.analyze_anisotropy',
                return_value=expected) as analyze,
          patch('flimkit_anisotropy.tool.load_irf_curve',
                side_effect=[parallel_irf, perpendicular_irf]) as load_irf,
          patch('flimkit_anisotropy.anisotropy.fit_polarized_decays',
                return_value=preferred_fit) as fit):
        result, _ = run_analysis(settings)

    assert result.polarized_fit is preferred_fit
    call = fit.call_args
    np.testing.assert_array_equal(call.kwargs['parallel_irf'], parallel_irf)
    np.testing.assert_array_equal(
        call.kwargs['perpendicular_irf'], perpendicular_irf)
    assert call.kwargs['repetition_period_ns'] == 1.51
    assert call.args[0].shape == (15,)
    assert call.args[1].shape == (15,)
    assert call.args[2].shape == (15,)
    assert call.kwargs['intensity_lifetime_ns'] == 3.0
    assert call.kwargs['initial_parallel_background'] == 2.0
    assert call.kwargs['initial_perpendicular_background'] == 3.0
    expected_effective_scale = (64.0 / 1.5) / (64.0 / 0.8)
    assert call.kwargs['g_factor'] == pytest.approx(expected_effective_scale)
    assert analyze.call_args.kwargs['g_factor'] == pytest.approx(
        expected_effective_scale)
    assert result.g_factor == pytest.approx(expected_effective_scale)
    assert load_irf.call_count == 2
    parallel_irf_call, perpendicular_irf_call = load_irf.call_args_list
    assert parallel_irf_call.kwargs['expected_period_ns'] == 1.51
    assert perpendicular_irf_call.kwargs['expected_period_ns'] == 1.51
    assert parallel_irf_call.kwargs['ptu_channel'] == 0
    assert perpendicular_irf_call.kwargs['ptu_channel'] == 0
    assert result.metadata['analysis_mode'] == 'global'
    assert result.metadata['parallel_irf_file'] == 'parallel.csv'
    assert result.metadata['perpendicular_irf_file'] == 'perpendicular.csv'
    assert result.metadata['repetition_period_ns'] == 1.51
    assert result.metadata['global_fit_bins'] == 15
    assert result.metadata['fixed_lifetime_ns'] == 3.0


def test_run_analysis_advanced_mode_passes_g_and_component_controls():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import run_analysis

    stack = np.ones((2, 2, 16), dtype=float)

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.time_ns = np.arange(16, dtype=float) * 0.1
            self.tcspc_res = 0.1e-9
            self.period_ns = 1.51

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stack

    advanced_fit = object()
    expected = SimpleNamespace(
        metadata={}, parallel_background=2.0,
        perpendicular_background=3.0, polarized_fit=None,
        multicomponent_fit=None)
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'parallel_irf_path': 'parallel.csv',
        'perpendicular_irf_path': 'perpendicular.csv',
        'analysis_mode': 'advanced',
        'fixed_lifetime_ns': 3.0,
        'max_components': 3,
        'multistart': 7,
        'component_range_mode': 'manual',
        'component_bounds_ns': (0.2, 1.3),
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 1.0,
        'auto_register': False,
        'background_bins': slice(0, 2),
        'g_factor': 2.75,
        'g_mode': 'assumed',
        'late_window_start_ns': 0.0,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 0.0,
        'min_map_photons': 0.0,
        'parallel_exposure': 1.0,
        'perpendicular_exposure': 1.0,
    }
    irf = np.eye(1, 15, 2).ravel()

    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit_anisotropy.anisotropy.analyze_anisotropy',
                return_value=expected),
          patch('flimkit_anisotropy.tool.load_irf_curve',
                side_effect=[irf, irf]),
          patch(
              'flimkit_anisotropy.anisotropy.fit_multicomponent_polarized_decays',
              return_value=advanced_fit) as fit):
        result, _ = run_analysis(settings)

    assert result.multicomponent_fit is advanced_fit
    assert result.polarized_fit is None
    assert fit.call_args.kwargs['g_factor'] == 2.75
    assert fit.call_args.kwargs['max_components'] == 3
    assert fit.call_args.kwargs['multistart'] == 7
    assert fit.call_args.kwargs['component_bounds_ns'] == settings[
        'component_bounds_ns']
    assert result.metadata['analysis_mode'] == 'advanced'
    assert result.metadata['advanced_max_components'] == 3
    assert result.metadata['advanced_component_range_mode'] == 'manual'


def test_run_analysis_applies_one_shared_late_window_scale():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import run_analysis

    stacks = {
        'parallel.ptu': np.array([[[10.0, 20.0, 30.0, 40.0]]]),
        'perpendicular.ptu': np.array([[[5.0, 10.0, 15.0, 20.0]]]),
    }

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.path = path
            self.time_ns = np.arange(4, dtype=float)
            self.period_ns = 4.0
            self.tcspc_res = 1e-9

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stacks[self.path]

    expected = SimpleNamespace(metadata={})
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'analysis_mode': 'direct',
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 3.0,
        'auto_register': False,
        'background_bins': slice(0, 1),
        'g_factor': 7.0,
        'g_mode': 'late_window',
        'late_window_start_ns': 2.0,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 0.0,
        'min_map_photons': 0.0,
        'parallel_exposure': 2.0,
        'perpendicular_exposure': 1.0,
    }

    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit_anisotropy.anisotropy.analyze_anisotropy',
                return_value=expected) as analyze):
        result, _ = run_analysis(settings)

    assert analyze.call_args.kwargs['g_factor'] == pytest.approx(1.0)
    assert result.g_factor == pytest.approx(1.0)
    assert result.late_window_stability.selected_scale == pytest.approx(1.0)
    assert result.metadata['shared_scale_source'] == 'late_window'
    assert result.metadata['shared_scale_interpretation'] == (
        'effective_late_window_scale')
    assert result.metadata['requested_shared_scale'] == 7.0


def test_effective_scale_requires_valid_matching_laser_periods():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import run_analysis

    class FakePTUFile:
        periods = {'parallel.ptu': None, 'perpendicular.ptu': None}

        def __init__(self, path, verbose=False):
            self.path = path
            self.time_ns = np.arange(4, dtype=float)
            self.period_ns = self.periods[path]
            self.tcspc_res = 1e-9

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return np.ones((1, 1, 4), dtype=float)

    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'analysis_mode': 'direct',
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 3.0,
        'auto_register': False,
        'background_bins': slice(0, 1),
        'g_factor': 1.0,
        'g_mode': 'late_window',
        'late_window_start_ns': 2.0,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 0.0,
        'min_map_photons': 0.0,
        'parallel_exposure': 1.0,
        'perpendicular_exposure': 1.0,
    }
    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit_anisotropy.anisotropy.analyze_anisotropy',
                return_value=SimpleNamespace(metadata={}))):
        with pytest.raises(ValueError, match='requires a laser period'):
            run_analysis(settings)
        FakePTUFile.periods = {
            'parallel.ptu': 4.0, 'perpendicular.ptu': 5.0}
        with pytest.raises(ValueError, match='matching laser periods'):
            run_analysis(settings)
        FakePTUFile.periods = {
            'parallel.ptu': 10.0, 'perpendicular.ptu': 10.0}
        with pytest.raises(ValueError, match='incompatible with the stored'):
            run_analysis(settings)


def test_assumed_scale_survives_unavailable_late_window_diagnostic():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import run_analysis

    stacks = {
        'parallel.ptu': np.ones((1, 1, 4), dtype=float),
        'perpendicular.ptu': np.ones((1, 1, 4), dtype=float),
    }

    class FakePTUFile:
        period = 1.0

        def __init__(self, path, verbose=False):
            self.path = path
            self.time_ns = np.arange(4, dtype=float)
            self.period_ns = self.period
            self.tcspc_res = 1e-9

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stacks[self.path]

    expected = SimpleNamespace(metadata={})
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'analysis_mode': 'direct',
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 3.0,
        'auto_register': False,
        'background_bins': slice(0, 1),
        'g_factor': 1.2,
        'g_mode': 'assumed',
        'late_window_start_ns': 9.6,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 0.0,
        'min_map_photons': 0.0,
        'parallel_exposure': 1.0,
        'perpendicular_exposure': 1.0,
    }
    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit_anisotropy.anisotropy.analyze_anisotropy',
                return_value=expected)):
        result, _ = run_analysis(settings)
        FakePTUFile.period = 4.0
        stacks['perpendicular.ptu'] = np.array([[[1.0, 1.0, 0.0, 0.0]]])
        settings['late_window_start_ns'] = 2.0
        zero_tail_result, _ = run_analysis(settings)

    assert result.g_factor == pytest.approx(1.2)
    assert result.late_window_stability is None
    assert result.metadata['late_window_diagnostic_available'] is False
    assert result.metadata['shared_scale_source'] == 'assumed'
    assert zero_tail_result.late_window_stability is None
    assert 'contain photons in both channels' in (
        zero_tail_result.metadata['late_window_diagnostic_error'])


def test_run_analysis_records_photon_thresholds():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import run_analysis

    stack = np.ones((2, 2, 4), dtype=float)

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.time_ns = np.arange(4, dtype=float)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stack

    expected = SimpleNamespace(metadata={})
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 2.0,
        'auto_register': False,
        'background_bins': slice(0, 1),
        'g_factor': 1.0,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 25.0,
        'min_map_photons': 500.0,
        'parallel_exposure': 1.0,
        'perpendicular_exposure': 1.0,
    }

    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit_anisotropy.anisotropy.analyze_anisotropy',
                return_value=expected)):
        result, _ = run_analysis(settings)

    assert result.metadata['min_bin_photons'] == 25.0
    assert result.metadata['min_map_photons'] == 500.0


def test_run_analysis_uses_g_and_exposure_normalized_peak():
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import run_analysis

    stacks = {
        'parallel.ptu': np.array([[[100.0, 0.0]]]),
        'perpendicular.ptu': np.array([[[0.0, 80.0]]]),
    }

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.path = path
            self.time_ns = np.arange(2, dtype=float)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stacks[self.path]

    expected = SimpleNamespace(metadata={})
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 0.5,
        'auto_register': False,
        'background_bins': slice(0, 1),
        'g_factor': 1.0,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 0.0,
        'min_map_photons': 0.0,
        'parallel_exposure': 1.0,
        'perpendicular_exposure': 100.0,
    }

    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit_anisotropy.anisotropy.analyze_anisotropy',
                return_value=expected)):
        _, peak_bin = run_analysis(settings)

    assert peak_bin == 0


def test_global_fit_csv_includes_models_residuals_and_parameters(tmp_path):
    import csv
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import AnisotropyTool

    fit = SimpleNamespace(
        parallel_model=np.array([9.0]),
        perpendicular_model=np.array([4.0]),
        parallel_residual=np.array([1.0]),
        perpendicular_residual=np.array([0.0]),
        intensity_lifetime_ns=3.2,
        rotational_correlation_ns=1.4,
        initial_anisotropy=0.32,
        poisson_deviance=5.0,
        common_irf_shift_bins=0.7,
        parallel_background=2.5,
        perpendicular_background=7.0,
    )
    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool.peak_bin = 0
    tool.status = SimpleNamespace(set=lambda value: None)
    tool.result = SimpleNamespace(
        time_ns=np.array([1.0, 2.0]),
        parallel_decay=np.array([8.0, 3.0]),
        perpendicular_decay=np.array([3.0, 1.0]),
        parallel_background=2.0,
        perpendicular_background=3.0,
        anisotropy_decay=np.array([0.2, 0.2]),
        polarized_fit=fit,
        g_factor=1.0,
        parallel_exposure=1.0,
        perpendicular_exposure=1.0,
        perpendicular_shift=(0.0, 0.0),
        spatial_window=1,
        stride=1,
        metadata={'analysis_mode': 'global'},
    )
    path = tmp_path / 'global.csv'

    with patch('flimkit_anisotropy.tool.filedialog.asksaveasfilename',
               return_value=str(path)):
        tool._save_csv()

    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['parallel_model'] == '9.0'
    assert rows[0]['parallel_observed'] == '10.0'
    assert rows[0]['perpendicular_observed'] == '6.0'
    assert rows[0]['perpendicular_residual'] == '0.0'
    assert rows[0]['intensity_lifetime_ns'] == '3.2'
    assert rows[0]['rotational_correlation_ns'] == '1.4'
    assert rows[0]['initial_anisotropy'] == '0.32'
    assert rows[0]['common_irf_shift_bins'] == '0.7'
    assert rows[0]['parallel_fit_background'] == '2.5'
    assert rows[0]['perpendicular_fit_background'] == '7.0'
    assert rows[1]['parallel_model'] == ''
    assert rows[1]['parallel_observed'] == ''


def test_advanced_fit_csv_includes_selection_components_and_warnings(tmp_path):
    import csv
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import AnisotropyTool

    start_record = SimpleNamespace(
        start_index=0, parameter_vector=np.arange(8, dtype=float),
        rotational_correlation_times_ns=np.array([0.2, 6.4]),
        component_weights=np.array([0.3, 0.7]), poisson_deviance=60.0,
        success=True, message='@SUM(1,1)',
        parameters_at_bounds=('common_irf_shift_bins',))
    candidate_one = SimpleNamespace(
        component_count=1, rotational_correlation_times_ns=np.array([1.5]),
        component_weights=np.array([1.0]), bic=120.0, aicc=115.0,
        poisson_deviance=100.0, identifiable=False,
        identifiability_warnings=('uncertainty validation unavailable',),
        success=True, component_bounds_ns=np.array([[0.2, 6.4]]),
        multistart_count=1, multistart_records=(), message='candidate converged',
        parameters_at_bounds=(), common_irf_shift_bins=0.1,
        parallel_background=2.0, perpendicular_background=3.0,
        initial_anisotropy=0.2, amplitude=100.0)
    candidate_two = SimpleNamespace(
        component_count=2,
        rotational_correlation_times_ns=np.array([0.2, 6.4]),
        component_weights=np.array([0.3, 0.7]), bic=90.0, aicc=85.0,
        poisson_deviance=60.0, identifiable=False,
        identifiability_warnings=(
            'fastest component is below the IRF resolution',), success=True,
        component_bounds_ns=np.array([[0.2, 6.4], [0.2, 6.4]]),
        parallel_model=np.array([9.0]),
        perpendicular_model=np.array([4.0]),
        parallel_residual=np.array([1.0]),
        perpendicular_residual=np.array([0.0]),
        intensity_lifetime_ns=3.2, initial_anisotropy=0.09,
        common_irf_shift_bins=0.7, parallel_background=2.5,
        perpendicular_background=7.0, message='+candidate converged',
        amplitude=123.0,
        parameters_at_bounds=('common_irf_shift_bins',), multistart_count=1,
        multistart_records=(start_record,),
        parameter_names=('time_1', 'time_2', 'weight_1', 'r0',
                         'log_amplitude', 'shift', 'parallel_bg',
                         'perpendicular_bg'))
    comparison = SimpleNamespace(
        max_components=2, selected_component_count=2,
        selected_fit=candidate_two, candidates=(candidate_one, candidate_two))
    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool.peak_bin = 0
    tool.status = SimpleNamespace(set=lambda value: None)
    tool.result = SimpleNamespace(
        time_ns=np.array([1.0, 2.0]),
        parallel_decay=np.array([8.0, 3.0]),
        perpendicular_decay=np.array([3.0, 1.0]),
        parallel_background=2.0, perpendicular_background=3.0,
        anisotropy_decay=np.array([0.2, 0.2]), polarized_fit=None,
        multicomponent_fit=comparison, late_window_stability=None,
        g_factor=1.0, parallel_exposure=1.0,
        perpendicular_exposure=1.0, perpendicular_shift=(0.0, 0.0),
        spatial_window=1, stride=1,
        metadata={
            'analysis_mode': 'advanced',
            'advanced_component_range_mode': 'auto',
            'parallel_file': '=HYPERLINK("https://example.invalid")',
            'parallel_irf_file': '=parallel_irf.ptu',
            'perpendicular_irf_file': 'perpendicular_irf.ptu',
            'repetition_period_ns': 12.8,
            'global_fit_bins': 132})
    path = tmp_path / 'advanced.csv'

    with patch('flimkit_anisotropy.tool.filedialog.asksaveasfilename',
               return_value=str(path)):
        tool._save_csv()

    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    first = rows[0]
    assert first['parallel_model'] == '9.0'
    assert first['advanced_selected_component_count'] == '2'
    assert first['advanced_candidate_1_times_ns'] == '1.5'
    assert first['advanced_candidate_2_times_ns'] == '0.2;6.4'
    assert first['advanced_candidate_2_weights'] == '0.3;0.7'
    assert first['advanced_candidate_2_bic'] == '90.0'
    assert first['advanced_candidate_2_identifiable'] == 'False'
    assert first['advanced_residual_type'] == 'signed_poisson_deviance'
    assert first['parallel_file'].startswith("'=")
    assert first['parallel_irf_file'] == "'=parallel_irf.ptu"
    assert first['perpendicular_irf_file'] == 'perpendicular_irf.ptu'
    assert first['repetition_period_ns'] == '12.8'
    assert first['global_fit_bins'] == '132'
    assert first['advanced_candidate_2_message'] == "'+candidate converged"
    assert first['advanced_candidate_2_initial_anisotropy'] == '0.09'
    assert first['advanced_candidate_2_amplitude'] == '123.0'
    assert first['advanced_candidate_2_parameter_names'].startswith(
        'time_1;time_2;weight_1;r0')
    assert first['advanced_candidate_2_common_irf_shift_bins'] == '0.7'
    assert first['advanced_candidate_2_parallel_background'] == '2.5'
    assert first['advanced_candidate_2_start_0_parameter_vector'] == (
        '0;1;2;3;4;5;6;7')
    assert first['advanced_candidate_2_start_0_times_ns'] == '0.2;6.4'
    assert first['advanced_candidate_2_start_0_weights'] == '0.3;0.7'
    assert first['advanced_candidate_2_start_0_message'] == "'@SUM(1,1)"
    assert 'below the IRF resolution' in first[
        'advanced_candidate_2_warnings']


def test_csv_export_includes_relative_time_and_provenance(tmp_path):
    import csv
    from types import SimpleNamespace
    import numpy as np
    from flimkit_anisotropy.tool import AnisotropyTool

    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool.peak_bin = 1
    tool.status = SimpleNamespace(set=lambda value: None)
    tool.result = SimpleNamespace(
        time_ns=np.array([1.0, 2.0]),
        parallel_decay=np.array([10.0, 5.0]),
        perpendicular_decay=np.array([4.0, 2.0]),
        anisotropy_decay=np.array([0.2, 0.2]),
        g_factor=0.9,
        parallel_exposure=1.0,
        perpendicular_exposure=2.0,
        perpendicular_shift=(0.25, -0.5),
        spatial_window=5,
        stride=1,
        late_window_stability=SimpleNamespace(
            nested_scale=np.array([3.5, 3.4]),
            rolling_scale=np.array([3.6, np.nan]),
            nested_standard_error=np.array([0.1, 0.2]),
            selected_start_bin=1,
            selected_scale=3.4,
            selected_parallel_photons=680.0,
            selected_perpendicular_photons=200.0),
        metadata={
            'parallel_file': 'parallel.ptu',
            'perpendicular_file': 'perpendicular.ptu',
            'parallel_channel': 0,
            'perpendicular_channel': 1,
            'background_start_bin': 2,
            'background_stop_bin': 10,
            'analysis_start_ns': 0.0,
            'analysis_stop_ns': 8.0,
            'min_bin_photons': 25.0,
            'min_map_photons': 500.0,
            'shared_scale_source': 'late_window',
            'shared_scale_interpretation': 'effective_late_window_scale',
            'requested_shared_scale': 1.0,
            'applied_shared_scale': 0.9,
            'late_window_selected_start_ns': 2.0,
        },
    )
    path = tmp_path / 'decay.csv'

    with patch('flimkit_anisotropy.tool.filedialog.asksaveasfilename',
               return_value=str(path)):
        tool._save_csv()

    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['time_after_peak_ns'] == '-1.0'
    assert rows[0]['g_factor'] == '0.9'
    assert rows[0]['parallel_file'] == 'parallel.ptu'
    assert rows[0]['perpendicular_channel'] == '1'
    assert rows[0]['min_map_photons'] == '500.0'
    assert rows[0]['shared_scale_source'] == 'late_window'
    assert rows[0]['shared_scale_interpretation'] == (
        'effective_late_window_scale')
    assert rows[0]['late_window_nested_scale'] == '3.5'
    assert rows[0]['late_window_rolling_scale'] == '3.6'
    assert rows[0]['late_window_nested_standard_error'] == '0.1'
    assert rows[0]['late_window_selected_scale'] == '3.4'
