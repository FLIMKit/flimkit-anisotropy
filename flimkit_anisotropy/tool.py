import csv
import queue
import textwrap
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np


def _spreadsheet_safe(value):
    if (isinstance(value, str)
            and value.lstrip().startswith(('=', '+', '-', '@'))):
        return "'" + value
    return value


class AnisotropyTool(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title('Time-Resolved Anisotropy')
        self.geometry('1180x820')
        self.minsize(980, 700)
        self.result = None
        self.peak_bin = None
        self._colorbar = None
        self._closed = False
        self._poll_after_id = None
        self.protocol('WM_DELETE_WINDOW', self._close)
        self._build_controls()
        self._build_plot()

    def _build_controls(self):
        self.parallel_path = tk.StringVar()
        self.perpendicular_path = tk.StringVar()
        self.parallel_irf_path = tk.StringVar()
        self.perpendicular_irf_path = tk.StringVar()
        self.analysis_mode = tk.StringVar(value='direct')
        self.fixed_lifetime_ns = tk.DoubleVar(value=3.0)
        self.g_factor = tk.DoubleVar(value=1.0)
        self.g_mode = tk.StringVar(value='Assumed scale')
        self.late_window_start_ns = tk.DoubleVar(value=9.6)
        self.max_components = tk.IntVar(value=1)
        self.multistart = tk.IntVar(value=6)
        self.component_range_mode = tk.StringVar(value='Auto')
        self.component_lower_ns = [tk.DoubleVar(value=0.2)]
        self.component_upper_ns = [tk.DoubleVar(value=6.0)]
        self.parallel_exposure = tk.DoubleVar(value=1.0)
        self.perpendicular_exposure = tk.DoubleVar(value=1.0)
        self.parallel_channel = tk.IntVar(value=0)
        self.perpendicular_channel = tk.IntVar(value=0)
        self.background_start = tk.IntVar(value=2)
        self.background_stop = tk.IntVar(value=10)
        self.analysis_start_ns = tk.DoubleVar(value=0.0)
        self.analysis_stop_ns = tk.DoubleVar(value=8.0)
        self.spatial_window = tk.IntVar(value=5)
        self.stride = tk.IntVar(value=1)
        self.min_bin_photons = tk.DoubleVar(value=25.0)
        self.min_map_photons = tk.DoubleVar(value=500.0)
        self.auto_register = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Choose the parallel and perpendicular PTU files.')

        self.toolbar = ttk.Frame(self, padding=(10, 8, 10, 4))
        self.toolbar.pack(fill='x')
        self.toggle_inputs_button = ttk.Button(
            self.toolbar, text='Hide inputs', command=self._toggle_inputs)
        self.toggle_inputs_button.pack(side='left')
        self.calculate_button = ttk.Button(
            self.toolbar, text='Calculate', command=self._start_analysis)
        self.calculate_button.pack(side='left', padx=(6, 0))
        self.scale_diagnostic_button = ttk.Button(
            self.toolbar, text='Scale diagnostic...',
            command=self._show_scale_diagnostic, state='disabled')
        self.scale_diagnostic_button.pack(side='left', padx=(6, 0))
        self.fit_details_button = ttk.Button(
            self.toolbar, text='Fit details...',
            command=self._show_advanced_details, state='disabled')
        self.fit_details_button.pack(side='left', padx=(6, 0))
        self.save_npz_button = ttk.Button(
            self.toolbar, text='Save NPZ...', command=self._save_npz,
            state='disabled')
        self.save_npz_button.pack(side='left', padx=(6, 0))
        self.save_csv_button = ttk.Button(
            self.toolbar, text='Save CSV...', command=self._save_csv,
            state='disabled')
        self.save_csv_button.pack(side='left', padx=(6, 0))
        self.status_label = ttk.Label(
            self.toolbar, textvariable=self.status, wraplength=650,
            justify='left')
        self.status_label.pack(side='left', padx=12)

        self.input_panel = ttk.Frame(self, padding=(10, 0, 10, 8))
        self.input_panel.pack(fill='x', after=self.toolbar)
        self.input_panel.columnconfigure(1, weight=1)

        self._file_row(self.input_panel, 0, 'Parallel PTU', self.parallel_path)
        self._file_row(
            self.input_panel, 1, 'Perpendicular PTU', self.perpendicular_path)
        self._file_row(
            self.input_panel, 2, 'Parallel IRF (PTU or export)',
            self.parallel_irf_path,
            file_kind='irf')
        self._file_row(
            self.input_panel, 3, 'Perpendicular IRF (PTU or export)',
            self.perpendicular_irf_path, file_kind='irf')

        modes = ttk.LabelFrame(
            self.input_panel, text='Analysis method', padding=8)
        modes.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        ttk.Radiobutton(
            modes, text='Direct r(t) diagnostic (no IRF)',
            variable=self.analysis_mode, value='direct').pack(side='left')
        ttk.Radiobutton(
            modes, text='Preferred global fit (Lakowicz Section 11.2.2)',
            variable=self.analysis_mode, value='global').pack(side='left', padx=12)
        ttk.Radiobutton(
            modes, text='Advanced 1/2/3-component fit',
            variable=self.analysis_mode, value='advanced').pack(side='left')
        ttk.Button(modes, text='Method info...',
                   command=self._show_method_info).pack(side='left')

        settings = ttk.LabelFrame(
            self.input_panel, text='Analysis settings', padding=8)
        settings.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        fields = [
            ('Known lifetime (ns, global fit)', self.fixed_lifetime_ns),
            ('Shared scale value', self.g_factor),
            ('Late-window start (ns)', self.late_window_start_ns),
            ('Parallel exposure (relative)', self.parallel_exposure),
            ('Perpendicular exposure (relative)', self.perpendicular_exposure),
            ('Parallel photon channel', self.parallel_channel),
            ('Perpendicular photon channel', self.perpendicular_channel),
            ('Background start bin', self.background_start),
            ('Background stop bin', self.background_stop),
            ('Post-peak start (ns)', self.analysis_start_ns),
            ('Post-peak stop (ns)', self.analysis_stop_ns),
            ('Spatial window', self.spatial_window),
            ('Stride', self.stride),
            ('Min photons / time bin', self.min_bin_photons),
            ('Min photons / map', self.min_map_photons),
        ]
        for index, (label, variable) in enumerate(fields):
            row, column = divmod(index, 3)
            base = column * 2
            ttk.Label(settings, text=label).grid(
                row=row, column=base, sticky='w', padx=(0, 4), pady=2)
            ttk.Entry(settings, textvariable=variable, width=12).grid(
                row=row, column=base + 1, sticky='w', padx=(0, 12), pady=2)
        ttk.Label(settings, text='Shared scale source').grid(
            row=5, column=0, sticky='w', padx=(0, 4), pady=(4, 0))
        self.scale_source_combo = ttk.Combobox(
            settings, textvariable=self.g_mode, state='readonly', width=27,
            values=('Assumed scale', 'Calibrated G',
                    'Effective late-window scale'))
        self.scale_source_combo.grid(
            row=5, column=1, columnspan=2, sticky='w', padx=(0, 12),
            pady=(4, 0))
        ttk.Checkbutton(
            settings, text='Auto-register perpendicular image to parallel image',
            variable=self.auto_register).grid(
                row=5, column=3, columnspan=3, sticky='w', pady=(4, 0))

        advanced = ttk.LabelFrame(
            self.input_panel, text='Advanced component settings', padding=8)
        self.advanced_settings_frame = advanced
        advanced.grid(
            row=6, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        ttk.Label(advanced, text='Maximum components').grid(
            row=0, column=0, sticky='w', padx=(0, 4))
        ttk.Combobox(
            advanced, textvariable=self.max_components, state='readonly',
            values=('1', '2', '3'), width=5).grid(
                row=0, column=1, sticky='w', padx=(0, 14))
        ttk.Label(advanced, text='Deterministic starts').grid(
            row=0, column=2, sticky='w', padx=(0, 4))
        ttk.Entry(advanced, textvariable=self.multistart, width=6).grid(
            row=0, column=3, sticky='w', padx=(0, 14))
        ttk.Label(advanced, text='Bounds mode').grid(
            row=0, column=4, sticky='w', padx=(0, 4))
        ttk.Combobox(
            advanced, textvariable=self.component_range_mode, state='readonly',
            values=('Auto', 'Manual'), width=9).grid(
                row=0, column=5, sticky='w')
        self.component_bound_entries = []
        ttk.Label(advanced, text='Shared time range').grid(
            row=1, column=0, sticky='w', pady=(3, 0))
        ttk.Label(advanced, text='lower ns').grid(
            row=1, column=1, sticky='e', pady=(3, 0))
        lower_entry = ttk.Entry(
            advanced, textvariable=self.component_lower_ns[0], width=7)
        lower_entry.grid(
            row=1, column=2, sticky='w', padx=(4, 12), pady=(3, 0))
        ttk.Label(advanced, text='upper ns').grid(
            row=1, column=3, sticky='e', pady=(3, 0))
        upper_entry = ttk.Entry(
            advanced, textvariable=self.component_upper_ns[0], width=7)
        upper_entry.grid(
            row=1, column=4, sticky='w', padx=(4, 12), pady=(3, 0))
        self.component_bound_entries.extend((lower_entry, upper_entry))
        g_hint = (
            'Manual mode applies this same range to every candidate. '
            'G defaults to 1. For a better G guess, first use the effective '
            'late-window scale feature. That estimate is effective, not calibrated.')
        ttk.Label(
            advanced, text=g_hint, foreground='#555555', wraplength=940,
            justify='left').grid(
                row=2, column=0, columnspan=6, sticky='w', pady=(6, 0))

        note = ('File roles are explicit; FLIMKit does not infer them from names. '
                'G=1 is an assumption unless calibrated independently. An '
                'effective late-window scale is not a calibrated physical G.')
        ttk.Label(self.input_panel, text=note, foreground='#555555').grid(
            row=7, column=0, columnspan=3, sticky='w', pady=(6, 0))
        self.analysis_mode.trace_add('write', self._update_advanced_controls)
        self.component_range_mode.trace_add(
            'write', self._update_component_range_controls)
        self._update_advanced_controls()
        self._update_component_range_controls()

    def _update_advanced_controls(self, *_args):
        if self.analysis_mode.get() == 'advanced':
            self.advanced_settings_frame.grid()
        else:
            self.advanced_settings_frame.grid_remove()

    def _update_component_range_controls(self, *_args):
        state = (
            'normal' if self.component_range_mode.get() == 'Manual'
            else 'disabled')
        for entry in self.component_bound_entries:
            entry.configure(state=state)

    def _toggle_inputs(self):
        self._set_inputs_visible(not bool(self.input_panel.winfo_manager()))

    def _set_inputs_visible(self, visible):
        if visible:
            if not self.input_panel.winfo_manager():
                self.input_panel.pack(fill='x', after=self.toolbar)
            self.toggle_inputs_button.configure(text='Hide inputs')
        else:
            self.input_panel.pack_forget()
            self.toggle_inputs_button.configure(text='Show inputs')

    def _file_row(self, parent, row, label, variable, file_kind='ptu'):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky='ew', padx=6, pady=2)
        ttk.Button(parent, text='Browse...',
                   command=lambda: self._browse(variable, file_kind=file_kind)).grid(
                       row=row, column=2, sticky='e', pady=2)

    def _build_plot(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(10.8, 6.0), dpi=100,
                             constrained_layout=True)
        self.figure.get_layout_engine().set(w_pad=0.08, h_pad=0.05)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)
        self._draw_empty()

    def _draw_empty(self):
        titles = ['Parallel intensity', 'Perpendicular intensity',
                  'Summed anisotropy decay', 'Anisotropy map']
        for axis, title in zip(self.axes.flat, titles):
            axis.clear()
            axis.set_title(title)
            axis.text(0.5, 0.5, 'No result yet', ha='center', va='center',
                      transform=axis.transAxes, color='#777777')
            axis.set_axis_off()
        self._style_plot_text()
        self.canvas.draw_idle()

    def _style_plot_text(self):
        text_color = '#222222'
        self.figure.set_facecolor('white')
        for axis in self.figure.axes:
            axis.set_facecolor('white')
            axis.tick_params(axis='both', colors=text_color)
            axis.title.set_color(text_color)
            axis.xaxis.label.set_color(text_color)
            axis.yaxis.label.set_color(text_color)
            for spine in axis.spines.values():
                spine.set_color(text_color)
            for text in axis.texts:
                text.set_color(text_color)
            legend = axis.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor('white')
                for text in legend.get_texts():
                    text.set_color(text_color)

    def _browse(self, variable, file_kind='ptu'):
        if file_kind == 'irf':
            title = 'Select measured IRF'
            filetypes = [
                ('Measured IRF',
                 ('*.ptu', '*.xlsx', '*.csv', '*.tsv', '*.txt', '*.dat',
                   '*.ascii', '*.asc')),
                ('All files', '*.*'),
            ]
        else:
            title = 'Select PTU file'
            filetypes = [
                ('PicoQuant PTU', '*.ptu'),
                ('All files', '*.*'),
            ]
        path = filedialog.askopenfilename(
            parent=self, title=title, filetypes=filetypes)
        if path:
            variable.set(path)

    def _show_advanced_details(self):
        if self.result is None:
            return None
        comparison = getattr(self.result, 'multicomponent_fit', None)
        if comparison is None or comparison.selected_fit is None:
            return None
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        details = tk.Toplevel(self)
        details.title('Advanced fit details')
        details.geometry('940x700')
        notebook = ttk.Notebook(details)
        notebook.pack(fill='both', expand=True, padx=8, pady=8)
        fit_tab = ttk.Frame(notebook, padding=6)
        components_tab = ttk.Frame(notebook, padding=12)
        identifiability_tab = ttk.Frame(notebook, padding=12)
        notebook.add(fit_tab, text='Fit curves')
        notebook.add(components_tab, text='Components')
        notebook.add(identifiability_tab, text='Identifiability')

        selected = comparison.selected_fit
        fit_bins = len(selected.parallel_model)
        time_relative = (
            self.result.time_ns[:fit_bins] - self.result.time_ns[self.peak_bin])
        parallel_observed = (
            self.result.parallel_decay[:fit_bins] + self.result.parallel_background)
        perpendicular_observed = (
            self.result.perpendicular_decay[:fit_bins]
            + self.result.perpendicular_background)
        figure = Figure(figsize=(8.8, 5.8), dpi=100, constrained_layout=True)
        axes = figure.subplots(2, 1, sharex=True)
        for axis, observed, model, label in (
                (axes[0], parallel_observed, selected.parallel_model, 'Parallel'),
                (axes[1], perpendicular_observed,
                 selected.perpendicular_model, 'Perpendicular')):
            axis.semilogy(
                time_relative, np.maximum(observed, 1e-3),
                color='#777777', linewidth=1.2, label='Measured')
            axis.semilogy(
                time_relative, np.maximum(model, 1e-3),
                color='#2468a2', linewidth=2.0, label='Selected model')
            axis.set_ylabel(f'{label} counts')
            axis.legend(
                fontsize=9, loc='upper right', bbox_to_anchor=(0.95, 0.98))
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel('Time after peak (ns)')
        canvas = FigureCanvasTkAgg(figure, master=fit_tab)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw_idle()

        component_lines = [
            f'BIC-selected model: {comparison.selected_component_count} components',
            '',
        ]
        for candidate in comparison.candidates:
            state = 'RESOLVED' if candidate.identifiable else 'NOT RESOLVED'
            component_lines.append(
                f'{candidate.component_count}-component candidate — '
                f'BIC {candidate.bic:.6g}; AICc {candidate.aicc:.6g}; {state}')
            for index, (correlation_time, weight) in enumerate(zip(
                    candidate.rotational_correlation_times_ns,
                    candidate.component_weights), start=1):
                lower, upper = candidate.component_bounds_ns[index - 1]
                component_lines.append(
                    f'  Component {index}: {correlation_time:.6g} ns, '
                    f'weight {weight:.6g}, bounds {lower:.6g}–{upper:.6g} ns')
            component_lines.append('')
        components_heading = ttk.Label(
            components_tab,
            text='Optimizer values only — not validated physical estimates',
            foreground='#b00020', font=('TkDefaultFont', 12, 'bold'))
        components_heading.pack(anchor='nw', fill='x', pady=(0, 8))
        components_label = ttk.Label(
            components_tab, text='\n'.join(component_lines), justify='left',
            wraplength=870)
        components_label.pack(anchor='nw', fill='x')

        identifiability_lines = [
            'A converged optimizer is not proof of separate physical motions.',
            'BIC helps compare models but is not a physical-resolution test.',
            '',
        ]
        for candidate in comparison.candidates:
            state = 'RESOLVED' if candidate.identifiable else 'NOT RESOLVED'
            identifiability_lines.append(
                f'{candidate.component_count}-component candidate: {state}')
            if candidate.identifiability_warnings:
                identifiability_lines.extend(
                    f'  • {warning}'
                    for warning in candidate.identifiability_warnings)
            else:
                identifiability_lines.append('  No resolution flags.')
        identifiability_heading = ttk.Label(
            identifiability_tab,
            text=(
                'No resolved rotational-component model'
                if not selected.identifiable
                else 'Selected model passed the reported checks'),
            foreground=('#b00020' if not selected.identifiable else '#2468a2'),
            font=('TkDefaultFont', 12, 'bold'))
        identifiability_heading.pack(anchor='nw', fill='x', pady=(0, 8))
        identifiability_label = ttk.Label(
            identifiability_tab, text='\n'.join(identifiability_lines),
            justify='left', wraplength=870)
        identifiability_label.pack(anchor='nw', fill='x')

        details.details_notebook = notebook
        details.details_figure = figure
        details.details_canvas = canvas
        details.components_heading = components_heading
        details.components_text = components_label
        details.identifiability_heading = identifiability_heading
        details.identifiability_text = identifiability_label
        return details

    def _show_scale_diagnostic(self):
        if self.result is None:
            return None
        stability = getattr(self.result, 'late_window_stability', None)
        if stability is None:
            return None
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        diagnostic = tk.Toplevel(self)
        diagnostic.title('Late-window stability')
        diagnostic.geometry('940x740')
        figure = Figure(figsize=(9.0, 5.6), dpi=100)
        figure.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.93)
        axis = figure.subplots()
        time_ns = stability.time_ns
        axis.plot(
            time_ns, stability.rolling_scale, color='#888888', linewidth=1.5,
            label=f'{stability.rolling_bins}-bin rolling ratio')
        axis.plot(
            time_ns, stability.nested_scale, color='#1769aa', linewidth=2.2,
            label='Nested ratio to period end')
        lower = stability.nested_scale - 1.96 * stability.nested_standard_error
        upper = stability.nested_scale + 1.96 * stability.nested_standard_error
        axis.fill_between(
            time_ns, lower, upper, color='#1769aa', alpha=0.18,
            label='Approx. 95% Poisson band')
        selected_time = time_ns[stability.selected_start_bin]
        axis.axvline(selected_time, color='#c62828', linestyle='--', linewidth=1.5)
        axis.axhline(
            stability.selected_scale, color='#c62828', linestyle=':',
            linewidth=1.5)
        axis.set_title('Late-window stability')
        axis.set_xlabel('Window start (ns)')
        axis.set_ylabel('Parallel / perpendicular scale')
        axis.grid(alpha=0.2)
        axis.legend(loc='best')

        source = self.result.metadata.get('shared_scale_source', 'assumed')
        if source == 'late_window':
            applied = f'Effective scale: {self.result.g_factor:.6g}'
        elif source == 'calibrated':
            applied = f'User-declared calibrated G: {self.result.g_factor:.6g}'
        else:
            applied = f'Uncalibrated assumed scale: {self.result.g_factor:.6g}'
        summary = (
            f'{applied}\n'
            f'Selected start: {selected_time:.6g} ns; '
            f'{stability.selected_parallel_photons:g} parallel, '
            f'{stability.selected_perpendicular_photons:g} perpendicular photons\n'
            'Late-window estimate assumes r(t) approaches zero; it is not a '
            'calibrated physical G.\n'
            'Shaded band: approximate delta-method Poisson 95% interval; '
            'unreliable at low counts.\n'
            'Nested windows share photons and are correlated. Raw counts are '
            'used without background subtraction; previous-pulse fluorescence '
            'remains included.')
        axis.set_facecolor('white')
        axis.tick_params(axis='both', colors='#222222')
        axis.title.set_color('#222222')
        axis.xaxis.label.set_color('#222222')
        axis.yaxis.label.set_color('#222222')
        figure.set_facecolor('white')
        summary_label = ttk.Label(
            diagnostic, text=summary, justify='left', wraplength=900,
            padding=(10, 4, 10, 10))
        summary_label.pack(side='bottom', fill='x')
        canvas = FigureCanvasTkAgg(figure, master=diagnostic)
        canvas.get_tk_widget().pack(
            side='top', fill='both', expand=True, padx=8, pady=(8, 0))
        canvas.draw_idle()
        diagnostic.scale_figure = figure
        diagnostic.scale_axes = (axis,)
        diagnostic.scale_summary = summary_label
        diagnostic.scale_canvas = canvas
        return diagnostic

    def _show_method_info(self):
        messagebox.showinfo(
            'Time-resolved anisotropy methods',
            'Direct r(t) diagnostic:\n'
            'r(t) = [I_parallel(t) - G I_perpendicular(t)] / '
            '[I_parallel(t) + 2 G I_perpendicular(t)]\n\n'
            'This ratio is useful for inspection and maps, but convolution and '
            'division do not commute. It should not be used to fit rotational '
            'correlation times near the IRF response.\n\n'
            'Preferred global fit:\n'
            'Fits the raw parallel and perpendicular photon counts together. '
            'It uses separate IRFs loaded directly from PTU files or from '
            'supported exports. PTU IRFs must match the sample timing resolution '
            'and laser period. The fit also uses a known fluorescence lifetime, '
            'one shared assumed, calibrated, or effective late-window scale, '
            'relative exposures, one rotational correlation time, one common IRF '
            'timing shift, and separate fitted backgrounds. Previous laser pulses '
            'are included. An effective late-window scale is not a calibrated '
            'physical G.\n\n'
            'Advanced 1/2/3-component fit:\n'
            'Compares models with one, two, or three ordered correlation times. '
            'Component weights are non-negative and sum to one. The raw parallel '
            'and perpendicular counts, separate IRFs, fixed fluorescence lifetime, '
            'shared G, exposures, periodic excitation, timing shift, and backgrounds '
            'are handled as in the preferred global fit. The reported model is the '
            'simplest candidate within 2 BIC units of the minimum. BIC alone does '
            'not prove distinct physical motions. No advanced candidate is labelled '
            'physically resolved until uncertainty and robustness validation are '
            'available. Numerical component values remain in Fit details only. '
            'Three-component values remain exploratory.\n\n'
            'G defaults to 1 as an uncalibrated assumption. For a better G guess, '
            'first use the effective late-window scale feature. That estimate is '
            'effective, not calibrated.\n\n'
            'The fitted r(0) is a time-zero model parameter. It is not '
            'automatically the fundamental anisotropy because very fast motion '
            'may be hidden by the IRF.\n\n'
            'Reference: Lakowicz, Principles of Fluorescence Spectroscopy, '
            'Chapter 11, Section 11.2.2, Preferred Analysis of TD Anisotropy Data.',
            parent=self)

    def _settings(self):
        parallel = Path(self.parallel_path.get()).expanduser()
        perpendicular = Path(self.perpendicular_path.get()).expanduser()
        if not parallel.is_file() or not perpendicular.is_file():
            raise ValueError('Choose two existing PTU files')
        if parallel.resolve() == perpendicular.resolve():
            raise ValueError('Parallel and perpendicular files must be different')
        analysis_mode = self.analysis_mode.get()
        if analysis_mode not in {'direct', 'global', 'advanced'}:
            raise ValueError('Choose a valid analysis method')
        parallel_irf = Path(self.parallel_irf_path.get()).expanduser()
        perpendicular_irf = Path(self.perpendicular_irf_path.get()).expanduser()
        if analysis_mode in {'global', 'advanced'}:
            if not parallel_irf.is_file() or not perpendicular_irf.is_file():
                raise ValueError(
                    'Global fitting requires parallel and perpendicular IRF files')
            if parallel_irf.resolve() == perpendicular_irf.resolve():
                raise ValueError('Choose a separate IRF file for each polarization')
        fixed_lifetime_ns = self.fixed_lifetime_ns.get()
        if (analysis_mode in {'global', 'advanced'}
                and (not np.isfinite(fixed_lifetime_ns)
                     or fixed_lifetime_ns <= 0)):
            raise ValueError(
                'Known fluorescence lifetime must be positive and finite')
        parallel_channel = self.parallel_channel.get()
        perpendicular_channel = self.perpendicular_channel.get()
        if parallel_channel < 0 or perpendicular_channel < 0:
            raise ValueError('Photon channels must be non-negative integers')
        background_start = self.background_start.get()
        background_stop = self.background_stop.get()
        if background_start < 0 or background_stop <= background_start:
            raise ValueError('Background bins must satisfy 0 <= start < stop')
        analysis_start = self.analysis_start_ns.get()
        analysis_stop = self.analysis_stop_ns.get()
        if (not np.isfinite(analysis_start) or analysis_start < 0
                or not np.isfinite(analysis_stop)
                or analysis_stop <= analysis_start):
            raise ValueError(
                'Post-peak times must be finite and satisfy 0 <= start < stop')
        window = self.spatial_window.get()
        stride = self.stride.get()
        if window < 1 or window % 2 == 0:
            raise ValueError('Spatial window must be a positive odd number')
        if stride < 1:
            raise ValueError('Stride must be positive')
        parallel_exposure = self.parallel_exposure.get()
        perpendicular_exposure = self.perpendicular_exposure.get()
        if (not np.isfinite(parallel_exposure) or parallel_exposure <= 0
                or not np.isfinite(perpendicular_exposure)
                or perpendicular_exposure <= 0):
            raise ValueError('Exposure values must be positive and finite')
        g_factor = self.g_factor.get()
        if not np.isfinite(g_factor) or g_factor <= 0:
            raise ValueError('Shared scale value must be positive and finite')
        g_mode_names = {
            'Assumed scale': 'assumed',
            'Calibrated G': 'calibrated',
            'Effective late-window scale': 'late_window',
        }
        g_mode = g_mode_names.get(self.g_mode.get(), self.g_mode.get())
        if g_mode not in {'assumed', 'calibrated', 'late_window'}:
            raise ValueError('Choose a valid shared scale source')
        late_window_start_ns = self.late_window_start_ns.get()
        if (not np.isfinite(late_window_start_ns)
                or late_window_start_ns < 0):
            raise ValueError('Late-window start must be finite and non-negative')
        min_bin_photons = self.min_bin_photons.get()
        min_map_photons = self.min_map_photons.get()
        if (not np.isfinite(min_bin_photons) or min_bin_photons < 0
                or not np.isfinite(min_map_photons)
                or min_map_photons < 0):
            raise ValueError('Photon thresholds must be finite and non-negative')
        max_components = self.max_components.get()
        multistart = self.multistart.get()
        if max_components not in {1, 2, 3}:
            raise ValueError('Maximum components must be 1, 2, or 3')
        if not 1 <= multistart <= 32:
            raise ValueError('Deterministic starts must be between 1 and at most 32')
        range_mode = self.component_range_mode.get()
        if range_mode not in {'Auto', 'Manual'}:
            raise ValueError('Choose Auto or Manual component time ranges')
        component_bounds_ns = None
        if analysis_mode == 'advanced' and range_mode == 'Manual':
            lower = self.component_lower_ns[0].get()
            upper = self.component_upper_ns[0].get()
            if (not np.isfinite(lower) or not np.isfinite(upper)
                    or lower <= 0 or lower >= upper):
                raise ValueError(
                    'Manual shared time range must contain positive increasing '
                    'bounds')
            component_bounds_ns = (float(lower), float(upper))
        fit_mode = analysis_mode in {'global', 'advanced'}
        return {
            'parallel_path': parallel,
            'perpendicular_path': perpendicular,
            'analysis_mode': analysis_mode,
            'parallel_irf_path': parallel_irf if fit_mode else None,
            'perpendicular_irf_path': perpendicular_irf if fit_mode else None,
            'fixed_lifetime_ns': fixed_lifetime_ns,
            'max_components': max_components,
            'multistart': multistart,
            'component_range_mode': range_mode.lower(),
            'component_bounds_ns': component_bounds_ns,
            'g_factor': g_factor,
            'g_mode': g_mode,
            'late_window_start_ns': late_window_start_ns,
            'parallel_exposure': parallel_exposure,
            'perpendicular_exposure': perpendicular_exposure,
            'parallel_channel': parallel_channel,
            'perpendicular_channel': perpendicular_channel,
            'background_bins': slice(background_start, background_stop),
            'analysis_start_ns': analysis_start,
            'analysis_stop_ns': analysis_stop,
            'spatial_window': window,
            'stride': stride,
            'min_bin_photons': min_bin_photons,
            'min_map_photons': min_map_photons,
            'auto_register': self.auto_register.get(),
        }

    def _clear_result_actions(self):
        self.result = None
        self.scale_diagnostic_button.configure(state='disabled')
        self.fit_details_button.configure(state='disabled')
        self.save_npz_button.configure(state='disabled')
        self.save_csv_button.configure(state='disabled')

    def _start_analysis(self):
        try:
            settings = self._settings()
        except Exception as exc:
            messagebox.showerror('Invalid settings', str(exc), parent=self)
            return
        self._clear_result_actions()
        self.calculate_button.configure(state='disabled')
        self.status.set('Reading and analysing PTUs...')
        self._result_queue = queue.Queue()
        threading.Thread(target=self._analysis_worker, args=(settings,),
                         daemon=True).start()
        self._poll_after_id = self.after(100, self._poll_worker)

    def _analysis_worker(self, settings):
        try:
            result, peak_bin = run_analysis(settings)
        except Exception as exc:
            self._result_queue.put(('error', exc))
            return
        self._result_queue.put(('success', result, peak_bin))

    def _poll_worker(self):
        self._poll_after_id = None
        if self._closed:
            return
        try:
            message = self._result_queue.get_nowait()
        except queue.Empty:
            self._poll_after_id = self.after(100, self._poll_worker)
            return
        if message[0] == 'error':
            self._analysis_failed(message[1])
        else:
            self._analysis_finished(*message[1:])

    def _close(self):
        self._closed = True
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self.destroy()

    def _analysis_failed(self, exc):
        self._clear_result_actions()
        self.calculate_button.configure(state='normal')
        self.status.set('Analysis failed.')
        messagebox.showerror('Anisotropy error', str(exc), parent=self)

    def _analysis_finished(self, result, peak_bin):
        self.result = result
        self.peak_bin = peak_bin
        self.calculate_button.configure(state='normal')
        diagnostic_state = (
            'normal' if getattr(result, 'late_window_stability', None) is not None
            else 'disabled')
        self.scale_diagnostic_button.configure(state=diagnostic_state)
        details_state = (
            'normal' if (
                getattr(result, 'multicomponent_fit', None) is not None
                and result.multicomponent_fit.selected_fit is not None)
            else 'disabled')
        self.fit_details_button.configure(state=details_state)
        self.save_npz_button.configure(state='normal')
        self.save_csv_button.configure(state='normal')
        shift_y, shift_x = result.perpendicular_shift
        status = f'Done. Shift: ({shift_y:.2f}, {shift_x:.2f}) px.'
        metadata = getattr(result, 'metadata', {})
        scale_source = metadata.get('shared_scale_source')
        applied_scale = getattr(result, 'g_factor', None)
        if scale_source == 'assumed' and applied_scale is not None:
            if np.isclose(applied_scale, 1.0):
                status += ' WARNING: G=1 is assumed, not calibrated.'
            else:
                status += f' Assumed scale {applied_scale:.4g}; not calibrated.'
        elif scale_source == 'late_window' and applied_scale is not None:
            status += (
                f' Effective scale {applied_scale:.4g}; '
                'not calibrated physical G.')
        elif scale_source == 'calibrated' and applied_scale is not None:
            status += f' User-declared calibrated G {applied_scale:.4g}.'
        comparison = getattr(result, 'multicomponent_fit', None)
        if comparison is not None:
            if comparison.selected_fit is None:
                status += ' No optimizer candidate converged. NOT RESOLVED.'
            else:
                status += (
                    f' BIC-selected {comparison.selected_component_count}-component '
                    'model.')
            if (comparison.selected_fit is not None
                    and not comparison.selected_fit.identifiable):
                status += ' WARNING: NOT RESOLVED; open Fit details.'
        self.status.set(status)
        self._set_inputs_visible(False)
        self._draw_result()

    def _draw_result(self):
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        for axis in self.axes.flat:
            axis.clear()
        if getattr(self.result, 'multicomponent_fit', None) is not None:
            self._draw_multicomponent_fit()
            self.canvas.draw_idle()
            return
        if getattr(self.result, 'polarized_fit', None) is not None:
            self._draw_global_fit()
            self.canvas.draw_idle()
            return
        self.figure.set_layout_engine('constrained')
        self.figure.get_layout_engine().set(w_pad=0.08, h_pad=0.05)
        self.axes[0, 0].imshow(self.result.parallel_intensity, cmap='gray')
        self.axes[0, 0].set_title('Parallel intensity')
        self.axes[0, 1].imshow(self.result.perpendicular_intensity, cmap='gray')
        self.axes[0, 1].set_title('Perpendicular intensity (registered)')
        for axis in self.axes[0]:
            axis.set_axis_off()

        time_relative = self.result.time_ns - self.result.time_ns[self.peak_bin]
        start_ns = self.result.metadata.get('analysis_start_ns', 0.0)
        stop_ns = self.result.metadata.get('analysis_stop_ns', 8.0)
        selected = (time_relative >= start_ns) & (time_relative < stop_ns)
        self.axes[1, 0].plot(
            time_relative[selected], self.result.anisotropy_decay[selected],
            color='#2468a2')
        self.axes[1, 0].set_xlim(start_ns, stop_ns)
        self.axes[1, 0].axhline(0.4, color='#999999', linestyle=':', linewidth=1)
        self.axes[1, 0].axhline(-0.2, color='#999999', linestyle=':', linewidth=1)
        self.axes[1, 0].set_xlabel('Time after peak (ns)')
        self.axes[1, 0].set_ylabel('Anisotropy r(t)')
        self.axes[1, 0].set_title('Summed anisotropy decay')

        finite_map = self.result.anisotropy_map[
            np.isfinite(self.result.anisotropy_map)]
        if finite_map.size:
            vmin = min(-0.2, float(np.percentile(finite_map, 2)))
            vmax = max(0.4, float(np.percentile(finite_map, 98)))
        else:
            vmin, vmax = -0.2, 0.4
        image = self.axes[1, 1].imshow(
            self.result.anisotropy_map, cmap='coolwarm', vmin=vmin, vmax=vmax)
        self.axes[1, 1].set_title(
            f'Anisotropy map\n{self.result.spatial_window}x'
            f'{self.result.spatial_window} window, stride {self.result.stride}')
        self.axes[1, 1].set_axis_off()
        self._colorbar = self.figure.colorbar(
            image, ax=self.axes[1, 1], fraction=0.04, pad=0.03,
            extend='both', location='left')
        self._style_plot_text()
        self.canvas.draw_idle()

    def _draw_multicomponent_fit(self):
        self.figure.set_layout_engine(None)
        self.figure.subplots_adjust(
            left=0.10, right=0.95, bottom=0.17, top=0.93,
            wspace=0.32, hspace=0.75)
        comparison = self.result.multicomponent_fit
        fit = comparison.selected_fit
        if fit is None:
            self.figure.set_layout_engine('constrained')
            for axis in self.axes.flat:
                axis.set_axis_off()
            self.axes[0, 0].text(
                0.5, 0.5,
                'No optimizer candidate converged\nNOT RESOLVED',
                ha='center', va='center', color='#b00020', fontsize=14,
                transform=self.axes[0, 0].transAxes)
            self._style_plot_text()
            return
        fit_bins = len(fit.parallel_model)
        time_relative = (
            self.result.time_ns[:fit_bins] - self.result.time_ns[self.peak_bin])
        parallel_observed = (
            self.result.parallel_decay[:fit_bins] + self.result.parallel_background)
        perpendicular_observed = (
            self.result.perpendicular_decay[:fit_bins]
            + self.result.perpendicular_background)
        channels = (
            (self.axes[0, 0], parallel_observed, fit.parallel_model,
             'Parallel advanced fit'),
            (self.axes[0, 1], perpendicular_observed, fit.perpendicular_model,
             'Perpendicular advanced fit'),
        )
        for axis, observed, model, title in channels:
            axis.semilogy(
                time_relative, np.maximum(observed, 1e-3),
                color='#777777', linewidth=1.2, label='Measured')
            axis.semilogy(
                time_relative, np.maximum(model, 1e-3),
                color='#2468a2', linewidth=2.0, label='Selected model')
            axis.set_title(title, fontsize=11)
            axis.set_xlabel('Time after peak (ns)', fontsize=9)
            axis.set_ylabel('Photon counts', fontsize=9)
            axis.tick_params(labelsize=8)
            axis.legend(
                fontsize=8, loc='upper right', bbox_to_anchor=(0.95, 0.98))

        self.axes[1, 0].plot(
            time_relative, fit.parallel_residual,
            color='#2468a2', linewidth=1.2, label='Parallel')
        self.axes[1, 0].plot(
            time_relative, fit.perpendicular_residual,
            color='#a34a28', linewidth=1.2, label='Perpendicular')
        self.axes[1, 0].axhline(0.0, color='#777777', linewidth=0.8)
        self.axes[1, 0].set_title('Residuals', fontsize=11)
        self.axes[1, 0].set_xlabel('Time after peak (ns)', fontsize=9)
        self.axes[1, 0].set_ylabel('Signed Poisson deviance', fontsize=9)
        self.axes[1, 0].tick_params(labelsize=8)
        self.axes[1, 0].legend(fontsize=8)

        summary_lines = [
            'Advanced anisotropy model comparison',
            f'Fixed fluorescence lifetime: {fit.intensity_lifetime_ns:.4g} ns',
            f'BIC-selected model: {comparison.selected_component_count} components',
        ]
        for candidate in comparison.candidates:
            state = 'resolved' if candidate.identifiable else 'not resolved'
            summary_lines.append(
                f'{candidate.component_count}-component: '
                f'BIC {candidate.bic:.4g} ({state})')
        if fit.identifiable:
            summary_lines.append(
                f'Time-zero anisotropy r(0): {fit.initial_anisotropy:.4g}')
            for index, (correlation_time, weight) in enumerate(zip(
                    fit.rotational_correlation_times_ns,
                    fit.component_weights), start=1):
                summary_lines.append(
                    f'Component {index}: {correlation_time:.4g} ns, '
                    f'weight {weight:.4g}')
            summary_lines.append('Physical resolution passed validated checks')
        else:
            summary_lines.append('No resolved rotational-component model')
            summary_lines.append('NOT RESOLVED — do not assign physical labels')
            for warning in fit.identifiability_warnings:
                summary_lines.append(f'WARNING: {warning}')
        summary_lines.append(f'Poisson deviance: {fit.poisson_deviance:.4g}')
        deviance_per_dof = getattr(
            fit, 'deviance_per_degree_of_freedom', None)
        if deviance_per_dof is not None:
            summary_lines.append(
                f'Deviance / degree of freedom: {deviance_per_dof:.4g}')
        summary_lines.extend([
            f'AICc: {fit.aicc:.4g}; BIC: {fit.bic:.4g}',
            f'Common IRF shift: {fit.common_irf_shift_bins:.4g} bins',
        ])
        scale_source = self.result.metadata.get(
            'shared_scale_source', 'assumed')
        if scale_source == 'late_window':
            summary_lines.extend([
                f'Effective late-window scale: {self.result.g_factor:.4g}',
                'WARNING: effective scale is not calibrated physical G',
            ])
        elif scale_source == 'calibrated':
            summary_lines.append(
                f'User-declared calibrated G: {self.result.g_factor:.4g}')
        elif np.isclose(self.result.g_factor, 1.0):
            summary_lines.append('WARNING: G=1 is assumed, not calibrated')
        else:
            summary_lines.extend([
                f'Uncalibrated assumed G: {self.result.g_factor:.4g}',
                'WARNING: assumed G is not calibrated',
            ])
        summary_lines.append(
            'Extra components are exploratory, not proof of distinct motions')
        summary = '\n'.join(summary_lines)
        summary_fontsize = 7.0 if len(summary_lines) <= 17 else 6.8
        self.axes[1, 1].set_position([0.60, 0.01, 0.37, 0.55])
        self.axes[1, 1].text(
            0.05, 0.95, summary, ha='left', va='top',
            fontsize=summary_fontsize,
            transform=self.axes[1, 1].transAxes)
        self.axes[1, 1].set_axis_off()
        self._style_plot_text()

    def _draw_global_fit(self):
        self.figure.set_layout_engine(None)
        self.figure.subplots_adjust(
            left=0.10, right=0.98, bottom=0.17, top=0.93,
            wspace=0.32, hspace=0.75)
        fit = self.result.polarized_fit
        fit_bins = len(fit.parallel_model)
        time_relative = (
            self.result.time_ns[:fit_bins] - self.result.time_ns[self.peak_bin])
        parallel_observed = (
            self.result.parallel_decay[:fit_bins] + self.result.parallel_background)
        perpendicular_observed = (
            self.result.perpendicular_decay[:fit_bins]
            + self.result.perpendicular_background)
        channels = (
            (self.axes[0, 0], parallel_observed, fit.parallel_model,
             'Parallel global fit'),
            (self.axes[0, 1], perpendicular_observed, fit.perpendicular_model,
             'Perpendicular global fit'),
        )
        for axis, observed, model, title in channels:
            axis.semilogy(time_relative, np.maximum(observed, 1e-3),
                          color='#777777', linewidth=1.2, label='Measured')
            axis.semilogy(time_relative, np.maximum(model, 1e-3),
                          color='#2468a2', linewidth=2.0, label='Global model')
            axis.set_title(title, fontsize=11)
            axis.set_xlabel('Time after peak (ns)', fontsize=9)
            axis.set_ylabel('Photon counts', fontsize=9)
            axis.tick_params(labelsize=8)
            axis.legend(fontsize=8)

        self.axes[1, 0].plot(
            time_relative, fit.parallel_residual,
            color='#2468a2', linewidth=1.2, label='Parallel')
        self.axes[1, 0].plot(
            time_relative, fit.perpendicular_residual,
            color='#a34a28', linewidth=1.2, label='Perpendicular')
        self.axes[1, 0].axhline(0.0, color='#777777', linewidth=0.8)
        self.axes[1, 0].set_title('Residuals', fontsize=11)
        self.axes[1, 0].set_xlabel('Time after peak (ns)', fontsize=9)
        self.axes[1, 0].set_ylabel('Observed - model', fontsize=9)
        self.axes[1, 0].tick_params(labelsize=8)
        self.axes[1, 0].legend(fontsize=8)

        summary_lines = [
            'Preferred global polarized-decay fit',
            f'Fixed fluorescence lifetime: {fit.intensity_lifetime_ns:.4g} ns',
            f'Rotational correlation: {fit.rotational_correlation_ns:.4g} ns',
            f'Resolved r(0): {fit.initial_anisotropy:.4g}',
            f'Common IRF shift: {fit.common_irf_shift_bins:.4g} bins',
            f'Fitted backgrounds: {fit.parallel_background:.4g}, '
            f'{fit.perpendicular_background:.4g}',
            f'Poisson deviance: {fit.poisson_deviance:.4g}',
        ]
        scale_source = self.result.metadata.get(
            'shared_scale_source', 'assumed')
        if scale_source == 'late_window':
            summary_lines.extend([
                f'Effective late-window scale: {self.result.g_factor:.4g}',
                'WARNING: effective scale is not calibrated physical G',
            ])
        elif scale_source == 'calibrated':
            summary_lines.extend([
                f'User-declared calibrated G: {self.result.g_factor:.4g}',
                'Calibration status supplied by user',
            ])
        else:
            summary_lines.extend([
                f'Uncalibrated assumed scale: {self.result.g_factor:.4g}',
                'WARNING: assumed scale is not calibrated',
            ])
        summary_lines.extend([
            'Lakowicz, Section 11.2.2',
            'Separate IRFs fitted simultaneously',
        ])
        if not getattr(fit, 'success', True):
            summary_lines.append('WARNING: optimizer did not converge:')
            summary_lines.extend(textwrap.wrap(
                str(getattr(fit, 'message', 'unknown reason')), width=48))
        parameters_at_bounds = getattr(fit, 'parameters_at_bounds', ())
        if parameters_at_bounds:
            summary_lines.append('WARNING: fit reached parameter bounds:')
            summary_lines.extend(textwrap.wrap(
                ', '.join(parameters_at_bounds), width=48))
        summary = '\n'.join(summary_lines)
        if len(summary_lines) <= 12:
            summary_fontsize = 8
        elif len(summary_lines) <= 15:
            summary_fontsize = 7
        else:
            summary_fontsize = 6
        self.axes[1, 1].set_position([0.60, 0.01, 0.37, 0.55])
        self.axes[1, 1].text(
            0.05, 0.95, summary, ha='left', va='top',
            fontsize=summary_fontsize,
            transform=self.axes[1, 1].transAxes)
        self.axes[1, 1].set_axis_off()
        self._style_plot_text()

    def _save_npz(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Save anisotropy result',
            defaultextension='.npz', filetypes=[('NPZ files', '*.npz')])
        if not path:
            return
        from .anisotropy import save_anisotropy_npz
        self.result.metadata['peak_bin'] = int(self.peak_bin)
        save_anisotropy_npz(self.result, path)
        self.status.set(f'Saved {Path(path).name}')

    def _save_csv(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Save summed anisotropy decay',
            defaultextension='.csv', filetypes=[('CSV files', '*.csv')])
        if not path:
            return
        metadata = self.result.metadata
        shift_y, shift_x = self.result.perpendicular_shift
        provenance = {
            'parallel_file': metadata.get('parallel_file', ''),
            'perpendicular_file': metadata.get('perpendicular_file', ''),
            'parallel_role': metadata.get('parallel_role', 'parallel'),
            'perpendicular_role': metadata.get('perpendicular_role', 'perpendicular'),
            'g_factor': self.result.g_factor,
            'parallel_exposure': self.result.parallel_exposure,
            'perpendicular_exposure': self.result.perpendicular_exposure,
            'parallel_channel': metadata.get('parallel_channel', ''),
            'perpendicular_channel': metadata.get('perpendicular_channel', ''),
            'background_start_bin': metadata.get('background_start_bin', ''),
            'background_stop_bin': metadata.get('background_stop_bin', ''),
            'analysis_start_ns': metadata.get('analysis_start_ns', ''),
            'analysis_stop_ns': metadata.get('analysis_stop_ns', ''),
            'min_bin_photons': metadata.get('min_bin_photons', ''),
            'min_map_photons': metadata.get('min_map_photons', ''),
            'shared_scale_source': metadata.get('shared_scale_source', ''),
            'shared_scale_interpretation': metadata.get(
                'shared_scale_interpretation', ''),
            'requested_shared_scale': metadata.get(
                'requested_shared_scale', ''),
            'applied_shared_scale': metadata.get(
                'applied_shared_scale', self.result.g_factor),
            'late_window_selected_start_ns': metadata.get(
                'late_window_selected_start_ns', ''),
            'analysis_mode': metadata.get('analysis_mode', ''),
            'parallel_irf_file': metadata.get('parallel_irf_file', ''),
            'perpendicular_irf_file': metadata.get(
                'perpendicular_irf_file', ''),
            'repetition_period_ns': metadata.get('repetition_period_ns', ''),
            'global_fit_bins': metadata.get('global_fit_bins', ''),
            'advanced_component_range_mode': metadata.get(
                'advanced_component_range_mode', ''),
            'perpendicular_shift_y': shift_y,
            'perpendicular_shift_x': shift_x,
            'spatial_window': self.result.spatial_window,
            'stride': self.result.stride,
        }
        fieldnames = [
            'time_ns', 'time_after_peak_ns', 'parallel', 'perpendicular',
            'anisotropy', 'valid',
        ]
        fit = getattr(self.result, 'polarized_fit', None)
        comparison = getattr(self.result, 'multicomponent_fit', None)
        if comparison is not None:
            fit = comparison.selected_fit
            provenance.update({
                'advanced_max_components': comparison.max_components,
                'advanced_selected_component_count': (
                    comparison.selected_component_count),
                'advanced_residual_type': 'signed_poisson_deviance',
            })
            for candidate in comparison.candidates:
                prefix = f'advanced_candidate_{candidate.component_count}'
                provenance.update({
                    f'{prefix}_times_ns': ';'.join(
                        f'{value:g}' for value in
                        candidate.rotational_correlation_times_ns),
                    f'{prefix}_weights': ';'.join(
                        f'{value:g}' for value in candidate.component_weights),
                    f'{prefix}_intensity_lifetime_ns': getattr(
                        candidate, 'intensity_lifetime_ns', ''),
                    f'{prefix}_initial_anisotropy': (
                        candidate.initial_anisotropy),
                    f'{prefix}_amplitude': candidate.amplitude,
                    f'{prefix}_poisson_deviance': candidate.poisson_deviance,
                    f'{prefix}_degrees_of_freedom': getattr(
                        candidate, 'degrees_of_freedom', ''),
                    f'{prefix}_deviance_per_degree_of_freedom': getattr(
                        candidate, 'deviance_per_degree_of_freedom', ''),
                    f'{prefix}_bic': candidate.bic,
                    f'{prefix}_aicc': candidate.aicc,
                    f'{prefix}_success': candidate.success,
                    f'{prefix}_eligible_for_selection': getattr(
                        candidate, 'eligible_for_selection', candidate.success),
                    f'{prefix}_message': candidate.message,
                    f'{prefix}_parameters_at_bounds': ';'.join(
                        candidate.parameters_at_bounds),
                    f'{prefix}_identifiable': candidate.identifiable,
                    f'{prefix}_warnings': ' | '.join(
                        candidate.identifiability_warnings),
                    f'{prefix}_component_bounds_ns': ';'.join(
                        f'{lower:g}:{upper:g}' for lower, upper in
                        candidate.component_bounds_ns),
                    f'{prefix}_common_irf_shift_bins': (
                        candidate.common_irf_shift_bins),
                    f'{prefix}_parallel_background': (
                        candidate.parallel_background),
                    f'{prefix}_perpendicular_background': (
                        candidate.perpendicular_background),
                    f'{prefix}_multistart_count': candidate.multistart_count,
                    f'{prefix}_parameter_names': ';'.join(getattr(
                        candidate, 'parameter_names', ())),
                })
                for record in candidate.multistart_records:
                    start_prefix = f'{prefix}_start_{record.start_index}'
                    provenance.update({
                        f'{start_prefix}_parameter_vector': ';'.join(
                            f'{value:g}' for value in record.parameter_vector),
                        f'{start_prefix}_times_ns': ';'.join(
                            f'{value:g}' for value in
                            record.rotational_correlation_times_ns),
                        f'{start_prefix}_weights': ';'.join(
                            f'{value:g}' for value in
                            record.component_weights),
                        f'{start_prefix}_poisson_deviance': (
                            record.poisson_deviance),
                        f'{start_prefix}_success': record.success,
                        f'{start_prefix}_message': record.message,
                        f'{start_prefix}_parameters_at_bounds': ';'.join(
                            record.parameters_at_bounds),
                    })
        stability = getattr(self.result, 'late_window_stability', None)
        late_fields = []
        if stability is not None:
            late_fields = [
                'late_window_nested_scale', 'late_window_rolling_scale',
                'late_window_nested_standard_error',
            ]
            provenance.update({
                'late_window_selected_scale': stability.selected_scale,
                'late_window_selected_parallel_photons': (
                    stability.selected_parallel_photons),
                'late_window_selected_perpendicular_photons': (
                    stability.selected_perpendicular_photons),
            })
        fit_fields = []
        if fit is not None:
            fit_fields = [
                'parallel_observed', 'perpendicular_observed',
                'parallel_model', 'perpendicular_model',
                'parallel_residual', 'perpendicular_residual',
            ]
            provenance.update({
                'intensity_lifetime_ns': fit.intensity_lifetime_ns,
                'initial_anisotropy': fit.initial_anisotropy,
                'common_irf_shift_bins': fit.common_irf_shift_bins,
                'parallel_fit_background': fit.parallel_background,
                'perpendicular_fit_background': fit.perpendicular_background,
                'poisson_deviance': fit.poisson_deviance,
            })
            if comparison is None:
                provenance['rotational_correlation_ns'] = (
                    fit.rotational_correlation_ns)
        fieldnames.extend([*fit_fields, *late_fields, *provenance])
        peak_time = self.result.time_ns[self.peak_bin]
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, (time_ns, parallel, perpendicular, anisotropy) in enumerate(zip(
                    self.result.time_ns, self.result.parallel_decay,
                    self.result.perpendicular_decay,
                    self.result.anisotropy_decay)):
                fit_values = {}
                if fit is not None and index < len(fit.parallel_model):
                    fit_values = {
                        'parallel_observed': (
                            self.result.parallel_decay[index]
                            + self.result.parallel_background),
                        'perpendicular_observed': (
                            self.result.perpendicular_decay[index]
                            + self.result.perpendicular_background),
                        'parallel_model': fit.parallel_model[index],
                        'perpendicular_model': fit.perpendicular_model[index],
                        'parallel_residual': fit.parallel_residual[index],
                        'perpendicular_residual': fit.perpendicular_residual[index],
                    }
                late_values = {}
                if stability is not None and index < len(stability.nested_scale):
                    late_values = {
                        'late_window_nested_scale': stability.nested_scale[index],
                        'late_window_rolling_scale': stability.rolling_scale[index],
                        'late_window_nested_standard_error': (
                            stability.nested_standard_error[index]),
                    }
                row = {
                    'time_ns': time_ns,
                    'time_after_peak_ns': time_ns - peak_time,
                    'parallel': parallel,
                    'perpendicular': perpendicular,
                    'anisotropy': anisotropy,
                    'valid': np.isfinite(anisotropy),
                    **fit_values,
                    **late_values,
                    **provenance,
                }
                writer.writerow({
                    key: _spreadsheet_safe(value)
                    for key, value in row.items()})
        self.status.set(f'Saved {Path(path).name}')


def load_irf_curve(path, n_bins, tcspc_res, expected_period_ns=None,
                   ptu_channel=None, ptu_reader_class=None):
    path = Path(path)
    if path.suffix.lower() == '.ptu':
        if ptu_reader_class is None:
            from flimkit.formats.PTU.reader import PTUFile
            ptu_reader_class = PTUFile
        with ptu_reader_class(path, verbose=False) as irf_file:
            if not np.isclose(
                    irf_file.tcspc_res, tcspc_res, rtol=1e-6, atol=0.0):
                raise ValueError(
                    'IRF PTU timing resolution does not match the sample PTU')
            if (expected_period_ns is not None
                    and not np.isclose(
                        irf_file.period_ns, expected_period_ns,
                        rtol=1e-6, atol=0.0)):
                raise ValueError(
                    'IRF PTU laser period does not match the sample PTU')
            decay = np.asarray(
                irf_file.summed_decay(channel=ptu_channel), dtype=float)
        if (decay.ndim != 1 or np.any(~np.isfinite(decay))
                or np.any(decay < 0) or decay.sum() <= 0):
            raise ValueError(
                'IRF PTU histogram must be finite, non-negative, and non-zero')
        if abs(decay.size - n_bins) > 1:
            raise ValueError(
                'IRF PTU bin count differs by more than one bin from the sample')
        curve = np.zeros(n_bins, dtype=float)
        curve[:min(n_bins, decay.size)] = decay[:n_bins]
        if curve.sum() <= 0:
            raise ValueError(
                'IRF PTU histogram must be finite, non-negative, and non-zero')
        return curve

    from flimkit.FLIM.irf_tools import irf_from_xlsx
    from flimkit.utils.xlsx_tools import load_irf_export

    exported = load_irf_export(path, debug=False)
    return irf_from_xlsx(exported, n_bins, tcspc_res)


def run_analysis(settings):
    from flimkit.formats.PTU.reader import PTUFile

    from .anisotropy import (
        analyze_anisotropy,
        calculate_late_window_stability,
        estimate_translation,
        fit_multicomponent_polarized_decays,
        fit_polarized_decays,
    )

    with PTUFile(settings['parallel_path'], verbose=False) as parallel_file:
        parallel = parallel_file.pixel_stack(
            channel=settings['parallel_channel'])
        time_ns = np.asarray(parallel_file.time_ns, dtype=float)
        parallel_period_ns = getattr(parallel_file, 'period_ns', None)
        tcspc_res = getattr(parallel_file, 'tcspc_res', None)
    with PTUFile(settings['perpendicular_path'], verbose=False) as perpendicular_file:
        perpendicular = perpendicular_file.pixel_stack(
            channel=settings['perpendicular_channel'])
        perpendicular_time = np.asarray(perpendicular_file.time_ns, dtype=float)
        perpendicular_period_ns = getattr(perpendicular_file, 'period_ns', None)
    if parallel.shape != perpendicular.shape:
        raise ValueError('Polarization PTUs must have matching image and time shapes')
    if not np.allclose(time_ns, perpendicular_time):
        raise ValueError('Polarization PTUs must have matching time axes')
    analysis_mode = settings.get('analysis_mode', 'direct')
    scale_source = settings.get('g_mode', 'assumed')
    interpretations = {
        'assumed': 'uncalibrated_assumption',
        'calibrated': 'user_declared_calibrated_g',
        'late_window': 'effective_late_window_scale',
    }
    if scale_source not in interpretations:
        raise ValueError('Choose a valid shared scale source')

    global_fit_bins = None
    repetition_period_ns = None
    period_bins = None
    period_error = None
    periods = np.asarray(
        [parallel_period_ns, perpendicular_period_ns], dtype=object)
    if any(value is None for value in periods):
        period_error = 'laser period is missing from a polarization PTU'
    else:
        numeric_periods = np.asarray(periods, dtype=float)
        if np.any(~np.isfinite(numeric_periods)) or np.any(numeric_periods <= 0):
            period_error = 'laser period must be finite and positive'
        elif not np.isclose(numeric_periods[0], numeric_periods[1]):
            period_error = 'Polarization PTUs must have matching laser periods'
        elif time_ns.size < 2:
            period_error = 'stored TCSPC time axis is too short'
        else:
            time_step_ns = float(np.median(np.diff(time_ns)))
            candidate_bins = int(round(float(numeric_periods[0]) / time_step_ns))
            if candidate_bins < 2 or candidate_bins > time_ns.size:
                period_error = (
                    'Laser period is incompatible with the stored TCSPC time axis')
            else:
                repetition_period_ns = float(numeric_periods[0])
                period_bins = candidate_bins

    fit_required = analysis_mode in {'global', 'advanced'}
    period_required = fit_required or scale_source == 'late_window'
    if period_required and period_error is not None:
        if ('missing' in period_error
                or 'finite and positive' in period_error):
            if scale_source == 'late_window':
                raise ValueError(
                    'Effective late-window scale requires a laser period')
            raise ValueError('Preferred global fitting requires a laser period')
        raise ValueError(period_error)
    if fit_required:
        if tcspc_res is None or not np.isfinite(tcspc_res) or tcspc_res <= 0:
            raise ValueError('Global fitting requires TCSPC resolution')
        if period_bins is None:
            raise ValueError('Global fitting requires a valid laser period')
        global_fit_bins = period_bins

    late_window_stability = None
    diagnostic_error = period_error
    if period_bins is not None:
        try:
            late_window_stability = calculate_late_window_stability(
                parallel.sum(axis=(0, 1))[:period_bins],
                perpendicular.sum(axis=(0, 1))[:period_bins],
                time_ns[:period_bins],
                parallel_exposure=settings['parallel_exposure'],
                perpendicular_exposure=settings['perpendicular_exposure'],
                selected_start_ns=settings.get(
                    'late_window_start_ns', float(time_ns[0])))
            diagnostic_error = None
        except ValueError as exc:
            if scale_source == 'late_window':
                raise
            diagnostic_error = str(exc)
    requested_scale = float(settings['g_factor'])
    if scale_source == 'late_window':
        if late_window_stability is None:
            raise RuntimeError('Late-window scale diagnostic was not calculated')
        applied_scale = late_window_stability.selected_scale
    else:
        applied_scale = requested_scale

    combined_decay = (
        parallel.sum(axis=(0, 1)) / settings['parallel_exposure']
        + 2.0 * applied_scale * perpendicular.sum(axis=(0, 1))
        / settings['perpendicular_exposure'])
    peak_bin = int(np.argmax(combined_decay))
    time_relative = time_ns - time_ns[peak_bin]
    selected = ((time_relative >= settings['analysis_start_ns'])
                & (time_relative < settings['analysis_stop_ns']))
    selected_bins = np.flatnonzero(selected)
    if selected_bins.size == 0:
        raise ValueError('Post-peak range contains no TCSPC bins')
    analysis_bins = slice(int(selected_bins[0]), int(selected_bins[-1]) + 1)

    shift_yx = (0.0, 0.0)
    if settings['auto_register']:
        shift_yx = estimate_translation(
            parallel.sum(axis=-1), perpendicular.sum(axis=-1))
    result = analyze_anisotropy(
        parallel, perpendicular, time_ns,
        background_bins=settings['background_bins'],
        analysis_bins=analysis_bins, g_factor=applied_scale,
        spatial_window=settings['spatial_window'], stride=settings['stride'],
        min_bin_photons=settings['min_bin_photons'],
        min_map_photons=settings['min_map_photons'],
        perpendicular_shift=shift_yx,
        parallel_exposure=settings['parallel_exposure'],
        perpendicular_exposure=settings['perpendicular_exposure'])
    result.g_factor = applied_scale
    result.late_window_stability = late_window_stability
    if fit_required:
        if global_fit_bins is None or repetition_period_ns is None:
            raise ValueError('Global fitting requires a valid laser period')
        fit_time_ns = time_ns[:global_fit_bins]
        parallel_irf = load_irf_curve(
            settings['parallel_irf_path'], global_fit_bins, tcspc_res,
            expected_period_ns=repetition_period_ns,
            ptu_channel=settings['parallel_channel'])
        perpendicular_irf = load_irf_curve(
            settings['perpendicular_irf_path'], global_fit_bins, tcspc_res,
            expected_period_ns=repetition_period_ns,
            ptu_channel=settings['perpendicular_channel'])
        fit_arguments = (
            parallel.sum(axis=(0, 1))[:global_fit_bins],
            perpendicular.sum(axis=(0, 1))[:global_fit_bins],
            fit_time_ns)
        fit_keywords = {
            'parallel_irf': parallel_irf,
            'perpendicular_irf': perpendicular_irf,
            'intensity_lifetime_ns': settings['fixed_lifetime_ns'],
            'g_factor': applied_scale,
            'parallel_exposure': settings['parallel_exposure'],
            'perpendicular_exposure': settings['perpendicular_exposure'],
            'initial_parallel_background': result.parallel_background,
            'initial_perpendicular_background': result.perpendicular_background,
            'repetition_period_ns': repetition_period_ns,
        }
        if analysis_mode == 'global':
            result.polarized_fit = fit_polarized_decays(
                *fit_arguments, **fit_keywords)
        else:
            result.multicomponent_fit = fit_multicomponent_polarized_decays(
                *fit_arguments, **fit_keywords,
                max_components=settings.get('max_components', 1),
                multistart=settings.get('multistart', 6),
                component_bounds_ns=settings.get('component_bounds_ns'))
    result.metadata.update({
        'parallel_file': Path(settings['parallel_path']).name,
        'perpendicular_file': Path(settings['perpendicular_path']).name,
        'parallel_role': 'parallel',
        'perpendicular_role': 'perpendicular',
        'parallel_channel': settings['parallel_channel'],
        'perpendicular_channel': settings['perpendicular_channel'],
        'background_start_bin': settings['background_bins'].start,
        'background_stop_bin': settings['background_bins'].stop,
        'analysis_start_ns': settings['analysis_start_ns'],
        'analysis_stop_ns': settings['analysis_stop_ns'],
        'min_bin_photons': settings['min_bin_photons'],
        'min_map_photons': settings['min_map_photons'],
        'auto_registration': settings['auto_register'],
        'analysis_mode': analysis_mode,
        'shared_scale_source': scale_source,
        'shared_scale_interpretation': interpretations[scale_source],
        'requested_shared_scale': requested_scale,
        'applied_shared_scale': applied_scale,
        'late_window_requested_start_ns': settings.get(
            'late_window_start_ns', float(time_ns[0])),
        'late_window_diagnostic_available': late_window_stability is not None,
        'late_window_diagnostic_error': diagnostic_error or '',
        'late_window_background_treatment': 'none',
        'late_window_previous_pulse_fluorescence_included': True,
    })
    if late_window_stability is not None:
        result.metadata.update({
            'late_window_selected_start_ns': float(
                late_window_stability.time_ns[
                    late_window_stability.selected_start_bin]),
            'late_window_parallel_photons': (
                late_window_stability.selected_parallel_photons),
            'late_window_perpendicular_photons': (
                late_window_stability.selected_perpendicular_photons),
            'late_window_poisson_standard_error': float(
                late_window_stability.nested_standard_error[
                    late_window_stability.selected_start_bin]),
        })
    if fit_required:
        result.metadata.update({
            'parallel_irf_file': Path(settings['parallel_irf_path']).name,
            'perpendicular_irf_file': Path(
                settings['perpendicular_irf_path']).name,
            'repetition_period_ns': repetition_period_ns,
            'global_fit_bins': global_fit_bins,
            'fixed_lifetime_ns': settings['fixed_lifetime_ns'],
            'global_fit_reference': 'Lakowicz, Chapter 11, Section 11.2.2',
        })
    if analysis_mode == 'global':
        result.metadata['global_fit_model'] = (
            'fixed single fluorescence lifetime; '
            'single rotational correlation')
    elif analysis_mode == 'advanced':
        result.metadata.update({
            'global_fit_model': (
                'fixed single fluorescence lifetime; compared ordered '
                'one-to-three-component rotational correlations'),
            'advanced_max_components': settings.get('max_components', 1),
            'advanced_multistart': settings.get('multistart', 6),
            'advanced_component_range_mode': settings.get(
                'component_range_mode', 'auto'),
            'advanced_component_bounds_ns': settings.get(
                'component_bounds_ns') or [],
        })
    return result, peak_bin


def show_anisotropy_tool(parent):
    return AnisotropyTool(parent)
