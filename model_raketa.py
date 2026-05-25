import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from scipy.integrate import solve_ivp
import json
import os
from datetime import datetime

# ============================================================
# КЛАСС РАКЕТНОГО СИМУЛЯТОРА
# ============================================================

class RocketSimulator:
    def __init__(self):
        self.params = {'m0': 1000.0, 'mk': 300.0, 'u': 2000.0, 'mdot': 50.0, 'Cx': 0.75, 'S': 0.1, 'theta0': 90.0, 'g0': 9.81, 'Rz': 6371000.0, 'rho0': 1.225, 'H': 8500.0}
        self.results = None
        self.time_span = (0, 30)
        self.dt = 0.01
    def atmosphere_density(self, h):
        return self.params['rho0'] * np.exp(-h / self.params['H'])
    def gravity(self, h):
        return self.params['g0'] * (self.params['Rz'] / (self.params['Rz'] + h))**2
    def rocket_equations(self, t, y):
        h, v, m = y
        if h < 0:
            h = 0
        rho = self.atmosphere_density(h)
        g = self.gravity(h)
        has_fuel = m > self.params['mk'] + 1e-6
        mdot_current = self.params['mdot'] if has_fuel else 0
        thrust = self.params['u'] * mdot_current if has_fuel else 0
        drag = 0.5 * rho * v * abs(v) * self.params['Cx'] * self.params['S']
        theta_rad = np.radians(self.params['theta0'])
        dhdt = v * np.sin(theta_rad)
        dmdt = -mdot_current
        if m > 1e-6:
            dvdt = (thrust - m * g - drag) / m
        else:
            dvdt = -g
        return [dhdt, dvdt, dmdt]
    def run_simulation(self):
        y0 = [0.0, 0.0, self.params['m0']]
        t_eval = np.arange(self.time_span[0], self.time_span[1] + self.dt, self.dt)
        sol = solve_ivp(self.rocket_equations, self.time_span, y0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-9)
        self.results = {'time': sol.t, 'height': sol.y[0], 'velocity': sol.y[1], 'mass': sol.y[2], 'thrust': np.zeros_like(sol.t), 'drag': np.zeros_like(sol.t), 'gravity': np.zeros_like(sol.t)}
        for i, t in enumerate(sol.t):
            h = sol.y[0, i]
            v = sol.y[1, i]
            m = sol.y[2, i]
            rho = self.atmosphere_density(h)
            g = self.gravity(h)
            has_fuel = m > self.params['mk'] + 1e-6
            mdot_current = self.params['mdot'] if has_fuel else 0
            self.results['thrust'][i] = self.params['u'] * mdot_current if has_fuel else 0
            self.results['drag'][i] = 0.5 * rho * v * abs(v) * self.params['Cx'] * self.params['S']
            self.results['gravity'][i] = m * g
        max_height_idx = np.argmax(self.results['height'])
        self.results['max_height'] = self.results['height'][max_height_idx]
        self.results['max_height_time'] = self.results['time'][max_height_idx]
        self.results['max_velocity'] = np.max(self.results['velocity'])
        self.results['engine_cutoff_time'] = (self.params['m0'] - self.params['mk']) / self.params['mdot']
        return self.results
    def plot_results(self, fig=None, theme='light'):
        if self.results is None:
            raise ValueError("Нет результатов для визуализации. Запустите симуляцию.")
        if fig is None:
            fig = plt.figure(figsize=(12, 8))
        if theme == 'dark':
            plt.style.use('dark_background')
            line_color = '#7392B5'
            grid_color = '#402B47'
            text_color = '#C9E0EB'
            bg_color = '#262423'
        else:
            plt.style.use('default')
            line_color = '#7392B5'
            grid_color = '#8350C4'
            text_color = '#262423'
            bg_color = '#C9E0EB'
        fig.patch.set_facecolor(bg_color)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor(bg_color)
        ax1.plot(self.results['time'], self.results['height'], color=line_color, linewidth=2)
        ax1.set_xlabel('Время (с)', fontsize=11, color=text_color)
        ax1.set_ylabel('Высота (м)', fontsize=11, color=text_color)
        ax1.set_title('Зависимость высоты от времени', fontsize=12, fontweight='bold', color=text_color)
        ax1.grid(True, color=grid_color, alpha=0.3)
        ax1.axvline(x=self.results['max_height_time'], color='#FF4C00', linestyle='--', label=f'Макс. высота: {self.results["max_height"]:.1f} м')
        ax1.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor(bg_color)
        ax2.plot(self.results['time'], self.results['velocity'], color='#FFE900', linewidth=2)
        ax2.set_xlabel('Время (с)', fontsize=11, color=text_color)
        ax2.set_ylabel('Скорость (м/с)', fontsize=11, color=text_color)
        ax2.set_title('Зависимость скорости от времени', fontsize=12, fontweight='bold', color=text_color)
        ax2.grid(True, color=grid_color, alpha=0.3)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor(bg_color)
        ax3.plot(self.results['time'], self.results['mass'], color='#00CB79', linewidth=2)
        ax3.set_xlabel('Время (с)', fontsize=11, color=text_color)
        ax3.set_ylabel('Масса (кг)', fontsize=11, color=text_color)
        ax3.set_title('Изменение массы со временем', fontsize=12, fontweight='bold', color=text_color)
        ax3.grid(True, color=grid_color, alpha=0.3)
        cutoff_time = (self.params['m0'] - self.params['mk']) / self.params['mdot']
        ax3.axvline(x=cutoff_time, color='#FF4C00', linestyle='--', label='Окончание топлива')
        ax3.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor(bg_color)
        ax4.plot(self.results['time'], self.results['thrust'], label='Тяга', color='#FF4C00', linewidth=2)
        ax4.plot(self.results['time'], self.results['drag'], label='Сопротивление', color='#FFE900', linewidth=2)
        ax4.plot(self.results['time'], self.results['gravity'], label='Сила тяжести', color='#5900CA', linewidth=2)
        ax4.set_xlabel('Время (с)', fontsize=11, color=text_color)
        ax4.set_ylabel('Сила (Н)', fontsize=11, color=text_color)
        ax4.set_title('Действующие силы', fontsize=12, fontweight='bold', color=text_color)
        ax4.grid(True, color=grid_color, alpha=0.3)
        ax4.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)
        fig.tight_layout()
        return fig

# ============================================================
# КЛАСС ГРАФИЧЕСКОГО ИНТЕРФЕЙСА
# ============================================================

class RocketSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Симулятор Ракеты v1.0")
        # Настройка темы
        self.current_theme = 'light'
        self.settings_file = 'rocket_sim_settings.json'
        # Загрузка настроек
        self.load_settings()
        # Настройка стилей
        self.setup_styles()
        # Установка размера окна
        self.root.geometry("1300x900")
        self.root.minsize(1200, 800)
        # Центрирование окна
        self.center_window()
        # Инициализируем экземпляр симулятора
        self.simulator = RocketSimulator()
        # Создание интерфейса
        self.create_menu()
        self.create_main_interface()
        self.create_status_bar()
        # Переменные для анимации
        self.animation = None
        self.animation_running = False
        self.current_frame = 0
        self.anim_data = None
        self.base_interval = 10  # Базовый интервал для анимации
        # Бинды клавиатуры
        self.root.bind('<F5>', lambda e: self.run_simulation())
        self.root.bind('<Control-s>', lambda e: self.export_results())
        self.root.bind('<Control-r>', lambda e: self.reset_parameters())
        self.root.bind('<Escape>', lambda e: self.quit_application())
        self.root.bind('<F1>', lambda e: self.show_docs())
        # Улучшенные настройки окна
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
    def center_window(self):
        """Центрирование окна на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    def center_child_window(self, child_window):
        """Центрирование дочернего окна относительно главного"""
        child_window.update_idletasks()
        width = child_window.winfo_width()
        height = child_window.winfo_height()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (height // 2)
        child_window.geometry(f'{width}x{height}+{x}+{y}')
    def setup_styles(self):
        """Настройка стилей приложения"""
        style = ttk.Style()
        if self.current_theme == 'dark':
            # Темная тема
            bg_color = '#262423'
            fg_color = '#8350C4'
            entry_bg = '#C9E0EB'
            button_bg = '#C9E0EB'
            button_fg = '#262423'
            accent_color = '#7392B5'
            frame_bg = '#402B47'
            self.root.configure(bg=bg_color)
            style.theme_use('clam')
            style.configure('.', background=bg_color, foreground=fg_color)
            style.configure('TFrame', background=bg_color)
            style.configure('TLabel', background=bg_color, foreground=fg_color, font=('Segoe UI', 10, 'normal'))
            style.configure('TButton', background=button_bg, foreground=button_fg, font=('Segoe UI', 10, 'normal'), borderwidth=1)
            style.map('TButton', background=[('active', accent_color), ('pressed', frame_bg)], foreground=[('active', button_fg), ('pressed', button_fg)])
            style.configure('TEntry', fieldbackground=entry_bg, foreground=fg_color, borderwidth=1, insertcolor=fg_color)
            style.configure('TLabelframe', background=frame_bg, foreground=accent_color, borderwidth=2, relief='solid')
            style.configure('TLabelframe.Label', background=frame_bg, foreground=accent_color, font=('Segoe UI', 11, 'bold'))
            style.configure('TNotebook', background=bg_color, borderwidth=0)
            style.configure('TNotebook.Tab', background=button_bg, foreground=button_fg, padding=[12, 6], font=('Segoe UI', 10, 'normal'))
            style.map('TNotebook.Tab', background=[('selected', accent_color), ('active', frame_bg)], foreground=[('selected', button_fg), ('active', button_fg)])
            style.configure('Horizontal.TProgressbar', background=accent_color, troughcolor=entry_bg)
            style.configure('TScale', background=bg_color)
            style.configure('Accent.TButton', background=accent_color, foreground=button_fg, font=('Segoe UI', 11, 'bold'))
            style.map('Accent.TButton', background=[('active', frame_bg), ('pressed', frame_bg)], foreground=[('active', button_fg), ('pressed', button_fg)])
        else:
            # Светлая тема
            bg_color = '#C9E0EB'
            fg_color = '#262423'
            entry_bg = '#402B47'
            button_bg = '#402B47'
            button_fg = '#262423'
            accent_color = '#7392B5'
            frame_bg = '#8350C4'
            self.root.configure(bg=bg_color)
            style.theme_use('vista')
            style.configure('.', background=bg_color, foreground=fg_color)
            style.configure('TFrame', background=bg_color)
            style.configure('TLabel', background=bg_color, foreground=fg_color, font=('Segoe UI', 10, 'normal'))
            style.configure('TButton', background=button_bg, foreground=button_fg, font=('Segoe UI', 10, 'normal'), borderwidth=1)
            style.map('TButton', background=[('active', accent_color), ('pressed', frame_bg)], foreground=[('active', button_fg), ('pressed', button_fg)])
            style.configure('TEntry', fieldbackground=entry_bg, foreground=fg_color, borderwidth=1, insertcolor=fg_color)
            style.configure('TLabelframe', background=frame_bg, foreground=accent_color, borderwidth=2, relief='solid')
            style.configure('TLabelframe.Label', background=frame_bg, foreground=accent_color, font=('Segoe UI', 11, 'bold'))
            style.configure('TNotebook', background=bg_color, borderwidth=0)
            style.configure('TNotebook.Tab', background=button_bg, foreground=button_fg, padding=[12, 6], font=('Segoe UI', 10, 'normal'))
            style.map('TNotebook.Tab', background=[('selected', accent_color), ('active', frame_bg)], foreground=[('selected', button_fg), ('active', button_fg)])
            style.configure('Horizontal.TProgressbar', background=accent_color, troughcolor=entry_bg)
            style.configure('TScale', background=bg_color)
            style.configure('Accent.TButton', background=accent_color, foreground=button_fg, font=('Segoe UI', 11, 'bold'))
            style.map('Accent.TButton', background=[('active', frame_bg), ('pressed', frame_bg)], foreground=[('active', button_fg), ('pressed', button_fg)])
    def create_menu(self):
        """Создание меню приложения"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="📁 Файл", menu=file_menu)
        file_menu.add_command(label="🚀 Новая симуляция", command=self.run_simulation, accelerator="F5")
        file_menu.add_command(label="💾 Экспорт результатов", command=self.export_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="📊 Экспорт графиков", command=self.export_all_plots)
        file_menu.add_separator()
        file_menu.add_command(label="🚪 Выход", command=self.quit_application, accelerator="Esc")
        # Меню Настройки
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="⚙ Настройки", menu=settings_menu)
        theme_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label="🎨 Тема интерфейса", menu=theme_menu)
        theme_menu.add_command(label="☀️ Светлая", command=lambda: self.change_theme('light'))
        theme_menu.add_command(label="🌙 Темная", command=lambda: self.change_theme('dark'))
        settings_menu.add_separator()
        settings_menu.add_command(label="💾 Сохранить настройки", command=self.save_settings)
        settings_menu.add_command(label="📂 Загрузить настройки", command=self.load_settings_from_file)
        settings_menu.add_separator()
        settings_menu.add_command(label="🔧 Расширенные настройки", command=self.show_advanced_settings)
        # Меню Справка
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="❓ Справка", menu=help_menu)
        help_menu.add_command(label="ℹ️ О программе", command=self.show_about)
        help_menu.add_command(label="📚 Документация", command=self.show_docs, accelerator="F1")
        help_menu.add_separator()
        help_menu.add_command(label="🎬 Руководство по анимации", command=self.show_animation_guide)
        help_menu.add_command(label="📊 Анализ результатов", command=self.show_analysis_guide)
    def create_main_interface(self):
        """Создание основного интерфейса"""
        # Создаем панель с вкладками с улучшенным стилем
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        # Создаем фреймы для вкладок
        self.params_frame = ttk.Frame(notebook)
        self.results_frame = ttk.Frame(notebook)
        self.plots_frame = ttk.Frame(notebook)
        self.animation_frame = ttk.Frame(notebook)
        # Добавляем вкладки с иконками
        notebook.add(self.params_frame, text='⚙ Параметры ракеты')
        notebook.add(self.results_frame, text='📊 Результаты симуляции')
        notebook.add(self.plots_frame, text='📈 Графики и анализ')
        notebook.add(self.animation_frame, text='🎬 Анимация полета')
        # Создаем интерфейсы для каждой вкладки
        self.create_params_interface()
        self.create_results_interface()
        self.create_plots_interface()
        self.create_animation_interface()
    def create_params_interface(self):
        """Улучшенный интерфейс вкладки параметров"""
        # Основной контейнер с прокруткой
        container = ttk.Frame(self.params_frame)
        container.pack(fill='both', expand=True)
        # Создаем Canvas для прокрутки
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        # Добавляем поддержку колесика мыши
        def _on_mousewheel(event):
            if event.num == 4 or event.delta > 0:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                canvas.yview_scroll(1, "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)
        # Заголовок
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill='x', pady=(15, 20), padx=25)
        title_label = ttk.Label(header_frame, text="🚀 Конфигурация ракеты", font=('Segoe UI', 18, 'bold'))
        title_label.pack(side='left')
        # Создаем фреймы для группировки параметров с улучшенным дизайном
        mass_frame = ttk.LabelFrame(scrollable_frame, text="⚖️ Массовые характеристики", padding=20)
        mass_frame.pack(fill='x', padx=25, pady=15)
        engine_frame = ttk.LabelFrame(scrollable_frame, text="🔥 Двигательная установка", padding=20)
        engine_frame.pack(fill='x', padx=25, pady=15)
        aero_frame = ttk.LabelFrame(scrollable_frame, text="🌪️ Аэродинамические параметры", padding=20)
        aero_frame.pack(fill='x', padx=25, pady=15)
        # Словарь для хранения виджетов ввода
        self.param_entries = {}
        # Параметры массы
        params_mass = [("Начальная масса:", "m0", 1000.0, "Полная начальная масса ракеты с топливом (кг)"), ("Масса конструкции:", "mk", 300.0, "Масса ракеты без топлива (кг)")]
        for label, key, default, tooltip in params_mass:
            frame = ttk.Frame(mass_frame)
            frame.pack(fill='x', pady=10)
            ttk.Label(frame, text=label, width=20, anchor='e', font=('Segoe UI', 10)).pack(side='left', padx=10)
            entry_frame = ttk.Frame(frame)
            entry_frame.pack(side='left', padx=5)
            entry = ttk.Entry(entry_frame, width=18, font=('Segoe UI', 10))
            entry.insert(0, str(default))
            entry.pack(side='left')
            self.create_tooltip(entry, tooltip)
            self.param_entries[key] = entry
            unit_label = ttk.Label(entry_frame, text="кг", font=('Segoe UI', 10))
            unit_label.pack(side='left', padx=5)
            # Добавляем кнопки регулировки
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(side='right', padx=10)
            ttk.Button(btn_frame, text="−", width=3, command=lambda e=entry, s=-10: self.adjust_value(e, s)).pack(side='left', padx=2)
            ttk.Button(btn_frame, text="+", width=3, command=lambda e=entry, s=10: self.adjust_value(e, s)).pack(side='left', padx=2)
        # Параметры двигателя
        params_engine = [("Скорость истечения:", "u", 2000.0, "Эффективная скорость выхлопных газов (м/с)"), ("Расход топлива:", "mdot", 50.0, "Массовый расход топлива (кг/с)"), ("Угол запуска:", "theta0", 90.0, "Угол запуска (90° = вертикально вверх)")]
        for label, key, default, tooltip in params_engine:
            frame = ttk.Frame(engine_frame)
            frame.pack(fill='x', pady=10)
            ttk.Label(frame, text=label, width=20, anchor='e', font=('Segoe UI', 10)).pack(side='left', padx=10)
            entry_frame = ttk.Frame(frame)
            entry_frame.pack(side='left', padx=5)
            entry = ttk.Entry(entry_frame, width=18, font=('Segoe UI', 10))
            entry.insert(0, str(default))
            entry.pack(side='left')
            self.create_tooltip(entry, tooltip)
            self.param_entries[key] = entry
            unit = "м/с" if key == 'u' else ("кг/с" if key == 'mdot' else "°")
            unit_label = ttk.Label(entry_frame, text=unit, font=('Segoe UI', 10))
            unit_label.pack(side='left', padx=5)
            # Добавляем кнопки регулировки
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(side='right', padx=10)
            step = 100 if key == 'u' else (0.5 if key == 'mdot' else 5)
            ttk.Button(btn_frame, text="−", width=3, command=lambda e=entry, s=-step: self.adjust_value(e, s)).pack(side='left', padx=2)
            ttk.Button(btn_frame, text="+", width=3, command=lambda e=entry, s=step: self.adjust_value(e, s)).pack(side='left', padx=2)
        # Параметры аэродинамики
        params_aero = [("Коэффициент Cx:", "Cx", 0.75, "Коэффициент аэродинамического сопротивления"), ("Площадь сечения:", "S", 0.1, "Площадь миделевого сечения ракеты (м²)")]
        for label, key, default, tooltip in params_aero:
            frame = ttk.Frame(aero_frame)
            frame.pack(fill='x', pady=10)
            ttk.Label(frame, text=label, width=20, anchor='e', font=('Segoe UI', 10)).pack(side='left', padx=10)
            entry_frame = ttk.Frame(frame)
            entry_frame.pack(side='left', padx=5)
            entry = ttk.Entry(entry_frame, width=18, font=('Segoe UI', 10))
            entry.insert(0, str(default))
            entry.pack(side='left')
            self.create_tooltip(entry, tooltip)
            self.param_entries[key] = entry
            unit_label = ttk.Label(entry_frame, text="м²" if key == 'S' else "", font=('Segoe UI', 10))
            unit_label.pack(side='left', padx=5)
            # Добавляем кнопки регулировки
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(side='right', padx=10)
            step = 0.05 if key == 'Cx' else 0.01
            ttk.Button(btn_frame, text="−", width=3, command=lambda e=entry, s=-step: self.adjust_value(e, s)).pack(side='left', padx=2)
            ttk.Button(btn_frame, text="+", width=3, command=lambda e=entry, s=step: self.adjust_value(e, s)).pack(side='left', padx=2)
        # Панель быстрых пресетов
        presets_frame = ttk.LabelFrame(scrollable_frame, text="💾 Быстрые пресеты", padding=20)
        presets_frame.pack(fill='x', padx=25, pady=20)
        presets_buttons = ttk.Frame(presets_frame)
        presets_buttons.pack(fill='x', pady=10)
        presets = [("🚀 Стандартная", self.load_standard_preset), ("⚡ Быстрая", self.load_fast_preset), ("🛰️ Высотная", self.load_high_altitude_preset), ("🧪 Экспериментальная", self.load_experimental_preset)]
        for text, command in presets:
            ttk.Button(presets_buttons, text=text, command=command, width=18).pack(side='left', padx=5, pady=5)
        # Кнопки управления
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=25, pady=25)
        ttk.Button(button_frame, text="🚀 ЗАПУСТИТЬ СИМУЛЯЦИЮ", command=self.run_simulation, style='Accent.TButton', cursor='hand2', width=25).pack(side='left', padx=10, pady=5, ipadx=20, ipady=10)
        ttk.Button(button_frame, text="🔄 Сбросить", command=self.reset_parameters, cursor='hand2', width=15).pack(side='left', padx=10, pady=5, ipadx=10, ipady=8)
        ttk.Button(button_frame, text="💾 Экспорт", command=self.export_results, cursor='hand2', width=15).pack(side='left', padx=10, pady=5, ipadx=10, ipady=8)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    def adjust_value(self, entry, step):
        """Регулировка значения в поле ввода"""
        try:
            current = float(entry.get())
            new_value = max(0.1, current + step)
            entry.delete(0, tk.END)
            entry.insert(0, f"{new_value:.2f}")
        except:
            pass
    def load_standard_preset(self):
        """Загрузка стандартного пресета"""
        preset = {'m0': 1000.0, 'mk': 300.0, 'u': 2000.0, 'mdot': 50.0, 'Cx': 0.75, 'S': 0.1, 'theta0': 90.0}
        self.load_preset(preset)
    def load_fast_preset(self):
        """Загрузка пресета быстрой ракеты"""
        preset = {'m0': 1000.0, 'mk': 500.0, 'u': 2500.0, 'mdot': 100.0, 'Cx': 0.6, 'S': 0.08, 'theta0': 85.0}
        self.load_preset(preset)
    def load_high_altitude_preset(self):
        """Загрузка пресета высотной ракеты"""
        preset = {'m0': 1200.0, 'mk': 350.0, 'u': 2200.0, 'mdot': 40.0, 'Cx': 0.5, 'S': 0.07, 'theta0': 90.0}
        self.load_preset(preset)
    def load_experimental_preset(self):
        """Загрузка экспериментального пресета"""
        preset = {'m0': 15000.0, 'mk': 4000.0, 'u': 3000.0, 'mdot': 800.0, 'Cx': 0.9, 'S': 0.12, 'theta0': 80.0}
        self.load_preset(preset)
    def load_preset(self, preset):
        """Загрузка пресета параметров"""
        for param, value in preset.items():
            if param in self.param_entries:
                self.param_entries[param].delete(0, tk.END)
                self.param_entries[param].insert(0, str(value))
        self.update_status(f"Загружен пресет: {preset.get('name', 'Стандартный')}")
    def create_results_interface(self):
        """Улучшенный интерфейс вкладки результатов"""
        main_frame = ttk.Frame(self.results_frame)
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        # Верхняя панель с ключевыми показателями
        metrics_frame = ttk.LabelFrame(main_frame, text="📊 Ключевые показатели", padding=20)
        metrics_frame.pack(fill='x', pady=(0, 15))
        # Создаем метрики в виде карточек
        self.metrics_cards = {}
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill='x')
        metrics = [("📈 Макс. высота", "max_height", "м"), ("⚡ Макс. скорость", "max_velocity", "м/с"), ("⏱ Время полета", "flight_time", "с"), ("⏳ Работа двигателя", "engine_time", "с"), ("⚖️ Конечная масса", "final_mass", "кг"), ("🔥 Израсходовано топлива", "fuel_used", "кг")]
        for i, (icon, key, unit) in enumerate(metrics):
            card_frame = ttk.Frame(metrics_grid, relief='solid', borderwidth=1)
            card_frame.grid(row=i//3, column=i%3, padx=10, pady=10, sticky='nsew')
            # Настраиваем вес колонок
            metrics_grid.columnconfigure(i%3, weight=1)
            icon_label = ttk.Label(card_frame, text=icon, font=('Segoe UI', 14))
            icon_label.pack(pady=(10, 5))
            value_label = ttk.Label(card_frame, text="—", font=('Segoe UI', 16, 'bold'))
            value_label.pack()
            unit_label = ttk.Label(card_frame, text=unit, font=('Segoe UI', 10))
            unit_label.pack(pady=(0, 10))
            self.metrics_cards[key] = value_label
        # Основная область с результатами
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill='both', expand=True)
        # Левая панель - детальные результаты
        left_frame = ttk.LabelFrame(results_frame, text="📋 Детальные результаты", padding=15)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        self.results_text = tk.Text(left_frame, height=20, width=50, font=('Consolas', 10), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(left_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        # Правая панель - статистика
        right_frame = ttk.LabelFrame(results_frame, text="📈 Расширенная статистика", padding=15)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        self.stats_text = tk.Text(right_frame, height=20, width=40, font=('Consolas', 10), wrap=tk.WORD)
        stats_scrollbar = ttk.Scrollbar(right_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        self.stats_text.pack(side='left', fill='both', expand=True)
        stats_scrollbar.pack(side='right', fill='y')
        # Инициализация текста
        self.results_text.insert('1.0', "Запустите симуляцию для отображения результатов...")
        self.stats_text.insert('1.0', "Статистика будет рассчитана после симуляции...")
    def create_plots_interface(self):
        """Улучшенный интерфейс вкладки графиков"""
        main_frame = ttk.Frame(self.plots_frame)
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        # Панель управления графиками
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 15))
        ttk.Button(control_frame, text="🔄 Обновить графики", command=self.update_plots, cursor='hand2', width=20).pack(side='left', padx=5)
        ttk.Button(control_frame, text="💾 Сохранить график", command=self.save_plot, cursor='hand2', width=20).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🎨 Сменить тему",  command=self.toggle_plot_theme, cursor='hand2', width=20).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📊 Все графики в PDF", command=self.export_all_plots, cursor='hand2', width=20).pack(side='left', padx=5)
        # Контейнер для графиков
        self.fig = plt.figure(figsize=(12, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    def create_animation_interface(self):
        """Улучшенный интерфейс вкладки анимации"""
        # Основной контейнер
        main_frame = ttk.Frame(self.animation_frame)
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        # Верхний фрейм для графика
        plot_frame = ttk.LabelFrame(main_frame, text="🎬 Анимация полета ракеты", padding=10)
        plot_frame.pack(fill='both', expand=True, pady=(0, 15))
        self.anim_fig = plt.figure(figsize=(11, 7))
        self.anim_canvas = FigureCanvasTkAgg(self.anim_fig, master=plot_frame)
        self.anim_canvas.get_tk_widget().pack(fill='both', expand=True)
        # Нижний фрейм для управления
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        # Панель скорости с улучшенным слайдером
        speed_frame = ttk.LabelFrame(control_frame, text="⚙ Управление анимацией", padding=15)
        speed_frame.pack(fill='x', pady=5)
        # Слайдер скорости
        slider_frame = ttk.Frame(speed_frame)
        slider_frame.pack(fill='x', pady=10)
        ttk.Label(slider_frame, text="Скорость:", font=('Segoe UI', 10)).pack(side='left', padx=(0, 15))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(slider_frame, from_=0.1, to=2.0, orient='horizontal', variable=self.speed_var, length=350, command=self.on_speed_change)
        speed_scale.pack(side='left', padx=5, fill='x', expand=True)
        self.speed_label = ttk.Label(slider_frame, text="1.0x", font=('Segoe UI', 10, 'bold'), width=8)
        self.speed_label.pack(side='left', padx=10)
        # Кнопки быстрой установки скорости
        speeds_frame = ttk.Frame(speed_frame)
        speeds_frame.pack(fill='x', pady=5)
        ttk.Label(speeds_frame, text="Быстрая установка:").pack(side='left', padx=(0, 10))
        for speed in [0.1, 0.25, 0.5, 1, 2]:
            ttk.Button(speeds_frame, text=f"{speed}x", width=5, command=lambda s=speed: self.set_animation_speed(s), cursor='hand2').pack(side='left', padx=2, pady=2)
        # Панель управления воспроизведением
        play_frame = ttk.Frame(control_frame)
        play_frame.pack(fill='x', pady=15)
        self.play_button = ttk.Button(play_frame, text="▶ Запуск", command=self.start_animation, state='disabled', width=15, cursor='hand2')
        self.play_button.pack(side='left', padx=5)
        self.pause_button = ttk.Button(play_frame, text="⏸ Пауза", command=self.pause_animation, state='disabled', width=15, cursor='hand2')
        self.pause_button.pack(side='left', padx=5)
        self.stop_button = ttk.Button(play_frame, text="⏹ Стоп", command=self.stop_animation, state='disabled', width=15, cursor='hand2')
        self.stop_button.pack(side='left', padx=5)
        ttk.Button(play_frame, text="🔄 Создать анимацию", command=self.prepare_animation, width=18, cursor='hand2').pack(side='left', padx=5)
        # Панель информации и прогресса
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill='x', pady=5)
        # Информация о кадрах
        frame_info_frame = ttk.Frame(info_frame)
        frame_info_frame.pack(side='left', padx=10)
        ttk.Label(frame_info_frame, text="Кадр:").pack(side='left')
        self.frame_info = ttk.Label(frame_info_frame, text="0/0", font=('Segoe UI', 10, 'bold'))
        self.frame_info.pack(side='left', padx=5)
        # Информация о времени
        time_info_frame = ttk.Frame(info_frame)
        time_info_frame.pack(side='left', padx=20)
        ttk.Label(time_info_frame, text="Время:").pack(side='left')
        self.time_info = ttk.Label(time_info_frame, text="0.0 с", font=('Segoe UI', 10, 'bold'))
        self.time_info.pack(side='left', padx=5)
        # Прогресс-бар
        progress_frame = ttk.Frame(info_frame)
        progress_frame.pack(side='right', fill='x', expand=True)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=250)
        self.progress_bar.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side='left')
    def create_status_bar(self):
        """Создание улучшенной статусной строки"""
        self.status_bar = ttk.Frame(self.root, relief='sunken', borderwidth=1)
        self.status_bar.pack(side='bottom', fill='x', padx=2, pady=2)
        # Левая часть - статус
        left_frame = ttk.Frame(self.status_bar)
        left_frame.pack(side='left', fill='x', expand=True)
        self.status_label = ttk.Label(left_frame, text="✅ Готов к работе")
        self.status_label.pack(side='left', padx=10)
        # Правая часть - информация
        right_frame = ttk.Frame(self.status_bar)
        right_frame.pack(side='right')
        # Индикатор темы
        self.theme_indicator = ttk.Label(right_frame, text="☀️ Светлая тема" if self.current_theme == 'light' else "🌙 Темная тема")
        self.theme_indicator.pack(side='right', padx=10)
        # Версия программы
        version_label = ttk.Label(right_frame, text="🚀 Симулятор Ракеты v1.0")
        version_label.pack(side='right', padx=10)
    def create_tooltip(self, widget, text):
        """Создание улучшенной всплывающей подсказки"""
        def enter(event):
            tooltip = tk.Toplevel(self.root)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+15}+{event.y_root+15}")
            if self.current_theme == 'dark':
                bg = "#262423"
                fg = "#8350C4"
                border = "#7392B5"
            else:
                bg = "#C9E0EB"
                fg = "#262423"
                border = "#7392B5"
            label = ttk.Label(tooltip, text=text, background=bg, foreground=fg, relief='solid', borderwidth=2, padding=8, font=('Segoe UI', 9), border=border)
            label.pack()
            widget.tooltip = tooltip
        def leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')
        
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    def update_status(self, message):
        """Обновление статусной строки"""
        self.status_label.config(text=message)
        self.root.update()
    def update_metrics_cards(self, results):
        """Обновление карточек с метриками"""
        if results is None:
            return
        # Обновляем значения в карточках
        if 'max_height' in self.metrics_cards:
            self.metrics_cards['max_height'].config(text=f"{results['max_height']:,.1f}")
        if 'max_velocity' in self.metrics_cards:
            self.metrics_cards['max_velocity'].config(text=f"{results['max_velocity']:.1f}")
        if 'flight_time' in self.metrics_cards:
            self.metrics_cards['flight_time'].config(text=f"{results['time'][-1]:.1f}")
        if 'engine_time' in self.metrics_cards:
            self.metrics_cards['engine_time'].config(text=f"{results['engine_cutoff_time']:.1f}")
        if 'final_mass' in self.metrics_cards:
            self.metrics_cards['final_mass'].config(text=f"{results['mass'][-1]:.1f}")
        if 'fuel_used' in self.metrics_cards:
            fuel_used = self.simulator.params['m0'] - self.simulator.params['mk']
            self.metrics_cards['fuel_used'].config(text=f"{fuel_used:.1f}")
    # ============================================================
    # ОСНОВНЫЕ МЕТОДЫ УПРАВЛЕНИЯ
    # ============================================================
    def get_parameters(self):
        """Получение параметров из интерфейса"""
        try:
            params = {}
            for param, entry in self.param_entries.items():
                value = float(entry.get())
                # Валидация
                if param in ['m0', 'mk', 'u', 'mdot', 'S'] and value <= 0:
                    raise ValueError(f"Параметр '{param}' должен быть положительным")
                if param == 'Cx' and (value <= 0 or value > 10):
                    raise ValueError("Коэффициент сопротивления должен быть между 0 и 10")
                if param == 'theta0' and not (0 <= value <= 180):
                    raise ValueError("Угол должен быть в диапазоне 0-180 градусов")
                params[param] = value
            if params['m0'] <= params['mk']:
                raise ValueError("Начальная масса должна быть больше массы конструкции")
            # Применяем параметры к симулятору
            for key, value in params.items():
                self.simulator.params[key] = value
            return True
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e), parent=self.root)
            return False
    def run_simulation(self):
        """Запуск симуляции"""
        self.update_status("🚀 Запуск симуляции...")
        # Показываем прогресс
        self.show_progress_dialog("Выполнение симуляции", "Расчет траектории...")
        if not self.get_parameters():
            self.update_status("❌ Ошибка ввода параметров")
            return
        try:
            results = self.simulator.run_simulation()
            self.display_results(results)
            self.update_metrics_cards(results)
            self.update_plots()
            self.update_status("✅ Симуляция успешно завершена!")
            self.hide_progress_dialog()
            messagebox.showinfo("Успех", "Симуляция успешно завершена!", parent=self.root)
        except Exception as e:
            self.update_status(f"❌ Ошибка: {str(e)}")
            self.hide_progress_dialog()
            messagebox.showerror("Ошибка", f"Ошибка при выполнении симуляции:\n{str(e)}", parent=self.root)
    def show_progress_dialog(self, title, message):
        """Показать диалог прогресса"""
        self.progress_dialog = tk.Toplevel(self.root)
        self.progress_dialog.title(title)
        self.progress_dialog.geometry("300x100")
        self.progress_dialog.transient(self.root)
        self.progress_dialog.grab_set()
        # Настраиваем тему для диалога
        if self.current_theme == 'dark':
            bg_color = '#262423'
            fg_color = '#8360C4'
        else:
            bg_color = '#C9E0EB'
            fg_color = '#262423'
        self.progress_dialog.configure(bg=bg_color)
        # Центрируем диалог
        self.center_child_window(self.progress_dialog)
        label = ttk.Label(self.progress_dialog, text=message, font=('Segoe UI', 10))
        label.pack(pady=20)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(self.progress_dialog, variable=progress_var, maximum=100, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill='x')
        progress_bar.start(10)
        self.progress_dialog.progress_bar = progress_bar
    def hide_progress_dialog(self):
        """Скрыть диалог прогресса"""
        if hasattr(self, 'progress_dialog'):
            if hasattr(self.progress_dialog, 'progress_bar'):
                self.progress_dialog.progress_bar.stop()
            self.progress_dialog.destroy()
            delattr(self, 'progress_dialog')
    def display_results(self, results):
        """Отображение результатов"""
        # Очистка текстовых полей
        self.results_text.delete('1.0', tk.END)
        self.stats_text.delete('1.0', tk.END)
        # Основные результаты
        text = "═"*55 + "\n"
        text += "РЕЗУЛЬТАТЫ СИМУЛЯЦИИ РАКЕТЫ\n"
        text += "═"*55 + "\n\n"
        text += f"📈 Максимальная высота: {results['max_height']:>12,.1f} м\n"
        text += f"⏱ Время достижения: {results['max_height_time']:>12.1f} с\n"
        text += f"⚡ Максимальная скорость: {results['max_velocity']:>12.1f} м/с\n"
        text += f"⏳ Время работы двигателя: {results['engine_cutoff_time']:>12.1f} с\n"
        text += f"⚖️ Конечная масса: {results['mass'][-1]:>12.1f} кг\n"
        text += f"🕐 Полное время полета: {results['time'][-1]:>12.1f} с\n"
        # Высота и скорость в момент выключения двигателя
        cutoff_idx = min(int(results['engine_cutoff_time'] / self.simulator.dt), len(results['time'])-1)
        if cutoff_idx >= 0:
            text += f"\n В момент выключения двигателя:\n"
            text += f"├ Высота: {results['height'][cutoff_idx]:>15.1f} м\n"
            text += f"├ Скорость: {results['velocity'][cutoff_idx]:>15.1f} м/с\n"
            text += f"└ Масса: {results['mass'][cutoff_idx]:>15.1f} кг\n"
        self.results_text.insert('1.0', text)
        # Статистика
        stats = "\n" + "─"*40 + "\n"
        stats += "РАСШИРЕННАЯ СТАТИСТИКА\n"
        stats += "─"*40 + "\n\n"
        # Анализ взлета
        ascent_time = results['max_height_time']
        avg_ascent_speed = results['max_height'] / ascent_time if ascent_time > 0 else 0
        stats += f"Средняя скорость подъема: {avg_ascent_speed:>8.1f} м/с\n"
        # Анализ спуска
        descent_start_idx = np.argmax(results['height'])
        descent_time = results['time'][-1] - results['time'][descent_start_idx]
        avg_descent_speed = results['max_height'] / descent_time if descent_time > 0 else 0
        stats += f"Средняя скорость спуска: {avg_descent_speed:>8.1f} м/с\n"
        # Потребление топлива
        fuel_used = self.simulator.params['m0'] - self.simulator.params['mk']
        fuel_efficiency = results['max_height'] / fuel_used if fuel_used > 0 else 0
        stats += f"Эффективность топлива: {fuel_efficiency:>8.1f} м/кг\n"
        # Ускорения
        acceleration = np.diff(results['velocity']) / self.simulator.dt
        if len(acceleration) > 0:
            max_accel = np.max(acceleration)
            max_decel = np.min(acceleration)
            avg_accel = np.mean(acceleration[:cutoff_idx]) if cutoff_idx > 0 else 0
            stats += f"Макс. ускорение: {max_accel:>8.2f} м/с²\n"
            stats += f"Макс. замедление: {max_decel:>8.2f} м/с²\n"
            stats += f"Среднее ускорение: {avg_accel:>8.2f} м/с²\n"
        # Энергетические показатели
        kinetic_energy = 0.5 * results['mass'] * results['velocity']**2
        potential_energy = results['mass'] * 9.81 * results['height']
        stats += f"Макс. кинетическая энергия: {np.max(kinetic_energy)/1000:>8.1f} кДж\n"
        stats += f"Макс. потенциальная энергия: {np.max(potential_energy)/1000:>8.1f} кДж\n"
        self.stats_text.insert('1.0', stats)
    def update_plots(self):
        """Обновление графиков"""
        if self.simulator.results is None:
            return
        self.fig.clear()
        self.simulator.plot_results(self.fig, theme=self.current_theme)
        self.canvas.draw()
        self.update_status("📊 Графики обновлены")
    def save_plot(self):
        """Сохранение графика в файл"""
        if self.simulator.results is None:
            messagebox.showwarning("Внимание", "Нет данных для сохранения", parent=self.root)
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG файлы", "*.png"), ("PDF файлы", "*.pdf"), ("SVG файлы", "*.svg"), ("JPEG файлы", "*.jpg"), ("Все файлы", "*.*")], title="Сохранить график как", initialfile=f"rocket_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if filename:
            try:
                dpi = 300 if filename.endswith('.png') or filename.endswith('.jpg') else 150
                self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=self.fig.get_facecolor())
                self.update_status(f"✅ График сохранен: {os.path.basename(filename)}")
                messagebox.showinfo("Успех", f"График успешно сохранен:\n{filename}", parent=self.root)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить график:\n{str(e)}", parent=self.root)
    def export_all_plots(self):
        """Экспорт всех графиков в PDF"""
        if self.simulator.results is None:
            messagebox.showwarning("Внимание", "Нет данных для экспорта", parent=self.root)
            return
        filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF файлы", "*.pdf")], title="Сохранить все графики как PDF", initialfile=f"rocket_sim_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if filename:
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(filename) as pdf:
                    # Сохраняем основные графики
                    fig1 = plt.figure(figsize=(11, 8.5))
                    self.simulator.plot_results(fig1, theme='light')
                    pdf.savefig(fig1, bbox_inches='tight')
                    plt.close(fig1)
                    # Дополнительные графики
                    fig2 = plt.figure(figsize=(11, 8.5))
                    self.create_additional_plots(fig2)
                    pdf.savefig(fig2, bbox_inches='tight')
                    plt.close(fig2)
                    # Добавляем метаданные
                    pdf.infodict()['Title'] = 'Отчет по симуляции ракеты'
                    pdf.infodict()['Author'] = 'Ракетный симулятор v3.0'
                    pdf.infodict()['Subject'] = 'Результаты симуляции полета ракеты'
                    pdf.infodict()['Keywords'] = 'ракета, симуляция, физика'
                    pdf.infodict()['CreationDate'] = datetime.now()
                self.update_status(f"✅ Все графики экспортированы в PDF")
                messagebox.showinfo("Успех", f"Все графики успешно экспортированы в PDF:\n{filename}", parent=self.root)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось экспортировать графики:\n{str(e)}", parent=self.root)
    def create_additional_plots(self, fig):
        """Создание дополнительных графиков для отчета"""
        if self.simulator.results is None:
            return
        # График энергии
        ax1 = fig.add_subplot(2, 2, 1)
        kinetic_energy = 0.5 * self.simulator.results['mass'] * self.simulator.results['velocity']**2
        potential_energy = self.simulator.results['mass'] * 9.81 * self.simulator.results['height']
        total_energy = kinetic_energy + potential_energy
        ax1.plot(self.simulator.results['time'], kinetic_energy/1000, label='Кинетическая', color='#FFE900')
        ax1.plot(self.simulator.results['time'], potential_energy/1000, label='Потенциальная', color='#00CB79')
        ax1.plot(self.simulator.results['time'], total_energy/1000, label='Полная', color='#FF4C00')
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Энергия (кДж)')
        ax1.set_title('Энергия ракеты')
        ax1.grid(True)
        ax1.legend()
        # График ускорения
        ax2 = fig.add_subplot(2, 2, 2)
        acceleration = np.diff(self.simulator.results['velocity']) / self.simulator.dt
        time_acc = self.simulator.results['time'][:-1] + self.simulator.dt/2
        ax2.plot(time_acc, acceleration, color='#5900CA')
        ax2.axhline(y=0, color='#00CB79', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Ускорение (м/с²)')
        ax2.set_title('Ускорение ракеты')
        ax2.grid(True)
        # График перегрузки
        ax3 = fig.add_subplot(2, 2, 3)
        g_load = acceleration / 9.81
        ax3.plot(time_acc, g_load, color='#FF4C00')
        ax3.axhline(y=0, color='#00CB79', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Время (с)')
        ax3.set_ylabel('Перегрузка (g)')
        ax3.set_title('Перегрузка ракеты')
        ax3.grid(True)
        # График эффективности
        ax4 = fig.add_subplot(2, 2, 4)
        efficiency = self.simulator.results['height'] / (self.simulator.params['m0'] - self.simulator.results['mass'])
        efficiency[~np.isfinite(efficiency)] = 0
        ax4.plot(self.simulator.results['time'], efficiency, color='#FFE900')
        ax4.set_xlabel('Время (с)')
        ax4.set_ylabel('Эффективность (м/кг)')
        ax4.set_title('Эффективность использования топлива')
        ax4.grid(True)
        fig.tight_layout()
    def toggle_plot_theme(self):
        """Переключение темы графиков"""
        self.current_theme = 'dark' if self.current_theme == 'light' else 'light'
        if self.simulator.results is not None:
            self.update_plots()
        self.update_status(f"🎨 Тема графиков изменена на {'темную' if self.current_theme == 'dark' else 'светлую'}")
    def prepare_animation(self):
        """Подготовка анимации с исправленной легендой"""
        if self.simulator.results is None:
            messagebox.showwarning("Внимание", "Сначала запустите симуляцию", parent=self.root)
            return
        self.update_status("🎬 Подготовка анимации...")
        try:
            # Очистка предыдущей анимации
            if self.animation is not None:
                self.animation.event_source.stop()
                self.animation = None
            # Подготовка данных
            self.anim_fig.clear()
            ax = self.anim_fig.add_subplot(111)
            # Настройка стиля в зависимости от темы
            if self.current_theme == 'dark':
                plt.style.use('dark_background')
                bg_color = '#262423'
                line_color = '#402B47'
                point_color = '#402B47'
                grid_color = '#8350C4'
                text_color = '#C9E0EB'
                info_bg = '#7392B5'
            else:
                plt.style.use('default')
                bg_color = '#C9E0EB'
                line_color = '#8350C4'
                point_color = '#402B47'
                grid_color = '#C9E0EB'
                text_color = '#262423'
                info_bg = '#7392B5'
            # Настройка графика
            max_time = self.simulator.results['time'][-1]
            max_height = self.simulator.results['max_height'] * 1.2
            ax.set_xlim(0, max(100, max_time))
            ax.set_ylim(0, max(1000, max_height))
            ax.set_xlabel('Время (с)', fontsize=12, color=text_color)
            ax.set_ylabel('Высота (м)', fontsize=12, color=text_color)
            ax.set_title('Анимация полета ракеты', fontsize=14, fontweight='bold', color=text_color, pad=20)
            ax.grid(True, color=grid_color, alpha=0.3, linestyle='--')
            # Фоновая сетка
            ax.set_facecolor(bg_color)
            self.anim_fig.patch.set_facecolor(bg_color)
            # Элементы анимации
            self.line, = ax.plot([], [], color=line_color, lw=3, label='Траектория', zorder=2, alpha=0.8)
            self.point, = ax.plot([], [], 'o', color=point_color, markersize=14, label='Ракета', zorder=3, markeredgecolor='white', markeredgewidth=2)
            # Информационная панель
            self.info_panel = ax.text(0.02, 0.98, '', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=info_bg, alpha=0.95, edgecolor='gray', linewidth=1.5), fontsize=10, color=text_color, zorder=100, verticalalignment='top')
            # Легенда
            legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor=text_color, facecolor=info_bg, frameon=True)
            legend.set_zorder(99)
            legend.get_frame().set_linewidth(1)
            # Данные для анимации
            self.anim_data = {'time': self.simulator.results['time'], 'height': self.simulator.results['height'], 'velocity': self.simulator.results['velocity'], 'mass': self.simulator.results['mass'], 'total_frames': len(self.simulator.results['time']), 'engine_cutoff_time': self.simulator.results['engine_cutoff_time']}
            # Включение кнопок управления
            self.play_button.config(state='normal')
            self.pause_button.config(state='disabled')
            self.stop_button.config(state='disabled')
            # Инициализация первого кадра
            self.current_frame = 0
            self.update_frame(0)
            self.anim_canvas.draw()
            # Обновление информации о кадрах
            self.frame_info.config(text=f"0/{self.anim_data['total_frames']}")
            self.time_info.config(text="0.0 с")
            self.update_status("✅ Анимация готова к запуску")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при подготовке анимации:\n{str(e)}", parent=self.root)
            self.update_status(f"❌ Ошибка: {str(e)}")
    def start_animation(self):
        """Запуск анимации"""
        if self.animation is not None and self.animation_running:
            return
        if self.animation is None:
            speed = self.speed_var.get()
            interval = max(1, int(self.base_interval / speed))
            self.animation = FuncAnimation(self.anim_fig, self.animate_frame, frames=self.anim_data['total_frames'], interval=interval, blit=True, repeat=False, cache_frame_data=False)
        self.animation_running = True
        self.play_button.config(state='disabled')
        self.pause_button.config(state='normal')
        self.stop_button.config(state='normal')
        self.update_status("▶ Анимация запущена")
    def pause_animation(self):
        """Пауза анимации"""
        if self.animation is not None and self.animation_running:
            self.animation.event_source.stop()
            self.animation_running = False
            self.play_button.config(state='normal')
            self.pause_button.config(state='disabled')
            self.update_status("⏸ Анимация на паузе")
    def stop_animation(self):
        """Остановка анимации"""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
        self.animation_running = False
        self.current_frame = 0
        if hasattr(self, 'anim_data') and self.anim_data:
            self.update_frame(0)
            self.anim_canvas.draw()
        self.play_button.config(state='normal')
        self.pause_button.config(state='disabled')
        self.stop_button.config(state='disabled')
        self.frame_info.config(text="0/0")
        self.time_info.config(text="0.0 с")
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self.update_status("⏹ Анимация остановлена")
    def animate_frame(self, frame):
        """Обновление кадра анимации"""
        if not self.animation_running:
            return [self.line, self.point, self.info_panel]
        self.current_frame = frame
        progress = (frame / self.anim_data['total_frames']) * 100
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{progress:.1f}%")
        self.frame_info.config(text=f"{frame+1}/{self.anim_data['total_frames']}")
        self.time_info.config(text=f"{self.anim_data['time'][frame]:.1f} с")
        return self.update_frame(frame)
    def update_frame(self, frame):
        """Обновление элементов кадра"""
        if frame >= self.anim_data['total_frames']:
            frame = self.anim_data['total_frames'] - 1
        # Данные для текущего кадра
        time_data = self.anim_data['time'][:frame+1]
        height_data = self.anim_data['height'][:frame+1]
        current_time = self.anim_data['time'][frame]
        current_height = self.anim_data['height'][frame]
        current_velocity = self.anim_data['velocity'][frame]
        current_mass = self.anim_data['mass'][frame]
        # Обновление траектории
        self.line.set_data(time_data, height_data)
        # Обновление точки (ракеты)
        self.point.set_data([current_time], [current_height])
        # Изменение цвета точки в зависимости от состояния двигателя
        engine_on = current_time < self.anim_data['engine_cutoff_time']
        point_color = '#00ff00' if engine_on else '#ff0000'
        self.point.set_color(point_color)
        self.point.set_markersize(16 if engine_on else 14)
        # Обновление информационной панели
        engine_status = "🟢 ВКЛ" if engine_on else "🔴 ВЫКЛ"
        self.info_panel.set_text(f'Время: {current_time:>7.1f} с\n' f'Высота: {current_height:>7,.0f} м\n' f'Скорость: {current_velocity:>7.1f} м/с\n' f'Масса: {current_mass:>7.1f} кг\n' f'Двигатель: {engine_status}')
        # Автоматическая прокрутка оси X
        ax = self.anim_fig.axes[0]
        current_xlim = ax.get_xlim()
        if current_time > current_xlim[1] * 0.85:
            new_xlim = (current_xlim[0], current_time * 1.15)
            ax.set_xlim(new_xlim)
        # Автоматическая прокрутка оси Y
        current_ylim = ax.get_ylim()
        if current_height > current_ylim[1] * 0.85:
            new_ylim = (current_ylim[0], current_height * 1.15)
            ax.set_ylim(new_ylim)
        self.info_panel.set_zorder(100)
        return [self.line, self.point, self.info_panel]
    def set_animation_speed(self, speed):
        """Установка скорости анимации"""
        self.speed_var.set(speed)
        self.speed_label.config(text=f"{speed:.1f}x")
        # Обновляем интервал анимации, если она запущена
        if self.animation and self.animation_running:
            new_interval = max(1, int(self.base_interval / speed))
            self.animation.event_source.interval = new_interval
    def on_speed_change(self, value):
        """Обработка изменения скорости"""
        try:
            speed = float(value)
            self.speed_label.config(text=f"{speed:.1f}x")
            # Обновляем интервал анимации, если она запущена
            if self.animation and self.animation_running:
                new_interval = max(1, int(self.base_interval / speed))
                self.animation.event_source.interval = new_interval
        except:
            pass
    def export_results(self):
        """Экспорт результатов в файл"""
        if self.simulator.results is None:
            messagebox.showwarning("Внимание", "Нет данных для экспорта", parent=self.root)
            return
        try:
            # Создание DataFrame с результатами
            df = pd.DataFrame({'Время_с': self.simulator.results['time'], 'Высота_м': self.simulator.results['height'], 'Скорость_м/с': self.simulator.results['velocity'], 'Масса_кг': self.simulator.results['mass'], 'Сила тяги_Н': self.simulator.results['thrust'], 'Сопротивление_Н': self.simulator.results['drag'], 'Сила тяжести_Н': self.simulator.results['gravity']})
            # Запрос имени файла
            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV файлы", "*.csv"), ("Excel файлы", "*.xlsx"), ("JSON файлы", "*.json"), ("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")], title="Сохранить результаты как", initialfile=f"rocket_sim_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            if filename:
                if filename.endswith('.xlsx'):
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Результаты', index=False)
                        # Добавляем лист с параметрами
                        params_df = pd.DataFrame([self.simulator.params])
                        params_df.to_excel(writer, sheet_name='Параметры', index=False)
                        # Добавляем лист со статистикой
                        stats_data = {'Макс_высота_м': [self.simulator.results['max_height']], 'Макс_скорость_мс': [self.simulator.results['max_velocity']], 'Время_полета_с': [self.simulator.results['time'][-1]], 'Время_двигателя_с': [self.simulator.results['engine_cutoff_time']]}
                        stats_df = pd.DataFrame(stats_data)
                        stats_df.to_excel(writer, sheet_name='Статистика', index=False)
                elif filename.endswith('.json'):
                    export_data = {'metadata': {'export_date': datetime.now().isoformat(), 'program_version': 'Ракетный симулятор v1.0', 'parameters': self.simulator.params, 'simulation_info': {'max_height': float(self.simulator.results['max_height']), 'max_velocity': float(self.simulator.results['max_velocity']), 'engine_cutoff_time': float(self.simulator.results['engine_cutoff_time']), 'flight_time': float(self.simulator.results['time'][-1])}}, 'data': df.to_dict('records')}
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                else:
                    df.to_csv(filename, index=False, encoding='utf-8')
                self.update_status(f"✅ Результаты экспортированы: {os.path.basename(filename)}")
                messagebox.showinfo("Успех", f"Результаты успешно экспортированы в файл:\n{filename}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте результатов:\n{str(e)}", parent=self.root)
    def reset_parameters(self):
        """Сброс параметров к значениям по умолчанию"""
        default_params = {'m0': 100.0, 'mk': 30.0, 'u': 2000.0, 'mdot': 5.0, 'Cx': 0.75, 'S': 0.1, 'theta0': 90.0}
        for param, value in default_params.items():
            if param in self.param_entries:
                self.param_entries[param].delete(0, tk.END)
                self.param_entries[param].insert(0, str(value))
        self.update_status("🔄 Параметры сброшены к значениям по умолчанию")
    def change_theme(self, theme):
        """Смена темы интерфейса"""
        self.current_theme = theme
        self.setup_styles()
        self.theme_indicator.config(text="☀️ Светлая тема" if theme == 'light' else "🌙 Темная тема")
        # Перерисовка графиков с новой темой
        if self.simulator.results is not None:
            self.update_plots()
            if hasattr(self, 'anim_fig'):
                self.prepare_animation()
        self.update_status(f"🎨 Тема изменена на {'темную' if theme == 'dark' else 'светлую'}")
    def save_settings(self):
        """Сохранение настроек приложения"""
        try:
            # Получаем текущие параметры из интерфейса
            if not self.get_parameters():
                raise ValueError("Некорректные параметры в настройках")
            # Собираем только пользовательские параметры
            params_to_save = ['m0', 'mk', 'u', 'mdot', 'Cx', 'S', 'theta0']
            filtered_params = {k: self.simulator.params[k] for k in params_to_save if k in self.simulator.params}
            settings = {'theme': self.current_theme, 'parameters': filtered_params, 'last_save': datetime.now().isoformat()}
            settings_dir = os.path.dirname(self.settings_file)
            if settings_dir and not os.path.exists(settings_dir):
                os.makedirs(settings_dir, exist_ok=True)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            self.update_status(f"💾 Настройки сохранены в {self.settings_file}")
            messagebox.showinfo("Настройки", "Настройки успешно сохранены", parent=self.root)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки:\n{str(e)}", parent=self.root)
    def load_settings(self):
        """Загрузка настроек приложения при запуске"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                # Сохраняем настройки для последующей загрузки
                self.settings_to_load = settings
                self.update_status(f"📂 Настройки загружены из {self.settings_file}")
            except Exception as e:
                self.update_status(f"❌ Ошибка загрузки настроек: {str(e)}")
                self.settings_to_load = None
        else:
            self.settings_to_load = None
    def load_settings_from_file(self):
        """Загрузка настроек из файла"""
        filename = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")], title="Загрузить настройки из файла")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                # Сохраняем старые значения на случай ошибки
                old_values = {}
                for param, entry in self.param_entries.items():
                    old_values[param] = entry.get()
                # Загрузка параметров
                if 'parameters' in settings:
                    for param, value in settings['parameters'].items():
                        if param in self.param_entries:
                            self.param_entries[param].delete(0, tk.END)
                            self.param_entries[param].insert(0, str(value))
                # Проверяем параметры
                if not self.get_parameters():
                    # Восстанавливаем старые значения при ошибке
                    for param, value in old_values.items():
                        self.param_entries[param].delete(0, tk.END)
                        self.param_entries[param].insert(0, value)
                    # Повторно загружаем старые параметры в симулятор
                    self.get_parameters()
                    raise ValueError("Некорректные параметры в файле настроек")
                # Загрузка темы
                if 'theme' in settings:
                    self.current_theme = settings['theme']
                    self.change_theme(settings['theme'])
                self.update_status(f"✅ Настройки загружены из {os.path.basename(filename)}")
                messagebox.showinfo("Настройки", "Настройки успешно загружены", parent=self.root)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить настройки:\n{str(e)}", parent=self.root)
    def show_advanced_settings(self):
        """Показать расширенные настройки"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("🔧 Расширенные настройки")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        # Настройка темы для окна
        if self.current_theme == 'dark':
            bg_color = '#262423'
            fg = '#8350C4'
        else:
            bg_color = '#C9E0EB'
            fg = '#262423'
        settings_window.configure(bg=bg_color)
        # Центрируем окно
        self.center_child_window(settings_window)
        # Настройки анимации
        anim_frame = ttk.LabelFrame(settings_window, text="Настройки анимации", padding=15)
        anim_frame.pack(fill='x', padx=20, pady=15)
        # Базовый интервал
        ttk.Label(anim_frame, text="Базовый интервал анимации (мс):").pack(anchor='w', pady=5)
        interval_var = tk.IntVar(value=self.base_interval)
        interval_scale = ttk.Scale(anim_frame, from_=1, to=50, orient='horizontal', variable=interval_var)
        interval_scale.pack(fill='x', pady=5)
        # Качество анимации
        ttk.Label(anim_frame, text="Качество анимации:").pack(anchor='w', pady=5)
        quality_var = tk.StringVar(value="Высокое")
        ttk.Combobox(anim_frame, textvariable=quality_var, values=["Низкое", "Среднее", "Высокое", "Очень высокое"]).pack(fill='x', pady=5)
        # Настройки симуляции
        sim_frame = ttk.LabelFrame(settings_window, text="Настройки симуляции", padding=15)
        sim_frame.pack(fill='x', padx=20, pady=15)
        # Шаг времени
        ttk.Label(sim_frame, text="Шаг времени (с):").pack(anchor='w', pady=5)
        dt_var = tk.DoubleVar(value=self.simulator.dt)
        dt_entry = ttk.Entry(sim_frame, textvariable=dt_var)
        dt_entry.pack(fill='x', pady=5)
        # Время симуляции
        ttk.Label(sim_frame, text="Макс. время симуляции (с):").pack(anchor='w', pady=5)
        time_var = tk.IntVar(value=self.simulator.time_span[1])
        time_entry = ttk.Entry(sim_frame, textvariable=time_var)
        time_entry.pack(fill='x', pady=5)
        # Кнопки
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill='x', padx=20, pady=20)
        def apply_settings():
            try:
                self.base_interval = interval_var.get()
                self.simulator.dt = dt_var.get()
                self.simulator.time_span = (0, time_var.get())
                settings_window.destroy()
                self.update_status("⚙ Расширенные настройки применены")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Некорректные значения настроек:\n{str(e)}", parent=settings_window)
        ttk.Button(button_frame, text="Применить", command=apply_settings).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Отмена", command=settings_window.destroy).pack(side='right', padx=5)
    def show_about(self):
        """Окно 'О программе'"""
        about_text = """
🚀 СИМУЛЯТОР РАКЕТЫ v1

Программа для реалистичного моделирования полета
ракеты с учетом физических законов:
• Изменения массы из-за расхода топлива
• Атмосферного сопротивления
• Изменения гравитации с высотой
• Угла запуска ракеты

ОСОБЕННОСТИ ВЕРСИИ:
✓ Улучшенный интерфейс с карточками метрик
✓ Интерактивная анимация с исправленной легендой
✓ Расширенная статистика и анализ
✓ Экспорт в различные форматы
✓ Быстрые пресеты и настройки
✓ Поддержка тем (светлая/темная)

ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ:
• Макс. скорость анимации: 2x
• Высокая точность расчетов
• Поддержка экспорта графиков
• Интерактивные элементы управления

РАЗРАБОТЧИКИ:
Шкарбаненко Ксения (телеграмм: @corpse83)

ВЕРСИЯ: 1.0 (2026)
ЛИЦЕНЗИЯ: Открытый исходный код

© Все права защищены. Моделируйте с умом!
        """
        about_window = tk.Toplevel(self.root)
        about_window.title("ℹ️ О программе")
        about_window.geometry("550x500")
        about_window.transient(self.root)
        # Настройка темы для окна
        if self.current_theme == 'dark':
            bg = '#262423'
            fg = '#8350C4'
        else:
            bg = '#C9E0EB'
            fg = '#262423'
        about_window.configure(bg=bg)
        # Центрируем окно и делаем его изменяемым
        self.center_child_window(about_window)
        about_window.minsize(450, 400)
        about_window.resizable(True, True)
        # Заголовок
        header = ttk.Label(about_window, text="🚀 Симулятор Ракеты v1.0", font=('Segoe UI', 16, 'bold'))
        header.pack(pady=20)
        # Основной текст
        text_widget = tk.Text(about_window, wrap=tk.WORD, font=('Segoe UI', 10), bg=bg, fg=fg, relief='flat', height=20, width=60)
        text_widget.insert('1.0', about_text)
        text_widget.config(state='disabled')
        text_widget.pack(fill='both', expand=True, padx=20, pady=10)
        ttk.Button(about_window, text="Закрыть", command=about_window.destroy).pack(pady=20)
    def show_docs(self):
        """Показать документацию"""
        docs_text = """
📚 РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ

КЛАВИАТУРНЫЕ СОКРАЩЕНИЯ:
• F5           - Запустить симуляцию
• Ctrl+S       - Экспорт результатов
• Ctrl+R       - Сброс параметров
• F1           - Открыть документацию
• Esc          - Выход из программы

ОСНОВНЫЕ ПАРАМЕТРЫ:
1. Начальная масса    - Полная масса ракеты с топливом (кг)
2. Масса конструкции  - Масса ракеты без топлива (кг)
3. Скорость истечения - Скорость выхлопных газов (м/с)
4. Расход топлива     - Массовый расход топлива (кг/с)
5. Угол запуска       - Направление полета (90° = вертикально)
6. Коэффициент Cx     - Аэродинамическое сопротивление
7. Площадь сечения    - Площадь поперечного сечения (м²)

УПРАВЛЕНИЕ АНИМАЦИЙ:
• Нажмите "Создать анимацию" для подготовки
• Используйте кнопки управления для запуска/паузы/остановки
• Регулируйте скорость от 0.1x до 2x
• Информационная панель показывает текущие параметры
• График автоматически масштабируется

ЭКСПОРТ ДАННЫХ:
• Результаты можно экспортировать в CSV, Excel, JSON
• Графики сохраняются в PNG, PDF, SVG, JPEG
• Полный отчет в PDF включает все графики и статистику

БЫСТРЫЕ ПРЕСЕТЫ:
• Стандартная   - Базовые параметры для обучения
• Быстрая       - Оптимизирована для скорости
• Высотная      - Максимальная высота полета
• Экспериментальная - Для тестирования граничных условий

СОВЕТЫ:
1. Начните со стандартного пресета
2. Постепенно изменяйте параметры
3. Следите за статистикой эффективности
4. Используйте анимацию для визуализации
5. Экспортируйте результаты для анализа
        """
        docs_window = tk.Toplevel(self.root)
        docs_window.title("📚 Документация")
        docs_window.geometry("650x600")
        docs_window.transient(self.root)
        # Настройка темы для окна
        if self.current_theme == 'dark':
            bg = '#19003C'
            fg = '#FFF8B8'
        else:
            bg = '#CBB1EF'
            fg = '#585000'
        docs_window.configure(bg=bg)
        # Центрируем окно и делаем его изменяемым
        self.center_child_window(docs_window)
        docs_window.minsize(550, 500)
        docs_window.resizable(True, True)
        text_widget = tk.Text(docs_window, wrap=tk.WORD, font=('Consolas', 10), bg=bg, fg=fg)
        scrollbar = ttk.Scrollbar(docs_window, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y')
        text_widget.insert('1.0', docs_text)
        text_widget.config(state='disabled')
        ttk.Button(docs_window, text="Закрыть", command=docs_window.destroy).pack(pady=10)
    def show_animation_guide(self):
        """Показать руководство по анимации"""
        guide_text = """
🎬 РУКОВОДСТВО ПО АНИМАЦИИ

ИНФОРМАЦИОННАЯ ПАНЕЛЬ:
• Всегда отображается в левом верхнем углу
• Показывает текущие параметры полета
• Цвет точки меняется при выключении двигателя
• Автоматическое масштабирование графика

УПРАВЛЕНИЕ СКОРОСТЬЮ:
• Диапазон от 0.1x до 2x
• Быстрые кнопки для стандартных скоростей
• Слайдер для точной регулировки
• FPS отображается в статусной строке

СОСТОЯНИЯ ДВИГАТЕЛЯ:
🟢 ЗЕЛЕНЫЙ - Двигатель работает
🔴 КРАСНЫЙ - Двигатель выключен

КНОПКИ УПРАВЛЕНИЯ:
▶ Запуск   - Начать воспроизведение
⏸ Пауза    - Приостановить анимацию
⏹ Стоп     - Остановить и сбросить
🔄 Создать - Подготовить новую анимацию

АВТОМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ:
• Ось X увеличивается при достижении 85% правой границы
• Ось Y увеличивается при достижении 85% верхней границы
• Траектория отображается полностью

СОВЕТЫ ПО ИСПОЛЬЗОВАНИЮ:
1. Сначала создайте анимацию
2. Установите нужную скорость
3. Запустите воспроизведение
4. Используйте паузу для детального изучения
5. Останавливайте для сброса
        """
        self.show_info_window("Руководство по анимации", guide_text, 500, 450)
    def show_analysis_guide(self):
        """Показать руководство по анализу"""
        guide_text = """
📊 РУКОВОДСТВО ПО АНАЛИЗУ РЕЗУЛЬТАТОВ

КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ:
• Макс. высота - Определяет успешность запуска
• Макс. скорость - Влияет на энергию ракеты
• Время работы двигателя - Продолжительность тяги
• Эффективность топлива - Высота на кг топлива

ГРАФИКИ ДЛЯ АНАЛИЗА:
1. Высота от времени - Основная траектория
2. Скорость от времени - Динамика изменения
3. Масса от времени - Расход топлива
4. Силы от времени - Баланс тяги и сопротивления

СТАТИСТИЧЕСКИЕ ДАННЫЕ:
• Средние скорости подъема и спуска
• Максимальные ускорения и замедления
• Энергетические показатели
• Эффективность использования топлива

ОПТИМИЗАЦИЯ ПАРАМЕТРОВ:
• Увеличение скорости истечения → Большая тяга
• Уменьшение Cx → Меньшее сопротивление
• Оптимальный угол → Максимальная высота
• Баланс массы → Эффективность

ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:
✅ Хорошие результаты:
   - Плавная траектория
   - Высокая эффективность
   - Оптимальное время работы

⚠️ Проблемные участки:
   - Резкие изменения скорости
   - Низкая эффективность
   - Короткое время полета

ЭКСПОРТ ДЛЯ АНАЛИЗА:
• Используйте CSV для обработки в Excel
• PDF отчет для презентаций
• JSON для програмmatic анализа
        """
        self.show_info_window("Руководство по анализу", guide_text, 500, 500)
    def show_info_window(self, title, text, width, height):
        """Показать информационное окно"""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry(f"{width}x{height}")
        window.transient(self.root)
        # Настройка темы для окна
        if self.current_theme == 'dark':
            bg = '#262423'
            fg = '#8350C4'
        else:
            bg = '#C9E0EB'
            fg = '#262423'
        window.configure(bg=bg)
        # Центрируем окно и делаем его изменяемым
        self.center_child_window(window)
        window.minsize(width, height)
        window.resizable(True, True)
        text_widget = tk.Text(window, wrap=tk.WORD, font=('Segoe UI', 10), bg=bg, fg=fg, relief='flat')
        scrollbar = ttk.Scrollbar(window, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y')
        text_widget.insert('1.0', text)
        text_widget.config(state='disabled')
        ttk.Button(window, text="Закрыть", command=window.destroy).pack(pady=10)
    def quit_application(self):
        """Выход из приложения"""
        if messagebox.askyesno("Выход", "Сохранить настройки перед выходом?", parent=self.root):
            self.save_settings()
        self.root.quit()

# ============================================================
# ТОЧКА ВХОДА
# ============================================================

def main():
    """Запуск приложения"""
    root = tk.Tk()
    # Настройка иконки (если есть)
    try:
        root.iconbitmap('rocket.ico')
    except:
        pass
    # Настройка заголовка
    root.title("🚀 Симулятор Ракеты v1.0")
    # Создание экземпляра приложения
    app = RocketSimulatorGUI(root)
    # Загрузка сохраненных параметров после создания интерфейса
    if hasattr(app, 'settings_to_load') and app.settings_to_load:
        try:
            # Загрузка параметров
            if 'parameters' in app.settings_to_load:
                for param, value in app.settings_to_load['parameters'].items():
                    if param in app.param_entries:
                        app.param_entries[param].delete(0, tk.END)
                        app.param_entries[param].insert(0, str(value))
                # Применяем загруженные параметры к симулятору
                app.get_parameters()
            # Загрузка темы
            if 'theme' in app.settings_to_load:
                app.change_theme(app.settings_to_load['theme'])
            app.update_status("✅ Настройки загружены из файла")
        except Exception as e:
            app.update_status(f"❌ Ошибка загрузки настроек: {str(e)}")
    # Запуск главного цикла
    root.mainloop()
if __name__ == "__main__":
    main()