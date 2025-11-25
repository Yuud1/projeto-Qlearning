import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from blackjack_env import BlackjackEnv, Action
from q_learning import QLearningAgent


class ModernQLearningGUI:    
    def __init__(self, root):
        self.root = root
        self.root.title("🎰 Q-Learning Blackjack - Visualização Interativa")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        style = ttk.Style()
        style.theme_use('clam')
        self._configure_styles()
        
        self.env = BlackjackEnv()
        self.agent = QLearningAgent(
            alpha=0.1,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        self.training = False
        self.training_thread = None
        
        self.reward_history = []
        self.win_rate_history = []
        self.epsilon_history = []
        self.episode_numbers = []
        
        self._create_widgets()
        
        self.update_all_visualizations()
        
    def _configure_styles(self):
        style = ttk.Style()
        
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 20, 'bold'),
                       background='#1a1a2e',
                       foreground='#00d4ff')
        
        style.configure('Modern.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       padding=10)
        
        style.configure('Card.TLabel',
                       font=('Segoe UI', 11),
                       background='#16213e',
                       foreground='#ffffff',
                       padding=8)
        
        style.configure('Stats.TLabel',
                       font=('Segoe UI', 12, 'bold'),
                       background='#16213e',
                       foreground='#00ff88',
                       padding=5)
        
    def _create_widgets(self):
        main_container = tk.Frame(self.root, bg='#1a1a2e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_frame = tk.Frame(main_container, bg='#1a1a2e')
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            title_frame,
            text="🎰 Q-LEARNING BLACKJACK",
            font=('Segoe UI', 24, 'bold'),
            bg='#1a1a2e',
            fg='#00d4ff'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Aprendizado por Reforço com Visualização Interativa",
            font=('Segoe UI', 12),
            bg='#1a1a2e',
            fg='#8e8e93'
        )
        subtitle_label.pack(pady=(5, 0))
        
        top_frame = tk.Frame(main_container, bg='#16213e', relief=tk.RAISED, bd=2)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        controls_frame = tk.LabelFrame(
            top_frame,
            text="⚙️ Controles",
            font=('Segoe UI', 11, 'bold'),
            bg='#16213e',
            fg='#00d4ff',
            padx=15,
            pady=10
        )
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        button_frame = tk.Frame(controls_frame, bg='#16213e')
        button_frame.pack()
        
        self.train_button = tk.Button(
            button_frame,
            text="🚀 Treinar (1K)",
            font=('Segoe UI', 10, 'bold'),
            bg='#00ff88',
            fg='#000000',
            activebackground='#00cc6a',
            activeforeground='#000000',
            relief=tk.RAISED,
            bd=3,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self.train_1000
        )
        self.train_button.grid(row=0, column=0, padx=5)
        
        self.train_custom_button = tk.Button(
            button_frame,
            text="⚡ Treinar (Custom)",
            font=('Segoe UI', 10, 'bold'),
            bg='#00d4ff',
            fg='#000000',
            activebackground='#00b8e6',
            activeforeground='#000000',
            relief=tk.RAISED,
            bd=3,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self.train_custom
        )
        self.train_custom_button.grid(row=0, column=1, padx=5)
        
        self.stop_button = tk.Button(
            button_frame,
            text="⏹️ Parar",
            font=('Segoe UI', 10, 'bold'),
            bg='#ff4444',
            fg='#ffffff',
            activebackground='#cc0000',
            activeforeground='#ffffff',
            relief=tk.RAISED,
            bd=3,
            padx=15,
            pady=8,
            cursor='hand2',
            state=tk.DISABLED,
            command=self.stop_training
        )
        self.stop_button.grid(row=0, column=2, padx=5)
        
        self.reset_button = tk.Button(
            button_frame,
            text="🔄 Resetar",
            font=('Segoe UI', 10, 'bold'),
            bg='#ffaa00',
            fg='#000000',
            activebackground='#cc8800',
            activeforeground='#000000',
            relief=tk.RAISED,
            bd=3,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self.reset_q_table
        )
        self.reset_button.grid(row=0, column=3, padx=5)
        
        stats_frame = tk.Frame(top_frame, bg='#16213e')
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        stats_title = tk.Label(
            stats_frame,
            text="📊 Estatísticas em Tempo Real",
            font=('Segoe UI', 11, 'bold'),
            bg='#16213e',
            fg='#00d4ff'
        )
        stats_title.pack(anchor=tk.W, pady=(0, 5))
        
        self.stats_container = tk.Frame(stats_frame, bg='#16213e')
        self.stats_container.pack(fill=tk.X)
        
        self.stats_labels = {}
        self._create_stat_cards()
        
        notebook_frame = tk.Frame(main_container, bg='#1a1a2e')
        notebook_frame.pack(fill=tk.BOTH, expand=True)
        
        style = ttk.Style()
        style.configure('TNotebook', background='#16213e', borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background='#0f3460',
                       foreground='#ffffff',
                       padding=[20, 10],
                       font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', '#00d4ff')],
                 foreground=[('selected', '#000000')])
        
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self._create_heatmap_tab()
        
        self._create_graphs_tab()
        
        self._create_table_tab()
        
    def _create_stat_cards(self):
        stats_info = [
            ('episodes', 'Episódios', '0'),
            ('avg_reward', 'Recompensa Média', '0.000'),
            ('win_rate', 'Taxa de Vitória', '0.0%'),
            ('epsilon', 'Epsilon', '1.000')
        ]
        
        for idx, (key, label, default) in enumerate(stats_info):
            card = tk.Frame(
                self.stats_container,
                bg='#0f3460',
                relief=tk.RAISED,
                bd=2
            )
            card.grid(row=0, column=idx, padx=5, sticky='ew')
            self.stats_container.columnconfigure(idx, weight=1)
            
            label_widget = tk.Label(
                card,
                text=label,
                font=('Segoe UI', 9),
                bg='#0f3460',
                fg='#8e8e93'
            )
            label_widget.pack(pady=(8, 2))
            
            value_label = tk.Label(
                card,
                text=default,
                font=('Segoe UI', 14, 'bold'),
                bg='#0f3460',
                fg='#00ff88'
            )
            value_label.pack(pady=(0, 8))
            
            self.stats_labels[key] = value_label
    
    def _create_heatmap_tab(self):
        heatmap_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(heatmap_frame, text='🔥 Heatmap Q-Matrix')
        
        heatmap_container = tk.Frame(heatmap_frame, bg='#1a1a2e')
        heatmap_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.heatmap_fig = Figure(figsize=(12, 6), facecolor='#1a1a2e')
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, heatmap_container)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = tk.Frame(heatmap_container, bg='#1a1a2e')
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.heatmap_canvas, toolbar_frame)
        toolbar.config(bg='#16213e')
        toolbar.update()
        
    def _create_graphs_tab(self):
        graphs_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(graphs_frame, text='📈 Gráficos de Aprendizado')
        
        graphs_container = tk.Frame(graphs_frame, bg='#1a1a2e')
        graphs_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.graphs_fig = Figure(figsize=(12, 8), facecolor='#1a1a2e')
        self.graphs_fig.subplots_adjust(hspace=0.4)
        
        self.graphs_canvas = FigureCanvasTkAgg(self.graphs_fig, graphs_container)
        self.graphs_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _create_table_tab(self):
        table_frame = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(table_frame, text='📋 Tabela Q-Matrix')
        
        canvas_frame = tk.Frame(table_frame, bg='#1a1a2e')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.table_canvas = tk.Canvas(canvas_frame, bg='#16213e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.table_canvas.yview)
        self.table_scrollable_frame = tk.Frame(self.table_canvas, bg='#16213e')
        
        self.table_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))
        )
        
        self.table_canvas.create_window((0, 0), window=self.table_scrollable_frame, anchor="nw")
        self.table_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.table_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def update_heatmap(self):
        self.heatmap_fig.clear()
        ax = self.heatmap_fig.add_subplot(111)
        ax.set_facecolor('#1a1a2e')
        
        Q_matrix = self.agent.Q
        states = [f'{i+4}' for i in range(self.agent.num_states)]
        actions = ['HIT', 'STAND']
        
        im = ax.imshow(Q_matrix.T, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states, color='white', fontsize=8)
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel('Valor da Mão (Estado)', color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ação', color='white', fontsize=12, fontweight='bold')
        ax.set_title('🔥 Matriz Q - Heatmap Interativo', 
                    color='#00d4ff', fontsize=14, fontweight='bold', pad=15)
        
        for i in range(len(actions)):
            for j in range(len(states)):
                text = ax.text(j, i, f'{Q_matrix[j, i]:.2f}',
                             ha="center", va="center", color="black", fontsize=7, fontweight='bold')
        
        cbar = self.heatmap_fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Valor Q', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('white')
        
        ax.tick_params(colors='white')
        self.heatmap_fig.patch.set_facecolor('#1a1a2e')
        self.heatmap_canvas.draw()
    
    def update_graphs(self):
        self.graphs_fig.clear()
        
        if len(self.episode_numbers) == 0:
            ax = self.graphs_fig.add_subplot(111)
            ax.set_facecolor('#1a1a2e')
            ax.text(0.5, 0.5, 'Inicie o treinamento para ver os gráficos',
                   ha='center', va='center', color='white', fontsize=14,
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#1a1a2e')
        else:
            ax1 = self.graphs_fig.add_subplot(221)
            ax1.set_facecolor('#1a1a2e')
            ax1.plot(self.episode_numbers, self.reward_history, 
                    color='#00ff88', linewidth=2, label='Recompensa Média')
            ax1.fill_between(self.episode_numbers, self.reward_history, 
                           alpha=0.3, color='#00ff88')
            ax1.set_xlabel('Episódio', color='white', fontweight='bold')
            ax1.set_ylabel('Recompensa', color='white', fontweight='bold')
            ax1.set_title('💰 Recompensa ao Longo do Tempo', 
                         color='#00ff88', fontweight='bold', pad=10)
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3, color='white')
            ax1.legend(loc='best', facecolor='#16213e', edgecolor='white', labelcolor='white')
            for spine in ax1.spines.values():
                spine.set_color('white')
            
            ax2 = self.graphs_fig.add_subplot(222)
            ax2.set_facecolor('#1a1a2e')
            ax2.plot(self.episode_numbers, self.win_rate_history, 
                    color='#00d4ff', linewidth=2, label='Taxa de Vitória')
            ax2.fill_between(self.episode_numbers, self.win_rate_history, 
                           alpha=0.3, color='#00d4ff')
            ax2.set_xlabel('Episódio', color='white', fontweight='bold')
            ax2.set_ylabel('Taxa (%)', color='white', fontweight='bold')
            ax2.set_title('🏆 Taxa de Vitória', 
                         color='#00d4ff', fontweight='bold', pad=10)
            ax2.set_ylim([0, 1])
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3, color='white')
            ax2.legend(loc='best', facecolor='#16213e', edgecolor='white', labelcolor='white')
            for spine in ax2.spines.values():
                spine.set_color('white')
            
            ax3 = self.graphs_fig.add_subplot(223)
            ax3.set_facecolor('#1a1a2e')
            ax3.plot(self.episode_numbers, self.epsilon_history, 
                    color='#ffaa00', linewidth=2, label='Epsilon')
            ax3.set_xlabel('Episódio', color='white', fontweight='bold')
            ax3.set_ylabel('Epsilon', color='white', fontweight='bold')
            ax3.set_title('🎯 Decaimento do Epsilon (Exploração)', 
                         color='#ffaa00', fontweight='bold', pad=10)
            ax3.tick_params(colors='white')
            ax3.grid(True, alpha=0.3, color='white')
            ax3.legend(loc='best', facecolor='#16213e', edgecolor='white', labelcolor='white')
            for spine in ax3.spines.values():
                spine.set_color('white')
            
            ax4 = self.graphs_fig.add_subplot(224)
            ax4.set_facecolor('#1a1a2e')
            
            states_range = range(self.agent.num_states)
            hit_values = [self.agent.Q[s, Action.HIT.value] for s in states_range]
            stand_values = [self.agent.Q[s, Action.STAND.value] for s in states_range]
            
            x = [s + 4 for s in states_range]
            width = 0.35
            x_pos = np.arange(len(x))
            
            ax4.bar(x_pos - width/2, hit_values, width, label='HIT', 
                   color='#00d4ff', alpha=0.8)
            ax4.bar(x_pos + width/2, stand_values, width, label='STAND', 
                   color='#00ff88', alpha=0.8)
            ax4.set_xlabel('Valor da Mão', color='white', fontweight='bold')
            ax4.set_ylabel('Valor Q', color='white', fontweight='bold')
            ax4.set_title('⚔️ Comparação HIT vs STAND por Estado', 
                         color='white', fontweight='bold', pad=10)
            ax4.set_xticks(x_pos[::3])
            ax4.set_xticklabels(x[::3], color='white')
            ax4.tick_params(colors='white')
            ax4.grid(True, alpha=0.3, color='white', axis='y')
            ax4.legend(loc='best', facecolor='#16213e', edgecolor='white', labelcolor='white')
            for spine in ax4.spines.values():
                spine.set_color('white')
        
        self.graphs_fig.patch.set_facecolor('#1a1a2e')
        self.graphs_canvas.draw()
    
    def update_table(self):
        for widget in self.table_scrollable_frame.winfo_children():
            widget.destroy()
        
        header_frame = tk.Frame(self.table_scrollable_frame, bg='#0f3460')
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        headers = ["Estado", "Mão", "Q(HIT)", "Q(STAND)", "Melhor Ação"]
        header_widths = [8, 8, 15, 15, 12]
        
        for i, (header, width) in enumerate(zip(headers, header_widths)):
            label = tk.Label(
                header_frame,
                text=header,
                font=('Segoe UI', 10, 'bold'),
                bg='#0f3460',
                fg='#00d4ff',
                width=width,
                anchor='w',
                padx=10,
                pady=10
            )
            label.grid(row=0, column=i, sticky='ew')
            header_frame.columnconfigure(i, weight=1)
        
        for state in range(self.agent.num_states):
            hand_value = state + 4
            hit_q = self.agent.Q[state, Action.HIT.value]
            stand_q = self.agent.Q[state, Action.STAND.value]
            best_action_idx = np.argmax(self.agent.Q[state, :])
            best_action = "HIT" if best_action_idx == Action.HIT.value else "STAND"
            
            bg_color = '#16213e' if state % 2 == 0 else '#1a1a2e'
            
            row_frame = tk.Frame(self.table_scrollable_frame, bg=bg_color)
            row_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(row_frame, text=str(state), font=('Segoe UI', 9),
                    bg=bg_color, fg='white', width=8, anchor='w', padx=10).grid(row=0, column=0, sticky='ew')
            
            tk.Label(row_frame, text=str(hand_value), font=('Segoe UI', 9, 'bold'),
                    bg=bg_color, fg='#00ff88', width=8, anchor='w', padx=10).grid(row=0, column=1, sticky='ew')
            
            hit_color = self._get_gradient_color(hit_q, Action.HIT.value)
            tk.Label(row_frame, text=f"{hit_q:.3f}", font=('Segoe UI', 9, 'bold'),
                    bg=hit_color, fg='#000000' if hit_q > 0 else '#ffffff',
                    width=15, anchor='w', padx=10).grid(row=0, column=2, sticky='ew')
            
            stand_color = self._get_gradient_color(stand_q, Action.STAND.value)
            tk.Label(row_frame, text=f"{stand_q:.3f}", font=('Segoe UI', 9, 'bold'),
                    bg=stand_color, fg='#000000' if stand_q > 0 else '#ffffff',
                    width=15, anchor='w', padx=10).grid(row=0, column=3, sticky='ew')
            
            action_color = '#00d4ff' if best_action == 'HIT' else '#00ff88'
            tk.Label(row_frame, text=best_action, font=('Segoe UI', 9, 'bold'),
                    bg=bg_color, fg=action_color, width=12, anchor='w', padx=10).grid(row=0, column=4, sticky='ew')
            
            for i in range(5):
                row_frame.columnconfigure(i, weight=1)
    
    def _get_gradient_color(self, q_value: float, action: int) -> str:
        normalized = (q_value + 2) / 4.0
        normalized = max(0, min(1, normalized))
        
        if action == Action.HIT.value:
            r = int(0 + normalized * 0)
            g = int(100 + normalized * 155)
            b = int(200 + normalized * 55)
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            r = int(0 + normalized * 0)
            g = int(200 + normalized * 55)
            b = int(100 + normalized * 100)
            return f"#{r:02x}{g:02x}{b:02x}"
    
    def update_stats(self):
        stats = self.agent.get_stats()
        if stats:
            self.stats_labels['episodes'].config(text=str(stats['total_episodes']))
            self.stats_labels['avg_reward'].config(text=f"{stats['avg_reward_recent']:.3f}")
            self.stats_labels['win_rate'].config(text=f"{stats['win_rate']*100:.1f}%")
            self.stats_labels['epsilon'].config(text=f"{stats['epsilon']:.3f}")
            
            if stats['total_episodes'] % 100 == 0:
                self.episode_numbers.append(stats['total_episodes'])
                self.reward_history.append(stats['avg_reward_recent'])
                self.win_rate_history.append(stats['win_rate'])
                self.epsilon_history.append(stats['epsilon'])
                
                if len(self.episode_numbers) > 1000:
                    self.episode_numbers.pop(0)
                    self.reward_history.pop(0)
                    self.win_rate_history.pop(0)
                    self.epsilon_history.pop(0)
    
    def update_all_visualizations(self):
        self.update_heatmap()
        self.update_graphs()
        self.update_table()
        self.update_stats()
    
    def train_1000(self):
        if self.training:
            messagebox.showwarning("Treinamento em andamento", 
                                 "Aguarde o treinamento atual terminar.")
            return
        
        self.training = True
        self.train_button.config(state=tk.DISABLED)
        self.train_custom_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.training_thread = threading.Thread(target=self._train, args=(1000,), daemon=True)
        self.training_thread.start()
    
    def train_custom(self):
        if self.training:
            messagebox.showwarning("Treinamento em andamento", 
                                 "Aguarde o treinamento atual terminar.")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Treinamento Customizado")
        dialog.geometry("350x200")
        dialog.configure(bg='#16213e')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Número de episódios:", 
                font=('Segoe UI', 11, 'bold'),
                bg='#16213e', fg='white').pack(pady=20)
        
        episodes_var = tk.StringVar(value="5000")
        entry = tk.Entry(dialog, textvariable=episodes_var, width=20,
                        font=('Segoe UI', 11), justify=tk.CENTER)
        entry.pack(pady=10)
        
        def start_custom_training():
            try:
                num_episodes = int(episodes_var.get())
                if num_episodes <= 0:
                    raise ValueError
                dialog.destroy()
                
                self.training = True
                self.train_button.config(state=tk.DISABLED)
                self.train_custom_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                
                self.training_thread = threading.Thread(target=self._train, 
                                                       args=(num_episodes,), 
                                                       daemon=True)
                self.training_thread.start()
            except ValueError:
                messagebox.showerror("Erro", "Por favor, insira um número válido de episódios.")
        
        button = tk.Button(dialog, text="Iniciar", command=start_custom_training,
                          font=('Segoe UI', 10, 'bold'),
                          bg='#00ff88', fg='#000000',
                          padx=20, pady=5, cursor='hand2')
        button.pack(pady=10)
        
        entry.bind("<Return>", lambda e: start_custom_training())
        entry.focus()
    
    def stop_training(self):
        self.training = False
        self.train_button.config(state=tk.NORMAL)
        self.train_custom_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def reset_q_table(self):
        if self.training:
            messagebox.showwarning("Treinamento em andamento", 
                                 "Pare o treinamento antes de resetar.")
            return
        
        if messagebox.askyesno("Confirmar", "Tem certeza que deseja resetar a matriz Q?"):
            self.agent.Q = np.zeros((self.agent.num_states, self.agent.num_actions))
            self.agent.episode_rewards = []
            self.agent.episode_lengths = []
            self.agent.total_episodes = 0
            self.agent.epsilon = 1.0
            self.reward_history = []
            self.win_rate_history = []
            self.epsilon_history = []
            self.episode_numbers = []
            self.update_all_visualizations()
    
    def _train(self, num_episodes: int):
        episodes_per_update = max(1, num_episodes // 50)
        
        for episode in range(num_episodes):
            if not self.training:
                break
            
            self.agent.train_episode(self.env)
            
            if (episode + 1) % episodes_per_update == 0 or (episode + 1) == num_episodes:
                progress = ((episode + 1) / num_episodes) * 100
                self.root.after(0, self.update_all_visualizations)
                self.root.after(0, lambda p=progress: self.root.title(
                    f"🎰 Q-Learning Blackjack - Visualização Interativa (Treinando... {p:.1f}%)"))
        
        self.root.after(0, lambda: self.root.title("🎰 Q-Learning Blackjack - Visualização Interativa"))
        self.root.after(0, self.update_all_visualizations)
        self.root.after(0, self.stop_training)
        self.root.after(0, lambda: messagebox.showinfo("Treinamento Completo", 
                                                      f"✅ Treinamento de {num_episodes} episódios concluído!"))


def main():
    root = tk.Tk()
    app = ModernQLearningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
