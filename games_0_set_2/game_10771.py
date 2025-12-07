import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:02:39.743108
# Source Brief: brief_00771.md
# Brief Index: 771
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build a neural network by placing nodes and creating connections. "
        "Propagate signals to achieve the target score before you run out of turns."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a neuron or create a connection. "
        "Press shift to cycle through available connection types."
    )
    auto_advance = False

    # --- Persistent state across resets ---
    _skill_tree_definitions = [
        {"name": "Linear", "mod": 1.0, "cost": 0, "color": (100, 100, 255)},
        {"name": "Amplifier", "mod": 1.5, "cost": 200, "color": (100, 255, 100)},
        {"name": "Splitter", "mod": 0.6, "cost": 500, "color": (255, 255, 100)},
        {"name": "Synergizer", "mod": 1.2, "cost": 1000, "color": (255, 100, 255)},
    ]
    _cumulative_score = 0
    _unlocked_skills = 1
    _target_score_base = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 480, 400
        self.UI_WIDTH = self.WIDTH - self.GRID_WIDTH
        self.GRID_COLS, self.GRID_ROWS = 12, 10
        self.CELL_W = self.GRID_WIDTH // self.GRID_COLS
        self.CELL_H = self.HEIGHT // self.GRID_ROWS
        self.MAX_TURNS = 50
        self.MAX_STEPS = self.MAX_TURNS * 20 # Generous step limit per turn

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_UI_BG = (20, 25, 45)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_CURSOR_INVALID = (255, 50, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_DIM = (100, 110, 130)
        self.COLOR_TEXT_ACCENT = (100, 200, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("sans-serif", 14)
        self.font_m = pygame.font.SysFont("sans-serif", 18)
        self.font_l = pygame.font.SysFont("sans-serif", 36, bold=True)
        
        # --- Initialize state variables (to be properly set in reset) ---
        self.steps = 0
        self.score = 0
        self.current_turn = 0
        self.target_score = 0
        self.game_over = False
        self.win_state = False
        self.game_phase = ""
        self.phase_timer = 0
        self.cursor_pos = [0, 0]
        self.neurons = []
        self.connections = []
        self.particles = []
        self.current_card = None
        self.newly_placed_neuron_id = -1
        self.selected_connection_idx = 0
        self.turn_reward = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.current_turn = self.MAX_TURNS
        self.target_score = self._target_score_base
        self.game_over = False
        self.win_state = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.neurons = []
        self.connections = []
        self.particles = []
        
        self._draw_new_card()
        self.game_phase = "PLACE_CARD"
        self.phase_timer = 0
        self.newly_placed_neuron_id = -1
        self.selected_connection_idx = 0
        
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.turn_reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        self._update_particles()
        
        if not self.game_over:
            self._update_game_phase(movement, space_press, shift_press)

        reward = self.turn_reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        
        if self.steps >= self.MAX_STEPS and not self.game_over: # Max steps reached
             self.game_over = True
             self.win_state = False
             reward -= 100.0
             terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_game_phase(self, movement, space_press, shift_press):
        if self.game_phase == "PLACE_CARD":
            self._handle_cursor_movement(movement)
            if space_press and self._is_valid_placement(self.cursor_pos):
                # SFX: card_place.wav
                self._place_card()
                if len(self.neurons) > 1:
                    self.game_phase = "SELECT_CONNECTION"
                    self.cursor_pos = self.neurons[self.newly_placed_neuron_id]['grid_pos']
                else:
                    self.game_phase = "PROPAGATE"
                    self.phase_timer = 60
        
        elif self.game_phase == "SELECT_CONNECTION":
            self._handle_cursor_movement(movement)
            if shift_press:
                # SFX: cycle_connection.wav
                self.selected_connection_idx = (self.selected_connection_idx + 1) % self._unlocked_skills
            
            target_neuron = self._get_neuron_at_cursor()
            if space_press and target_neuron and target_neuron['id'] != self.newly_placed_neuron_id:
                # SFX: connect.wav
                self._create_connection(self.newly_placed_neuron_id, target_neuron['id'])
                self.game_phase = "PROPAGATE"
                self.phase_timer = 120

        elif self.game_phase == "PROPAGATE":
            self.phase_timer -= 1
            if self.phase_timer == 119 or (self.phase_timer == 59 and len(self.neurons) == 1):
                # SFX: propagation_start.wav
                propagation_reward = self._propagate_signal()
                self.turn_reward += propagation_reward
                self._check_unlocks()
            
            if self.phase_timer <= 0:
                self.game_phase = "TURN_END"
                self.phase_timer = 15

        elif self.game_phase == "TURN_END":
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.current_turn -= 1
                if self.score >= self.target_score:
                    # SFX: win.wav
                    self.game_over = True
                    self.win_state = True
                    self.turn_reward += 100.0
                    GameEnv._cumulative_score += self.score
                    GameEnv._target_score_base += 50
                elif self.current_turn <= 0:
                    # SFX: lose.wav
                    self.game_over = True
                    self.win_state = False
                    self.turn_reward -= 100.0
                else:
                    self._draw_new_card()
                    self.game_phase = "PLACE_CARD"

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_connections()
        self._render_neurons()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turn": self.MAX_TURNS - self.current_turn,
            "target_score": self.target_score,
            "phase": self.game_phase,
        }

    # --- Game Logic Helpers ---

    def _draw_new_card(self):
        self.current_card = {
            "base_strength": self.np_random.integers(5, 16),
            "type": self.np_random.choice(["A", "B", "C"])
        }

    def _handle_cursor_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

    def _is_valid_placement(self, grid_pos):
        return not any(n['grid_pos'] == grid_pos for n in self.neurons)

    def _get_neuron_at_cursor(self):
        for n in self.neurons:
            if n['grid_pos'] == self.cursor_pos:
                return n
        return None

    def _place_card(self):
        neuron_id = len(self.neurons)
        pos = (
            (self.cursor_pos[0] + 0.5) * self.CELL_W,
            (self.cursor_pos[1] + 0.5) * self.CELL_H
        )
        neuron = {
            "id": neuron_id,
            "grid_pos": list(self.cursor_pos),
            "pos": pos,
            "base_strength": self.current_card["base_strength"],
            "type": self.current_card["type"],
            "current_strength": 0,
            "anim_strength": 0,
        }
        self.neurons.append(neuron)
        self.newly_placed_neuron_id = neuron_id
        self._create_effect(pos, self.COLOR_CURSOR, 20)

    def _create_connection(self, start_id, end_id):
        conn_type_def = self._skill_tree_definitions[self.selected_connection_idx]
        connection = {
            "start_id": start_id,
            "end_id": end_id,
            "type": conn_type_def["name"],
            "mod": conn_type_def["mod"],
        }
        self.connections.append(connection)

    def _propagate_signal(self):
        for n in self.neurons:
            n["current_strength"] = 0

        q = [(self.newly_placed_neuron_id, self.neurons[self.newly_placed_neuron_id]['base_strength'])]
        visited = {self.newly_placed_neuron_id}
        
        total_signal_increase = 0

        while q:
            neuron_id, incoming_signal = q.pop(0)
            
            neuron = self.neurons[neuron_id]
            
            # Synergizer bonus
            synergy_bonus = 0
            if any(c['type'] == 'Synergizer' for c in self.connections if c['start_id'] == neuron_id or c['end_id'] == neuron_id):
                 synergy_bonus = sum(1 for n_neighbor in self._get_neighbors(neuron_id) if n_neighbor['type'] == neuron['type']) * 2


            new_strength = incoming_signal + synergy_bonus
            
            if new_strength > neuron["current_strength"]:
                increase = new_strength - neuron["current_strength"]
                neuron["current_strength"] = new_strength
                total_signal_increase += increase
                
                # Create particles for this propagation step
                start_pos = self._get_source_pos(neuron_id)
                self._create_signal_particles(start_pos, neuron['pos'], new_strength)

                # Find outgoing connections
                for conn in self.connections:
                    if conn['start_id'] == neuron_id and conn['end_id'] not in visited:
                        outgoing_signal = neuron["current_strength"] * conn["mod"]
                        q.append((conn['end_id'], outgoing_signal))
                        visited.add(conn['end_id'])
                    # Splitter logic
                    if conn['type'] == 'Splitter' and conn['start_id'] == neuron_id:
                        targets = [c['end_id'] for c in self.connections if c['start_id'] == neuron_id and c['end_id'] not in visited]
                        for target_id in targets:
                            outgoing_signal = (neuron["current_strength"] * conn["mod"]) / max(1, len(targets))
                            if (target_id, outgoing_signal) not in q:
                                q.append((target_id, outgoing_signal))
                            visited.add(target_id)
        
        self.score += total_signal_increase
        return total_signal_increase / 10 # Scaled reward

    def _get_neighbors(self, neuron_id):
        neighbors = []
        for c in self.connections:
            if c['start_id'] == neuron_id:
                neighbors.append(self.neurons[c['end_id']])
            elif c['end_id'] == neuron_id:
                neighbors.append(self.neurons[c['start_id']])
        return neighbors

    def _get_source_pos(self, neuron_id):
        # Find where the signal came from to draw particles
        for conn in self.connections:
            if conn['end_id'] == neuron_id:
                return self.neurons[conn['start_id']]['pos']
        # If it's the source node, particles emanate from center
        return self.neurons[neuron_id]['pos']

    def _check_unlocks(self):
        current_unlocks = self._unlocked_skills
        while (self._unlocked_skills < len(self._skill_tree_definitions) and
               self._cumulative_score + self.score >= self._skill_tree_definitions[self._unlocked_skills]['cost']):
            self._unlocked_skills += 1
        
        if self._unlocked_skills > current_unlocks:
            # SFX: unlock.wav
            self.turn_reward += 5.0 * (self._unlocked_skills - current_unlocks)


    # --- Particle and Effect Helpers ---

    def _create_effect(self, pos, color, max_radius):
        self.particles.append({"pos": pos, "type": "burst", "radius": 0, "max_radius": max_radius, "color": color, "life": 1})

    def _create_signal_particles(self, start_pos, end_pos, strength):
        num_particles = min(10, int(strength / 10))
        dist_vec = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        dist = math.hypot(*dist_vec)
        if dist == 0: return

        for _ in range(num_particles):
            delay = self.np_random.uniform(0, 0.7) * self.phase_timer
            speed = dist / (self.phase_timer * (1.0 - delay/self.phase_timer))
            self.particles.append({
                "type": "signal",
                "start_pos": list(start_pos),
                "pos": list(start_pos),
                "end_pos": list(end_pos),
                "life": self.phase_timer,
                "delay": delay,
                "speed": speed,
                "strength": strength,
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            if p['type'] == 'burst':
                p['radius'] += p['max_radius'] / 15 # Burst lasts 15 frames
                if p['radius'] >= p['max_radius']:
                    self.particles.remove(p)
            
            elif p['type'] == 'signal':
                if p['delay'] > 0:
                    p['delay'] -= 1
                else:
                    vec = (p['end_pos'][0] - p['start_pos'][0], p['end_pos'][1] - p['start_pos'][1])
                    dist = math.hypot(*vec)
                    if dist > 1:
                        p['pos'][0] += (vec[0] / dist) * p['speed']
                        p['pos'][1] += (vec[1] / dist) * p['speed']
                    
                    if math.hypot(p['pos'][0] - p['end_pos'][0], p['pos'][1] - p['end_pos'][1]) < p['speed']:
                        self.particles.remove(p)

    # --- Rendering ---

    def _render_grid(self):
        for i in range(1, self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * self.CELL_W, 0), (i * self.CELL_W, self.HEIGHT))
        for i in range(1, self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i * self.CELL_H), (self.GRID_WIDTH, i * self.CELL_H))

    def _render_connections(self):
        for conn in self.connections:
            start_n = self.neurons[conn['start_id']]
            end_n = self.neurons[conn['end_id']]
            
            conn_def = next(item for item in self._skill_tree_definitions if item["name"] == conn["type"])
            color = conn_def['color']
            
            # Animate color with signal strength
            avg_strength = (start_n['anim_strength'] + end_n['anim_strength']) / 2
            final_color = self._get_signal_color(avg_strength, base_color=color)
            
            pygame.draw.aaline(self.screen, final_color, start_n['pos'], end_n['pos'], 1)

    def _render_neurons(self):
        for n in self.neurons:
            # Animate strength for smooth color transition
            n['anim_strength'] += (n['current_strength'] - n['anim_strength']) * 0.1
            
            color = self._get_signal_color(n['anim_strength'])
            pos_int = (int(n['pos'][0]), int(n['pos'][1]))
            radius = 10 + min(10, int(n['base_strength']/2))
            
            # Glow effect
            glow_radius = int(radius * (1.2 + min(1, n['anim_strength'] / 50)))
            glow_alpha = min(100, int(n['anim_strength']))
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)
            
            # Type indicator
            type_text = self.font_s.render(n['type'], True, self.COLOR_BG)
            self.screen.blit(type_text, (pos_int[0] - type_text.get_width()//2, pos_int[1] - type_text.get_height()//2))

    def _render_particles(self):
        for p in self.particles:
            if p['type'] == 'burst':
                alpha = 255 * (1 - p['radius']/p['max_radius'])
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*p['color'], alpha))
            elif p['type'] == 'signal':
                color = self._get_signal_color(p['strength'])
                radius = int(2 + min(3, p['strength']/20))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_cursor(self):
        if self.game_phase not in ["PLACE_CARD", "SELECT_CONNECTION"]:
            return
            
        pos = (
            (self.cursor_pos[0] + 0.5) * self.CELL_W,
            (self.cursor_pos[1] + 0.5) * self.CELL_H
        )
        pos_int = (int(pos[0]), int(pos[1]))
        is_valid = (self.game_phase == "PLACE_CARD" and self._is_valid_placement(self.cursor_pos)) or \
                   (self.game_phase == "SELECT_CONNECTION" and self._get_neuron_at_cursor() is not None and self._get_neuron_at_cursor()['id'] != self.newly_placed_neuron_id)

        color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        radius = 15
        
        # Draw ghost card
        if self.game_phase == "PLACE_CARD":
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 12, (*color, 50))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 12, (*color, 100))

        # Draw cursor brackets
        for i in range(4):
            angle = i * math.pi / 2 + math.pi / 4
            start = (pos[0] + radius * math.cos(angle), pos[1] + radius * math.sin(angle))
            end = (pos[0] + (radius+5) * math.cos(angle), pos[1] + (radius+5) * math.sin(angle))
            pygame.draw.aaline(self.screen, color, start, end, 2)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.GRID_WIDTH, 0, self.UI_WIDTH, self.HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_WIDTH, 0), (self.GRID_WIDTH, self.HEIGHT), 2)
        
        y = 15
        # Score
        self._draw_text("SCORE", self.GRID_WIDTH + 15, y, self.font_s, self.COLOR_TEXT_DIM)
        y += 15
        self._draw_text(f"{int(self.score)}", self.GRID_WIDTH + 15, y, self.font_l, self.COLOR_TEXT)
        y += 40
        
        # Target
        self._draw_text("TARGET", self.GRID_WIDTH + 15, y, self.font_s, self.COLOR_TEXT_DIM)
        y += 15
        self._draw_text(f"{self.target_score}", self.GRID_WIDTH + 15, y, self.font_m, self.COLOR_TEXT_ACCENT)
        y += 30
        
        # Turns
        self._draw_text("TURNS LEFT", self.GRID_WIDTH + 15, y, self.font_s, self.COLOR_TEXT_DIM)
        y += 15
        self._draw_text(f"{self.current_turn}", self.GRID_WIDTH + 15, y, self.font_l, self.COLOR_TEXT)
        y += 40
        
        # Current Action
        if self.game_phase == "PLACE_CARD" and self.current_card:
            self._draw_text("DRAWN CARD", self.GRID_WIDTH + 15, y, self.font_s, self.COLOR_TEXT_DIM)
            y += 20
            self._draw_text(f"Strength: {self.current_card['base_strength']}", self.GRID_WIDTH + 20, y, self.font_m, self.COLOR_TEXT)
            y += 20
            self._draw_text(f"Type: {self.current_card['type']}", self.GRID_WIDTH + 20, y, self.font_m, self.COLOR_TEXT)
            y += 30

        if self.game_phase == "SELECT_CONNECTION":
            self._draw_text("CONNECTION TYPE", self.GRID_WIDTH + 15, y, self.font_s, self.COLOR_TEXT_DIM)
            y += 20
            conn_def = self._skill_tree_definitions[self.selected_connection_idx]
            self._draw_text(f"{conn_def['name']}", self.GRID_WIDTH + 20, y, self.font_m, conn_def['color'])
            y+= 20
            self._draw_text(f"Mod: x{conn_def['mod']}", self.GRID_WIDTH + 20, y, self.font_s, self.COLOR_TEXT)
            y += 30

        # Skill Tree
        self._draw_text("SKILL TREE", self.GRID_WIDTH + 15, y, self.font_s, self.COLOR_TEXT_DIM)
        y += 20
        for i, skill in enumerate(self._skill_tree_definitions):
            color = skill['color'] if i < self._unlocked_skills else self.COLOR_TEXT_DIM
            prefix = "✓ " if i < self._unlocked_skills else "🔒 "
            text = f"{prefix}{skill['name']}"
            if i >= self._unlocked_skills:
                text += f" ({skill['cost']})"
            self._draw_text(text, self.GRID_WIDTH + 20, y, self.font_s, color)
            y += 18

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(s, (0, 0))
        
        msg = "SUCCESS" if self.win_state else "FAILURE"
        color = (100, 255, 150) if self.win_state else (255, 100, 100)
        
        text = self.font_l.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
        self.screen.blit(text, text_rect)
        
        score_text = self.font_m.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
        self.screen.blit(score_text, score_rect)

    def _draw_text(self, text, x, y, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def _get_signal_color(self, strength, base_color=None):
        strength = max(0, strength)
        if base_color:
            r, g, b = base_color
        else: # Default signal color
            r, g, b = 80, 80, 200 # Base blue
        
        # Green component increases with strength
        g = min(255, g + strength * 2)
        # Red component increases at higher strength
        r = min(255, r + max(0, strength - 30) * 3)
        # Blue component fades slightly
        b = max(50, b - strength)
        
        return (int(r), int(g), int(b))
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neuronet Propagator")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()