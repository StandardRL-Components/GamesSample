import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:58:56.413156
# Source Brief: brief_02499.md
# Brief Index: 2499
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment simulating the growth and defense of a bioluminescent cell.
    The player matches DNA strands to grow the cell, manipulates gravity to direct growth,
    and defends against waves of harmful mutations.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Grow a bioluminescent cell by matching DNA sequences while defending it from waves of harmful mutations. "
        "Manipulate gravity to guide the cell's growth."
    )
    user_guide = (
        "Controls: Use ↑/↓ arrow keys to select a DNA strand. Press space to confirm a match and grow the cell. "
        "Press shift to flip gravity and change the direction of growth."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.np_random = None

        # --- Visual & Color Palette ---
        self.COLOR_BG_DARK = (10, 0, 20)
        self.COLOR_BG_LIGHT = (30, 0, 50)
        self.COLOR_CELL = (0, 255, 150)
        self.COLOR_CELL_GLOW = (0, 255, 150, 50)
        self.COLOR_MUTATION = (255, 50, 50)
        self.COLOR_MUTATION_GLOW = (255, 50, 50, 60)
        self.COLOR_DNA_TARGET = (100, 150, 255)
        self.COLOR_DNA_AVAILABLE = (100, 200, 255)
        self.COLOR_DNA_SELECT = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.DNA_SYMBOLS = ['circle', 'square', 'triangle', 'diamond']

        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.MAX_HEALTH = 100.0
        self.INITIAL_WAVE_TIMER = 300 # Steps until first wave
        self.WAVE_INTERVAL = 300 # Steps between waves

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cell_health = 0.0
        self.gravity_direction = 0
        self.cell_nodes = []
        self.cell_connections = []
        self.dna_target_sequence = []
        self.dna_available_strands = []
        self.dna_selection_index = 0
        self.dna_sequence_length = 0
        self.mutations = []
        self.mutation_wave_timer = 0
        self.mutation_speed = 0.0
        self.unlocked_genes = []
        self.last_space_held = False
        self.last_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cell_health = self.MAX_HEALTH
        self.gravity_direction = 1  # 1 for down, -1 for up
        
        # Initialize cell with a central node
        self.cell_nodes = [[self.screen_width / 2, self.screen_height / 2]]
        self.cell_connections = []

        self.mutations = []
        self.mutation_wave_timer = self.INITIAL_WAVE_TIMER
        self.mutation_speed = 1.0

        self.dna_sequence_length = 3
        self._generate_new_dna_puzzle()

        self.unlocked_genes = []
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # --- 1. Unpack Actions & Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # Action: Cycle DNA selection
        if len(self.dna_available_strands) > 0:
            if movement == 1: # Up
                self.dna_selection_index = (self.dna_selection_index - 1) % len(self.dna_available_strands)
            elif movement == 2: # Down
                self.dna_selection_index = (self.dna_selection_index + 1) % len(self.dna_available_strands)

        # Action: Flip Gravity
        if shift_press:
            self.gravity_direction *= -1

        # Action: Confirm DNA Match
        if space_press and self.dna_target_sequence and self.dna_available_strands:
            if self.dna_available_strands[self.dna_selection_index] == self.dna_target_sequence[0]:
                # Correct Match
                reward += 0.1
                self.score += 0.1
                self.dna_target_sequence.pop(0)
                self.dna_available_strands.pop(self.dna_selection_index)
                self._grow_cell()
                if not self.dna_target_sequence:
                    reward += 2.0 # Bonus for completing sequence
                    self.score += 2.0
                    self._generate_new_dna_puzzle()
            else:
                # Incorrect Match
                reward -= 0.2
                self.score -= 0.2

            if self.dna_available_strands:
                self.dna_selection_index %= len(self.dna_available_strands)
            else:
                self.dna_selection_index = 0

        # --- 2. Update Game Logic ---
        self._update_mutations()
        damage, wave_cleared = self._check_mutation_collisions()
        if damage > 0:
            self.cell_health -= damage
            reward -= 0.5 * damage
            self.score -= 0.5 * damage
        if wave_cleared:
            reward += 5.0
            self.score += 5.0
            if len(self.unlocked_genes) < 2 and self.score > (len(self.unlocked_genes) + 1) * 20:
                self.unlocked_genes.append(f"GENE-{len(self.unlocked_genes)+1}")
                reward += 10.0
                self.score += 10.0

        # --- 3. Difficulty Scaling ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.mutation_speed = min(3.0, self.mutation_speed + 0.1)
        if self.steps > 0 and self.steps % 500 == 0:
            self.dna_sequence_length = min(6, self.dna_sequence_length + 1)

        # --- 4. Check Termination ---
        terminated = self.cell_health <= 0
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.cell_health <= 0:
                reward -= 100.0 # Penalty for dying
            elif self.steps >= self.MAX_STEPS:
                reward += 100.0 # Bonus for surviving

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_background()
        self._render_cell()
        self._render_mutations()
        self._render_dna_ui()
        self._render_hud()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.cell_health}

    def _generate_new_dna_puzzle(self):
        self.dna_target_sequence = list(self.np_random.choice(self.DNA_SYMBOLS, size=self.dna_sequence_length))
        shuffled_strands = list(self.dna_target_sequence)
        self.np_random.shuffle(shuffled_strands)
        self.dna_available_strands = shuffled_strands
        self.dna_selection_index = 0

    def _grow_cell(self):
        if not self.cell_nodes: return
        
        last_node = self.cell_nodes[-1]
        
        parent_node_idx = len(self.cell_nodes) - 1
        
        angle = (math.pi / 2) * self.gravity_direction + (self.np_random.random() - 0.5) * (math.pi / 2)
        length = self.np_random.uniform(20, 40)
        
        new_node_pos = [
            last_node[0] + math.cos(angle) * length,
            last_node[1] + math.sin(angle) * length,
        ]

        new_node_pos[0] = np.clip(new_node_pos[0], 10, self.screen_width - 10)
        new_node_pos[1] = np.clip(new_node_pos[1], 10, self.screen_height - 10)
        
        self.cell_nodes.append(new_node_pos)
        new_node_idx = len(self.cell_nodes) - 1
        self.cell_connections.append((parent_node_idx, new_node_idx))

    def _update_mutations(self):
        self.mutation_wave_timer -= 1
        if self.mutation_wave_timer <= 0:
            self.mutation_wave_timer = self.WAVE_INTERVAL
            num_mutations = 5 + self.steps // 150
            for _ in range(num_mutations):
                edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top':
                    pos = [self.np_random.uniform(0, self.screen_width), -10]
                    vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(0.5, 1.5)]
                elif edge == 'bottom':
                    pos = [self.np_random.uniform(0, self.screen_width), self.screen_height + 10]
                    vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-1.5, -0.5)]
                elif edge == 'left':
                    pos = [-10, self.np_random.uniform(0, self.screen_height)]
                    vel = [self.np_random.uniform(0.5, 1.5), self.np_random.uniform(-1, 1)]
                else: # right
                    pos = [self.screen_width + 10, self.np_random.uniform(0, self.screen_height)]
                    vel = [self.np_random.uniform(-1.5, -0.5), self.np_random.uniform(-1, 1)]
                self.mutations.append({'pos': pos, 'vel': vel})

        for m in self.mutations:
            m['pos'][0] += m['vel'][0] * self.mutation_speed
            m['pos'][1] += m['vel'][1] * self.mutation_speed
        
        self.mutations = [m for m in self.mutations if 0 < m['pos'][0] < self.screen_width and 0 < m['pos'][1] < self.screen_height]

    def _check_mutation_collisions(self):
        damage_taken = 0
        initial_mutation_count = len(self.mutations)
        
        surviving_mutations = []
        for m in self.mutations:
            collided = False
            for node_pos in self.cell_nodes:
                dist = math.hypot(m['pos'][0] - node_pos[0], m['pos'][1] - node_pos[1])
                if dist < 10: # 5 radius for node + 5 for mutation
                    damage_taken += 1
                    collided = True
                    break
            if not collided:
                surviving_mutations.append(m)
        
        self.mutations = surviving_mutations
        wave_cleared = initial_mutation_count > 0 and len(self.mutations) == 0
        return damage_taken, wave_cleared

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_DARK)
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        max_radius = int(math.hypot(center_x, center_y))
        for r in range(max_radius, 0, -5):
            alpha = 255 * (1 - r / max_radius)**3
            color = (*self.COLOR_BG_LIGHT, int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, r, color)

    def _render_cell(self):
        for start_idx, end_idx in self.cell_connections:
            p1 = self.cell_nodes[start_idx]
            p2 = self.cell_nodes[end_idx]
            pygame.draw.line(self.screen, self.COLOR_CELL, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 2)

        for x, y in self.cell_nodes:
            ix, iy = int(x), int(y)
            pygame.gfxdraw.filled_circle(self.screen, ix, iy, 8, self.COLOR_CELL_GLOW)
            pygame.gfxdraw.aacircle(self.screen, ix, iy, 8, self.COLOR_CELL_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, ix, iy, 5, self.COLOR_CELL)
            pygame.gfxdraw.aacircle(self.screen, ix, iy, 5, self.COLOR_CELL)

    def _render_mutations(self):
        for m in self.mutations:
            ix, iy = int(m['pos'][0]), int(m['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, ix, iy, 7, self.COLOR_MUTATION_GLOW)
            pygame.gfxdraw.aacircle(self.screen, ix, iy, 7, self.COLOR_MUTATION_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, ix, iy, 4, self.COLOR_MUTATION)
            pygame.gfxdraw.aacircle(self.screen, ix, iy, 4, self.COLOR_MUTATION)
            
    def _render_dna_ui(self):
        self._draw_text("TARGET:", (30, 80), self.font_small, self.COLOR_UI_TEXT)
        for i, symbol in enumerate(self.dna_target_sequence):
            self._draw_dna_symbol(symbol, (60, 110 + i * 30), self.COLOR_DNA_TARGET, is_target=True)

        self._draw_text("AVAILABLE:", (self.screen_width - 150, 80), self.font_small, self.COLOR_UI_TEXT)
        if self.dna_available_strands:
            for i, symbol in enumerate(self.dna_available_strands):
                is_selected = (i == self.dna_selection_index)
                color = self.COLOR_DNA_SELECT if is_selected else self.COLOR_DNA_AVAILABLE
                self._draw_dna_symbol(symbol, (self.screen_width - 120, 110 + i * 30), color, is_selected)

    def _draw_dna_symbol(self, symbol, pos, color, selected=False, is_target=False):
        x, y = pos
        size = 10
        if selected:
            pygame.gfxdraw.box(self.screen, (x - 15, y - 15, 30, 30), (*color, 60))
        
        points = []
        if symbol == 'circle':
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)
            if is_target: pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
        elif symbol == 'square':
            pygame.draw.rect(self.screen, color, (x - size, y - size, size*2, size*2), 1 if not is_target else 0)
        elif symbol == 'triangle':
            points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            if is_target: pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif symbol == 'diamond':
            points = [(x, y - size), (x - size, y), (x, y + size), (x + size, y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            if is_target: pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
    def _render_hud(self):
        self._draw_text(f"SCORE: {self.score:.1f}", (20, 15), self.font_large, self.COLOR_UI_TEXT)
        health_percent = self.cell_health / self.MAX_HEALTH
        health_color = (int(255 * (1 - health_percent)), int(255 * health_percent), 50)
        pygame.draw.rect(self.screen, (50,50,50), (20, 45, 200, 20))
        pygame.draw.rect(self.screen, health_color, (20, 45, max(0, 200 * health_percent), 20))
        self._draw_text(f"HEALTH: {int(self.cell_health)}%", (25, 47), self.font_small, (255,255,255))
        
        self._draw_text(f"NEXT WAVE: {max(0, self.mutation_wave_timer)}", (self.screen_width - 200, 20), self.font_small, self.COLOR_UI_TEXT)
        
        grav_text = "GRAVITY: DOWN" if self.gravity_direction == 1 else "GRAVITY: UP"
        self._draw_text(grav_text, (self.screen_width / 2, self.screen_height - 30), self.font_large, self.COLOR_DNA_SELECT, center=True)
        
        for i, gene in enumerate(self.unlocked_genes):
             self._draw_text(gene, (20 + i * 100, self.screen_height - 30), self.font_small, self.COLOR_CELL)

    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Create a window to display the game
    pygame.display.set_caption("DNA Evolution Environment")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    running = True
    total_reward = 0
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    while running:
        action.fill(0) # Reset actions each frame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- ENV RESET ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])
        
    env.close()