import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:57:50.447134
# Source Brief: brief_02017.md
# Brief Index: 2017
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cellular automata tower defense game.

    The agent must place and evolve cellular automata towers to defend a central
    core against waves of mutating attackers. The core gameplay involves strategic
    tower placement, managing resources (Energy), and selecting the right cellular
    automata rules to create effective defensive barriers.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): 0=none, 1=up, 2=down, 3=left, 4=right. Controls the placement cursor.
    - `actions[1]` (Place Tower): 0=released, 1=held. Places a tower seed at the cursor.
    - `actions[2]` (Cycle Rule): 0=released, 1=held. Cycles through available tower rules.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - `+0.1` for each mutation destroyed.
    - `+5.0` for completing a wave.
    - `-1.0` each time the core is hit by a mutation.
    - `+100.0` for surviving all 50 waves.
    - `-100.0` if the core is destroyed.
    - `-0.001` per step to encourage efficiency.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your core against waves of mutating attackers by strategically placing and evolving cellular automata towers."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a tower and shift to cycle through tower rules."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 64, 40
    CELL_SIZE = 10
    MAX_STEPS = 5000
    TOTAL_WAVES = 50

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_CORE = (0, 255, 255)
    COLOR_CORE_DMG = (255, 100, 100)
    COLOR_TOWER_BASE = (255, 200, 0)
    COLOR_TOWER_CELL = (200, 255, 200)
    COLOR_MUTATION = (255, 50, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_ENERGY = (50, 150, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Cellular Automata Rules
        self.ca_rules = [
            {'name': "Game of Life", 'B': [3], 'S': [2, 3]},
            {'name': "HighLife", 'B': [3, 6], 'S': [2, 3]},
            {'name': "Seeds", 'B': [2], 'S': []},
            {'name': "Replicator", 'B': [1, 3, 5, 7], 'S': [1, 3, 5, 7]},
            {'name': "Day & Night", 'B': [3, 6, 7, 8], 'S': [3, 4, 6, 7, 8]},
        ]

        # Initialize state variables
        self._initialize_state()
        
    def _initialize_state(self):
        """Initializes or resets all game state variables."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        self.tower_info = {} # (gx, gy) -> {'rule_index': int}
        
        self.core_pos_grid = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.core_pos_px = pygame.math.Vector2(
            self.core_pos_grid[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
            self.core_pos_grid[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        )
        self.grid[self.core_pos_grid] = 2 # 2 = Core
        
        self.core_health = 100
        self.max_core_health = 100
        self.energy = 50
        self.tower_cost = 25

        self.wave_number = 0
        self.wave_timer = 150 # Steps until next wave
        self.wave_prep_time = 150 
        
        self.mutations = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 4]
        self.unlocked_rules = [0]
        self.selected_rule_index = 0

        self.last_space_held = False
        self.last_shift_held = False
        self.last_core_hit_step = -100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        self._start_next_wave()
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001 # Small penalty per step
        self.steps += 1
        
        self._handle_input(action)
        
        self.wave_timer -= 1
        if self.wave_timer <= 0:
            self._start_next_wave()
            
        reward += self._update_game_logic()
        
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # --- Place Tower (on press) ---
        if space_held and not self.last_space_held:
            if self.energy >= self.tower_cost and self.grid[tuple(self.cursor_pos)] == 0:
                self.energy -= self.tower_cost
                gx, gy = self.cursor_pos
                self.grid[gx, gy] = 3 # 3 = Tower Base
                self.tower_info[(gx, gy)] = {'rule_index': self.selected_rule_index}
                # Sound: Place tower sfx
                # Spawn some placement particles
                for _ in range(10):
                    self.particles.append(self._create_particle(
                        (gx * self.CELL_SIZE + 5, gy * self.CELL_SIZE + 5), self.COLOR_TOWER_BASE, 20
                    ))


        # --- Cycle Rule (on press) ---
        if shift_held and not self.last_shift_held:
            self.selected_rule_index = (self.selected_rule_index + 1) % len(self.unlocked_rules)
            # Sound: UI cycle sfx
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_logic(self):
        step_reward = 0
        
        # Update towers (Cellular Automata) every 15 steps
        if self.steps % 15 == 0:
            self._update_towers()

        self._update_mutations()
        step_reward += self._check_collisions()
        self._update_particles()
        
        # Check for wave completion
        if self.wave_timer > self.wave_prep_time and not self.mutations:
            step_reward += 5.0 # Wave complete reward
            self.wave_timer = self.wave_prep_time # Start prep for next wave
            
            # Unlock new rule every 5 waves
            if self.wave_number % 5 == 0 and self.wave_number > 0:
                if len(self.unlocked_rules) < len(self.ca_rules):
                    self.unlocked_rules.append(len(self.unlocked_rules))
                    # Sound: Unlock sfx
        
        return step_reward

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_timer = 30 * 20 # 20 seconds per wave
        
        num_mutations = 5 + (self.wave_number - 1)
        speed = 0.5 + (self.wave_number - 1) * 0.05
        
        for _ in range(num_mutations):
            self._spawn_mutation(speed)
        # Sound: Wave start horn

    def _spawn_mutation(self, speed):
        edge = random.randint(0, 3)
        if edge == 0: # Top
            pos = pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), -10)
        elif edge == 1: # Bottom
            pos = pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10)
        elif edge == 2: # Left
            pos = pygame.math.Vector2(-10, random.uniform(0, self.SCREEN_HEIGHT))
        else: # Right
            pos = pygame.math.Vector2(self.SCREEN_WIDTH + 10, random.uniform(0, self.SCREEN_HEIGHT))
        
        self.mutations.append({
            'pos': pos,
            'speed': speed,
            'radius': 5
        })
        
    def _update_towers(self):
        next_grid = self.grid.copy()
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                # Count neighbors
                neighbors = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                            if self.grid[nx, ny] == 4: # 4 = Tower Cell
                                neighbors += 1
                
                # Apply rules
                state = self.grid[x, y]
                # Find the rule for this region. This is a simplification.
                # A better model would track which tower base a cell belongs to.
                # For now, we use the selected rule for all new growth.
                rule = self.ca_rules[self.unlocked_rules[self.selected_rule_index]]
                
                if state == 4: # Alive cell
                    if neighbors not in rule['S']:
                        next_grid[x, y] = 0 # Dies
                elif state == 0 or state == 3: # Empty or Base
                    if neighbors in rule['B']:
                        next_grid[x, y] = 4 # Born
                        
        self.grid = next_grid

    def _update_mutations(self):
        for m in self.mutations:
            # Vector to core
            dir_to_core = (self.core_pos_px - m['pos']).normalize()
            
            # Repulsion from towers
            repulsion = pygame.math.Vector2(0, 0)
            repulsion_radius = 50
            tower_cells = np.argwhere((self.grid == 3) | (self.grid == 4))
            
            for gx, gy in tower_cells:
                tower_px = pygame.math.Vector2(gx * self.CELL_SIZE + 5, gy * self.CELL_SIZE + 5)
                dist_vec = m['pos'] - tower_px
                if 0 < dist_vec.length_squared() < repulsion_radius**2:
                    repulsion += dist_vec.normalize() / max(0.1, dist_vec.length())

            # Combine forces
            if repulsion.length() > 0:
                repulsion.normalize_ip()
            
            final_dir = dir_to_core + repulsion * 1.5
            if final_dir.length() > 0:
                final_dir.normalize_ip()
            
            m['pos'] += final_dir * m['speed']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

    def _check_collisions(self):
        reward = 0
        mutations_to_remove = []
        for i, m in enumerate(self.mutations):
            # Mutation vs Core
            if m['pos'].distance_to(self.core_pos_px) < m['radius'] + 10:
                self.core_health -= 10
                self.last_core_hit_step = self.steps
                reward -= 1.0
                mutations_to_remove.append(i)
                # Sound: Core damage sfx
                for _ in range(20):
                    self.particles.append(self._create_particle(m['pos'], self.COLOR_CORE_DMG, 40))
                continue
                
            # Mutation vs Tower
            mx, my = int(m['pos'].x / self.CELL_SIZE), int(m['pos'].y / self.CELL_SIZE)
            if 0 <= mx < self.GRID_COLS and 0 <= my < self.GRID_ROWS:
                if self.grid[mx, my] == 3 or self.grid[mx, my] == 4:
                    reward += 0.1
                    self.energy += 1
                    mutations_to_remove.append(i)
                    # Sound: Mutation destroyed sfx
                    for _ in range(15):
                        self.particles.append(self._create_particle(m['pos'], self.COLOR_MUTATION, 30))
                    continue

            # Despawn if off-screen
            if not self.screen.get_rect().inflate(50, 50).collidepoint(m['pos']):
                mutations_to_remove.append(i)

        if mutations_to_remove:
            self.mutations = [m for i, m in enumerate(self.mutations) if i not in mutations_to_remove]
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
            
        if self.core_health <= 0:
            self.score -= 100
            self.game_over = True
            return True
            
        if self.wave_number > self.TOTAL_WAVES:
            self.score += 100
            self.game_over = True
            return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "core_health": self.core_health,
            "energy": self.energy,
            "wave": self.wave_number,
            "mutations": len(self.mutations)
        }
        
    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw grid cells
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                cell_type = self.grid[x, y]
                if cell_type == 3: # Tower Base
                    pygame.draw.rect(self.screen, self.COLOR_TOWER_BASE, rect.inflate(-2, -2))
                elif cell_type == 4: # Tower Cell
                    pygame.draw.rect(self.screen, self.COLOR_TOWER_CELL, rect.inflate(-2, -2))

        # Draw Core
        core_color = self.COLOR_CORE if self.steps - self.last_core_hit_step > 15 else self.COLOR_CORE_DMG
        core_radius = 10 + 3 * math.sin(self.steps * 0.1)
        pygame.gfxdraw.filled_circle(self.screen, int(self.core_pos_px.x), int(self.core_pos_px.y), int(core_radius), core_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.core_pos_px.x), int(self.core_pos_px.y), int(core_radius), core_color)

        # Draw Mutations
        for m in self.mutations:
            pos = (int(m['pos'].x), int(m['pos'].y))
            radius = int(m['radius'])
            # Glow effect
            glow_color = (*self.COLOR_MUTATION, 60)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 3, glow_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 3, glow_color)
            # Main circle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_MUTATION)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_MUTATION)

        # Draw Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_life']))))
            color = (*p['color'], alpha)
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Draw Cursor
        cursor_rect = pygame.Rect(
            self.cursor_pos[0] * self.CELL_SIZE, 
            self.cursor_pos[1] * self.CELL_SIZE, 
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.core_health / self.max_core_health
        health_bar_width = 200
        health_bar_rect = pygame.Rect(self.SCREEN_WIDTH // 2 - health_bar_width // 2, 10, health_bar_width * health_ratio, 15)
        pygame.draw.rect(self.screen, self.COLOR_CORE, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2 - health_bar_width // 2, 10, health_bar_width, 15), 1)
        
        # UI Text
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 30))

        energy_text = self.font_small.render(f"ENERGY: {self.energy}", True, self.COLOR_ENERGY)
        self.screen.blit(energy_text, (self.SCREEN_WIDTH - energy_text.get_width() - 10, 10))

        rule_idx = self.unlocked_rules[self.selected_rule_index]
        rule_name = self.ca_rules[rule_idx]['name']
        rule_text = self.font_small.render(f"RULE: {rule_name}", True, self.COLOR_TEXT)
        self.screen.blit(rule_text, (self.SCREEN_WIDTH - rule_text.get_width() - 10, 30))
        
        if self.game_over:
            outcome = "VICTORY" if self.wave_number > self.TOTAL_WAVES else "CORE DESTROYED"
            end_text = self.font_large.render(outcome, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _create_particle(self, pos, color, max_life):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        return {
            'pos': pygame.math.Vector2(pos),
            'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
            'lifespan': random.randint(max_life // 2, max_life),
            'max_life': max_life,
            'color': color,
            'radius': random.uniform(2, 5)
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    clock = pygame.time.Clock()
    
    # Create a display for manual playing
    pygame.display.set_caption("Cellular Automata Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = [0, 0, 0] # no-op, release, release
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        # Actions
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']}")
            obs, info = env.reset() # Auto-reset after a game over

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()