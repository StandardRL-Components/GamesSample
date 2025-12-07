import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment: Crystal Core Defense.

    In this action-puzzle game, the player defends a central, immobile core
    against waves of encroaching crystal creatures. The playfield is a cave
    that continuously shrinks, adding time pressure.

    The primary mechanic involves placing and growing different types of crystals
    on a grid. Activating a crystal triggers a chain reaction with adjacent
    crystals of the same type, destroying nearby enemies. The goal is to
    survive for a set duration by strategically managing crystal placement
    and triggering massive chain reactions.

    - **Action Space**: `MultiDiscrete([5, 2, 2])`
        - `action[0]`: Move cursor (0:None, 1:Up, 2:Down, 3:Left, 4:Right)
        - `action[1]`: Grow/Activate Crystal (Spacebar)
        - `action[2]`: Shrink/Reclaim Crystal (Shift)

    - **Observation Space**: A 640x400 RGB image of the game state.

    - **Rewards**:
        - Positive rewards for destroying enemies, triggering chain reactions,
          and unlocking new crystal types.
        - Negative rewards for spending energy.
        - Large terminal rewards for winning (surviving) or losing.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend a central core from encroaching crystal creatures by placing and activating crystals on a grid. "
        "Survive for as long as you can as the cave walls close in."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place, grow, or activate a crystal. "
        "Hold shift to reclaim a crystal."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    GRID_COLS = 11
    GRID_ROWS = 7
    CELL_SIZE = 40
    
    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_WALLS = (40, 35, 50)
    COLOR_GRID = (50, 45, 70)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_ENEMY = (210, 40, 40)
    COLOR_ENEMY_GLOW = (210, 40, 40, 80)
    COLOR_EXIT = (255, 255, 255)
    CRYSTAL_COLORS = [
        (60, 220, 180),   # Cyan
        (255, 200, 50),   # Yellow
        (220, 50, 220),   # Magenta
        (100, 255, 100),  # Green
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        self.grid_width = self.GRID_COLS * self.CELL_SIZE
        self.grid_height = self.GRID_ROWS * self.CELL_SIZE
        self.grid_origin_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_origin_y = (self.SCREEN_HEIGHT - self.grid_height) // 2

        self.cursor_pos = [0, 0]
        self.grid = []
        self.enemies = []
        self.particles = []
        
        self.walls = {}
        self.wall_shrink_rate = 0.08
        
        self.player_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.player_radius = 15

        self.crystal_energy = 0
        self.max_crystal_energy = 100
        self.energy_cost_per_grow = 5
        self.energy_gain_per_shrink = 3

        self.unlocked_crystal_types = 1
        self.active_crystal_type_idx = 0

        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 80
        self.enemy_speed = 0.5
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.grid = [[(0, 0) for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)] # (type_idx, size)
        
        self.enemies = []
        self.particles = []
        
        self.walls = {
            'left': 0, 'right': self.SCREEN_WIDTH,
            'top': 0, 'bottom': self.SCREEN_HEIGHT
        }
        
        self.crystal_energy = self.max_crystal_energy
        self.unlocked_crystal_types = 1
        self.active_crystal_type_idx = 0

        self.enemy_spawn_timer = 0
        self.enemy_speed = 0.5
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        self._handle_input(action)
        self._update_game_state()
        self._check_collisions_and_termination()
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
            # Victory condition
            self.reward_this_step += 100
            self.game_over = True

        reward = self.reward_this_step
        self.score += reward

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
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)
        
        cx, cy = self.cursor_pos
        crystal_type, crystal_size = self.grid[cx][cy]

        # --- Actions ---
        if space_held: # Grow or Activate
            if crystal_size == 0 and self.crystal_energy >= self.energy_cost_per_grow:
                # Place new crystal
                self.grid[cx][cy] = (self.active_crystal_type_idx, 1)
                self.crystal_energy -= self.energy_cost_per_grow
                self.reward_this_step -= 0.1 # Cost for spending energy
            elif 0 < crystal_size < 3 and self.crystal_energy >= self.energy_cost_per_grow:
                # Grow existing crystal
                self.grid[cx][cy] = (crystal_type, crystal_size + 1)
                self.crystal_energy -= self.energy_cost_per_grow
                self.reward_this_step -= 0.1
            elif crystal_size == 3:
                # Activate max-size crystal
                self._trigger_chain_reaction(cx, cy, crystal_type)

        if shift_held and crystal_size > 0: # Shrink
            self.grid[cx][cy] = (crystal_type, crystal_size - 1)
            self.crystal_energy = min(self.max_crystal_energy, self.crystal_energy + self.energy_gain_per_shrink)

    def _update_game_state(self):
        # Shrink walls
        self.walls['left'] += self.wall_shrink_rate
        self.walls['right'] -= self.wall_shrink_rate
        self.walls['top'] += self.wall_shrink_rate
        self.walls['bottom'] -= self.wall_shrink_rate

        # Update and spawn enemies
        self._update_enemies()
        self._spawn_enemies()
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # Unlock new crystals and scale difficulty
        if self.steps > 0:
            if self.steps % 400 == 0 and self.unlocked_crystal_types < len(self.CRYSTAL_COLORS):
                self.unlocked_crystal_types += 1
                self.active_crystal_type_idx = (self.active_crystal_type_idx + 1) % self.unlocked_crystal_types
                self.reward_this_step += 5.0 # Reward for unlock
            if self.steps % 50 == 0:
                self.enemy_speed += 0.05
            if self.steps % 200 == 0:
                self.enemy_spawn_rate = max(20, self.enemy_spawn_rate - 10)

    def _update_enemies(self):
        for enemy in self.enemies:
            direction = np.array(self.player_pos) - np.array(enemy['pos'])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            enemy['pos'] += direction * self.enemy_speed
            enemy['angle'] += 0.05 # Simple rotation animation

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self.enemy_spawn_timer = self.enemy_spawn_rate
            
            side = random.randint(0, 3)
            if side == 0: # Left
                x, y = self.walls['left'] - 20, random.uniform(self.walls['top'], self.walls['bottom'])
            elif side == 1: # Right
                x, y = self.walls['right'] + 20, random.uniform(self.walls['top'], self.walls['bottom'])
            elif side == 2: # Top
                x, y = random.uniform(self.walls['left'], self.walls['right']), self.walls['top'] - 20
            else: # Bottom
                x, y = random.uniform(self.walls['left'], self.walls['right']), self.walls['bottom'] + 20
            
            self.enemies.append({'pos': np.array([x, y]), 'radius': 8, 'angle': 0})

    def _trigger_chain_reaction(self, start_x, start_y, chain_type):
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        chain_count = 0

        while q:
            cx, cy = q.popleft()
            
            if not (0 <= cx < self.GRID_COLS and 0 <= cy < self.GRID_ROWS): continue
            
            crystal_type, crystal_size = self.grid[cx][cy]
            if crystal_type != chain_type or crystal_size == 0: continue
            
            chain_count += 1
            
            cell_center_x = self.grid_origin_x + cx * self.CELL_SIZE + self.CELL_SIZE / 2
            cell_center_y = self.grid_origin_y + cy * self.CELL_SIZE + self.CELL_SIZE / 2
            self._create_explosion(cell_center_x, cell_center_y, self.CRYSTAL_COLORS[chain_type], 40)
            
            explosion_radius = self.CELL_SIZE * 1.5
            enemies_destroyed_this_pop = 0
            for enemy in self.enemies[:]:
                dist = np.linalg.norm(enemy['pos'] - np.array([cell_center_x, cell_center_y]))
                if dist < explosion_radius:
                    self.enemies.remove(enemy)
                    enemies_destroyed_this_pop += 1
                    self._create_explosion(enemy['pos'][0], enemy['pos'][1], self.COLOR_ENEMY, 20)
            
            if enemies_destroyed_this_pop > 0:
                self.reward_this_step += 1.0 * enemies_destroyed_this_pop
            
            self.grid[cx][cy] = (0, 0)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        
        if chain_count > 1:
            self.reward_this_step += 2.0 * (chain_count - 1) # Reward for a chain reaction

    def _check_collisions_and_termination(self):
        if self.game_over: return
        
        for enemy in self.enemies:
            dist = np.linalg.norm(enemy['pos'] - np.array(self.player_pos))
            if dist < self.player_radius + enemy['radius']:
                self.game_over = True
                self.reward_this_step -= 100
                return
                
        if (self.player_pos[0] - self.player_radius < self.walls['left'] or
            self.player_pos[0] + self.player_radius > self.walls['right'] or
            self.player_pos[1] - self.player_radius < self.walls['top'] or
            self.player_pos[1] + self.player_radius > self.walls['bottom']):
            self.game_over = True
            self.reward_this_step -= 100
            return

    def _create_explosion(self, x, y, color, num_particles):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_WALLS, (0, 0, self.SCREEN_WIDTH, self.walls['top']))
        pygame.draw.rect(self.screen, self.COLOR_WALLS, (0, self.walls['bottom'], self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.walls['bottom']))
        pygame.draw.rect(self.screen, self.COLOR_WALLS, (0, 0, self.walls['left'], self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALLS, (self.walls['right'], 0, self.SCREEN_WIDTH - self.walls['right'], self.SCREEN_HEIGHT))

        for x in range(self.GRID_COLS + 1):
            start = (self.grid_origin_x + x * self.CELL_SIZE, self.grid_origin_y)
            end = (self.grid_origin_x + x * self.CELL_SIZE, self.grid_origin_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_ROWS + 1):
            start = (self.grid_origin_x, self.grid_origin_y + y * self.CELL_SIZE)
            end = (self.grid_origin_x + self.grid_width, self.grid_origin_y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        for cx in range(self.GRID_COLS):
            for cy in range(self.GRID_ROWS):
                ctype, csize = self.grid[cx][cy]
                if csize > 0:
                    center_x = self.grid_origin_x + cx * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.grid_origin_y + cy * self.CELL_SIZE + self.CELL_SIZE // 2
                    radius = int(csize * 4 + 4)
                    color = self.CRYSTAL_COLORS[ctype]
                    
                    surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(surface, radius, radius, radius, (*color, 100))
                    pygame.gfxdraw.aacircle(surface, radius, radius, radius, (*color, 200))
                    self.screen.blit(surface, (center_x - radius, center_y - radius))

        cursor_x = self.grid_origin_x + self.cursor_pos[0] * self.CELL_SIZE
        cursor_y = self.grid_origin_y + self.cursor_pos[1] * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE), 2)
        
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        glow_surf = pygame.Surface((self.player_radius * 4, self.player_radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.player_radius*2, self.player_radius*2), self.player_radius*2)
        self.screen.blit(glow_surf, (px - self.player_radius*2, py - self.player_radius*2))
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.player_radius, self.COLOR_PLAYER)
        
        if self.steps > self.MAX_STEPS - 300:
            alpha = min(255, int((self.steps - (self.MAX_STEPS - 300)) / 300 * 255))
            exit_rect = pygame.Rect(self.player_pos[0] - 50, self.player_pos[1] - 50, 100, 100)
            s = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_EXIT, alpha), s.get_rect(), 4, border_radius=10)
            self.screen.blit(s, exit_rect.topleft)

        for enemy in self.enemies:
            ex, ey = int(enemy['pos'][0]), int(enemy['pos'][1])
            radius = int(enemy['radius'])
            glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_ENEMY_GLOW, (radius*2, radius*2), radius*2)
            self.screen.blit(glow_surf, (ex - radius*2, ey - radius*2))
            points = []
            for i in range(5):
                angle = enemy['angle'] + (i * 2 * math.pi / 5)
                points.append((ex + math.cos(angle) * radius, ey + math.sin(angle) * radius))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(5 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_EXIT)
        steps_text = self.font_small.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_EXIT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 30))

        energy_text = self.font_small.render("ENERGY", True, self.COLOR_EXIT)
        bar_w = 150
        bar_h = 15
        bar_x = self.SCREEN_WIDTH - bar_w - 10
        bar_y = 10
        fill_w = (self.crystal_energy / self.max_crystal_energy) * bar_w
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (bar_x, bar_y, bar_w, bar_h), 1)
        self.screen.blit(energy_text, (bar_x, bar_y + bar_h + 5))
        
        ui_cx = self.SCREEN_WIDTH // 2
        ui_cy = self.SCREEN_HEIGHT - 30
        
        for i in range(self.unlocked_crystal_types):
            color = self.CRYSTAL_COLORS[i]
            x_pos = ui_cx + (i - (self.unlocked_crystal_types - 1) / 2) * 40
            radius = 12
            
            surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(surface, radius, radius, radius, (*color, 150))
            pygame.gfxdraw.aacircle(surface, radius, radius, radius, color)
            self.screen.blit(surface, (x_pos - radius, ui_cy - radius))
            
            if i == self.active_crystal_type_idx:
                pygame.gfxdraw.aacircle(self.screen, int(x_pos), int(ui_cy), radius + 4, self.COLOR_EXIT)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.crystal_energy,
            "enemies": len(self.enemies),
            "unlocked_crystals": self.unlocked_crystal_types
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Core Defense")
    
    running = True
    clock = pygame.time.Clock()
    total_reward = 0
    
    while running:
        action = [0, 0, 0]  # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30)
        
    env.close()