
# Generated: 2025-08-27T19:56:20.762620
# Source Brief: brief_02300.md
# Brief Index: 2300

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Adapted user-facing control string
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place a block."
    )

    # Adapted user-facing description of the game
    game_description = (
        "Build a fortress of blocks to defend your core against waves of enemies. "
        "Strategically place walls to create a long path for the attackers."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Game Constants ---
    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (50, 50, 70)
    COLOR_CORE = (0, 150, 255)
    COLOR_CORE_GLOW = (0, 150, 255, 50)
    COLOR_BLOCK = (100, 255, 100)
    COLOR_BLOCK_GLOW = (100, 255, 100, 40)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Screen & Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 24
    TILE_WIDTH_HALF = 12
    TILE_HEIGHT_HALF = 6
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 60

    # Game Mechanics
    MAX_STEPS = 10000 # Increased to allow for 10 waves
    TOTAL_WAVES = 10
    BASE_HEALTH_START = 100
    INTER_WAVE_DURATION = 90  # 3 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables are initialized in reset()
        self.cursor_pos = None
        self.base_pos = None
        self.base_health = None
        self.blocks = None
        self.enemies = None
        self.particles = None
        self.flow_field = None
        self.wave_number = None
        self.wave_active = None
        self.inter_wave_timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = False
        
        self.np_random = None # Initialized in reset

        # This will call reset() for the first time
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.base_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.base_health = self.BASE_HEALTH_START
        
        self.blocks = set()
        self.enemies = []
        self.particles = []

        self.wave_number = 0
        self.wave_active = False
        self.inter_wave_timer = self.INTER_WAVE_DURATION // 2

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False

        self._update_flow_field()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Player Input ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        if space_held and not self.prev_space_held:
            pos_tuple = tuple(self.cursor_pos)
            if pos_tuple not in self.blocks and pos_tuple != self.base_pos:
                self.blocks.add(pos_tuple)
                self._update_flow_field()
                # sound: block_place.wav
        self.prev_space_held = space_held

        # --- Update Game State ---
        self._update_timers()
        reward += self._update_enemies()
        self._update_particles()
        
        self.steps += 1
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            return
            
        self.wave_active = True
        num_enemies = 2 + self.wave_number
        enemy_speed = 0.02 + self.wave_number * 0.005
        enemy_health = 50 + self.wave_number * 10

        spawn_edges = (
            [(x, 0) for x in range(self.GRID_WIDTH)] +
            [(x, self.GRID_HEIGHT - 1) for x in range(self.GRID_WIDTH)] +
            [(0, y) for y in range(1, self.GRID_HEIGHT - 1)] +
            [(self.GRID_WIDTH - 1, y) for y in range(1, self.GRID_HEIGHT - 1)]
        )
        
        for _ in range(num_enemies):
            grid_pos = self.np_random.choice(spawn_edges)
            screen_pos = self._iso_to_screen(grid_pos[0], grid_pos[1])
            self.enemies.append({
                "grid_pos": list(grid_pos),
                "screen_pos": list(screen_pos),
                "health": enemy_health,
                "speed": enemy_speed
            })
        # sound: wave_start.wav

    def _update_timers(self):
        if not self.wave_active and not self.game_over:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                self._start_next_wave()

    def _update_enemies(self):
        if not self.wave_active:
            return 0
        
        reward = 0
        enemies_to_remove = []

        for i, enemy in enumerate(self.enemies):
            gx, gy = int(enemy["grid_pos"][0]), int(enemy["grid_pos"][1])

            if (gx, gy) == self.base_pos:
                self.base_health -= 10
                self._create_particles(enemy["screen_pos"], self.COLOR_CORE, 20)
                enemies_to_remove.append(i)
                reward -= 1.0 # Penalty for hitting base
                # sound: base_damage.wav
                continue

            if self.flow_field[gy][gx] == -1: # Trapped
                continue

            # Find best next move from flow field
            best_move = (gx, gy)
            min_dist = self.flow_field[gy][gx]
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.flow_field[ny][nx] < min_dist:
                    min_dist = self.flow_field[ny][nx]
                    best_move = (nx, ny)
            
            # Move screen position towards target grid position
            target_screen_pos = self._iso_to_screen(best_move[0], best_move[1])
            direction_x = target_screen_pos[0] - enemy["screen_pos"][0]
            direction_y = target_screen_pos[1] - enemy["screen_pos"][1]
            dist = math.hypot(direction_x, direction_y)

            if dist < 1.0:
                enemy["grid_pos"] = list(best_move)
            else:
                enemy["screen_pos"][0] += (direction_x / dist) * enemy["speed"] * self.TILE_WIDTH_HALF
                enemy["screen_pos"][1] += (direction_y / dist) * enemy["speed"] * self.TILE_HEIGHT_HALF
        
        # This part is for a tower-defense where towers shoot, adapting for this game
        # We can say enemies that hit a block are destroyed
        # This is a simplification from the brief to fit the TD funneling style
        # For this version, let's stick to pathfinding. Damage comes from hypothetical towers.
        # To make it playable, let's say enemies are destroyed when they hit the base.
        # But for an RL agent, it needs a way to kill them.
        # Let's add a simple kill mechanism: if an enemy is forced to move into a block, it dies.
        # This is an indirect way for the player to "attack".
        
        # Let's re-implement enemy update to be more direct.
        # When an enemy moves to a new grid cell, check if it's a block. If so, it dies.
        # This is not standard, but makes the block placement the "attack".
        
        # The above logic is fine, let's add a reward for funneling.
        # Reward for each step an enemy is alive and not at the base?
        # Let's stick to the brief's reward structure.
        # So, how are enemies destroyed? Let's assume an invisible auto-turret on the base.
        for i, enemy in enumerate(self.enemies):
            if i in enemies_to_remove: continue
            dist_to_base = math.hypot(enemy["screen_pos"][0] - self.base_screen_pos[0], enemy["screen_pos"][1] - self.base_screen_pos[1])
            if dist_to_base < 80: # "Turret" range
                enemy["health"] -= 1
                if enemy["health"] <= 0:
                    enemies_to_remove.append(i)
                    self.score += 10
                    reward += 0.1
                    self._create_particles(enemy["screen_pos"], self.COLOR_ENEMY, 15)
                    # sound: enemy_destroy.wav

        if enemies_to_remove:
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in sorted(enemies_to_remove, reverse=True)]

        if self.wave_active and not self.enemies:
            self.wave_active = False
            self.inter_wave_timer = self.INTER_WAVE_DURATION
            self.score += 100 * self.wave_number
            reward += 1.0 # Wave clear bonus
            # sound: wave_clear.wav
            
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True, -100.0
        if self.wave_number > self.TOTAL_WAVES and not self.enemies:
            self.game_over = True
            return True, 50.0
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, 0.0
        return False, 0.0

    def _update_flow_field(self):
        self.flow_field = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), -1, dtype=int)
        q = deque([(self.base_pos, 0)])
        visited = {self.base_pos}

        while q:
            (x, y), dist = q.popleft()
            self.flow_field[y][x] = dist

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited and (nx, ny) not in self.blocks:
                    visited.add((nx, ny))
                    q.append(((nx, ny), dist + 1))

    def _iso_to_screen(self, x, y):
        return (
            self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF,
            self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        )

    def _draw_iso_cube(self, surface, x, y, color, glow_color=None):
        screen_x, screen_y = self._iso_to_screen(x, y)
        
        points = [
            (screen_x, screen_y - self.TILE_HEIGHT_HALF),
            (screen_x + self.TILE_WIDTH_HALF, screen_y),
            (screen_x, screen_y + self.TILE_HEIGHT_HALF),
            (screen_x - self.TILE_WIDTH_HALF, screen_y)
        ]
        
        # Glow effect
        if glow_color:
            pygame.gfxdraw.filled_polygon(surface, [(p[0], p[1]+2) for p in points], glow_color)
            
        # Top face
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF), (sx + self.TILE_WIDTH_HALF, sy),
                    (sx, sy + self.TILE_HEIGHT_HALF), (sx - self.TILE_WIDTH_HALF, sy)
                ]
                pygame.draw.lines(self.screen, self.COLOR_GRID, True, points, 1)

        # Draw placed blocks
        for x, y in self.blocks:
            self._draw_iso_cube(self.screen, x, y, self.COLOR_BLOCK, self.COLOR_BLOCK_GLOW)

        # Draw base
        self.base_screen_pos = self._iso_to_screen(self.base_pos[0], self.base_pos[1])
        self._draw_iso_cube(self.screen, self.base_pos[0], self.base_pos[1], self.COLOR_CORE, self.COLOR_CORE_GLOW)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["screen_pos"][0]), int(enemy["screen_pos"][1]))
            size = 6 + math.sin(self.steps * 0.2) * 1 # Pulse animation
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size+3), self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size), self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(size), self.COLOR_ENEMY)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            pygame.draw.circle(self.screen, color, [int(c) for c in p['pos']], int(p['life'] * 0.2))

        # Draw cursor
        self._draw_iso_cube(self.screen, self.cursor_pos[0], self.cursor_pos[1], self.COLOR_CURSOR)

        # Draw UI
        self._render_text(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", self.font_large, self.COLOR_TEXT, (10, 10))
        self._render_text(f"SCORE: {self.score}", self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 10))
        
        # Base Health Bar
        health_pct = max(0, self.base_health / self.BASE_HEALTH_START)
        health_bar_color = (255 * (1 - health_pct), 255 * health_pct, 50)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 45, 200, 20))
        pygame.draw.rect(self.screen, health_bar_color, (10, 45, 200 * health_pct, 20))
        self._render_text("CORE INTEGRITY", self.font_small, self.COLOR_TEXT, (12, 47), shadow=False)
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "VICTORY" if self.wave_number > self.TOTAL_WAVES else "CORE DESTROYED"
            color = self.COLOR_BLOCK if msg == "VICTORY" else self.COLOR_ENEMY
            self._render_text(msg, pygame.font.Font(None, 80), color, (self.SCREEN_WIDTH/2 - 250, self.SCREEN_HEIGHT/2 - 50))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "enemies_left": len(self.enemies),
        }

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 2
            life = 20 + self.np_random.integers(0, 20)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': life,
                'max_life': life
            })
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example usage:
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset(seed=random.randint(0, 1000))
    done = False
    
    # Re-initialize pygame for display
    pygame.display.set_caption("Isometric Base Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, no-space, no-shift
    
    while not done:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
            
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(30)

    env.close()
    print(f"Game Over. Final Score: {info['score']}")