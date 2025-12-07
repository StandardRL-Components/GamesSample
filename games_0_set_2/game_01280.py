
# Generated: 2025-08-27T16:38:05.690838
# Source Brief: brief_01280.md
# Brief Index: 1280

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place a defensive block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your isometric fortress from waves of enemies by strategically placing blocks to create a deadly maze."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Visuals & Game Constants
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_CURSOR = (255, 255, 0, 100)
        self.COLOR_CORE = (0, 150, 255)
        self.COLOR_CORE_GLOW = (0, 150, 255, 50)
        self.COLOR_BLOCK = (0, 200, 100)
        self.COLOR_BLOCK_SIDE = (0, 150, 75)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_SIDE = (200, 40, 40)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_MSG = (255, 255, 0)
        
        self.font_ui = pygame.font.SysFont("Consolas", 18)
        self.font_msg = pygame.font.SysFont("Consolas", 36, bold=True)
        
        # Grid and Isometric Projection
        self.GRID_W, self.GRID_H = 22, 16
        self.TILE_W, self.TILE_H = 32, 16
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100
        self.BLOCK_HEIGHT = 12

        # Game Parameters
        self.MAX_STEPS = 3000 # Increased from 1000 to allow for 10 waves
        self.MAX_WAVES = 10
        self.INITIAL_BLOCKS = 30
        self.CORE_MAX_HEALTH = 10
        self.INTERMISSION_TIME = 90 # 3 seconds at 30fps

        # Initialize state variables
        self.grid = None
        self.enemies = []
        self.particles = []
        self.cursor_pos = None
        self.last_space_held = False
        self.reward_this_step = 0.0
        self.np_random = None
        
        self.reset()
        # self.validate_implementation() # Uncomment for testing

    def _world_to_iso(self, x, y):
        iso_x = self.ORIGIN_X + (x - y) * (self.TILE_W / 2)
        iso_y = self.ORIGIN_Y + (x + y) * (self.TILE_H / 2)
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, grid_x, grid_y, color_top, color_side, height):
        iso_x, iso_y = self._world_to_iso(grid_x, grid_y)
        
        # Points for the top face
        p_top = (iso_x, iso_y - height)
        p_left = (iso_x - self.TILE_W / 2, iso_y - height + self.TILE_H / 2)
        p_right = (iso_x + self.TILE_W / 2, iso_y - height + self.TILE_H / 2)
        p_bottom = (iso_x, iso_y - height + self.TILE_H)

        # Draw top face
        pygame.gfxdraw.filled_polygon(surface, [p_top, p_left, p_bottom, p_right], color_top)
        pygame.gfxdraw.aapolygon(surface, [p_top, p_left, p_bottom, p_right], color_top)

        # Draw side faces
        if height > 0:
            base_y = iso_y + self.TILE_H
            p_base_left = (p_left[0], p_left[1] + height)
            p_base_bottom = (p_bottom[0], p_bottom[1] + height)
            p_base_right = (p_right[0], p_right[1] + height)
            
            # Left side
            pygame.gfxdraw.filled_polygon(surface, [p_left, p_bottom, p_base_bottom, p_base_left], color_side)
            # Right side
            pygame.gfxdraw.filled_polygon(surface, [p_right, p_bottom, p_base_bottom, p_base_right], color_side)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Grid state: 0=empty, 1=block, 2=core
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.core_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.grid[self.core_pos] = 2
        self.core_health = self.CORE_MAX_HEALTH
        self.core_last_hit_time = -1000

        # Player state
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2 - 4]
        self.blocks_remaining = self.INITIAL_BLOCKS
        self.last_space_held = False

        # Game flow
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.wave_num = 0
        self.time_since_wave_clear = 0
        self.is_intermission = True

        # Entities
        self.enemies = []
        self.particles = []

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0.0
        
        if self.game_over:
            return self._get_observation(), self.reward_this_step, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        
        self.steps += 1
        reward = self.reward_this_step
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100.0
                self.score += 100
            else:
                reward -= 100.0
                self.score -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Place block on key press (not hold)
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.blocks_remaining > 0:
            cx, cy = self.cursor_pos
            if self.grid[cx, cy] == 0: # Can only place on empty tiles
                self.grid[cx, cy] = 1
                self.blocks_remaining -= 1
                # sfx: block_place.wav
                self._create_particles(self._world_to_iso(cx, cy), self.COLOR_BLOCK, 10)

        self.last_space_held = space_held

    def _update_game_state(self):
        # Update intermission
        if self.is_intermission:
            self.time_since_wave_clear += 1
            if self.time_since_wave_clear > self.INTERMISSION_TIME:
                self._spawn_wave()
        
        # Update enemies
        for enemy in list(self.enemies):
            enemy['cooldown'] = max(0, enemy['cooldown'] - 1)
            if enemy['cooldown'] > 0:
                continue

            gx, gy = int(round(enemy['pos'][0])), int(round(enemy['pos'][1]))
            
            # Determine target: move towards core
            dx, dy = self.core_pos[0] - gx, self.core_pos[1] - gy
            
            # Choose next grid step
            next_gx, next_gy = gx, gy
            if abs(dx) > abs(dy):
                next_gx += np.sign(dx)
            else:
                next_gy += np.sign(dy)

            # Check target cell
            if 0 <= next_gx < self.GRID_W and 0 <= next_gy < self.GRID_H:
                target_cell = self.grid[next_gx, next_gy]
                
                if target_cell == 2: # Hit core
                    self.core_health -= 1
                    self.core_last_hit_time = self.steps
                    self.enemies.remove(enemy)
                    self.reward_this_step -= 5.0 # Penalty for core hit
                    self.score -= 5
                    # sfx: core_hit.wav
                    iso_pos = self._world_to_iso(self.core_pos[0], self.core_pos[1])
                    self._create_particles(iso_pos, self.COLOR_ENEMY, 30)
                    continue

                elif target_cell == 1: # Hit block
                    enemy['health'] -= 1
                    enemy['cooldown'] = 10 # Stunned after hitting block
                    # sfx: enemy_hit_block.wav
                    iso_pos = self._world_to_iso(next_gx, next_gy)
                    self._create_particles(iso_pos, self.COLOR_BLOCK, 5)
                    if enemy['health'] <= 0:
                        self.enemies.remove(enemy)
                        self.reward_this_step += 0.1
                        self.score += 1
                        # sfx: enemy_die.wav
                        self._create_particles(iso_pos, self.COLOR_ENEMY, 20)
                    continue
            
            # Move if not blocked
            enemy['pos'][0] += np.sign(dx) * enemy['speed']
            enemy['pos'][1] += np.sign(dy) * enemy['speed']

        # Update particles
        for p in list(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Check for wave clear
        if not self.enemies and not self.is_intermission:
            self.is_intermission = True
            self.time_since_wave_clear = 0
            if self.wave_num > 0:
                self.reward_this_step += 1.0
                self.score += 10
                # sfx: wave_clear.wav
            if self.wave_num >= self.MAX_WAVES:
                self.win = True

    def _spawn_wave(self):
        if self.wave_num >= self.MAX_WAVES:
            return
        self.wave_num += 1
        self.is_intermission = False
        
        num_enemies = 2 + self.wave_num
        enemy_speed = 0.04 + self.wave_num * 0.005
        enemy_health = 1 + self.wave_num
        
        for _ in range(num_enemies):
            # Spawn at random edge
            side = self.np_random.integers(4)
            if side == 0: x, y = self.np_random.integers(self.GRID_W), 0
            elif side == 1: x, y = self.np_random.integers(self.GRID_W), self.GRID_H - 1
            elif side == 2: x, y = 0, self.np_random.integers(self.GRID_H)
            else: x, y = self.GRID_W - 1, self.np_random.integers(self.GRID_H)
            
            if self.grid[x, y] != 0: # Avoid spawning in blocks/core
                x, y = 0, 0

            self.enemies.append({
                'pos': np.array([float(x), float(y)]),
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'cooldown': 0
            })
        # sfx: new_wave.wav

    def _check_termination(self):
        return self.core_health <= 0 or self.steps >= self.MAX_STEPS or self.win

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                iso_x, iso_y = self._world_to_iso(x, y)
                points = [
                    (iso_x, iso_y),
                    (iso_x - self.TILE_W / 2, iso_y + self.TILE_H / 2),
                    (iso_x, iso_y + self.TILE_H),
                    (iso_x + self.TILE_W / 2, iso_y + self.TILE_H / 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw blocks and core
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.grid[x, y] == 1: # Block
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_BLOCK, self.COLOR_BLOCK_SIDE, self.BLOCK_HEIGHT)
                elif self.grid[x, y] == 2: # Core
                    pulse = abs(math.sin(self.steps * 0.05))
                    glow_radius = int(self.TILE_W * (1.5 + pulse * 0.5))
                    
                    # Core hit flash
                    if self.steps - self.core_last_hit_time < 10:
                        flash_alpha = 150 * (1 - (self.steps - self.core_last_hit_time) / 10)
                        pygame.gfxdraw.filled_circle(self.screen, self.ORIGIN_X, self.HEIGHT-30, self.WIDTH, (255,0,0, int(flash_alpha)))

                    # Glow
                    pygame.gfxdraw.filled_circle(self.screen, *self._world_to_iso(x, y), glow_radius, self.COLOR_CORE_GLOW)
                    
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_CORE, self.COLOR_CORE, self.BLOCK_HEIGHT * 1.5)

        # Draw cursor
        cx, cy = self.cursor_pos
        iso_x, iso_y = self._world_to_iso(cx, cy)
        cursor_points = [
            (iso_x, iso_y),
            (iso_x - self.TILE_W / 2, iso_y + self.TILE_H / 2),
            (iso_x, iso_y + self.TILE_H),
            (iso_x + self.TILE_W / 2, iso_y + self.TILE_H / 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, cursor_points, self.COLOR_CURSOR)
        
        # Draw enemies
        for enemy in self.enemies:
            self._draw_iso_cube(self.screen, enemy['pos'][0], enemy['pos'][1], self.COLOR_ENEMY, self.COLOR_ENEMY_SIDE, self.BLOCK_HEIGHT * 0.8)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['life'] / p['max_life'] * p['size']), color)

    def _render_ui(self):
        # UI text
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        wave_text = self.font_ui.render(f"WAVE: {self.wave_num}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        blocks_text = self.font_ui.render(f"BLOCKS: {self.blocks_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(wave_text, (10, 30))
        self.screen.blit(blocks_text, (10, 50))
        
        # Core Health Bar
        health_bar_w = 100
        health_bar_h = 10
        health_pct = self.core_health / self.CORE_MAX_HEALTH
        pygame.draw.rect(self.screen, (80, 0, 0), (self.WIDTH - health_bar_w - 10, 10, health_bar_w, health_bar_h))
        pygame.draw.rect(self.screen, self.COLOR_CORE, (self.WIDTH - health_bar_w - 10, 10, health_bar_w * health_pct, health_bar_h))

        # Game Over / Win Message
        if self.game_over:
            msg_text = "VICTORY" if self.win else "FORTRESS LOST"
            rendered_msg = self.font_msg.render(msg_text, True, self.COLOR_MSG)
            msg_rect = rendered_msg.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(rendered_msg, msg_rect)
        elif self.is_intermission and self.wave_num < self.MAX_WAVES:
            msg_text = f"WAVE {self.wave_num+1} INCOMING"
            rendered_msg = self.font_msg.render(msg_text, True, self.COLOR_MSG)
            msg_rect = rendered_msg.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(rendered_msg, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_num,
            "core_health": self.core_health,
            "blocks_remaining": self.blocks_remaining,
        }

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(10, 25)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Fortress Defense")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Human input mapping ---
        movement = 0 # none
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
        
        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}")
    env.close()