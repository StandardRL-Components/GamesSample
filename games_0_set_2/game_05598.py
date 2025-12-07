import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to place a standard block and shift to place a reinforced block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your fortress from falling projectiles by strategically placing reinforcing blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 12
        self.CELL_SIZE = 30
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = 30
        self.MAX_STEPS = 1800  # 60 seconds * 30 FPS
        self.STARTING_BLOCKS = 50
        self.BASE_HEALTH_MAX = 100

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_BASE = (46, 204, 113)
        self.COLOR_BASE_DMG = (231, 76, 60)
        self.COLOR_PROJECTILE = (231, 76, 60)
        self.COLOR_PROJECTILE_GLOW = (255, 120, 100, 100)
        self.COLOR_BLOCK_STD = (52, 152, 219)
        self.COLOR_BLOCK_STD_DARK = (41, 128, 185)
        self.COLOR_BLOCK_REIN = (155, 89, 182)
        self.COLOR_BLOCK_REIN_DARK = (142, 68, 173)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_CURSOR_VALID = (46, 204, 113, 150)
        self.COLOR_CURSOR_INVALID = (231, 76, 60, 150)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # Initialize state variables
        self.cursor_pos = None
        self.base_health = None
        self.available_blocks = None
        self.projectiles = None
        self.placed_blocks = None
        self.particles = None
        self.projectile_spawn_rate = None
        self.action_cooldown = None
        self.steps = None
        self.score = None
        self.game_over = None

        # self.reset() is called here to set up initial state
        # self.validate_implementation() is called in the original code but removed
        # for submission as it's a testing utility. If you wish to run it,
        # you can uncomment it after the reset call.
        # self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.action_cooldown = 0

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.base_health = self.BASE_HEALTH_MAX
        self.available_blocks = self.STARTING_BLOCKS
        self.projectiles = []
        self.particles = []

        # Procedurally generate the initial fortress
        self.placed_blocks = {}
        base_y = self.GRID_HEIGHT - 2
        for x in range(self.GRID_WIDTH // 2 - 3, self.GRID_WIDTH // 2 + 3):
            self.placed_blocks[(x, base_y)] = {'type': 1, 'health': 1}
        self.placed_blocks[(self.GRID_WIDTH // 2 - 2, base_y - 1)] = {'type': 1, 'health': 1}
        self.placed_blocks[(self.GRID_WIDTH // 2 + 1, base_y - 1)] = {'type': 1, 'health': 1}
        self.placed_blocks[(self.GRID_WIDTH // 2 - 1, base_y - 2)] = {'type': 2, 'health': 3}
        self.placed_blocks[(self.GRID_WIDTH // 2, base_y - 2)] = {'type': 2, 'health': 3}

        self.projectile_spawn_rate = 0.02

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # -- Handle Actions --
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1

        if self.action_cooldown > 0:
            self.action_cooldown -= 1

        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Block placement (prioritize reinforced)
        can_place = self.action_cooldown == 0 and tuple(self.cursor_pos) not in self.placed_blocks

        if shift_pressed and can_place and self.available_blocks >= 2:
            self.placed_blocks[tuple(self.cursor_pos)] = {'type': 2, 'health': 3}
            self.available_blocks -= 2
            reward -= 0.02 # Cost for placing a reinforced block
            self.action_cooldown = 5 # Cooldown after placing
            self._create_particles(self._grid_to_pixel(self.cursor_pos), self.COLOR_BLOCK_REIN, 10)
        elif space_pressed and can_place and self.available_blocks >= 1:
            self.placed_blocks[tuple(self.cursor_pos)] = {'type': 1, 'health': 1}
            self.available_blocks -= 1
            reward -= 0.01 # Cost for placing a standard block
            self.action_cooldown = 5 # Cooldown after placing
            self._create_particles(self._grid_to_pixel(self.cursor_pos), self.COLOR_BLOCK_STD, 10)

        # -- Update Game Logic --
        # Spawn projectiles
        self.projectile_spawn_rate += 0.001 / 30 # Increase spawn rate per second
        if self.np_random.random() < self.projectile_spawn_rate:
            self._spawn_projectile()

        # Update projectiles
        projectiles_to_remove = []
        for proj in self.projectiles:
            proj['pos'] += proj['vel']
            proj['trail'].append(proj['pos'].copy())
            if len(proj['trail']) > 10:
                proj['trail'].pop(0)

            px, py = proj['pos']

            # Check collision with base
            base_rect = self._get_base_rect()
            if base_rect.collidepoint(px, py):
                self.base_health -= 10
                self._create_particles((px, py), self.COLOR_PROJECTILE, 30)
                projectiles_to_remove.append(proj)
                continue

            # Check collision with blocks
            grid_pos = self._pixel_to_grid((px,py))
            if grid_pos in self.placed_blocks:
                block = self.placed_blocks[grid_pos]
                block['health'] -= 1
                reward += 0.1 # Reward for deflection
                self.score += 1
                self._create_particles((px, py), self.COLOR_PROJECTILE, 20)
                projectiles_to_remove.append(proj)
                if block['health'] <= 0:
                    del self.placed_blocks[grid_pos]
                continue

            # Check out of bounds
            if py > self.HEIGHT:
                projectiles_to_remove.append(proj)

        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # -- Check Termination --
        terminated = False
        truncated = False
        if self.base_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += 100
            terminated = True # Game ends in success
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "base_health": self.base_health,
            "available_blocks": self.available_blocks,
        }

    # -- Helper and Rendering Methods --

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _pixel_to_grid(self, pixel_pos):
        gx = (pixel_pos[0] - self.GRID_OFFSET_X) // self.CELL_SIZE
        gy = (pixel_pos[1] - self.GRID_OFFSET_Y) // self.CELL_SIZE
        if 0 <= gx < self.GRID_WIDTH and 0 <= gy < self.GRID_HEIGHT:
            return (int(gx), int(gy))
        return None

    def _get_base_rect(self):
        base_y_pos = self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE
        return pygame.Rect(self.GRID_OFFSET_X, base_y_pos, self.GRID_WIDTH * self.CELL_SIZE, 10)

    def _spawn_projectile(self):
        x = self.np_random.uniform(self.GRID_OFFSET_X, self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE)
        y = 0.0
        speed = self.np_random.uniform(1.5, 3.0) + (self.steps / self.MAX_STEPS) * 2.0
        angle = self.np_random.uniform(-0.1, 0.1) # slight angle
        vel = np.array([math.sin(angle) * speed, math.cos(angle) * speed])
        self.projectiles.append({'pos': np.array([x, y], dtype=np.float64), 'vel': vel, 'trail': []})

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': np.array(pos, dtype=np.float64), 'vel': vel, 'life': life, 'color': color})

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw base
        base_rect = self._get_base_rect()
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, self.COLOR_BASE_DMG, base_rect, 2)

        # Draw placed blocks
        for pos, block in self.placed_blocks.items():
            rect = pygame.Rect(self.GRID_OFFSET_X + pos[0] * self.CELL_SIZE,
                               self.GRID_OFFSET_Y + pos[1] * self.CELL_SIZE,
                               self.CELL_SIZE, self.CELL_SIZE)

            color = self.COLOR_BLOCK_STD if block['type'] == 1 else self.COLOR_BLOCK_REIN
            dark_color = self.COLOR_BLOCK_STD_DARK if block['type'] == 1 else self.COLOR_BLOCK_REIN_DARK

            pygame.draw.rect(self.screen, dark_color, rect)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect)

            if block['type'] == 2: # Reinforced visual
                pygame.draw.line(self.screen, dark_color, inner_rect.topleft, inner_rect.bottomright, 2)
                pygame.draw.line(self.screen, dark_color, inner_rect.topright, inner_rect.bottomleft, 2)
                if block['health'] < 3: # Damage indicator
                    crack_pos = (inner_rect.centerx, inner_rect.centery)
                    pygame.gfxdraw.aacircle(self.screen, int(crack_pos[0]), int(crack_pos[1]), 4, (0,0,0,100))
                if block['health'] < 2:
                    pygame.gfxdraw.aacircle(self.screen, int(crack_pos[0]), int(crack_pos[1]), 8, (0,0,0,100))

        # Draw projectiles
        for proj in self.projectiles:
            # Trail
            if len(proj['trail']) > 1:
                trail_points = [(int(p[0]), int(p[1])) for p in proj['trail']]
                pygame.draw.aalines(self.screen, self.COLOR_PROJECTILE, False, trail_points)
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 10, self.COLOR_PROJECTILE_GLOW)
            # Core
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 5, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 5, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            try:
                color = p['color'] + (alpha,)
            except TypeError: # if color already has alpha
                color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            size = max(1, int(p['life'] * 0.2))
            pygame.draw.rect(self.screen, color, (int(p['pos'][0]), int(p['pos'][1]), size, size))

        # Draw cursor
        cursor_rect = pygame.Rect(self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
                                  self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
                                  self.CELL_SIZE, self.CELL_SIZE)
        is_valid = tuple(self.cursor_pos) not in self.placed_blocks
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID

        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, cursor_color, (0, 0, self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.rect(cursor_surface, (255,255,255,200), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 2)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.base_health / self.BASE_HEALTH_MAX)
        health_bar_width = (self.GRID_WIDTH * self.CELL_SIZE) - 4
        current_health_width = int(health_bar_width * health_ratio)
        health_bar_rect = pygame.Rect(self.GRID_OFFSET_X + 2, self._get_base_rect().top - 15, health_bar_width, 10)

        # Interpolate color from green to red
        bar_color = (
            min(255, int(231 + (46-231) * health_ratio)),
            min(255, int(76 + (204-76) * health_ratio)),
            min(255, int(60 + (113-60) * health_ratio))
        )

        pygame.draw.rect(self.screen, (50, 50, 50), health_bar_rect, border_radius=3)
        if current_health_width > 0:
            pygame.draw.rect(self.screen, bar_color, (health_bar_rect.left, health_bar_rect.top, current_health_width, health_bar_rect.height), border_radius=3)

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30.0
        timer_text = f"TIME: {max(0, time_left):.1f}s"
        timer_surf = self.font_main.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 5))

        # Block Count
        block_text = f"BLOCKS: {self.available_blocks}"
        block_surf = self.font_main.render(block_text, True, self.COLOR_TEXT)
        self.screen.blit(block_surf, (10, 5))

        # Score
        score_text = f"DEFLECTED: {self.score}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 5))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It is not part of the required solution but is useful for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Fortress")
    
    total_reward = 0
    
    while not done:
        # Default action is no-op
        action = np.array([0, 0, 0])

        # Poll for events and keyboard state
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Combine actions. This allows moving while holding space/shift
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(30)
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()