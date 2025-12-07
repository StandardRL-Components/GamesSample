
# Generated: 2025-08-28T03:34:55.418498
# Source Brief: brief_04966.md
# Brief Index: 4966

        
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
        "Controls: Arrow keys to move the cursor. Press Space to squash a bug."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade action! Squash 20 bugs of different point values before the 60-second timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CONSTANTS ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = self.WIDTH // self.GRID_WIDTH
        
        self.MAX_TIME = 60.0
        self.TARGET_SQUASHES = 20
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_CURSOR = (0, 255, 255) # Cyan
        self.COLOR_CURSOR_GLOW = (0, 150, 150)
        self.BUG_COLORS = {
            1: (100, 255, 100), # Green
            2: (255, 100, 100), # Red
            3: (100, 100, 255), # Blue
        }
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables
        self.np_random = None
        self.cursor_pos = None
        self.bugs = []
        self.splats = []
        self.time_remaining = 0
        self.score = 0
        self.bugs_squashed = 0
        self.game_over = False
        self.win = False
        self.last_space_held = False
        self.base_respawn_time = 0.5
        self.respawn_timer = 0.0
        self.steps = 0
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.bugs_squashed = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_TIME
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_space_held = False
        
        self.bugs = []
        self.splats = []
        
        self.base_respawn_time = 0.5
        self.respawn_timer = 0.0

        for _ in range(10):
            self._spawn_bug()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        dt = 1.0 / self.FPS
        self.steps += 1
        self.time_remaining -= dt
        reward = 0.0
        
        # Unpack factorized action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- 1. Calculate distance before action ---
        dist_before = self._get_dist_to_nearest_bug(self.cursor_pos)
        
        # --- 2. Handle Movement ---
        if movement != 0:
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            
            # Grid wraparound
            self.cursor_pos[0] %= self.GRID_WIDTH
            self.cursor_pos[1] %= self.GRID_HEIGHT

        # --- 3. Handle Squash Action ---
        squashed_bug_this_step = False
        if space_held and not self.last_space_held:
            bug_to_squash = next((bug for bug in self.bugs if bug['pos'] == self.cursor_pos), None)
            
            if bug_to_squash:
                # SQUASH!
                reward += bug_to_squash['score']
                self.score += bug_to_squash['score']
                self.bugs_squashed += 1
                self._create_splat(bug_to_squash['pos'], self.BUG_COLORS[bug_to_squash['score']])
                self.bugs.remove(bug_to_squash)
                squashed_bug_this_step = True
                # sfx: squash.wav

        self.last_space_held = space_held

        # --- 4. Update Game Logic ---
        self._update_splats(dt)
        
        # Bug spawning
        self.respawn_timer -= dt
        if self.respawn_timer <= 0 or len(self.bugs) < 5:
             current_respawn_time = self.base_respawn_time - 0.02 * (self.bugs_squashed // 5)
             self.respawn_timer = max(0.1, current_respawn_time)
             self._spawn_bug()

        # --- 5. Calculate Shaping Reward ---
        dist_after = self._get_dist_to_nearest_bug(self.cursor_pos)
        if movement != 0 and not squashed_bug_this_step and dist_after < dist_before:
            reward += 0.1
        
        # --- 6. Check Termination ---
        terminated = False
        if self.bugs_squashed >= self.TARGET_SQUASHES:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 50
        elif self.time_remaining <= 0:
            self.win = False
            self.game_over = True
            terminated = True
            reward -= 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_splats()
        self._render_bugs()
        self._render_cursor()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "bugs_squashed": self.bugs_squashed,
            "cursor_pos": self.cursor_pos,
        }

    # --- Rendering Methods ---
    def _render_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_splats(self):
        for splat in self.splats:
            for p in splat['particles']:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _render_bugs(self):
        for bug in self.bugs:
            px, py = self._grid_to_pixel(bug['pos'])
            pulse = math.sin(self.steps * 0.3 + bug['pulse']) * 2
            radius = int(self.CELL_SIZE * 0.35 + pulse)
            
            color = self.BUG_COLORS[bug['score']]
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, (0,0,0)) # Outline for clarity

    def _render_cursor(self):
        px, py = self._grid_to_pixel(self.cursor_pos)
        size = self.CELL_SIZE
        
        # Pulsating glow
        glow_alpha = 100 + math.sin(self.steps * 0.2) * 50
        glow_color = (*self.COLOR_CURSOR_GLOW, glow_alpha)
        s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (size, size), size // 2)
        self.screen.blit(s, (px - size, py - size), special_flags=pygame.BLEND_RGBA_ADD)

        # Main cursor
        rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px, rect.top), (px, rect.bottom))
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (rect.left, py), (rect.right, py))

    def _render_ui(self):
        ui_bar = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 8))

        time_text = self.font_ui.render(f"TIME: {max(0, self.time_remaining):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 8))

        bugs_text = self.font_ui.render(f"BUGS: {self.bugs_squashed}/{self.TARGET_SQUASHES}", True, self.COLOR_TEXT)
        self.screen.blit(bugs_text, (self.WIDTH // 2 - bugs_text.get_width() // 2, 8))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "TIME'S UP!"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            shadow_text = self.font_game_over.render(msg, True, (0,0,0))
            self.screen.blit(shadow_text, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(over_text, text_rect)

    # --- Helper Methods ---
    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _spawn_bug(self):
        if len(self.bugs) >= self.GRID_WIDTH * self.GRID_HEIGHT: return
            
        rand_val = self.np_random.random()
        score = 1 if rand_val < 0.5 else (2 if rand_val < 0.8 else 3)

        occupied_pos = {tuple(bug['pos']) for bug in self.bugs}
        occupied_pos.add(tuple(self.cursor_pos))
        
        for _ in range(100):
            pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            if tuple(pos) not in occupied_pos:
                self.bugs.append({'pos': pos, 'score': score, 'pulse': self.np_random.random() * 2 * math.pi})
                return

    def _create_splat(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        particles = []
        for _ in range(30):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 50 + 20
            particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.random() * 4 + 2,
                'life': self.np_random.random() * 0.5 + 0.3,
            })
        self.splats.append({'particles': particles, 'life': 1.0, 'color': color})

    def _update_splats(self, dt):
        for splat in self.splats[:]:
            splat['life'] -= dt * 2
            if splat['life'] <= 0:
                self.splats.remove(splat)
                continue
            
            for p in splat['particles']:
                p['pos'][0] += p['vel'][0] * dt
                p['pos'][1] += p['vel'][1] * dt
                p['radius'] -= dt * 3
                p['life'] -= dt
            
            splat['particles'] = [p for p in splat['particles'] if p['radius'] > 0 and p['life'] > 0]

    def _get_dist_to_nearest_bug(self, from_pos):
        if not self.bugs: return float('inf')
        
        min_dist = float('inf')
        for bug in self.bugs:
            dx = abs(bug['pos'][0] - from_pos[0])
            dy = abs(bug['pos'][1] - from_pos[1])
            dist_x = min(dx, self.GRID_WIDTH - dx)
            dist_y = min(dy, self.GRID_HEIGHT - dy)
            min_dist = min(min_dist, dist_x + dist_y)
        return min_dist

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    env.reset()
    
    running = True
    game_over_screen_timer = 3 * env.FPS

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bug Squasher")

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated:
            if game_over_screen_timer > 0:
                game_over_screen_timer -= 1
            else:
                print(f"Game Over! Final Score: {info['score']}")
                env.reset()
                game_over_screen_timer = 3 * env.FPS

    env.close()