
# Generated: 2025-08-27T19:12:07.500058
# Source Brief: brief_02078.md
# Brief Index: 2078

        
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

    user_guide = (
        "Controls: Press space to jump. Avoid the red gaps."
    )

    game_description = (
        "Minimalist arcade jumper. Tap to jump between ascending lines, avoiding gaps to reach the top. The higher you go, the faster it gets."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.WIN_LINE = 100

        # --- Colors ---
        self.COLOR_BG_TOP = (15, 25, 50)
        self.COLOR_BG_BOTTOM = (30, 60, 110)
        self.COLOR_LINE = (255, 255, 255)
        self.COLOR_GAP = (220, 50, 50)
        self.COLOR_PLAYER = (50, 255, 255)
        self.COLOR_PLAYER_GLOW = (50, 255, 255, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (200, 200, 220)

        # --- Game Physics & Mechanics ---
        self.GRAVITY = 0.8
        self.JUMP_VELOCITY = -13.5  # Tuned to reach ~5 lines (40px/line)
        self.LINE_HEIGHT = 40
        self.PLAYER_SIZE = 12
        self.PLAYER_SQUASH_FACTOR = 0.4
        self.PLAYER_SQUASH_RECOVER_RATE = 0.15

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- Internal State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        self.player_x = 0
        self.player_y = 0
        self.player_dy = 0
        self.on_ground = False
        self.player_anim = {'squash': 1.0, 'land_timer': 0}
        
        self.lines = []
        self.particles = []
        self.highest_line_reached = 0
        
        self.line_speed = 0.0
        self.gap_frequency = 0.0
        
        self.space_was_held = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_x = self.WIDTH // 2
        self.player_y = self.HEIGHT - self.LINE_HEIGHT * 2
        self.player_dy = 0
        self.on_ground = True
        self.player_anim = {'squash': 1.0, 'land_timer': 0}

        self.highest_line_reached = 0
        self.line_speed = 1.0
        self.gap_frequency = 0.10

        self.lines = []
        self.particles = []
        self._generate_initial_lines()
        
        self.space_was_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.02  # Time penalty

        # --- Handle Input ---
        if space_held and not self.space_was_held and self.on_ground:
            self.player_dy = self.JUMP_VELOCITY
            self.on_ground = False
            self.player_anim['squash'] = -self.PLAYER_SQUASH_FACTOR # Stretch on jump
            # sfx: jump
            self._spawn_particles(self.player_x, self.player_y + self.PLAYER_SIZE, 8, 'jump')


        # --- Update Game Logic ---
        self._update_player()
        landed, lines_advanced = self._check_collisions()
        if landed:
            reward += 0.1 * lines_advanced
            if lines_advanced > 0: # sfx: land_success
                pass
            else: # sfx: land_thud
                pass
        
        self._update_lines()
        self._update_particles()
        self._update_difficulty()

        # --- Update Animation ---
        if self.player_anim['squash'] != 0:
            self.player_anim['squash'] -= np.sign(self.player_anim['squash']) * self.PLAYER_SQUASH_RECOVER_RATE
            if abs(self.player_anim['squash']) < self.PLAYER_SQUASH_RECOVER_RATE:
                self.player_anim['squash'] = 0

        # --- Termination and Score ---
        terminated = False
        if self.player_y > self.HEIGHT + self.PLAYER_SIZE:
            terminated = True
            reward = -10.0
            # sfx: fall_game_over
        elif self.highest_line_reached >= self.WIN_LINE:
            terminated = True
            reward = 100.0
            # sfx: win
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score = self.highest_line_reached
        self.steps += 1
        self.space_was_held = space_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_player(self):
        if not self.on_ground:
            self.player_dy += self.GRAVITY
        
        # Store previous y for collision detection
        self.prev_player_y = self.player_y
        self.player_y += self.player_dy

    def _check_collisions(self):
        lines_advanced = 0
        landed = False
        if self.player_dy < 0: # Can't land while moving up
            return False, 0
            
        for line in self.lines:
            line_top = line['y'] - 2
            line_bottom = line['y'] + 2
            
            # Check for vertical intersection
            if self.prev_player_y + self.PLAYER_SIZE / 2 <= line_top and \
               self.player_y + self.PLAYER_SIZE / 2 >= line_top:
                
                # Check for horizontal position (gap)
                gap_start, gap_end = line['gap']
                if not (gap_start <= self.player_x <= gap_end):
                    self.on_ground = True
                    self.player_dy = 0
                    self.player_y = line['y'] - self.PLAYER_SIZE / 2
                    self.player_anim['squash'] = self.PLAYER_SQUASH_FACTOR # Squash on land
                    
                    lines_advanced = max(0, line['number'] - self.highest_line_reached)
                    self.highest_line_reached = max(self.highest_line_reached, line['number'])
                    landed = True
                    self._spawn_particles(self.player_x, self.player_y + self.PLAYER_SIZE, 12, 'land')
                    break # Stop checking other lines
        return landed, lines_advanced

    def _update_lines(self):
        for line in self.lines:
            line['y'] += self.line_speed
        
        self.lines = [line for line in self.lines if line['y'] < self.HEIGHT + self.LINE_HEIGHT]

        while len(self.lines) < (self.HEIGHT / self.LINE_HEIGHT) + 2:
            highest_num = 0
            if self.lines:
                highest_num = max(l['number'] for l in self.lines)
            
            lowest_y = 0
            if self.lines:
                lowest_y = min(l['y'] for l in self.lines)

            self._generate_line(highest_num + 1, lowest_y - self.LINE_HEIGHT)

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['dx']
            p['y'] += p['dy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_difficulty(self):
        self.line_speed = 1.0 + (self.highest_line_reached / 100) * 0.01 * self.FPS
        self.gap_frequency = min(0.8, 0.10 + (self.highest_line_reached / 50) * 0.01)

    def _get_observation(self):
        self._render_background()
        self._render_lines()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Rendering Helpers ---

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_lines(self):
        for line in self.lines:
            y = int(line['y'])
            gap_start, gap_end = line['gap']
            
            # Main line
            pygame.draw.line(self.screen, self.COLOR_LINE, (0, y), (self.WIDTH, y), 4)
            
            # Gap
            if gap_start < gap_end:
                pygame.draw.line(self.screen, self.COLOR_GAP, (gap_start, y), (gap_end, y), 6)

    def _render_player(self):
        size_w = self.PLAYER_SIZE * (1 - self.player_anim['squash'])
        size_h = self.PLAYER_SIZE * (1 + self.player_anim['squash'])
        
        x, y = int(self.player_x), int(self.player_y)
        
        # Glow
        glow_radius = int(size_w * 1.5)
        if glow_radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, x, y, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Body
        player_rect = pygame.Rect(0, 0, size_w, size_h)
        player_rect.center = (x, y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*self.COLOR_PARTICLE, alpha)
            
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['x']-p['size']), int(p['y']-p['size'])))

    def _render_ui(self):
        line_text = self.font_ui.render(f"Line: {self.highest_line_reached}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(line_text, (10, 10))
        self.screen.blit(score_text, (10, 35))

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))

        status = "YOU WIN!" if self.highest_line_reached >= self.WIN_LINE else "GAME OVER"
        text = self.font_game_over.render(status, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

    # --- Game Logic Helpers ---

    def _generate_line(self, line_number, y_pos):
        has_gap = self.np_random.random() < self.gap_frequency if line_number > 5 else False

        if not has_gap:
            gap_tuple = (0, 0)
        else:
            base_width = 50
            gap_width = min(self.WIDTH * 0.6, base_width + (line_number / 2))
            
            # Ensure path always exists
            min_edge_margin = 20
            gap_start = self.np_random.integers(
                min_edge_margin, self.WIDTH - gap_width - min_edge_margin
            )
            gap_end = gap_start + gap_width
            gap_tuple = (gap_start, gap_end)

        self.lines.append({'y': y_pos, 'gap': gap_tuple, 'number': line_number})

    def _generate_initial_lines(self):
        for i in range(int(self.HEIGHT / self.LINE_HEIGHT) + 2):
            y_pos = self.HEIGHT - i * self.LINE_HEIGHT
            self._generate_line(i, y_pos)
        
        # Ensure the player starts on a solid line
        for line in self.lines:
            if abs(line['y'] - (self.HEIGHT - self.LINE_HEIGHT)) < 10:
                line['gap'] = (0, 0)
                break
        
        self.highest_line_reached = 1

    def _spawn_particles(self, x, y, count, p_type):
        for _ in range(count):
            if p_type == 'land':
                angle = self.np_random.uniform(math.pi, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                dx = math.cos(angle) * speed
                dy = math.sin(angle) * speed
            elif p_type == 'jump':
                angle = self.np_random.uniform(0, math.pi)
                speed = self.np_random.uniform(0.5, 2)
                dx = math.cos(angle) * speed
                dy = math.sin(angle) * speed
            
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'x': x, 'y': y, 'dx': dx, 'dy': dy,
                'life': life, 'max_life': life,
                'size': self.np_random.integers(1, 4)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Line Jumper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        space_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_pressed = True
        
        action = [0, 1 if space_pressed else 0, 0] # Movement=None, Space=pressed/released, Shift=released
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    pygame.quit()