
# Generated: 2025-08-28T03:31:02.058852
# Source Brief: brief_02041.md
# Brief Index: 2041

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # User-facing control string
    user_guide = (
        "Controls: ↑↓ to move paddle. ←→ to change paddle color."
    )

    # User-facing game description
    game_description = (
        "An isometric 2D arcade game where players control a color-changing paddle to match the incoming ball's color for bonus points."
    )

    # Auto-advance frames for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (25, 30, 50)
    COLOR_WALL = (40, 45, 70)
    COLOR_SHADOW = (10, 12, 22, 150)
    COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255)    # Blue
    ]
    COLOR_TEXT = (220, 220, 240)
    COLOR_MATCH_GLOW = (255, 255, 150)

    # Court dimensions (logical)
    COURT_WIDTH = 300
    COURT_HEIGHT = 200

    # Isometric projection scaling
    ISO_X_SCALE = 1.0
    ISO_Y_SCALE = 0.5

    # Game rules
    MAX_SCORE = 20
    MAX_MISSES = 4
    MAX_STEPS = 1000
    BALL_COLOR_CHANGE_HITS = 3
    
    # Paddle properties
    PADDLE_LOGICAL_WIDTH = 8
    PADDLE_LOGICAL_HEIGHT = 50
    PADDLE_SPEED = 8

    # Ball properties
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 5
    BALL_SPEED_INCREMENT = 0.5 # Per 5 points as per brief (0.05 is too slow)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Pre-calculate screen center for projection
        self.screen_offset = pygame.math.Vector2(
            self.SCREEN_WIDTH / 2, 
            self.SCREEN_HEIGHT / 2 - 50
        )
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.ball_speed = 0
        self.ball_color_index = 0
        self.ball_hit_counter = 0
        self.paddle_y = 0
        self.paddle_color_index = 0
        self.last_score_milestone = 0
        self.color_change_lock = False
        self.particles = []
        self.screen_flash = 0
        
        self.reset()

        # Run validation check
        # self.validate_implementation()

    def _iso_transform(self, x, y):
        """Converts logical court coordinates to screen coordinates."""
        iso_x = (x - y) * self.ISO_X_SCALE
        iso_y = (x + y) * self.ISO_Y_SCALE
        return pygame.math.Vector2(iso_x, iso_y) + self.screen_offset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        # Paddle state
        self.paddle_y = self.COURT_HEIGHT / 2
        self.paddle_color_index = self.np_random.integers(0, len(self.COLORS))
        
        # Ball state
        self.ball_pos = pygame.math.Vector2(self.COURT_WIDTH / 2, self.COURT_HEIGHT / 2)
        initial_angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        self.ball_vel = pygame.math.Vector2(
            -math.cos(initial_angle), 
            math.sin(initial_angle)
        )
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.ball_color_index = self.np_random.integers(0, len(self.COLORS))
        self.ball_hit_counter = 0
        
        # Misc state
        self.last_score_milestone = 0
        self.color_change_lock = False
        self.particles = []
        self.screen_flash = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()
            
        # --- 1. Process Action ---
        movement = action[0]
        
        # Paddle movement
        if movement == 1:  # Up
            self.paddle_y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_y += self.PADDLE_SPEED
        self.paddle_y = np.clip(self.paddle_y, self.PADDLE_LOGICAL_HEIGHT / 2, self.COURT_HEIGHT - self.PADDLE_LOGICAL_HEIGHT / 2)

        # Paddle color change
        is_color_change_action = movement in [3, 4]
        if is_color_change_action and not self.color_change_lock:
            if movement == 3:  # Left
                self.paddle_color_index = (self.paddle_color_index - 1) % len(self.COLORS)
            elif movement == 4:  # Right
                self.paddle_color_index = (self.paddle_color_index + 1) % len(self.COLORS)
            self.color_change_lock = True
        elif not is_color_change_action:
            self.color_change_lock = False

        # --- 2. Update Game Logic ---
        reward = 0
        self._update_ball()
        self._update_particles()
        if self.screen_flash > 0:
            self.screen_flash -= 1

        # Reward for keeping paddle near ball's y-coord
        y_dist = abs(self.paddle_y - self.ball_pos.y)
        reward -= 0.1 * (y_dist / self.COURT_HEIGHT)

        # --- 3. Check for paddle collision ---
        paddle_edge_x = self.COURT_WIDTH - self.BALL_RADIUS
        if self.ball_pos.x >= paddle_edge_x and self.ball_vel.x > 0:
            paddle_top = self.paddle_y - self.PADDLE_LOGICAL_HEIGHT / 2
            paddle_bottom = self.paddle_y + self.PADDLE_LOGICAL_HEIGHT / 2

            if paddle_top <= self.ball_pos.y <= paddle_bottom:
                # --- HIT ---
                self.ball_pos.x = paddle_edge_x
                self.ball_vel.x *= -1
                
                # Add vertical deflection based on where it hit the paddle
                deflect_factor = (self.ball_pos.y - self.paddle_y) / (self.PADDLE_LOGICAL_HEIGHT / 2)
                self.ball_vel.y += deflect_factor * 0.5
                self.ball_vel.normalize_ip()

                # Score and rewards
                reward += 0.1  # Base hit reward
                self.ball_hit_counter += 1
                color_match = self.ball_color_index == self.paddle_color_index
                if color_match:
                    self.score += 1
                    reward += 3  # Color match bonus reward
                    self._create_particles(self.ball_pos, self.COLOR_MATCH_GLOW, 20, 5)
                
                # Visuals & Sound placeholder
                self._create_particles(self.ball_pos, self.COLORS[self.ball_color_index], 10, 3)
                # play_sound('hit')
                
                # Change ball color every few hits
                if self.ball_hit_counter % self.BALL_COLOR_CHANGE_HITS == 0:
                    self.ball_color_index = (self.ball_color_index + self.np_random.integers(1, len(self.COLORS))) % len(self.COLORS)

            else:
                # --- MISS ---
                self.misses += 1
                reward -= 1  # Miss penalty
                self.screen_flash = 5 # Flash red
                self._reset_ball()
                # play_sound('miss')
        
        # --- 4. Update Difficulty ---
        current_milestone = self.score // 5
        if current_milestone > self.last_score_milestone:
            self.last_score_milestone = current_milestone
            self.ball_speed += self.BALL_SPEED_INCREMENT

        # --- 5. Check Termination ---
        self.steps += 1
        terminated = (
            self.score >= self.MAX_SCORE or 
            self.misses >= self.MAX_MISSES or 
            self.steps >= self.MAX_STEPS
        )
        if terminated:
            self.game_over = True
            if self.score >= self.MAX_SCORE:
                reward += 100  # Victory bonus
            if self.misses >= self.MAX_MISSES:
                reward -= 100  # Defeat penalty
        
        # --- 6. Return Step Tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _update_ball(self):
        self.ball_pos += self.ball_vel * self.ball_speed
        
        # Wall collisions
        if self.ball_pos.y <= self.BALL_RADIUS and self.ball_vel.y < 0:
            self.ball_pos.y = self.BALL_RADIUS
            self.ball_vel.y *= -1
            self._create_particles(self.ball_pos, (200,200,200), 5, 2)
            # play_sound('bounce')
        elif self.ball_pos.y >= self.COURT_HEIGHT - self.BALL_RADIUS and self.ball_vel.y > 0:
            self.ball_pos.y = self.COURT_HEIGHT - self.BALL_RADIUS
            self.ball_vel.y *= -1
            self._create_particles(self.ball_pos, (200,200,200), 5, 2)
            # play_sound('bounce')

        if self.ball_pos.x <= self.BALL_RADIUS and self.ball_vel.x < 0:
            self.ball_pos.x = self.BALL_RADIUS
            self.ball_vel.x *= -1
            self._create_particles(self.ball_pos, (200,200,200), 5, 2)
            # play_sound('bounce')

    def _reset_ball(self):
        self.ball_pos = pygame.math.Vector2(self.COURT_WIDTH / 2, self.COURT_HEIGHT / 2)
        initial_angle = self.np_random.uniform(math.pi * 3/4, math.pi * 5/4) # Send towards opponent
        self.ball_vel = pygame.math.Vector2(math.cos(initial_angle), math.sin(initial_angle))

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
            "misses": self.misses,
        }

    def _render_game(self):
        # Draw court
        self._draw_court()
        
        # Draw shadows first
        self._draw_shadows()
        
        # Draw particles
        self._draw_particles()

        # Draw ball
        ball_screen_pos = self._iso_transform(self.ball_pos.x, self.ball_pos.y)
        ball_color = self.COLORS[self.ball_color_index]
        highlight_color = tuple(min(255, c + 80) for c in ball_color)
        pygame.gfxdraw.aacircle(self.screen, int(ball_screen_pos.x), int(ball_screen_pos.y), self.BALL_RADIUS, ball_color)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_screen_pos.x), int(ball_screen_pos.y), self.BALL_RADIUS, ball_color)
        pygame.gfxdraw.aacircle(self.screen, int(ball_screen_pos.x-2), int(ball_screen_pos.y-2), self.BALL_RADIUS // 2, highlight_color)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_screen_pos.x-2), int(ball_screen_pos.y-2), self.BALL_RADIUS // 2, highlight_color)

        # Draw paddle
        paddle_color = self.COLORS[self.paddle_color_index]
        paddle_glow_color = tuple(c // 2 for c in paddle_color)
        
        p_top_y = self.paddle_y - self.PADDLE_LOGICAL_HEIGHT / 2
        p_bot_y = self.paddle_y + self.PADDLE_LOGICAL_HEIGHT / 2
        p_x = self.COURT_WIDTH

        p1 = self._iso_transform(p_x, p_top_y)
        p2 = self._iso_transform(p_x - self.PADDLE_LOGICAL_WIDTH, p_top_y)
        p3 = self._iso_transform(p_x - self.PADDLE_LOGICAL_WIDTH, p_bot_y)
        p4 = self._iso_transform(p_x, p_bot_y)

        # Glow effect
        glow_points = [(p[0], p[1]) for p in [p1, p2, p3, p4]]
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, paddle_glow_color)
        
        # Main paddle
        pygame.gfxdraw.aapolygon(self.screen, glow_points, paddle_color)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, paddle_color)
        
    def _draw_court(self):
        # Court corners in logical space
        c1 = self._iso_transform(0, 0)
        c2 = self._iso_transform(self.COURT_WIDTH, 0)
        c3 = self._iso_transform(self.COURT_WIDTH, self.COURT_HEIGHT)
        c4 = self._iso_transform(0, self.COURT_HEIGHT)

        # Floor
        pygame.gfxdraw.filled_polygon(self.screen, [c1, c2, c3, c4], self.COLOR_GRID)
        pygame.gfxdraw.aapolygon(self.screen, [c1, c2, c3, c4], self.COLOR_WALL)

        # Walls
        wall_height = 80
        wc1 = c1 - (0, wall_height)
        wc4 = c4 - (0, wall_height)
        pygame.gfxdraw.filled_polygon(self.screen, [c1, c4, wc4, wc1], self.COLOR_WALL)
        pygame.gfxdraw.aapolygon(self.screen, [c1, c4, wc4, wc1], self.COLOR_BG)
        
        wc2 = c2 - (0, wall_height)
        pygame.gfxdraw.filled_polygon(self.screen, [c1, c2, wc2, wc1], self.COLOR_WALL)
        pygame.gfxdraw.aapolygon(self.screen, [c1, c2, wc2, wc1], self.COLOR_BG)

    def _draw_shadows(self):
        shadow_surface = self.screen.copy()
        shadow_surface.fill((255, 0, 255)) # Colorkey
        shadow_surface.set_colorkey((255, 0, 255))

        # Ball shadow
        ball_screen_pos = self._iso_transform(self.ball_pos.x, self.ball_pos.y)
        pygame.gfxdraw.filled_ellipse(shadow_surface, int(ball_screen_pos.x), int(ball_screen_pos.y + self.BALL_RADIUS * 1.5), self.BALL_RADIUS, self.BALL_RADIUS // 2, self.COLOR_SHADOW)
        
        # Paddle shadow
        p_top_y = self.paddle_y - self.PADDLE_LOGICAL_HEIGHT / 2
        p_bot_y = self.paddle_y + self.PADDLE_LOGICAL_HEIGHT / 2
        p_x = self.COURT_WIDTH

        p1_s = self._iso_transform(p_x, p_top_y) + (0, 10)
        p2_s = self._iso_transform(p_x - self.PADDLE_LOGICAL_WIDTH, p_top_y) + (0, 10)
        p3_s = self._iso_transform(p_x - self.PADDLE_LOGICAL_WIDTH, p_bot_y) + (0, 10)
        p4_s = self._iso_transform(p_x, p_bot_y) + (0, 10)
        pygame.gfxdraw.filled_polygon(shadow_surface, [p1_s, p2_s, p3_s, p4_s], self.COLOR_SHADOW)
        
        shadow_surface.set_alpha(150)
        self.screen.blit(shadow_surface, (0,0))
        
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))

        # Misses
        miss_text = self.font_main.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (20, 20))
        for i in range(self.MAX_MISSES - 1):
            color = self.COLORS[0] if i < self.misses else (50, 50, 50)
            pygame.draw.circle(self.screen, color, (110 + i * 25, 33), 8)
        
        # Game Over / Victory Text
        if self.game_over:
            if self.score >= self.MAX_SCORE:
                end_text_str = "VICTORY!"
                end_color = self.COLOR_MATCH_GLOW
            else:
                end_text_str = "GAME OVER"
                end_color = self.COLORS[0]
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

        # Screen flash on miss
        if self.screen_flash > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            flash_surface.fill(self.COLORS[0])
            flash_surface.set_alpha(self.screen_flash * 20)
            self.screen.blit(flash_surface, (0,0))

    def _create_particles(self, pos_logical, color, count, speed_mult):
        pos_screen = self._iso_transform(pos_logical.x, pos_logical.y)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos_screen.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'life': self.np_random.integers(10, 20)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.9  # friction
            p['life'] -= 1
            p['radius'] -= 0.2
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 15)))
            color = (*p['color'], alpha)
            if p['radius'] > 1:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color
                )

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Color Pong")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # The other actions are unused in this game
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame rendering ---
        # The observation is already a rendered frame
        # We just need to blit it to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()