
# Generated: 2025-08-28T00:34:35.517204
# Source Brief: brief_03828.md
# Brief Index: 3828

        
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
        "Controls: Use ← and → to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "An isometric brick-breaker. Use the paddle to destroy all the bricks with the ball before you run out of lives."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PADDLE = (52, 152, 219)
    COLOR_PADDLE_SHADOW = (41, 128, 185)
    COLOR_BALL = (236, 240, 241)
    COLOR_BALL_GLOW = (236, 240, 241, 50)
    COLOR_TEXT = (236, 240, 241)
    BRICK_COLORS = [
        ((231, 76, 60), (192, 57, 43)),   # Red
        ((241, 196, 15), (243, 156, 18)),  # Yellow
        ((46, 204, 113), (39, 174, 96)),   # Green
        ((155, 89, 182), (142, 68, 173)),  # Purple
    ]

    # Game parameters
    PADDLE_WIDTH = 80
    PADDLE_HEIGHT = 10
    PADDLE_SPEED = 8
    PADDLE_Y_OFFSET = 30 # from bottom
    BALL_RADIUS = 6
    BALL_SPEED_INITIAL = 5.0
    BALL_SPIN_FACTOR = 0.08
    MAX_STEPS = 3000
    
    BRICK_ROWS = 4
    BRICK_COLS = 10
    BRICK_WIDTH = 28
    BRICK_HEIGHT = 14
    BRICK_DEPTH = 10
    BRICK_START_Y = 100
    BRICK_SPACING = 4

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)
        
        self.paddle_move_action = 0
        self.launch_action = False
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.lives = 3
        
        self.paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        
        self.ball_launched = False
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self._reset_ball()

        self.bricks = self._initialize_bricks()
        self.total_bricks = len(self.bricks)
        self.bricks_destroyed = 0
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _initialize_bricks(self):
        bricks = []
        grid_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_SPACING) - self.BRICK_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) / 2

        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                color_pair = self.BRICK_COLORS[r % len(self.BRICK_COLORS)]
                brick_x = start_x + c * (self.BRICK_WIDTH + self.BRICK_SPACING)
                brick_y = self.BRICK_START_Y + r * (self.BRICK_HEIGHT + self.BRICK_SPACING)
                
                bricks.append({
                    "rect": pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT),
                    "color": color_pair[0],
                    "shadow_color": color_pair[1],
                    "active": True,
                    "points": (self.BRICK_ROWS - r) * 10
                })
        return bricks
        
    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos.x = self.paddle_x + self.PADDLE_WIDTH / 2
        self.ball_pos.y = self.SCREEN_HEIGHT - self.PADDLE_Y_OFFSET - self.PADDLE_HEIGHT - self.BALL_RADIUS - 1
        self.ball_vel = pygame.math.Vector2(0, 0)

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage speed
        
        self._handle_input(action)
        self._update_paddle()
        
        if not self.ball_launched and self.launch_action:
            self.ball_launched = True
            angle = self.rng.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards
            self.ball_vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED_INITIAL
            # sfx: ball_launch
        
        self._update_ball()
        reward += self._handle_collisions()
        
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win_state:
                reward += 100
            elif self.lives <= 0:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_btn, _ = action
        self.paddle_move_action = 0
        if movement == 3: # Left
            self.paddle_move_action = -1
        elif movement == 4: # Right
            self.paddle_move_action = 1
        
        self.launch_action = (space_btn == 1)

    def _update_paddle(self):
        self.paddle_x += self.paddle_move_action * self.PADDLE_SPEED
        self.paddle_x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle_x))

    def _update_ball(self):
        if self.ball_launched:
            self.ball_pos += self.ball_vel
        else:
            self.ball_pos.x = self.paddle_x + self.PADDLE_WIDTH / 2
            self.ball_pos.y = self.SCREEN_HEIGHT - self.PADDLE_Y_OFFSET - self.PADDLE_HEIGHT - self.BALL_RADIUS - 1

    def _handle_collisions(self):
        if not self.ball_launched:
            return 0
        
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        # Wall collisions
        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.SCREEN_WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # sfx: wall_bounce
        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sfx: wall_bounce

        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_x, self.SCREEN_HEIGHT - self.PADDLE_Y_OFFSET - self.PADDLE_HEIGHT, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if self.ball_vel.y > 0 and paddle_rect.colliderect(ball_rect):
            self.ball_vel.y *= -1
            self.ball_pos.y = paddle_rect.top - self.BALL_RADIUS

            dist_from_center = (self.ball_pos.x - paddle_rect.centerx)
            self.ball_vel.x += dist_from_center * self.BALL_SPIN_FACTOR
            if self.ball_vel.length() > 0:
                self.ball_vel.scale_to_length(self.BALL_SPEED_INITIAL)
            # sfx: paddle_bounce
        
        # Brick collisions
        for brick in self.bricks:
            if brick["active"] and brick["rect"].colliderect(ball_rect):
                brick["active"] = False
                self.bricks_destroyed += 1
                reward += 1.0 # Base reward for destruction
                self.score += brick["points"]
                
                self._spawn_particles(brick["rect"].center, brick["color"], 20)
                # sfx: brick_destroy
                
                prev_ball_pos = self.ball_pos - self.ball_vel
                brick_center = pygame.math.Vector2(brick["rect"].center)
                
                dx = abs(prev_ball_pos.x - brick_center.x) - brick["rect"].width / 2
                dy = abs(prev_ball_pos.y - brick_center.y) - brick["rect"].height / 2

                if dx > dy:
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1
                break

        # Ball out of bounds (lose life)
        if self.ball_pos.y > self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 1.0
            if self.lives > 0:
                self._reset_ball()
                # sfx: lose_life
            else:
                self.game_over = True
                # sfx: game_over
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.bricks_destroyed == self.total_bricks:
            self.win_state = True
            self.game_over = True
            return True
        if self.lives <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.math.Vector2(pos),
                "vel": vel,
                "life": self.rng.integers(20, 40),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_bricks()
        self._render_particles()
        self._render_paddle()
        self._render_ball()

    def _draw_iso_cube(self, surface, rect, depth, top_color, side_color):
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        
        # Top face
        top_points = [
            (x, y + h / 2),
            (x + w / 2, y),
            (x + w, y + h / 2),
            (x + w / 2, y + h)
        ]
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)

        # Right side face
        side_points_right = [
            (x + w, y + h / 2),
            (x + w / 2, y + h),
            (x + w / 2, y + h + depth),
            (x + w, y + h / 2 + depth)
        ]
        pygame.gfxdraw.aapolygon(surface, side_points_right, side_color)
        pygame.gfxdraw.filled_polygon(surface, side_points_right, side_color)
        
        # Bottom side face
        side_points_bottom = [
            (x + w / 2, y + h),
            (x, y + h / 2),
            (x, y + h / 2 + depth),
            (x + w / 2, y + h + depth)
        ]
        pygame.gfxdraw.aapolygon(surface, side_points_bottom, side_color)
        pygame.gfxdraw.filled_polygon(surface, side_points_bottom, side_color)

    def _render_bricks(self):
        for brick in self.bricks:
            if brick["active"]:
                self._draw_iso_cube(self.screen, brick["rect"], self.BRICK_DEPTH, brick["color"], brick["shadow_color"])

    def _render_paddle(self):
        paddle_rect = pygame.Rect(self.paddle_x, self.SCREEN_HEIGHT - self.PADDLE_Y_OFFSET - self.PADDLE_HEIGHT, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self._draw_iso_cube(self.screen, paddle_rect, 5, self.COLOR_PADDLE, self.COLOR_PADDLE_SHADOW)

    def _render_ball(self):
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Main ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for p in self.particles:
            radius = int(self.BALL_RADIUS * 0.5 * (p["life"] / 40.0))
            if radius > 0:
                pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), radius)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))
        
        if self.game_over:
            if self.win_state:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_msg.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": self.total_bricks - self.bricks_destroyed,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Breakout")
    clock = pygame.time.Clock()
    
    while running:
        movement = 0 # No-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1

        if terminated:
            if any(keys):
                obs, info = env.reset()
                terminated = False
        else:
            action = [movement, space_held, 0] # shift is not used
            obs, reward, terminated, truncated, info = env.step(action)

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()