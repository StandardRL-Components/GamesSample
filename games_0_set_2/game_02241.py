
# Generated: 2025-08-28T04:11:34.044432
# Source Brief: brief_02241.md
# Brief Index: 2241

        
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
        "Controls: Use ← and → to move the paddle."
    )

    game_description = (
        "Breakout Bandit: an isometric breakout where risky plays yield high scores. Clear all bricks across 3 stages to win."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PADDLE = (0, 200, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GRID = (40, 45, 60)
    BRICK_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80)]
    
    # Game world
    LOGICAL_WIDTH = 12
    LOGICAL_HEIGHT = 20
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    ORIGIN_X = WIDTH // 2
    ORIGIN_Y = 60

    # Paddle
    PADDLE_WIDTH = 3.0
    PADDLE_HEIGHT = 0.5
    PADDLE_Y_POS = 17.0
    PADDLE_SPEED = 0.4
    
    # Ball
    BALL_RADIUS = 0.3
    INITIAL_BALL_SPEED = 6.0 / FPS # units per step
    
    # Bricks
    BRICK_ROWS = 5
    BRICK_COLS = 6
    BRICK_AREA_X_START = (LOGICAL_WIDTH - BRICK_COLS) / 2
    BRICK_AREA_Y_START = 3

    # Game rules
    MAX_LIVES = 3
    NUM_STAGES = 3
    STAGE_TIME_LIMIT = 60 # seconds
    MAX_STEPS = STAGE_TIME_LIMIT * FPS * NUM_STAGES

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        self.paddle_pos = 0.0
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.bricks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.steps = 0
        self.stage_timer = 0.0
        self.game_over = False
        self.game_won = False
        self.ball_speed_multiplier = 1.0

        self.reset()
        
        self.validate_implementation()

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_rect(self, surface, color, x, y, w, h, border_color=None, border_width=1):
        points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x + w, y),
            self._iso_to_screen(x + w, y + h),
            self._iso_to_screen(x, y + h)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if border_color:
            pygame.gfxdraw.aapolygon(surface, points, border_color)

    def _setup_stage(self, is_new_game=False):
        if is_new_game:
            self.stage = 1
            self.score = 0
            self.lives = self.MAX_LIVES
            self.ball_speed_multiplier = 1.0
        else:
            self.stage += 1
            self.ball_speed_multiplier += 0.2
        
        self.stage_timer = self.STAGE_TIME_LIMIT
        
        # Reset paddle
        self.paddle_pos = self.LOGICAL_WIDTH / 2
        
        # Reset ball
        self.ball_pos = np.array([self.paddle_pos, self.PADDLE_Y_POS - 1.0], dtype=np.float32)
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        ball_speed = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier
        self.ball_vel = np.array([math.cos(angle) * ball_speed, math.sin(angle) * ball_speed], dtype=np.float32)

        # Create bricks
        self.bricks = []
        for r in range(self.BRICK_ROWS):
            row = []
            for c in range(self.BRICK_COLS):
                brick_x = self.BRICK_AREA_X_START + c
                brick_y = self.BRICK_AREA_Y_START + r
                color_index = (r + c) % len(self.BRICK_COLORS)
                row.append({
                    "pos": np.array([brick_x, brick_y], dtype=np.float32),
                    "active": True,
                    "color": self.BRICK_COLORS[color_index]
                })
            self.bricks.append(row)
        
        self.particles.clear()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self._setup_stage(is_new_game=True)
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over and not self.game_won:
            # 1. Handle Input
            movement = action[0]
            if movement == 3:  # Left
                self.paddle_pos -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle_pos += self.PADDLE_SPEED
            
            paddle_half_width = self.PADDLE_WIDTH / 2
            self.paddle_pos = np.clip(self.paddle_pos, paddle_half_width, self.LOGICAL_WIDTH - paddle_half_width)

            # 2. Update Ball
            self.ball_pos += self.ball_vel
            
            # 3. Handle Collisions
            # Walls
            if self.ball_pos[0] < self.BALL_RADIUS or self.ball_pos[0] > self.LOGICAL_WIDTH - self.BALL_RADIUS:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.LOGICAL_WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce
            if self.ball_pos[1] < self.BALL_RADIUS:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.LOGICAL_HEIGHT - self.BALL_RADIUS)
                # sfx: wall_bounce

            # Paddle
            paddle_y_check = self.PADDLE_Y_POS - self.BALL_RADIUS
            if self.ball_vel[1] > 0 and paddle_y_check <= self.ball_pos[1] < paddle_y_check + abs(self.ball_vel[1]):
                paddle_left = self.paddle_pos - self.PADDLE_WIDTH / 2
                paddle_right = self.paddle_pos + self.PADDLE_WIDTH / 2
                if paddle_left <= self.ball_pos[0] <= paddle_right:
                    self.ball_pos[1] = paddle_y_check
                    
                    hit_pos_norm = (self.ball_pos[0] - self.paddle_pos) / (self.PADDLE_WIDTH / 2)
                    
                    # Risk/Reward calculation
                    if abs(hit_pos_norm) > 0.7:
                        reward += 0.1 # Risky play
                    elif abs(hit_pos_norm) < 0.3:
                        reward -= 0.02 # Safe play
                        
                    base_angle = -math.pi / 2
                    deflection = hit_pos_norm * (math.pi / 3) # Max 60 degree deflection
                    new_angle = base_angle - deflection
                    
                    ball_speed = np.linalg.norm(self.ball_vel)
                    self.ball_vel[0] = math.cos(new_angle) * ball_speed
                    self.ball_vel[1] = math.sin(new_angle) * ball_speed
                    # sfx: paddle_hit

            # Bricks
            bricks_cleared = True
            for r in range(self.BRICK_ROWS):
                for c in range(self.BRICK_COLS):
                    brick = self.bricks[r][c]
                    if not brick["active"]:
                        continue
                    bricks_cleared = False
                    
                    brick_x, brick_y = brick["pos"]
                    if (brick_x <= self.ball_pos[0] <= brick_x + 1 and
                        brick_y <= self.ball_pos[1] <= brick_y + 1):
                        brick["active"] = False
                        reward += 1
                        self.score += 10
                        
                        # sfx: brick_destroy
                        # Spawn particles
                        for _ in range(10):
                            angle = self.np_random.uniform(0, 2 * math.pi)
                            speed = self.np_random.uniform(1, 4)
                            p_vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                            p_pos = self._iso_to_screen(brick_x + 0.5, brick_y + 0.5)
                            self.particles.append({
                                "pos": list(p_pos),
                                "vel": p_vel,
                                "life": self.np_random.uniform(0.5, 1.0),
                                "color": brick["color"]
                            })

                        # Simple bounce logic
                        dist_x = self.ball_pos[0] - (brick_x + 0.5)
                        dist_y = self.ball_pos[1] - (brick_y + 0.5)
                        if abs(dist_x) > abs(dist_y):
                            self.ball_vel[0] *= -1
                        else:
                            self.ball_vel[1] *= -1
                        break # Only break one brick per frame
                if not brick["active"]: break

            # Out of bounds
            if self.ball_pos[1] > self.LOGICAL_HEIGHT:
                self.lives -= 1
                reward -= 1
                # sfx: lose_life
                if self.lives <= 0:
                    self.game_over = True
                else:
                    self._setup_stage(is_new_game=False) # Reset ball/paddle for new life
                    self.stage -= 1 # stay on same stage

            # Stage Clear
            if bricks_cleared:
                reward += 10
                self.score += 100
                if self.stage >= self.NUM_STAGES:
                    self.game_won = True
                    reward += 100
                    self.score += 1000
                else:
                    # sfx: stage_clear
                    self._setup_stage(is_new_game=False)

            # Timer
            self.stage_timer -= 1 / self.FPS
            if self.stage_timer <= 0:
                self.game_over = True
                # sfx: time_up

        # 4. Update Particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1 / self.FPS

        # 5. Update Termination
        self.steps += 1
        terminated = self.game_over or self.game_won or self.steps >= self.MAX_STEPS
        if terminated:
            self.particles.clear()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines for depth
        for i in range(self.LOGICAL_WIDTH + 1):
            p1 = self._iso_to_screen(i, 0)
            p2 = self._iso_to_screen(i, self.LOGICAL_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
        for i in range(self.LOGICAL_HEIGHT + 1):
            p1 = self._iso_to_screen(0, i)
            p2 = self._iso_to_screen(self.LOGICAL_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
            
        # Draw bricks
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                brick = self.bricks[r][c]
                if brick["active"]:
                    darker_color = tuple(max(0, val - 40) for val in brick["color"])
                    self._draw_iso_rect(self.screen, brick["color"], brick["pos"][0], brick["pos"][1], 1, 1, darker_color)

        # Draw paddle
        paddle_x = self.paddle_pos - self.PADDLE_WIDTH / 2
        self._draw_iso_rect(self.screen, self.COLOR_PADDLE, paddle_x, self.PADDLE_Y_POS, self.PADDLE_WIDTH, self.PADDLE_HEIGHT, (200,255,255))

        # Draw ball
        ball_screen_pos = self._iso_to_screen(self.ball_pos[0], self.ball_pos[1])
        ball_screen_radius = int(self.BALL_RADIUS * self.TILE_WIDTH_HALF)
        pygame.gfxdraw.filled_circle(self.screen, ball_screen_pos[0], ball_screen_pos[1], ball_screen_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_screen_pos[0], ball_screen_pos[1], ball_screen_radius, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * 255)))
            color_with_alpha = p["color"] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p["pos"][0]), int(p["pos"][1])))

    def _render_ui(self):
        # UI background bar
        ui_bar = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill((10, 15, 30, 180))
        self.screen.blit(ui_bar, (0, 0))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH / 2 - stage_text.get_width() / 2, 10))
        
        # Lives
        lives_text = self.font_small.render(f"BALLS: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 100, 10))

        # Timer
        timer_text = self.font_small.render(f"TIME: {int(self.stage_timer):02d}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - 200, 10))
        
        # Game Over / Victory messages
        if self.game_over:
            msg = "GAME OVER"
            color = (255, 50, 50)
        elif self.game_won:
            msg = "VICTORY!"
            color = (50, 255, 50)
        else:
            return
            
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        shadow = self.font_large.render(msg, True, (0,0,0))
        shadow_rect = shadow.get_rect(center=(self.WIDTH / 2 + 3, self.HEIGHT / 2 + 3))
        
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's not part of the gym environment but is useful for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Breakout Bandit")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to restart
            
        clock.tick(env.FPS)
        
    env.close()