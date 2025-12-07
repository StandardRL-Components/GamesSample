
# Generated: 2025-08-28T06:38:56.723975
# Source Brief: brief_02979.md
# Brief Index: 2979

        
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
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Clear all blocks across 3 stages to win. Lose all your balls, and it's game over. Each stage has a 60-second time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # As specified by "no-ops at a rate of 30fps"
        self.MAX_STEPS = 3 * 60 * self.FPS # 3 stages * 60 seconds/stage

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 200, 0)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            "red": {"color": (255, 50, 50), "points": 10, "reward": 1},
            "green": {"color": (50, 255, 50), "points": 20, "reward": 2},
            "blue": {"color": (50, 150, 255), "points": 30, "reward": 3},
            "yellow": {"color": (255, 255, 0), "points": 50, "reward": 5},
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Initialize state variables
        self.paddle = None
        self.balls = []
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = 0
        self.current_stage = 0
        self.stage_timer = 0
        self.last_paddle_x = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def _setup_stage(self, stage_num):
        self.current_stage = stage_num
        self.stage_timer = 60 * self.FPS
        self.blocks.clear()
        self.particles.clear()
        self.balls.clear()

        # Generate blocks
        block_width, block_height = 40, 20
        rows, cols = 6, 14
        top_offset = 60
        
        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() < 0.2: # Create some gaps
                    continue
                
                block_type_rand = self.np_random.random()
                if block_type_rand < 0.4:
                    block_type = "red"
                elif block_type_rand < 0.7:
                    block_type = "green"
                elif block_type_rand < 0.9:
                    block_type = "blue"
                else:
                    block_type = "yellow"

                x = c * (block_width + 5) + 30
                y = r * (block_height + 5) + top_offset
                rect = pygame.Rect(x, y, block_width, block_height)
                
                speed = 0
                if self.current_stage == 2:
                    speed = self.np_random.choice([-1, 1]) * 1
                elif self.current_stage == 3:
                    speed = self.np_random.choice([-1, 1]) * 3

                self.blocks.append({
                    "rect": rect,
                    "type": block_type,
                    "speed": speed,
                    "color": self.BLOCK_COLORS[block_type]["color"]
                })
        
        self._spawn_ball()

    def _spawn_ball(self):
        ball_radius = 8
        ball_speed = 5
        ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - ball_radius - 5)
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * ball_speed
        self.balls.append({"pos": ball_pos, "vel": ball_vel, "radius": ball_radius})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = 3
        
        self.paddle = pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 30, 100, 15)
        self.last_paddle_x = self.paddle.x
        
        self._setup_stage(1)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1 # Survival reward
        self.steps += 1
        self.stage_timer -= 1
        
        # 1. Handle player input
        movement = action[0]
        paddle_speed = 10
        
        prev_paddle_x = self.paddle.x
        if movement == 3:  # Left
            self.paddle.x -= paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += paddle_speed
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.paddle.width)
        
        # Reward for safe/small movements
        if abs(self.paddle.x - prev_paddle_x) < paddle_speed / 2:
            reward -= 0.02

        # 2. Update game logic
        reward += self._update_balls()
        self._update_blocks()
        self._update_particles()

        # 3. Check for stage/game state changes
        terminated = False
        if not self.blocks: # Stage clear
            reward += 50
            if self.current_stage < 3:
                self._setup_stage(self.current_stage + 1)
            else: # Game won
                reward += 100
                self.game_over = True
                terminated = True

        if self.stage_timer <= 0 or self.balls_remaining <= 0:
            self.game_over = True
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_blocks(self):
        for block in self.blocks:
            if block["speed"] != 0:
                block["rect"].x += block["speed"]
                if block["rect"].left < 0 or block["rect"].right > self.WIDTH:
                    block["speed"] *= -1
                    block["rect"].x += block["speed"] # Prevent sticking

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"].y += 0.1 # Gravity
            p["lifespan"] -= 1

    def _update_balls(self):
        reward = 0
        
        for ball in self.balls[:]:
            ball["pos"] += ball["vel"]
            ball_rect = pygame.Rect(ball["pos"].x - ball["radius"], ball["pos"].y - ball["radius"], ball["radius"]*2, ball["radius"]*2)

            # Wall collisions
            if ball["pos"].x - ball["radius"] < 0 or ball["pos"].x + ball["radius"] > self.WIDTH:
                ball["vel"].x *= -1
                ball["pos"].x = np.clip(ball["pos"].x, ball["radius"], self.WIDTH - ball["radius"])
                # sfx: wall_bounce
            if ball["pos"].y - ball["radius"] < 0:
                ball["vel"].y *= -1
                ball["pos"].y = np.clip(ball["pos"].y, ball["radius"], self.HEIGHT - ball["radius"])
                # sfx: wall_bounce

            # Paddle collision
            if ball_rect.colliderect(self.paddle) and ball["vel"].y > 0:
                ball["vel"].y *= -1
                ball["pos"].y = self.paddle.top - ball["radius"]
                
                # Add "spin" based on hit location
                offset = (ball["pos"].x - self.paddle.centerx) / (self.paddle.width / 2)
                ball["vel"].x += offset * 2.5
                ball["vel"].normalize_ip()
                ball["vel"] *= 5 # Maintain speed
                # sfx: paddle_hit

            # Block collisions
            for block in self.blocks[:]:
                if ball_rect.colliderect(block["rect"]):
                    # sfx: block_break
                    block_info = self.BLOCK_COLORS[block["type"]]
                    self.score += block_info["points"]
                    reward += block_info["reward"]
                    
                    self._create_particles(block["rect"].center, block["color"])
                    self.blocks.remove(block)

                    # Determine bounce direction
                    # A simple but effective method:
                    # Check overlap and reverse velocity component of the smallest overlap
                    overlap_x = min(ball_rect.right, block["rect"].right) - max(ball_rect.left, block["rect"].left)
                    overlap_y = min(ball_rect.bottom, block["rect"].bottom) - max(ball_rect.top, block["rect"].top)

                    if overlap_x < overlap_y:
                        ball["vel"].x *= -1
                    else:
                        ball["vel"].y *= -1
                    break # One block per ball per frame

            # Ball out of bounds
            if ball["pos"].y - ball["radius"] > self.HEIGHT:
                self.balls.remove(ball)
                # sfx: ball_lost
                
        # Handle ball respawn/loss
        if not self.balls and not self.game_over:
            self.balls_remaining -= 1
            reward -= 50
            
            # Yellow block penalty
            if any(b["type"] == "yellow" for b in self.blocks):
                reward -= 1
            
            if self.balls_remaining > 0:
                self._spawn_ball()
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])))

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], 2)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Render balls (antialiased)
        for ball in self.balls:
            pygame.gfxdraw.aacircle(self.screen, int(ball["pos"].x), int(ball["pos"].y), int(ball["radius"]), self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(ball["pos"].x), int(ball["pos"].y), int(ball["radius"]), self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_main.render(f"STAGE: {self.current_stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))
        
        # Balls remaining
        ball_icon_radius = 8
        for i in range(self.balls_remaining):
            x = self.WIDTH - 20 - i * (ball_icon_radius * 2 + 5)
            y = 10 + ball_icon_radius
            pygame.gfxdraw.aacircle(self.screen, x, y, ball_icon_radius, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, ball_icon_radius, self.COLOR_BALL)

        # Timer
        time_left = max(0, self.stage_timer / self.FPS)
        timer_color = self.COLOR_TEXT if time_left > 10 else (255, 80, 80)
        timer_text = self.font_main.render(f"{time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, self.HEIGHT - 55))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = not self.blocks and self.current_stage == 3
            message = "YOU WIN!" if win else "GAME OVER"
            
            end_text = self.font_main.render(message, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "balls_left": self.balls_remaining,
            "time_left": round(self.stage_timer / self.FPS, 2)
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # To display the game, we need a separate Pygame window
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Main game loop
    running = True
    while running:
        # Get action from keyboard (for human play)
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # In a real scenario, you might wait for a reset action
            # For this example, we just let the loop continue showing the final screen
            # until 'r' is pressed to reset.

    env.close()