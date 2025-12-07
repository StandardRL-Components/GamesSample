
# Generated: 2025-08-27T16:03:37.635531
# Source Brief: brief_01106.md
# Brief Index: 1106

        
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
        "Controls: ←→ to aim the launcher. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Aim your shots to clear all the blocks before you run out of balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 60  # 60 seconds at 30fps
        self.INITIAL_BALLS = 3
        self.BALL_SPEED = 8
        self.LAUNCHER_WIDTH = 80
        self.LAUNCHER_HEIGHT = 20
        self.BALL_RADIUS = 7
        self.MAX_LAUNCH_ANGLE = 80 # degrees from vertical

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (200, 200, 255)
        self.COLOR_LAUNCHER = (180, 180, 200)
        self.COLOR_WALL = (80, 80, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (255, 89, 94), (255, 202, 58), (138, 201, 38),
            (25, 130, 196), (106, 76, 147)
        ]

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
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Initialize state variables
        self.launcher_pos = None
        self.launcher_angle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_state = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = None
        
        self.reset()
        
        # This check is for development and ensures the implementation adheres to the API.
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.launcher_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - self.LAUNCHER_HEIGHT)
        self.launcher_angle = 0  # 0 is straight up
        
        self.ball_state = "ready"
        self._reset_ball()

        self._create_blocks()
        self.particles = []

        self.balls_left = self.INITIAL_BALLS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        if self.ball_state == "ready":
            if movement == 3:  # Left
                self.launcher_angle = min(self.launcher_angle + 2, self.MAX_LAUNCH_ANGLE)
            elif movement == 4:  # Right
                self.launcher_angle = max(self.launcher_angle - 2, -self.MAX_LAUNCH_ANGLE)
            
            # Launch on space press (rising edge)
            if space_held and not self.prev_space_held:
                self._launch_ball()
                # sound: launch_ball.wav

        self.prev_space_held = space_held

        # --- Update Game State ---
        hit_block_this_step = self._update_ball()
        if hit_block_this_step:
            reward += 1.0
            self.score += 10
        else:
            # Small penalty for time passing without progress
            if self.ball_state == "in_play":
                reward -= 0.01

        self._update_particles()
        
        # --- Check Termination Conditions ---
        if self.game_over: # Set in _update_ball if balls run out
            terminated = True
        elif not self.blocks: # Win condition
            terminated = True
            reward += 100.0
            self.score += 1000
            # sound: win_game.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        else:
            terminated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _reset_ball(self):
        self.ball_state = "ready"
        self.ball_pos = pygame.Vector2(self.launcher_pos.x, self.launcher_pos.y - self.LAUNCHER_HEIGHT/2 - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
    
    def _launch_ball(self):
        if self.ball_state == "ready":
            self.ball_state = "in_play"
            angle_rad = math.radians(self.launcher_angle - 90) # Convert to standard angle
            self.ball_vel = pygame.Vector2(
                math.cos(angle_rad) * self.BALL_SPEED,
                math.sin(angle_rad) * self.BALL_SPEED
            )

    def _create_blocks(self):
        self.blocks = []
        block_rows = 5
        block_cols = 10
        block_width = (self.WIDTH - 40) / block_cols
        block_height = 20
        
        for i in range(block_rows):
            for j in range(block_cols):
                rect = pygame.Rect(
                    20 + j * block_width,
                    40 + i * (block_height + 5),
                    block_width - 5,
                    block_height
                )
                color = random.choice(self.BLOCK_COLORS)
                self.blocks.append({"rect": rect, "color": color})

    def _update_ball(self):
        if self.ball_state != "in_play":
            return False
        
        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.ball_pos.x, self.WIDTH - self.BALL_RADIUS))
            # sound: bounce_wall.wav
        if ball_rect.top <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = max(self.BALL_RADIUS, self.ball_pos.y)
            # sound: bounce_wall.wav

        # Launcher collision
        launcher_rect = pygame.Rect(self.launcher_pos.x - self.LAUNCHER_WIDTH/2, self.launcher_pos.y - self.LAUNCHER_HEIGHT/2, self.LAUNCHER_WIDTH, self.LAUNCHER_HEIGHT)
        if ball_rect.colliderect(launcher_rect) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            # Add horizontal velocity based on hit location
            offset = (self.ball_pos.x - self.launcher_pos.x) / (self.LAUNCHER_WIDTH / 2)
            self.ball_vel.x += offset * 2
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.BALL_SPEED
            # sound: bounce_paddle.wav
        
        # Block collisions
        hit_block = False
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                self._create_particles(block["rect"].center, block["color"])
                self.blocks.remove(block)
                self.ball_vel.y *= -1 # Simple vertical reflection
                hit_block = True
                # sound: break_block.wav
                break # Only break one block per frame

        # Lost ball
        if ball_rect.top > self.HEIGHT:
            self.balls_left -= 1
            # sound: lose_ball.wav
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
        
        return hit_block

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
            lifespan = random.randint(15, 30)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "color": color, "lifespan": lifespan})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            
        # Draw Launcher
        launcher_surf = pygame.Surface((self.LAUNCHER_WIDTH, self.LAUNCHER_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(launcher_surf, self.COLOR_LAUNCHER, (0, 0, self.LAUNCHER_WIDTH, self.LAUNCHER_HEIGHT), border_radius=5)
        rotated_launcher = pygame.transform.rotate(launcher_surf, self.launcher_angle)
        launcher_rect = rotated_launcher.get_rect(center=self.launcher_pos)
        self.screen.blit(rotated_launcher, launcher_rect)

        # Draw Aiming Line
        if self.ball_state == "ready":
            angle_rad = math.radians(self.launcher_angle - 90)
            start_pos = self.ball_pos
            for i in range(10):
                end_pos = start_pos + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * 15
                pygame.draw.line(self.screen, self.COLOR_LAUNCHER, start_pos, end_pos, 2)
                start_pos = end_pos + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * 10
        
        # Draw Ball
        if self.ball_state != "off_screen":
            # Glow effect
            glow_radius = int(self.BALL_RADIUS * 1.8)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_BALL_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (int(self.ball_pos.x) - glow_radius, int(self.ball_pos.y) - glow_radius))

            # Ball itself (anti-aliased)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            size = max(0, int(6 * (p["lifespan"] / 30)))
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p["pos"].x - size/2), int(p["pos"].y - size/2)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Balls Left
        balls_text = self.font_large.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - 150, 5))
        for i in range(self.balls_left):
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 60 + i * 20, 17, 6, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 60 + i * 20, 17, 6, self.COLOR_BALL)
            
        # Game Over / Win Message
        if self.game_over and self.balls_left <= 0:
            msg = "GAME OVER"
        elif not self.blocks:
            msg = "YOU WIN!"
        else:
            msg = None
            
        if msg:
            text_surf = self.font_large.render(msg, True, self.COLOR_BALL)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks)
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
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run with a human player ---
    # This requires pygame to be installed with display support.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Block Breaker")
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        
        env.close()
        
    except pygame.error as e:
        print("Could not create Pygame display. This is expected on a headless server.")
        print("Running a short automated test instead.")
        
        # --- Automated test for headless servers ---
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break
        print(f"Automated test finished. Total reward: {total_reward}, Info: {info}")
        env.close()