
# Generated: 2025-08-27T13:06:08.536721
# Source Brief: brief_00260.md
# Brief Index: 260

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Destroy all the blocks with the ball to win. You have 3 balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 8
        self.BALL_SPEED_INITIAL = 5
        self.BALL_SPEED_MAX = 10

        # --- Colors ---
        self.COLOR_BG_TOP = (15, 20, 35)
        self.COLOR_BG_BOTTOM = (30, 40, 60)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = {
            10: (50, 205, 50),   # Green
            20: (65, 105, 225),  # Blue
            30: (220, 20, 60)    # Red
        }

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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # --- Game State Initialization ---
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.ball_on_paddle = True
        self.consecutive_hits = 0
        
        self.reset()
        
        # --- Self-Validation ---
        # self.validate_implementation() # Uncomment for debugging

    def _generate_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        gap = 4
        rows = 5
        cols = self.WIDTH // (block_width + gap)
        
        start_x = (self.WIDTH - cols * (block_width + gap) + gap) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                points = self.np_random.choice(list(self.BLOCK_COLORS.keys()))
                color = self.BLOCK_COLORS[points]
                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                rect = pygame.Rect(x, y, block_width, block_height)
                self.blocks.append({"rect": rect, "color": color, "points": points})

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top - 2
        self.ball_vel = [0, 0]
        self.consecutive_hits = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.particles = []

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        self._generate_blocks()
        self._reset_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small penalty per step to encourage speed

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        if self.paddle.left < 0:
            self.paddle.left = 0
            reward -= 0.2 # Penalty for hitting side walls
        if self.paddle.right > self.WIDTH:
            self.paddle.right = self.WIDTH
            reward -= 0.2

        if self.ball_on_paddle:
            self.ball.centerx = self.paddle.centerx
            if space_held:
                # sfx: launch_ball
                self.ball_on_paddle = False
                angle = self.np_random.uniform(-math.pi/4, math.pi/4) # Random angle up to 45 deg
                self.ball_vel = [
                    self.BALL_SPEED_INITIAL * math.sin(angle),
                    -self.BALL_SPEED_INITIAL * math.cos(angle)
                ]
        else:
            # --- Ball Physics ---
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]
            reward += self._handle_collisions()

        # --- Update Particles ---
        self._update_particles()
        
        # --- Update State ---
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if not self.blocks: # Win condition
                reward += 100
            elif self.balls_left <= 0: # Lose condition
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_collisions(self):
        collision_reward = 0

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.WIDTH, self.ball.right)
            # sfx: ball_hit_wall
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = 0
            # sfx: ball_hit_wall

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: ball_hit_paddle
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on impact point
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2
            
            # Speed up ball slightly on each paddle hit
            speed = math.hypot(*self.ball_vel)
            if speed < self.BALL_SPEED_MAX:
                self.ball_vel[0] *= 1.02
                self.ball_vel[1] *= 1.02

            # Clamp speed
            speed = math.hypot(*self.ball_vel)
            if speed > self.BALL_SPEED_MAX:
                scale = self.BALL_SPEED_MAX / speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale
                
            self.consecutive_hits = 0 # Reset combo on paddle hit

        # Block collisions
        for i in range(len(self.blocks) - 1, -1, -1):
            block_data = self.blocks[i]
            if self.ball.colliderect(block_data["rect"]):
                # sfx: block_break
                collision_reward += block_data["points"]
                self.score += block_data["points"]
                
                self.consecutive_hits += 1
                if self.consecutive_hits > 1:
                    combo_bonus = 5 * (self.consecutive_hits - 1)
                    collision_reward += combo_bonus
                    self.score += combo_bonus

                self._create_particles(block_data["rect"].center, block_data["color"])
                
                # Determine collision side to correctly reflect the ball
                self._reflect_off_block(block_data["rect"])
                
                self.blocks.pop(i)
                break # Only handle one block collision per frame

        # Bottom wall (lose ball)
        if self.ball.top >= self.HEIGHT:
            # sfx: lose_ball
            self.balls_left -= 1
            collision_reward -= 5
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
        
        return collision_reward

    def _reflect_off_block(self, block_rect):
        # A simple but effective way to handle reflection
        overlap = self.ball.clip(block_rect)
        
        if overlap.width < overlap.height:
            # Horizontal collision is more likely
            self.ball_vel[0] *= -1
            # Nudge ball out of collision
            if self.ball.centerx < block_rect.centerx:
                self.ball.right = block_rect.left
            else:
                self.ball.left = block_rect.right
        else:
            # Vertical collision is more likely
            self.ball_vel[1] *= -1
            # Nudge ball out of collision
            if self.ball.centery < block_rect.centery:
                self.ball.bottom = block_rect.top
            else:
                self.ball.top = block_rect.bottom

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "life": 60, # frames
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.05 # gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.balls_left <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
        
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw ball with glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (self.ball.centerx - glow_radius, self.ball.centery - glow_radius))
        pygame.draw.circle(self.screen, self.COLOR_BALL, self.ball.center, self.BALL_RADIUS)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 60))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:06}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        # Balls left
        for i in range(self.balls_left):
            x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            y = 10 + self.BALL_RADIUS
            pygame.draw.circle(self.screen, self.COLOR_BALL, (x, y), self.BALL_RADIUS)
            
        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (100, 255, 100) if not self.blocks else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Controls ---
    # To use manual controls, you need a window.
    # The environment is designed to be headless, but for testing, we can create a window.
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused in this game
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Frame Rate ---
        env.clock.tick(60) # Run at 60 FPS for smooth human play
        
        if terminated:
            print(f"Game Over. Final Score: {info['score']}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()