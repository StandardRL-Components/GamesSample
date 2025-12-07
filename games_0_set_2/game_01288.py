
# Generated: 2025-08-27T16:39:21.157051
# Source Brief: brief_01288.md
# Brief Index: 1288

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A retro arcade game where you control a paddle to bounce a ball and destroy a grid of blocks. "
        "Clear all blocks to win, but lose a life if you miss the ball. You have 3 lives."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_GRID = (30, 30, 55)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            1: (0, 200, 100),  # Green
            2: (0, 150, 255),  # Blue
            3: (255, 50, 50),  # Red
        }

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 2.0
        self.MAX_BALL_SPEED = 5.0
        self.PARTICLE_LIFETIME = 20
        self.PARTICLE_COUNT = 15

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.ball_attached = True
        self.blocks = []
        self.particles = []
        self.blocks_destroyed_total = 0
        self.blocks_destroyed_since_paddle = 0

        self.reset()
        # self.validate_implementation() # Optional: call for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False

        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        self.ball_attached = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.blocks_destroyed_total = 0
        self.blocks_destroyed_since_paddle = 0

        self.blocks = []
        block_width, block_height = 58, 20
        for i in range(10):  # Rows
            for j in range(10):  # Columns
                points = 1
                if i < 2: points = 3
                elif i < 5: points = 2
                
                block_rect = pygame.Rect(
                    j * (block_width + 6) + 28,
                    i * (block_height + 5) + 50,
                    block_width,
                    block_height,
                )
                self.blocks.append({
                    "rect": block_rect, 
                    "points": points, 
                    "color": self.BLOCK_COLORS[points]
                })

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small penalty for time to encourage efficiency

        self._handle_input(action)

        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
        else:
            event_reward = self._update_ball()
            reward += event_reward

        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if not self.blocks:  # Win condition
                reward += 100
            else:  # Loss condition
                reward += -50
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if space_held and self.ball_attached:
            self.ball_attached = False
            # Sound: Ball launch
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = [math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed]

    def _update_ball(self):
        reward = 0
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = np.clip(ball_rect.left, 0, self.WIDTH - ball_rect.width)
            self.ball_pos[0] = ball_rect.centerx
            # Sound: Wall bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = 0
            self.ball_pos[1] = ball_rect.centery
            # Sound: Wall bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on hit location
            hit_offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += hit_offset * 2.0
            
            # Normalize speed
            current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            self.ball_vel = [(v / current_speed) * self.ball_speed for v in self.ball_vel]
            
            # Reward for paddle hit quality
            if abs(hit_offset) < 0.2: # Center 20%
                reward += 0.1
            elif abs(hit_offset) > 0.8: # Outer 10% on each side
                reward -= 0.2
            
            # Reset combo counter
            self.blocks_destroyed_since_paddle = 0
            # Sound: Paddle bounce

        # Block collisions
        hit_block = None
        for block in self.blocks:
            if ball_rect.colliderect(block["rect"]):
                hit_block = block
                break

        if hit_block:
            # Sound: Block break
            reward += hit_block["points"]
            if self.blocks_destroyed_since_paddle >= 1:
                reward += 1 # Combo bonus
            self.blocks_destroyed_since_paddle += 1
            
            self._create_particles(pygame.Vector2(ball_rect.center), hit_block["color"], self.PARTICLE_COUNT)
            self.blocks.remove(hit_block)
            self.score += hit_block["points"]
            self.blocks_destroyed_total += 1

            # Determine bounce direction
            # A simple vertical bounce is most common and predictable
            self.ball_vel[1] *= -1
            
            # Increase ball speed every 20 blocks
            if self.blocks_destroyed_total > 0 and self.blocks_destroyed_total % 20 == 0:
                self.ball_speed = min(self.MAX_BALL_SPEED, self.ball_speed + 0.05 * self.ball_speed)


        # Missed ball (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            # Sound: Lose life
            if self.lives > 0:
                self.ball_attached = True
                self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
                self.ball_vel = [0, 0]
            else:
                self.game_over = True
        
        return reward

    def _check_termination(self):
        return self.lives <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": self.PARTICLE_LIFETIME,
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Drag
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], 2, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / self.PARTICLE_LIFETIME))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color
            )

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow effect
        pygame.gfxdraw.filled_circle(
            self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW
        )
        # Solid ball
        pygame.gfxdraw.filled_circle(
            self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL
        )
        pygame.gfxdraw.aacircle(
            self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL
        )


    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Lives
        lives_text = self.font.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 180, 10))
        for i in range(self.lives):
            life_paddle_rect = pygame.Rect(self.WIDTH - 80 + (i * 25), 16, 20, 8)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_paddle_rect, border_radius=2)
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font.render(message, True, self.COLOR_BALL)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Breakout Arcade")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    print(env.game_description)

    while not done:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling (for closing the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Frame rate ---
        clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()