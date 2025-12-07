
# Generated: 2025-08-27T12:44:22.096155
# Source Brief: brief_00148.md
# Brief Index: 148

        
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
        "Controls: Use ← and → to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Destroy all blocks to advance to the next stage."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto-advance logic
        self.MAX_STEPS = 5000
        self.TOTAL_STAGES = 3
        self.INITIAL_LIVES = 3

        # Colors (Vibrant Retro Arcade)
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (255, 0, 128), (0, 255, 255), (0, 255, 0),
            (255, 128, 0), (128, 0, 255)
        ]

        # Game object properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 8
        self.BLOCK_ROWS, self.BLOCK_COLS = 10, 12
        self.BLOCK_WIDTH = self.WIDTH // self.BLOCK_COLS
        self.BLOCK_HEIGHT = 20
        self.BLOCK_AREA_TOP = 50

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State ---
        # These are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.lives = None
        self.stage = None
        self.game_over = None
        self.bounces_since_last_hit = None
        self.active_blocks_count = None
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def _setup_stage(self):
        """Creates the block layout for the current stage."""
        self.blocks = []
        base_block_count = 75
        block_count = int(base_block_count * (1.1 ** (self.stage - 1)))
        
        possible_positions = []
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                possible_positions.append((r, c))
        
        # Ensure we don't try to place more blocks than available spots
        block_count = min(block_count, len(possible_positions))
        
        chosen_positions = self.np_random.choice(len(possible_positions), block_count, replace=False)

        for idx in chosen_positions:
            r, c = possible_positions[idx]
            x = c * self.BLOCK_WIDTH
            y = self.BLOCK_AREA_TOP + r * self.BLOCK_HEIGHT
            rect = pygame.Rect(x, y, self.BLOCK_WIDTH - 2, self.BLOCK_HEIGHT - 2)
            color = self.np_random.choice(len(self.BLOCK_COLORS))
            self.blocks.append({"rect": rect, "color": self.BLOCK_COLORS[color], "active": True})
        self.active_blocks_count = len(self.blocks)

    def _reset_ball_and_paddle(self):
        """Resets the paddle and ball to their starting positions for a new life."""
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        self.ball_launched = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)
        self.bounces_since_last_hit = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.game_over = False
        self.particles = []
        
        self._setup_stage()
        self._reset_ball_and_paddle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for every step to encourage speed
        
        # --- Action Handling ---
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean (we care about the press, not hold)
        
        # --- Game Logic Update ---
        self._update_paddle(movement)
        
        if not self.ball_launched:
            self.ball_pos[0] = self.paddle.centerx
            if space_pressed:
                self.ball_launched = True
                # Launch with a random horizontal component
                initial_vx = self.np_random.uniform(-0.5, 0.5)
                self.ball_vel = np.array([initial_vx, -1.0])
                self.ball_vel /= np.linalg.norm(self.ball_vel)
                self.ball_vel *= self.BALL_SPEED
                # Sound: Ball Launch
        else:
            # Update ball position
            self.ball_pos += self.ball_vel

            # --- Collision Detection ---
            # Walls
            if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                self.bounces_since_last_hit += 1
                # Sound: Wall Bounce
            if self.ball_pos[1] <= self.BALL_RADIUS:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                self.bounces_since_last_hit += 1
                # Sound: Wall Bounce
            
            # Paddle
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
                reward += 0.1 # Reward for successful return
                self.bounces_since_last_hit = 0
                
                # Change ball angle based on where it hits the paddle
                hit_offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = hit_offset * 1.2 # Max horizontal velocity component
                self.ball_vel[1] *= -1
                self.ball_vel /= np.linalg.norm(self.ball_vel)
                self.ball_vel *= self.BALL_SPEED
                # Sound: Paddle Hit
            
            # Blocks
            for block in self.blocks:
                if block["active"] and block["rect"].colliderect(ball_rect):
                    block["active"] = False
                    self.active_blocks_count -= 1
                    reward += 1  # Reward for breaking a block
                    self.score += 10
                    self.bounces_since_last_hit = 0
                    self._create_particles(block["rect"].center, block["color"])
                    # Sound: Block Break

                    # Determine bounce direction
                    # Simple but effective: reverse velocity component based on which side was hit
                    # A more robust method would check penetration depth
                    dx = self.ball_pos[0] - block["rect"].centerx
                    dy = self.ball_pos[1] - block["rect"].centery
                    if abs(dx / block["rect"].width) > abs(dy / block["rect"].height):
                        self.ball_vel[0] *= -1
                    else:
                        self.ball_vel[1] *= -1
                    break # Only handle one block collision per frame

            # Anti-softlock mechanism
            if self.bounces_since_last_hit > 20:
                self.ball_vel[1] += self.np_random.uniform(-0.1, 0.1)
                self.ball_vel /= np.linalg.norm(self.ball_vel)
                self.ball_vel *= self.BALL_SPEED
                self.bounces_since_last_hit = 0

            # Out of bounds (lose life)
            if self.ball_pos[1] > self.HEIGHT:
                self.lives -= 1
                reward -= 10 # Penalty for losing a life
                # Sound: Lose Life
                if self.lives <= 0:
                    self.game_over = True
                else:
                    self._reset_ball_and_paddle()

        # Update particles
        self._update_particles()
        
        # --- Stage/Game End Conditions ---
        if self.active_blocks_count <= 0 and not self.game_over:
            reward += 10 # Reward for clearing stage
            self.score += 100
            self.stage += 1
            if self.stage > self.TOTAL_STAGES:
                reward += 50 # Bonus for winning the game
                self.score += 500
                self.game_over = True
            else:
                self._setup_stage()
                self._reset_ball_and_paddle()
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        self.steps += 1
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _create_particles(self, position, color):
        # Sound: Particle Burst
        for _ in range(20):
            vel = self.np_random.uniform(-2, 2, size=2)
            self.particles.append({
                "pos": np.array(position, dtype=float),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.uniform(1, 4)
            })
            
    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color_with_alpha = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

        # Blocks
        for block in self.blocks:
            if block["active"]:
                pygame.draw.rect(self.screen, block["color"], block["rect"])
                # Add a slight inner bevel for depth
                pygame.draw.rect(self.screen, (255,255,255, 40), block["rect"].inflate(-6,-6), 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        # Glow effect for paddle
        paddle_glow = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(paddle_glow.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE, 50), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, paddle_glow.topleft)

        # Ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow effect for ball
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 4, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 2, (*self.COLOR_BALL, 100))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))
        
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "YOU WIN!" if self.active_blocks_count <= 0 else "GAME OVER"
            status_text = self.font_big.render(status_text_str, True, self.COLOR_PADDLE)
            self.screen.blit(status_text, (self.WIDTH // 2 - status_text.get_width() // 2, self.HEIGHT // 2 - 50))
            
            final_score_text = self.font_ui.render(f"FINAL SCORE: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text, (self.WIDTH // 2 - final_score_text.get_width() // 2, self.HEIGHT // 2 + 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "active_blocks": self.active_blocks_count,
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
    
    # --- Manual Play ---
    # This requires a window, so we'll re-init pygame for display
    pygame.quit() # Close the headless instance
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        movement = 0 # No-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()
    pygame.quit()