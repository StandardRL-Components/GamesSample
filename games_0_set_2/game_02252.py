
# Generated: 2025-08-27T19:45:17.240764
# Source Brief: brief_02252.md
# Brief Index: 2252

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a 60-second barrage of falling blocks by deflecting them with your paddle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # As per auto_advance recommendation
        self.MAX_STEPS = 60 * self.FPS # 60 seconds

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (50, 255, 255),  # Cyan
            (255, 50, 255),  # Magenta
            (255, 255, 50),  # Yellow
            (50, 255, 50),   # Lime
            (255, 100, 50),  # Orange
        ]

        # Game parameters
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BLOCK_SIZE = 20
        self.INITIAL_BLOCK_SPEED = 2.5
        self.DIFFICULTY_INTERVAL = 10 * self.FPS # 10 seconds
        self.DIFFICULTY_SPEED_INCREASE = 0.5
        self.MIN_SPAWN_INTERVAL = 10
        self.MAX_SPAWN_INTERVAL = 25

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.paddle_x = 0
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.block_speed = 0
        self.difficulty_timer = 0
        self.block_spawn_timer = 0
        
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.block_speed = self.INITIAL_BLOCK_SPEED
        self.difficulty_timer = 0
        self.block_spawn_timer = self.np_random.integers(self.MIN_SPAWN_INTERVAL, self.MAX_SPAWN_INTERVAL)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        
        reward = 0.0
        terminated = False

        # --- Game Logic ---
        if not self.game_over:
            # 1. Handle Input
            if movement == 3:  # Left
                self.paddle_x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle_x += self.PADDLE_SPEED
            
            # Clamp paddle position
            self.paddle_x = max(0, min(self.paddle_x, self.WIDTH - self.PADDLE_WIDTH))

            # 2. Update Timers and State
            self.steps += 1
            self.difficulty_timer += 1
            self.block_spawn_timer -= 1
            reward += 0.01 # Small reward for surviving a frame

            # 3. Increase Difficulty
            if self.difficulty_timer >= self.DIFFICULTY_INTERVAL:
                self.block_speed += self.DIFFICULTY_SPEED_INCREASE
                self.difficulty_timer = 0
            
            # 4. Spawn new blocks
            if self.block_spawn_timer <= 0:
                self._spawn_block()
                self.block_spawn_timer = self.np_random.integers(self.MIN_SPAWN_INTERVAL, self.MAX_SPAWN_INTERVAL)

            # 5. Update blocks and check for collisions
            reward, terminated = self._update_blocks(reward)

            # 6. Update particles
            self._update_particles()
            
            # 7. Check for win condition
            if self.steps >= self.MAX_STEPS:
                terminated = True
                reward = 100.0 # Large reward for winning

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_block(self):
        block_x = self.np_random.integers(0, self.WIDTH - self.BLOCK_SIZE)
        color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
        block = {
            "rect": pygame.Rect(block_x, -self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE),
            "color": self.BLOCK_COLORS[color_index]
        }
        self.blocks.append(block)

    def _update_blocks(self, reward):
        paddle_rect = pygame.Rect(self.paddle_x, self.HEIGHT - self.PADDLE_HEIGHT - 10, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        remaining_blocks = []
        for block in self.blocks:
            block["rect"].y += self.block_speed
            
            # Check for paddle collision
            if block["rect"].colliderect(paddle_rect):
                reward += 1.0
                self.score += 1
                self._create_particles(block["rect"].center, block["color"])
                # Sound: Block break
                continue # Block is destroyed

            # Check for bottom edge collision (game over)
            if block["rect"].top > self.HEIGHT:
                self.game_over = True
                reward = -100.0 # Large penalty for losing
                # Sound: Game over
                return reward, True

            remaining_blocks.append(block)
        
        self.blocks = remaining_blocks
        return reward, self.game_over

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            particle = {
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            }
            self.particles.append(particle)

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = p["color"]
            # Use gfxdraw for anti-aliased circles for a softer particle look
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color + (alpha,))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            # Add a slight inner border for depth
            inner_color = tuple(max(0, c - 40) for c in block["color"])
            pygame.draw.rect(self.screen, inner_color, block["rect"], 2)

        # Draw paddle
        paddle_rect = pygame.Rect(self.paddle_x, self.HEIGHT - self.PADDLE_HEIGHT - 10, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        # Add a subtle glow/highlight
        highlight_color = (200, 200, 255, 100)
        highlight_rect = paddle_rect.inflate(6, 6)
        highlight_surf = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(highlight_surf, highlight_color, highlight_surf.get_rect(), border_radius=6)
        self.screen.blit(highlight_surf, highlight_rect.topleft)

    def _render_ui(self):
        # Render score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Render game over/victory message
        if self.game_over:
            msg = "GAME OVER"
            color = (255, 50, 50)
        elif self.steps >= self.MAX_STEPS:
            msg = "VICTORY!"
            color = (50, 255, 50)
        else:
            return

        end_text = self.font_large.render(msg, True, color)
        end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen_human = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Barrage")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # --- Game Loop for Human Play ---
    while not done:
        # Action defaults
        movement = 0 # no-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # Action is always submitted, even if it's a no-op
        action = [movement, 0, 0] # Space and Shift are not used
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the human-facing screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_human.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()