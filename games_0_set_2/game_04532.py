
# Generated: 2025-08-28T02:41:23.819111
# Source Brief: brief_04532.md
# Brief Index: 4532

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = "Controls: Use ← and → to move the paddle left and right."

    # User-facing game description
    game_description = "A retro block-breaker. Deflect falling blocks with your paddle to destroy them. Don't let any blocks reach the bottom!"

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WALL_THICKNESS = 10
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BLOCK_SIZE = 24
    BLOCK_PADDING = 4
    TOTAL_BLOCKS = 50
    MAX_HITS = 3
    MAX_LIVES = 3
    INITIAL_BLOCK_FALL_SPEED = 1.5
    BLOCK_SPEED_INCREASE = 0.05
    MAX_BLOCK_SPEED = 5.0
    MAX_STEPS = 10000

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (100, 100, 120)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_ACCENT = (200, 200, 255)
    COLOR_TEXT = (220, 220, 240)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]
    COLOR_WHITE = (255, 255, 255)

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.paddle = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.blocks_destroyed_count = 0
        self.current_block_speed = self.INITIAL_BLOCK_FALL_SPEED

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.blocks_destroyed_count = 0
        self.current_block_speed = self.INITIAL_BLOCK_FALL_SPEED

        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - self.WALL_THICKNESS - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        self.blocks = self._create_blocks()
        self.particles = []

        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        blocks = []
        cols = 10
        rows = self.TOTAL_BLOCKS // cols
        block_total_width = cols * (self.BLOCK_SIZE + self.BLOCK_PADDING) - self.BLOCK_PADDING
        start_x = (self.SCREEN_WIDTH - block_total_width) / 2
        start_y = 60

        for i in range(self.TOTAL_BLOCKS):
            row = i // cols
            col = i % cols
            x = start_x + col * (self.BLOCK_SIZE + self.BLOCK_PADDING)
            y = start_y + row * (self.BLOCK_SIZE + self.BLOCK_PADDING)
            
            base_color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
            base_color = self.BLOCK_COLORS[base_color_idx]
            
            blocks.append({
                "rect": pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE),
                "base_color": base_color,
                "vx": 0,
                "vy": self.current_block_speed,
                "hits": 0,
            })
        return blocks

    def step(self, action):
        reward = -0.01  # Time penalty

        # --- Handle Action ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        self.paddle.x = np.clip(
            self.paddle.x,
            self.WALL_THICKNESS,
            self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS,
        )

        # --- Update Game State ---
        self._update_difficulty()
        reward += self._update_blocks()
        self._update_particles()
        
        # --- Check Termination ---
        win = not self.blocks
        lose = self.lives <= 0
        timeout = self.steps >= self.MAX_STEPS

        if win:
            reward += 50
            # // Sound effect: Win
            self.game_over = True
        elif lose or timeout:
            self.game_over = True
            
        self.steps += 1
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )
    
    def _update_difficulty(self):
        speed_increase_steps = self.blocks_destroyed_count // 5
        self.current_block_speed = min(
            self.MAX_BLOCK_SPEED,
            self.INITIAL_BLOCK_FALL_SPEED + speed_increase_steps * self.BLOCK_SPEED_INCREASE
        )

    def _update_blocks(self):
        reward = 0
        for block in self.blocks[:]:
            # Apply base falling speed if not hit yet
            if block['vy'] >= 0 and block['vx'] == 0:
                block['vy'] = self.current_block_speed

            block["rect"].x += block["vx"]
            block["rect"].y += block["vy"]

            # Wall collisions
            if block["rect"].left <= self.WALL_THICKNESS or block["rect"].right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
                block["vx"] *= -1
                block["rect"].x = np.clip(block["rect"].x, self.WALL_THICKNESS, self.SCREEN_WIDTH - self.WALL_THICKNESS - self.BLOCK_SIZE)
                # // Sound effect: Wall bounce

            # Top wall collision (destruction)
            if block["rect"].top <= self.WALL_THICKNESS:
                reward += self._destroy_block(block)

            # Paddle collision
            if block["rect"].colliderect(self.paddle) and block["vy"] > 0:
                # // Sound effect: Paddle hit
                block["hits"] += 1
                if block["hits"] >= self.MAX_HITS:
                    reward += self._destroy_block(block)
                else:
                    # Bounce
                    block["rect"].bottom = self.paddle.top - 1
                    
                    # Calculate bounce angle based on hit position
                    hit_pos = (block["rect"].centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                    hit_pos = np.clip(hit_pos, -1, 1) # Normalize
                    
                    angle = math.radians(90 - hit_pos * 75) # 15 to 165 degrees
                    speed = math.sqrt(block['vx']**2 + block['vy']**2) * 1.05 # slight speed up
                    
                    block["vx"] = math.cos(angle) * speed
                    block["vy"] = -math.sin(angle) * speed
                    
            # Bottom wall collision (life loss)
            if block["rect"].top >= self.SCREEN_HEIGHT:
                self.lives -= 1
                reward -= 5
                # // Sound effect: Life lost
                self._spawn_particles(block["rect"].center, (255, 50, 50), 50, is_explosion=True)
                self.blocks.remove(block)

        return reward

    def _destroy_block(self, block):
        # // Sound effect: Block break
        self._spawn_particles(block["rect"].center, block["base_color"], 20)
        self.blocks.remove(block)
        self.score += 1
        self.blocks_destroyed_count += 1
        return 1 # Return reward for destruction

    def _spawn_particles(self, pos, color, count, is_explosion=False):
        for _ in range(count):
            if is_explosion:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                life = self.np_random.integers(20, 40)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(0.5, 2.5)
                life = self.np_random.integers(15, 30)

            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color,
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
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
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            radius = int(3 * (p["life"] / p["max_life"]))
            if radius > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, (*p["color"], alpha)
                )

        # Blocks
        for block in self.blocks:
            damage_factor = block["hits"] / self.MAX_HITS
            current_color = tuple(int(c + (255 - c) * damage_factor * 0.7) for c in block["base_color"])
            pygame.draw.rect(self.screen, current_color, block["rect"], border_radius=3)
            darker_color = tuple(max(0, c - 40) for c in current_color)
            pygame.draw.rect(self.screen, darker_color, block["rect"], width=2, border_radius=3)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        accent_rect = self.paddle.inflate(-6, -6)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_ACCENT, accent_rect, border_radius=3)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 15, self.WALL_THICKNESS + 5))

        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - self.WALL_THICKNESS - 15, self.WALL_THICKNESS + 5))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if not self.blocks:
                end_text_str = "YOU WIN!"
            else:
                end_text_str = "GAME OVER"

            end_text = self.font_large.render(end_text_str, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless

    env = GameEnv()
    
    # --- Test Reset ---
    print("--- Testing Reset ---")
    obs, info = env.reset(seed=42)
    print(f"Initial Info: {info}")
    assert info['score'] == 0
    assert info['steps'] == 0
    assert info['lives'] == GameEnv.MAX_LIVES
    assert info['blocks_remaining'] == GameEnv.TOTAL_BLOCKS
    print("Reset test passed.")

    # --- Test a few steps ---
    print("\n--- Testing Step ---")
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample() # Random actions
        action[0] = 4 # Move right
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i + 1) % 20 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")
        if terminated:
            print(f"Episode terminated at step {i+1}.")
            break
    print("Step test completed.")

    # --- Test win condition ---
    print("\n--- Testing Win Condition ---")
    env.reset()
    env.blocks = [] # Manually remove all blocks
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"Win state: Terminated={terminated}, Reward={reward:.2f}, Info={info}")
    assert terminated
    assert reward > 40 # Should get win bonus + step penalty
    print("Win condition test passed.")

    # --- Test loss condition ---
    print("\n--- Testing Loss Condition ---")
    env.reset()
    env.lives = 1
    # Manually move a block to the bottom
    if env.blocks:
        env.blocks[0]['rect'].y = GameEnv.SCREEN_HEIGHT
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"Loss state: Terminated={terminated}, Reward={reward:.2f}, Info={info}")
        assert terminated
        assert info['lives'] == 0
        assert reward < 0 # Should get life loss penalty
        print("Loss condition test passed.")

    env.close()