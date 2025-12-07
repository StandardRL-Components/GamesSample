
# Generated: 2025-08-27T13:33:03.197567
# Source Brief: brief_00400.md
# Brief Index: 400

        
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

    # User-facing control string
    user_guide = (
        "Controls: Use ← and → to move the falling block. Press Space to drop it immediately."
    )

    # User-facing description of the game
    game_description = (
        "Stack blocks to build the tallest tower possible. A wobbly placement can lead to a total collapse!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.BLOCK_SIZE = 20
        self.WIN_HEIGHT = 20
        self.MAX_STEPS = 5000
        self.MOVE_SPEED = 5
        self.INITIAL_FALL_SPEED = 1.0
        self.MAX_FALL_SPEED = 3.0
        
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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TARGET_LINE = (0, 100, 0)
        self.BLOCK_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
            (255, 165, 0),  # Orange
            (0, 255, 0),    # Lime
        ]

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stacked_blocks = []
        self.falling_block = None
        self.falling_block_color = None
        self.falling_block_pos_y_float = 0.0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.tower_height = 0
        self.particles = []
        self.space_was_pressed = False
        self.reward_this_step = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.tower_height = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.space_was_pressed = False
        self.particles.clear()
        
        # Create base platform
        self.stacked_blocks = []
        base_block = pygame.Rect(
            self.WIDTH // 2 - 5 * self.BLOCK_SIZE,
            self.HEIGHT - self.BLOCK_SIZE,
            10 * self.BLOCK_SIZE,
            self.BLOCK_SIZE
        )
        self.stacked_blocks.append((base_block, (80, 80, 80)))
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        self.reward_this_step = 0
        
        if not self.game_over:
            # --- Handle Input ---
            if movement == 3:  # Left
                self.falling_block.x -= self.MOVE_SPEED
            if movement == 4:  # Right
                self.falling_block.x += self.MOVE_SPEED
            
            # Clamp block to screen bounds
            self.falling_block.x = max(0, min(self.WIDTH - self.BLOCK_SIZE, self.falling_block.x))

            # Instant drop on space press (rising edge)
            if space_pressed and not self.space_was_pressed:
                # Sound effect placeholder: # sfx_drop
                # Find the highest block below the falling one to drop onto
                highest_y = self.HEIGHT
                for block, _ in self.stacked_blocks:
                    if self.falling_block.colliderect(pygame.Rect(block.x, 0, block.width, self.HEIGHT)):
                        highest_y = min(highest_y, block.top)
                self.falling_block_pos_y_float = highest_y - self.BLOCK_SIZE
            
            self.space_was_pressed = space_pressed
            
            # --- Update Game Logic ---
            self.steps += 1
            self.fall_speed = min(self.MAX_FALL_SPEED, self.INITIAL_FALL_SPEED + 0.02 * (self.steps // 500))
            self.falling_block_pos_y_float += self.fall_speed
            self.falling_block.y = int(self.falling_block_pos_y_float)

            # --- Collision and Placement ---
            self._check_placement()

            # --- Update Particles ---
            self._update_particles()
        
        # --- Termination and Rewards ---
        terminated = self._check_termination()
        if terminated:
            if self.win:
                self.reward_this_step += 100
                # Sound effect placeholder: # sfx_win
            elif self.game_over:
                self.reward_this_step += -100
                # Sound effect placeholder: # sfx_lose

        self.score += self.reward_this_step
        
        if self.auto_advance:
            self.clock.tick(30)
            
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_new_block(self):
        start_x = self.np_random.integers(self.WIDTH // 4, self.WIDTH * 3 // 4)
        self.falling_block = pygame.Rect(start_x, 0, self.BLOCK_SIZE, self.BLOCK_SIZE)
        self.falling_block_pos_y_float = 0.0
        self.falling_block_color = self.np_random.choice(self.BLOCK_COLORS, p=[0.2]*5)

    def _check_placement(self):
        # Check for collision with any stacked block
        for support_block, _ in self.stacked_blocks:
            if self.falling_block.colliderect(support_block) and self.falling_block.bottom >= support_block.top:
                # Snap to top of support block
                self.falling_block.bottom = support_block.top
                
                # Check placement stability
                dx = self.falling_block.centerx - support_block.centerx
                is_stable = abs(dx) <= support_block.width / 2
                
                if is_stable:
                    # Sound effect placeholder: # sfx_place_block
                    self.stacked_blocks.append((self.falling_block.copy(), self.falling_block_color))
                    self.reward_this_step += 0.1  # Base placement reward

                    # Create particles
                    self._create_particles(self.falling_block.midbottom)

                    # Update tower height and reward
                    old_height = self.tower_height
                    self._update_tower_height()
                    if self.tower_height > old_height:
                        self.reward_this_step += 1.0

                    self._spawn_new_block()
                else:
                    # Unstable placement, game over
                    self.game_over = True
                return # Exit after first collision

        # Check if block fell off the bottom (missed the tower)
        if self.falling_block.top > self.HEIGHT:
            # Sound effect placeholder: # sfx_miss
            self._spawn_new_block()

    def _update_tower_height(self):
        # Tower height is number of blocks above the base
        self.tower_height = len(self.stacked_blocks) - 1

    def _create_particles(self, pos, count=10):
        for _ in range(count):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "size": self.np_random.uniform(2, 5),
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _check_termination(self):
        if self.game_over:
            return True
        if self.tower_height >= self.WIN_HEIGHT:
            self.win = True
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Render Game Elements ---
        # Target height line
        target_y = self.HEIGHT - self.BLOCK_SIZE * (self.WIN_HEIGHT + 1)
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.WIDTH, target_y), 2)
        
        # Stacked blocks
        for block, color in self.stacked_blocks:
            darker_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, (0,0,0), block.inflate(4, 4))
            pygame.draw.rect(self.screen, darker_color, block)
        
        # Falling block
        if self.falling_block and not self.game_over:
            pygame.draw.rect(self.screen, (0,0,0), self.falling_block.inflate(4, 4))
            pygame.draw.rect(self.screen, self.falling_block_color, self.falling_block)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30.0))
            color = (255, 255, 255, alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(p["size"])
            pygame.gfxdraw.box(self.screen, (pos[0], pos[1], size, size), color)

        # --- Render UI ---
        height_text = self.font.render(f"Height: {self.tower_height}/{self.WIN_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        score_text = self.font.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        if self.game_over:
            status_text_str = "YOU WIN!" if self.win else "GAME OVER"
            status_text = self.font.render(status_text_str, True, (255, 50, 50))
            text_rect = status_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(status_text, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.tower_height,
            "win": self.win,
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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set auto_advance to False for manual play, True to watch a random agent
    env.auto_advance = False 
    
    total_reward = 0
    
    print(env.user_guide)
    
    # Main game loop
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keyboard keys to the MultiDiscrete action space
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not env.auto_advance:
            # Manual step on any key press for turn-based feel
            if any(keys):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}, Height: {info['height']}")
                    obs, info = env.reset()
        else:
            # Auto-advancing step for watching agents
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Height: {info['height']}")
                obs, info = env.reset()

        # Render the observation to the screen
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        # Create a display if one doesn't exist
        try:
            display = pygame.display.get_surface()
            if display is None:
                raise Exception
            display.blit(render_surface, (0, 0))
        except Exception:
            display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            display.blit(render_surface, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30)
        
    env.close()