import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the block. Hold Space to drop it quickly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks to reach the target height. Risky placements earn more points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.BLOCK_WIDTH = 80
        self.BLOCK_HEIGHT = 20
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Game constants
        self.PLAYER_SPEED = 8
        self.TARGET_HEIGHT = 10
        self.MAX_STEPS = 1000
        self.BASE_FALL_SPEED = 2.0
        self.FAST_DROP_SPEED = 20.0

        # Colors
        self.COLOR_BG = (44, 62, 80)
        self.COLOR_GRID = (52, 73, 94)
        self.COLOR_GROUND = (127, 140, 141)
        self.COLOR_TARGET_LINE = (46, 204, 113)
        self.COLOR_TEXT = (236, 240, 241)
        self.BLOCK_COLORS = [
            (231, 76, 60), (230, 126, 34), (241, 196, 15),
            (26, 188, 156), (52, 152, 219), (155, 89, 182)
        ]

        # State variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.stacked_blocks = []
        self.falling_block = None
        self.falling_block_color = None
        self.particles = []
        self.current_fall_speed = 0.0
        self.np_random = None
        
        # Initialize state variables
        # A seed is not passed here, so the environment will be initialized with a random seed.
        # A specific seed can be passed to reset() later.
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.particles = []
        self.current_fall_speed = self.BASE_FALL_SPEED

        ground_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.BLOCK_HEIGHT, self.SCREEN_WIDTH, self.BLOCK_HEIGHT)
        self.stacked_blocks = [(ground_rect, self.COLOR_GROUND)]

        self._spawn_new_block()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update game logic: Handle input
        if movement == 3:  # Left
            self.falling_block.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.falling_block.x += self.PLAYER_SPEED
        
        self.falling_block.x = max(0, min(self.SCREEN_WIDTH - self.BLOCK_WIDTH, self.falling_block.x))

        # Update game logic: Physics and difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.current_fall_speed += 0.2
        
        drop_speed = self.FAST_DROP_SPEED if space_held else self.current_fall_speed
        self.falling_block.y += drop_speed

        # Check for landing
        landed = False
        support_block_rect = None
        for block_rect, _ in reversed(self.stacked_blocks):
            if self.falling_block.colliderect(block_rect):
                self.falling_block.bottom = block_rect.top
                support_block_rect = block_rect
                landed = True
                break

        # Process game events (landing, falling off, etc.)
        if landed:
            # sfx: THUD_SOUND
            reward += self._calculate_placement_reward(support_block_rect)
            self.stacked_blocks.append((self.falling_block.copy(), self.falling_block_color))
            self._create_particles(self.falling_block.midbottom, self.falling_block_color)
            
            if self._check_win_condition():
                reward += 100
                terminated = True
                # sfx: WIN_SOUND
            else:
                self._spawn_new_block()
        
        # Check for loss condition
        if not landed and self.falling_block.top > self.SCREEN_HEIGHT:
            reward = -10.0
            terminated = True
            # sfx: LOSE_SOUND
            
        # Update counters and check for timeout
        self.steps += 1
        self.score += reward
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
        
        self.game_over = terminated
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_placement_reward(self, support_block):
        # Ground placement is a neutral event, base reward.
        if support_block.width >= self.SCREEN_WIDTH:
            return 0.1

        # For stacked blocks, reward based on risk.
        is_overhang = (self.falling_block.left < support_block.left or
                       self.falling_block.right > support_block.right)

        if is_overhang:
            return 1.0  # High reward for risky placement
        else:
            return 0.2  # Small reward for safe, centered placement

    def _check_win_condition(self):
        return len(self.stacked_blocks) - 1 >= self.TARGET_HEIGHT

    def _spawn_new_block(self):
        x_pos = self.np_random.integers(self.BLOCK_WIDTH, self.SCREEN_WIDTH - self.BLOCK_WIDTH * 2)
        self.falling_block = pygame.Rect(x_pos, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
        
        color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
        self.falling_block_color = self.BLOCK_COLORS[color_index]

    def _create_particles(self, pos, color):
        for _ in range(15):
            particle = {
                "pos": list(pos),
                "vel": [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -0.5)],
                "size": self.np_random.uniform(3, 7),
                "life": 30,
                "color": color,
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["size"] -= 0.2
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0 and p["size"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": len(self.stacked_blocks) - 1,
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.BLOCK_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.BLOCK_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw target line
        target_y = self.SCREEN_HEIGHT - (self.TARGET_HEIGHT + 1) * self.BLOCK_HEIGHT
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.SCREEN_WIDTH, target_y), 2)

        # Draw stacked blocks
        for block_rect, color in self.stacked_blocks:
            self._draw_block(block_rect, color)

        # Draw falling block
        if self.falling_block and not self.game_over:
            self._draw_block(self.falling_block, self.falling_block_color)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * (255 / 30))))
            s = pygame.Surface((int(p["size"]), int(p["size"])), pygame.SRCALPHA)
            r, g, b = p["color"]
            pygame.draw.rect(s, (r, g, b, alpha), s.get_rect())
            self.screen.blit(s, (int(p["pos"][0] - p["size"]/2), int(p["pos"][1] - p["size"]/2)))

    def _draw_block(self, rect, color):
        dark_color = tuple(max(0, c - 40) for c in color)
        # Main face
        pygame.draw.rect(self.screen, color, rect)
        # 3D effect
        pygame.draw.line(self.screen, dark_color, rect.bottomleft, rect.bottomright, 4)
        pygame.draw.line(self.screen, dark_color, rect.topright, rect.bottomright, 4)
        # Outline
        pygame.draw.rect(self.screen, (0,0,0), rect, 1)

    def _render_ui(self):
        height = len(self.stacked_blocks) - 1
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30)

        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        height_text = self.font_ui.render(f"HEIGHT: {height}/{self.TARGET_HEIGHT}", True, self.COLOR_TEXT)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}s", True, self.COLOR_TEXT)

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(height_text, (self.SCREEN_WIDTH // 2 - height_text.get_width() // 2, 10))
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            if self._check_win_condition():
                end_text_str = "YOU WIN!"
                color = self.COLOR_TARGET_LINE
            else:
                end_text_str = "GAME OVER"
                color = self.BLOCK_COLORS[0]
            
            end_text = self.font_game_over.render(end_text_str, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        # Test specific reward values
        assert self._calculate_placement_reward(pygame.Rect(0,0,self.SCREEN_WIDTH, 20)) == 0.1
        assert self._calculate_placement_reward(pygame.Rect(100, 200, 80, 20)) < 1.1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for human play and is not part of the environment validation.
    # It will not be run by the evaluation server.
    # To switch from headless mode to visual mode, comment out the following line:
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Block Stacker")
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    action = [0, 0, 0] # No-op
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            mov = 0 # No-op
            if keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4
                
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0 # Unused in this env
            
            action = [mov, space, shift]

            obs, reward, terminated, truncated, info = env.step(action)
        else:
            # Allow reset on key press after game over
            keys = pygame.key.get_pressed()
            if any(keys): # Reset on any key press
                obs, info = env.reset()
                terminated = False

        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()