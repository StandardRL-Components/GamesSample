
# Generated: 2025-08-27T23:30:11.416060
# Source Brief: brief_03486.md
# Brief Index: 3486

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling block. Press space to drop it quickly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks to reach the target height. "
        "Place blocks precisely for score bonuses, but be careful! "
        "Unsupported parts of blocks will be lost, and if a block falls off the screen, you lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # --- Visuals & Style ---
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_BIG = pygame.font.Font(None, 48)
        
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.COLOR_TARGET_LINE = (255, 255, 255)
        self.BLOCK_COLORS = [
            ((227, 68, 68), (181, 54, 54)),   # Red
            ((68, 135, 227), (54, 108, 181)), # Blue
            ((100, 227, 68), (80, 181, 54)),  # Green
            ((227, 216, 68), (181, 173, 54)), # Yellow
            ((180, 68, 227), (144, 54, 181))  # Purple
        ]

        # --- Game Mechanics ---
        self.MAX_STEPS = 5000
        self.TARGET_HEIGHT = 20
        self.INITIAL_FALL_SPEED = 1.0
        self.DIFFICULTY_INCREMENT = 0.05
        self.DROP_SPEED = 15.0
        self.PLAYER_SPEED = 6.0
        self.BLOCK_HEIGHT = 15
        self.BASE_WIDTH = 180
        self.BASE_HEIGHT = 20
        self.BLOCK_WIDTH_RANGE = (60, 140)
        self.PERFECT_PLACEMENT_TOLERANCE = 2

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stack_height = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.last_space_held = False
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.last_space_held = False
        self.particles = []

        # Create base platform
        base_rect = pygame.Rect(
            (self.WIDTH - self.BASE_WIDTH) / 2,
            self.HEIGHT - self.BASE_HEIGHT,
            self.BASE_WIDTH,
            self.BASE_HEIGHT
        )
        base_color = ((100, 100, 100), (80, 80, 80))
        self.stacked_blocks = [{'rect': base_rect, 'color': base_color}]
        self.stack_height = 1

        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.game_over = False
        
        self.steps += 1
        
        self._handle_input(action)
        reward += self._update_game_state()
        self._update_particles()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Max steps reached
            # No specific reward change for timeout
            pass

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        if self.falling_block['state'] == 'preview':
            if movement == 3:  # Left
                self.falling_block['rect'].x -= self.PLAYER_SPEED
            elif movement == 4:  # Right
                self.falling_block['rect'].x += self.PLAYER_SPEED
            
            # Clamp to screen bounds
            self.falling_block['rect'].x = max(0, min(self.WIDTH - self.falling_block['rect'].width, self.falling_block['rect'].x))

        # Detect space press (transition from not held to held)
        if space_held and not self.last_space_held and self.falling_block['state'] == 'preview':
            self.falling_block['state'] = 'dropping'
            # SFX: Whoosh sound
        
        self.last_space_held = space_held
    
    def _update_game_state(self):
        reward = 0
        if self.falling_block is None:
            return 0
        
        # Update falling block position
        speed = self.DROP_SPEED if self.falling_block['state'] == 'dropping' else self.fall_speed
        self.falling_block['rect'].y += speed
        
        # Check for termination conditions
        if self.falling_block['rect'].top > self.HEIGHT:
            self.game_over = True
            reward -= 10 # Lose penalty
            # SFX: Fail sound
            return reward
            
        if self.stack_height >= self.TARGET_HEIGHT:
            self.game_over = True
            reward += 100 # Win reward
            # SFX: Win fanfare
            return reward

        # Check for collision with stack
        support_surface_y = self.HEIGHT
        support_block = None
        for block in self.stacked_blocks:
            if self.falling_block['rect'].colliderect(block['rect']):
                # Find the highest point of collision
                if block['rect'].top < support_surface_y:
                    support_surface_y = block['rect'].top
                    support_block = block
        
        if self.falling_block['rect'].bottom >= support_surface_y:
            # Place the block
            self.falling_block['rect'].bottom = support_surface_y
            
            # --- Calculate Overhang and Placement ---
            if support_block:
                original_rect = self.falling_block['rect'].copy()
                clipped_rect = self.falling_block['rect'].clip(support_block['rect'])
                
                if clipped_rect.width <= 0: # Completely missed the support
                    # Continue falling, this logic is handled by the termination check
                    return reward

                # Block is placed, calculate rewards and effects
                reward += 0.1 # Base placement reward
                
                # Penalize for overhang
                if clipped_rect.width < original_rect.width:
                    reward -= 0.2
                
                # Reward for perfect placement
                center_diff = abs(clipped_rect.centerx - support_block['rect'].centerx)
                if center_diff <= self.PERFECT_PLACEMENT_TOLERANCE:
                    reward += 1.0
                    self._spawn_particles(self.falling_block['rect'].midbottom, self.falling_block['color'][0], 20, is_perfect=True)
                
                # Finalize block and add to stack
                self.falling_block['rect'] = clipped_rect
                self.stacked_blocks.append(self.falling_block)
                self.stack_height += 1
                self.score += reward
                
                self._spawn_particles(self.falling_block['rect'].midbottom, self.falling_block['color'][0], 10)
                # SFX: Block place thud
                self._spawn_new_block()

        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed = min(self.fall_speed + self.DIFFICULTY_INCREMENT, self.DROP_SPEED / 2)
            
        return reward

    def _spawn_new_block(self):
        width = self.np_random.integers(self.BLOCK_WIDTH_RANGE[0], self.BLOCK_WIDTH_RANGE[1] + 1)
        x = self.np_random.integers(0, self.WIDTH - width + 1)
        rect = pygame.Rect(x, -self.BLOCK_HEIGHT, width, self.BLOCK_HEIGHT)
        color_pair = random.choice(self.BLOCK_COLORS)
        
        self.falling_block = {
            'rect': rect,
            'color': color_pair,
            'state': 'preview' # 'preview' or 'dropping'
        }

    def _spawn_particles(self, pos, color, count, is_perfect=False):
        for _ in range(count):
            if is_perfect:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                size = self.np_random.integers(3, 6)
                life = self.np_random.integers(20, 30)
            else:
                vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)]
                size = self.np_random.integers(2, 5)
                life = self.np_random.integers(15, 25)
            
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': size
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw target height line
        target_y = self.HEIGHT - self.BASE_HEIGHT - (self.TARGET_HEIGHT - 1) * self.BLOCK_HEIGHT
        if target_y > 0:
            for x in range(0, self.WIDTH, 20):
                 pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + 10, target_y), 1)

        # Draw stacked blocks
        for block in self.stacked_blocks:
            self._draw_3d_rect(block['rect'], block['color'])

        # Draw falling block
        if self.falling_block:
            self._draw_3d_rect(self.falling_block['rect'], self.falling_block['color'])
            # Draw a faint drop shadow
            shadow_pos = self.falling_block['rect'].copy()
            shadow_pos.bottom = self.HEIGHT
            for block in self.stacked_blocks:
                if shadow_pos.colliderect(block['rect']):
                    shadow_pos.bottom = min(shadow_pos.bottom, block['rect'].top)
            pygame.draw.rect(self.screen, (0,0,0,50), shadow_pos, border_radius=2)


        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            r = pygame.Rect(int(p['pos'][0]), int(p['pos'][1]), p['size'], p['size'])
            shape_surf = pygame.Surface(r.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            self.screen.blit(shape_surf, r)

    def _draw_3d_rect(self, rect, color_pair):
        main_color, shadow_color = color_pair
        pygame.draw.rect(self.screen, main_color, rect)
        
        # Draw shadow lines for 3D effect
        p = rect.bottomleft
        points = [
            (p[0], p[1]-1),
            (p[0] + rect.width, p[1]-1),
            (p[0] + rect.width, p[1] - rect.height),
        ]
        pygame.draw.line(self.screen, shadow_color, (rect.left, rect.bottom-1), (rect.right-1, rect.bottom-1), 2)
        pygame.draw.line(self.screen, shadow_color, (rect.right-1, rect.top), (rect.right-1, rect.bottom-1), 2)
        
        # Antialiased outline for crispness
        pygame.gfxdraw.rectangle(self.screen, rect, (0,0,0,100))

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 1, pos[1] + 1))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        draw_text(score_text, self.FONT_UI, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Height
        height_text = f"HEIGHT: {self.stack_height} / {self.TARGET_HEIGHT}"
        text_width = self.FONT_UI.size(height_text)[0]
        draw_text(height_text, self.FONT_UI, self.COLOR_TEXT, (self.WIDTH - text_width - 10, 10), self.COLOR_TEXT_SHADOW)
        
        # Steps
        steps_text = f"STEPS: {self.steps} / {self.MAX_STEPS}"
        text_width = self.FONT_UI.size(steps_text)[0]
        draw_text(steps_text, self.FONT_UI, self.COLOR_TEXT, ((self.WIDTH - text_width) / 2, self.HEIGHT - 30), self.COLOR_TEXT_SHADOW)

        # Game Over / Win message
        if self.game_over:
            if self.stack_height >= self.TARGET_HEIGHT:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            text_width, text_height = self.FONT_BIG.size(msg)
            draw_text(msg, self.FONT_BIG, color, ((self.WIDTH - text_width) / 2, (self.HEIGHT - text_height) / 2), self.COLOR_TEXT_SHADOW)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.stack_height,
        }

    def close(self):
        pygame.font.quit()
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
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)

    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Info={info}, Terminated={terminated}")
        if terminated:
            print("Episode finished.")
            break
    
    # Example of manual control for visualization
    # To run this, comment out the `os.environ` line above
    # and install pygame if you haven't already.
    
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Block Stacker")
    # clock = pygame.time.Clock()
    # running = True
    # total_reward = 0

    # while running:
    #     movement = 0 # No-op
    #     space = 0 # Released
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         movement = 3
    #     elif keys[pygame.K_RIGHT]:
    #         movement = 4
        
    #     if keys[pygame.K_SPACE]:
    #         space = 1

    #     action = np.array([movement, space, 0]) # shift is unused
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward

    #     # Convert obs back to a Pygame surface to display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    #         pygame.time.wait(2000)
    #         obs, info = env.reset()
    #         total_reward = 0

    #     clock.tick(30) # Limit to 30 FPS

    env.close()