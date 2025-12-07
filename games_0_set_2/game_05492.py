import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the block. Hold space for a fast drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as you can. Place blocks perfectly for a score bonus. The game ends if a block falls off the stack."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Visuals & Fonts
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (30, 45, 60)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.COLOR_BASE = (100, 120, 140)

        self.BLOCK_TYPES = [
            {'width': 60, 'color': (227, 85, 85)},   # Red
            {'width': 80, 'color': (85, 227, 159)},  # Green
            {'width': 100, 'color': (85, 159, 227)}, # Blue
            {'width': 70, 'color': (227, 227, 85)},  # Yellow
            {'width': 90, 'color': (159, 85, 227)},  # Purple
        ]
        
        # Game Parameters
        self.BLOCK_HEIGHT = 20
        self.FALL_SPEED_NORMAL = 3
        self.FALL_SPEED_FAST = 20
        self.MOVE_SPEED = 10
        self.MAX_STEPS = 1000
        self.WIN_HEIGHT = 25
        self.PERFECT_PLACEMENT_THRESHOLD = 2

        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.base_platform = None
        self.np_random = None
        
        # Initialize state variables
        # self.reset() is called by gym.make, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.particles = []
        
        base_width = self.WIDTH * 0.8
        self.base_platform = pygame.Rect(
            (self.WIDTH - base_width) / 2, 
            self.HEIGHT - self.BLOCK_HEIGHT, 
            base_width, 
            self.BLOCK_HEIGHT
        )
        self.stacked_blocks = [(self.base_platform, self.COLOR_BASE)]
        
        self._spawn_new_block()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _spawn_new_block(self):
        block_type = self.np_random.choice(self.BLOCK_TYPES)
        width = block_type['width']
        color = block_type['color']
        
        spawn_x_range = (self.WIDTH * 0.25, self.WIDTH * 0.75 - width)
        x = self.np_random.uniform(spawn_x_range[0], spawn_x_range[1])
        
        self.falling_block = {
            'rect': pygame.Rect(x, -self.BLOCK_HEIGHT, width, self.BLOCK_HEIGHT),
            'color': color,
        }

    def step(self, action):
        reward = 0.0
        
        if self.game_over or self.win:
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # --- 1. Handle Input ---
        if movement == 3:  # Left
            self.falling_block['rect'].x -= self.MOVE_SPEED
        elif movement == 4:  # Right
            self.falling_block['rect'].x += self.MOVE_SPEED
        
        self.falling_block['rect'].x = max(0, self.falling_block['rect'].x)
        self.falling_block['rect'].x = min(self.WIDTH - self.falling_block['rect'].width, self.falling_block['rect'].x)

        # --- 2. Update Game Logic ---
        self.steps += 1
        fall_speed = self.FALL_SPEED_FAST if space_held else self.FALL_SPEED_NORMAL
        self.falling_block['rect'].y += fall_speed
        self._update_particles()

        # --- 3. Collision and Landing Logic ---
        landed_on = None
        for block_rect, _ in self.stacked_blocks:
            if self.falling_block['rect'].colliderect(block_rect) and self.falling_block['rect'].bottom > block_rect.top:
                if landed_on is None or block_rect.top > landed_on.top: # Land on the highest block
                    landed_on = block_rect
        
        if landed_on:
            self.falling_block['rect'].bottom = landed_on.top
            
            # Check if landing is valid (fully supported)
            is_supported = (self.falling_block['rect'].left >= landed_on.left and
                            self.falling_block['rect'].right <= landed_on.right)

            if is_supported:
                # --- Successful Placement ---
                self.stacked_blocks.append((self.falling_block['rect'].copy(), self.falling_block['color']))
                self._create_particles(self.falling_block['rect'].midbottom, self.falling_block['color'], 20, 3)

                # Calculate reward
                placement_reward = 0.1  # Base reward for successful placement
                center_diff = abs(self.falling_block['rect'].centerx - landed_on.centerx)
                if center_diff <= self.PERFECT_PLACEMENT_THRESHOLD:
                    placement_reward += 1.0  # Perfectly centered bonus
                    self._create_particles(self.falling_block['rect'].midbottom, (255, 255, 100), 30, 5)
                else:
                    placement_reward += -0.2  # Off-center penalty
                
                reward += placement_reward
                self.score += placement_reward

                # Check for win condition
                if len(self.stacked_blocks) - 1 >= self.WIN_HEIGHT:
                    self.win = True
                    reward += 100
                else:
                    self._spawn_new_block()
            else:
                # --- Failed Placement (fell off) ---
                self.game_over = True
                reward += -10
                self._create_particles(self.falling_block['rect'].center, (255, 50, 50), 50, 1)
        
        # Check for falling off the bottom of the screen
        if not self.game_over and self.falling_block['rect'].top > self.HEIGHT:
            self.game_over = True
            reward += -10

        # --- 4. Check Termination ---
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS and not (self.game_over or self.win)
        
        if truncated:
            terminated = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_game(self):
        # Draw stacked blocks
        for rect, color in self.stacked_blocks:
            self._draw_block(rect, color)
        
        # Draw falling block
        if not self.game_over and not self.win and self.falling_block:
             self._draw_block(self.falling_block['rect'], self.falling_block['color'])
        
        # Draw particles
        for p in self.particles:
            if p['alpha'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                radius = int(p['radius'])
                # Use standard pygame.draw.circle instead of gfxdraw
                # Manually blend color with background to simulate alpha transparency
                alpha_ratio = p['alpha'] / 255.0
                blended_color = tuple(
                    int(c_p * alpha_ratio + c_bg * (1 - alpha_ratio))
                    for c_p, c_bg in zip(p['color'], self.COLOR_BG)
                )
                pygame.draw.circle(self.screen, blended_color, pos, radius)


    def _draw_block(self, rect, color):
        shadow_color = tuple(max(0, c - 50) for c in color)
        highlight_color = tuple(min(255, c + 50) for c in color)
        
        # Main body
        pygame.draw.rect(self.screen, color, rect)
        
        # 3D effect
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(self.screen, shadow_color, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, shadow_color, rect.topright, rect.bottomright, 2)

    def _render_ui(self):
        # Score
        self._draw_text(f"Score: {self.score:.1f}", (10, 10), self.font_medium)
        
        # Height
        height = len(self.stacked_blocks) - 1
        height_text = f"Height: {height}/{self.WIN_HEIGHT}"
        text_width = self.font_medium.size(height_text)[0]
        self._draw_text(height_text, (self.WIDTH - text_width - 10, 10), self.font_medium)
        
        # Game Over / Win message
        if self.game_over:
            self._draw_text("GAME OVER", (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_large, center=True)
        elif self.win:
            self._draw_text("YOU WIN!", (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=None, shadow_color=None, center=False):
        if color is None: color = self.COLOR_TEXT
        if shadow_color is None: shadow_color = self.COLOR_TEXT_SHADOW

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, position, color, count, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': list(position),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'life': self.np_random.integers(20, 40),
                'alpha': 255
            })

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            p['alpha'] = max(0, 255 * (p['life'] / 40.0))
            if p['life'] <= 0:
                particles_to_remove.append(p)
        
        for p in particles_to_remove:
            self.particles.remove(p)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": len(self.stacked_blocks) - 1,
            "win": self.win,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: This will create a window and not run headless.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display for interactive mode
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Height: {info['height']}")
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Final Height: {info['height']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            
        env.clock.tick(30) # Control the frame rate

    env.close()