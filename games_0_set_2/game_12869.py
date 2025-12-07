import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:19:54.599502
# Source Brief: brief_02869.md
# Brief Index: 2869
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player stacks falling blocks to build a tower.

    The player controls a horizontal catcher at the bottom of the screen. Blocks fall
    from the top, and the player must move the catcher to catch them, adding them to
    a growing stack. Oscillating platforms act as obstacles that can deflect the
    falling blocks.

    The goal is to complete 5 levels by building the tower to a height of 100 units
    in each level. Difficulty increases with each level, with more blocks falling
    simultaneously, at higher speeds, and with more aggressive platform movement.

    The environment prioritizes visual quality and "game feel", featuring smooth
    animations, a clean minimalist aesthetic, particle effects, and a dynamic
    camera that follows the action.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Stack falling blocks to build a tower, avoiding oscillating platforms that can knock them away."
    user_guide = "Use the ← and → arrow keys to move the catcher left and right to catch the falling blocks."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CATCHER_WIDTH, CATCHER_HEIGHT = 100, 15
    BLOCK_WIDTH, BLOCK_HEIGHT = 50, 20
    PLATFORM_WIDTH, PLATFORM_HEIGHT = 80, 10
    CATCHER_SPEED = 8.0
    BASE_Y = 380
    WIN_LEVEL_HEIGHT = 100
    MAX_LEVELS = 5
    MAX_STEPS = 5000

    # --- Colors ---
    COLOR_BG_TOP = (15, 23, 42)
    COLOR_BG_BOTTOM = (4, 12, 30)
    COLOR_CATCHER = (240, 240, 255)
    COLOR_CATCHER_GLOW = (100, 100, 255, 50)
    COLOR_PLATFORM = (100, 116, 139)
    COLOR_TEXT = (226, 232, 240)
    BLOCK_COLORS = [
        (250, 100, 100), (100, 250, 100), (100, 100, 250),
        (250, 250, 100), (100, 250, 250), (250, 100, 250)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

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
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # Game state variables are initialized in reset()
        self.catcher_x = 0
        self.stacked_blocks = []
        self.falling_blocks = []
        self.platforms = []
        self.particles = []
        self.stack_height = 0
        self.level = 1
        self.levels_completed = 0
        self.camera_y = 0.0
        self.time_elapsed = 0
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.levels_completed = 0
        self.stack_height = 0
        self.camera_y = 0.0
        self.time_elapsed = 0
        self.catcher_x = self.WIDTH / 2 - self.CATCHER_WIDTH / 2
        
        self.stacked_blocks = []
        self.falling_blocks = []
        self.platforms = []
        self.particles = []

        self._setup_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_elapsed += 1 / 30.0 # Assume 30 FPS for physics consistency

        reward = 0
        self._update_catcher(action)
        self._update_platforms()
        self._update_particles()
        
        block_reward = self._update_falling_blocks()
        reward += block_reward
        
        # If a block was missed, the game is over
        if self.game_over:
            self.score += reward
            return self._get_observation(), reward, True, False, self._get_info()

        # Check for level completion
        if self.stack_height >= self.WIN_LEVEL_HEIGHT:
            self.levels_completed += 1
            reward += 10.0  # +10 for completing a level
            
            if self.levels_completed >= self.MAX_LEVELS:
                self.game_over = True
                reward += 100.0 # +100 for winning the game
            else:
                self.level += 1
                self.stack_height = 0
                self.stacked_blocks.clear()
                self._setup_level()
                # sound: level_up.wav

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _setup_level(self):
        self.falling_blocks.clear()
        self.platforms.clear()

        # Difficulty scaling
        num_platforms = min(6, 2 + self.level)
        
        # Create platforms
        for i in range(num_platforms):
            platform_y = self.BASE_Y - 100 - i * 70
            self.platforms.append({
                "rect": pygame.Rect(0, platform_y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT),
                "center_x": self.np_random.uniform(self.PLATFORM_WIDTH / 2, self.WIDTH - self.PLATFORM_WIDTH / 2),
                "amplitude": self.np_random.uniform(50, self.WIDTH / 2 - self.PLATFORM_WIDTH / 2),
                "frequency": (0.5 + 0.1 * (self.level - 1)) * self.np_random.uniform(0.8, 1.2),
                "phase": self.np_random.uniform(0, 2 * math.pi)
            })
        
        # Ensure correct number of falling blocks
        self._ensure_falling_blocks()

    def _ensure_falling_blocks(self):
        num_to_spawn = self.level - len(self.falling_blocks)
        for _ in range(num_to_spawn):
            self._spawn_falling_block()

    def _spawn_falling_block(self):
        fall_speed = 1.5 + 0.2 * (self.level - 1)
        start_x = self.np_random.uniform(0, self.WIDTH - self.BLOCK_WIDTH)
        start_y = -self.BLOCK_HEIGHT - self.camera_y - self.np_random.uniform(20, 100)
        color = random.choice(self.BLOCK_COLORS)
        
        self.falling_blocks.append({
            "rect": pygame.Rect(start_x, start_y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            "color": color,
            "vy": fall_speed
        })

    def _update_catcher(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.catcher_x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            self.catcher_x += self.CATCHER_SPEED
        
        self.catcher_x = np.clip(self.catcher_x, 0, self.WIDTH - self.CATCHER_WIDTH)

    def _update_platforms(self):
        for p in self.platforms:
            p['rect'].x = p['center_x'] - p['amplitude'] * math.sin(self.time_elapsed * p['frequency'] + p['phase']) - p['rect'].width / 2
            p['rect'].x = np.clip(p['rect'].x, 0, self.WIDTH - p['rect'].width)

    def _update_falling_blocks(self):
        reward = 0
        blocks_to_remove = []

        for block in self.falling_blocks:
            block['rect'].y += block['vy']

            # Collision with platforms
            for p in self.platforms:
                if block['rect'].colliderect(p['rect']):
                    self.game_over = True
                    reward -= 1.0  # -1 for losing a block
                    # sound: block_hit_obstacle.wav
                    self._spawn_particles(block['rect'].centerx, block['rect'].centery, (150, 150, 150), 30)
                    return reward

            # Determine landing surface
            if not self.stacked_blocks:
                landing_surface = pygame.Rect(self.catcher_x, self.BASE_Y, self.CATCHER_WIDTH, self.CATCHER_HEIGHT)
            else:
                landing_surface = self.stacked_blocks[-1][0]

            # Check for landing
            if block['rect'].bottom >= landing_surface.top and block['rect'].colliderect(landing_surface):
                # Check for horizontal alignment
                overlap = max(0, min(block['rect'].right, landing_surface.right) - max(block['rect'].left, landing_surface.left))
                if overlap / self.BLOCK_WIDTH > 0.2: # Must have at least 20% overlap
                    new_block_rect = block['rect'].copy()
                    new_block_rect.bottom = landing_surface.top
                    
                    self.stacked_blocks.append((new_block_rect, block['color']))
                    self.stack_height += new_block_rect.height
                    reward += 0.1 # +0.1 for each caught block
                    blocks_to_remove.append(block)
                    # sound: block_land.wav
                    self._spawn_particles(new_block_rect.centerx, new_block_rect.top, block['color'])
                else: # Misaligned landing
                    self.game_over = True
                    reward -= 1.0
                    return reward

            # Check for miss (falling off screen)
            elif block['rect'].top > self.BASE_Y + self.CATCHER_HEIGHT:
                self.game_over = True
                reward -= 1.0
                return reward
        
        # Remove caught blocks and spawn new ones
        if blocks_to_remove:
            self.falling_blocks = [b for b in self.falling_blocks if b not in blocks_to_remove]
            self._ensure_falling_blocks()

        return reward
    
    def _spawn_particles(self, x, y, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.uniform(15, 30),
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        # Update camera smoothly
        target_camera_y = max(0, self.stack_height - (self.HEIGHT - 150))
        self.camera_y += (target_camera_y - self.camera_y) * 0.1

        self._render_background()
        self._render_game_objects()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_objects(self):
        cam_y = int(self.camera_y)

        # Catcher
        catcher_rect = pygame.Rect(int(self.catcher_x), self.BASE_Y - cam_y, self.CATCHER_WIDTH, self.CATCHER_HEIGHT)
        glow_rect = catcher_rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_CATCHER_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=5)

        # Stacked Blocks
        for block_rect, color in self.stacked_blocks:
            r = block_rect.move(0, -cam_y)
            pygame.draw.rect(self.screen, color, r, border_radius=3)
            # Add a slight dark edge for definition
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), r, width=1, border_radius=3)

        # Falling Blocks
        for block in self.falling_blocks:
            r = block['rect'].move(0, -cam_y)
            pygame.draw.rect(self.screen, block['color'], r, border_radius=3)

        # Platforms
        for p in self.platforms:
            r = p['rect'].move(0, -cam_y)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, r, border_radius=5)

    def _render_particles(self):
        cam_y = int(self.camera_y)
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30.0))))
            color_with_alpha = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1] - cam_y))
            size = int(p['size'])
            
            # Use a small surface for alpha blending
            part_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(part_surf, color_with_alpha, (0, 0, size, size))
            self.screen.blit(part_surf, (pos[0] - size//2, pos[1] - size//2))

    def _render_ui(self):
        # Height display
        height_text = self.font_large.render(f"Height: {self.stack_height:.0f}/{self.WIN_LEVEL_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        # Level display
        level_text = self.font_large.render(f"Level: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

        # Score display
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "stack_height": self.stack_height,
            "levels_completed": self.levels_completed,
        }

    def close(self):
        pygame.quit()

# Example usage to test the environment
if __name__ == '__main__':
    # The following code is for local testing and will not be executed by the grader.
    # It allows you to play the game with keyboard controls.
    
    # Un-comment the line below to run with a visible display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Tower Stacker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        keys = pygame.key.get_pressed()
        action[0] = 0 # No-op
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for playability

    print("Game Over!")
    print(f"Final Info: {info}")
    env.close()