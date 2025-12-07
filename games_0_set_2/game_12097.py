import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:05:21.009374
# Source Brief: brief_02097.md
# Brief Index: 2097
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent stacks falling recursive color blocks.
    The goal is to build a tower of 10 blocks of the same color.
    The game ends if a mismatched color is placed, a block falls unsupported,
    the stack gets too high, or the max steps are reached.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Stack falling blocks of the same color to build a tower. Place 10 matching blocks to win, but avoid unstable or mismatched stacks."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move the block and press space to drop it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.PLAY_AREA_WIDTH = 240
        self.PLAY_AREA_LEFT = (self.SCREEN_WIDTH - self.PLAY_AREA_WIDTH) // 2
        self.PLAY_AREA_RIGHT = self.PLAY_AREA_LEFT + self.PLAY_AREA_WIDTH
        
        self.BLOCK_WIDTH = 50
        self.BLOCK_HEIGHT = 20
        self.BLOCK_MOVE_SPEED = 10
        self.INITIAL_FALL_SPEED = 1.0  # pixels per step
        self.FALL_SPEED_INCREMENT = 0.1
        self.DIFFICULTY_INTERVAL = 200 # steps to increase difficulty

        self.MAX_STEPS = 1000
        self.WIN_SCORE = 10
        self.TOP_BOUNDARY = 50

        # Colors
        self.COLOR_BG_TOP = (15, 20, 35)
        self.COLOR_BG_BOTTOM = (30, 40, 65)
        self.COLOR_WALL = (60, 70, 95)
        self.COLOR_TEXT = (230, 230, 240)
        self.BLOCK_COLORS = {
            "red": (255, 80, 80),
            "green": (80, 255, 80),
            "blue": (80, 120, 255),
            "yellow": (255, 230, 80),
            "purple": (200, 80, 255),
        }
        self.COLOR_NAMES = list(self.BLOCK_COLORS.keys())
        self.RECURSIVE_OUTLINE_COLOR = (255, 255, 255)

        # --- Gymnasium Spaces ---
        # Observation space is the screen pixel data
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        # Action space: [movement, drop, unused]
        # movement: 0-2 no-op, 3 left, 4 right
        # drop: 0 no-op, 1 drop
        # unused: 0 no-op, 1 unused
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.stacked_blocks = None
        self.falling_block = None
        self.particles = None
        self.fall_speed = None
        self.stack_base_color = None
        self.last_reward = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stacked_blocks = []
        self.particles = []
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.stack_base_color = None
        self.last_reward = 0.0

        self._spawn_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.last_reward = 0.0
        
        self._handle_input(action)
        
        drop_triggered = (action[1] == 1)

        if drop_triggered:
            # SFX: Block drop whoosh
            self._process_landing(fast_drop=True)
        else:
            self.falling_block['rect'].y += self.fall_speed
            if self._check_landing():
                self._process_landing(fast_drop=False)

        self.steps += 1
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.fall_speed += self.FALL_SPEED_INCREMENT

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        terminated = self.game_over

        return (
            self._get_observation(),
            self.last_reward,
            terminated,
            False, # Truncated is handled by the wrapper
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.falling_block['rect'].x -= self.BLOCK_MOVE_SPEED
        elif movement == 4:  # Right
            self.falling_block['rect'].x += self.BLOCK_MOVE_SPEED

        # Clamp position to play area
        self.falling_block['rect'].x = max(
            self.PLAY_AREA_LEFT, self.falling_block['rect'].x
        )
        self.falling_block['rect'].x = min(
            self.PLAY_AREA_RIGHT - self.BLOCK_WIDTH, self.falling_block['rect'].x
        )

    def _spawn_block(self):
        color_name = self.np_random.choice(self.COLOR_NAMES)
        is_recursive = self.np_random.random() < 0.20
        start_x = self.np_random.integers(
            self.PLAY_AREA_LEFT, self.PLAY_AREA_RIGHT - self.BLOCK_WIDTH + 1
        )
        self.falling_block = {
            'rect': pygame.Rect(start_x, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            'color_name': color_name,
            'color': self.BLOCK_COLORS[color_name],
            'is_recursive': is_recursive,
            'inner_block_scale': 0.6 if is_recursive else 0.0
        }

    def _check_landing(self):
        highest_y = self.SCREEN_HEIGHT
        for block in self.stacked_blocks:
            if self.falling_block['rect'].colliderect(block['rect']):
                highest_y = min(highest_y, block['rect'].top)
        
        if self.falling_block['rect'].bottom >= highest_y:
            self.falling_block['rect'].bottom = highest_y
            return True
        return False

    def _process_landing(self, fast_drop):
        # Determine landing surface
        landing_y = self.SCREEN_HEIGHT
        support_blocks = []
        for block in self.stacked_blocks:
            if (self.falling_block['rect'].left < block['rect'].right and
                self.falling_block['rect'].right > block['rect'].left):
                landing_y = min(landing_y, block['rect'].top)
        
        if fast_drop:
            self.falling_block['rect'].bottom = landing_y

        for block in self.stacked_blocks:
            if (block['rect'].top == landing_y and
                self.falling_block['rect'].left < block['rect'].right and
                self.falling_block['rect'].right > block['rect'].left):
                support_blocks.append(block)

        # 1. Check for stability (fell into a gap)
        if not support_blocks and landing_y != self.SCREEN_HEIGHT:
            # SFX: Block fall and break sound
            self.last_reward = -100.0
            self.game_over = True
            return

        # 2. Check stack height limit
        if self.falling_block['rect'].top < self.TOP_BOUNDARY:
            # SFX: Failure buzz
            self.last_reward = -100.0
            self.game_over = True
            return

        # 3. Check color match
        current_color = self.falling_block['color_name']
        if not self.stack_base_color:
            self.stack_base_color = current_color
        
        if current_color != self.stack_base_color:
            # SFX: Mismatch error sound
            self.last_reward = -100.0
            self.game_over = True
            return

        # --- Successful Placement ---
        # SFX: Block placement click
        self.last_reward += 1.0
        self._create_particles(
            self.falling_block['rect'].midbottom, self.falling_block['color']
        )
        self.stacked_blocks.append(self.falling_block)

        # Handle recursion
        if self.falling_block['is_recursive']:
            inner_block_width = self.BLOCK_WIDTH * self.falling_block['inner_block_scale']
            inner_block_rect = pygame.Rect(
                self.falling_block['rect'].centerx - inner_block_width / 2,
                self.falling_block['rect'].top - self.BLOCK_HEIGHT,
                inner_block_width,
                self.BLOCK_HEIGHT
            )
            inner_block = {
                'rect': inner_block_rect,
                'color_name': self.falling_block['color_name'],
                'color': self.falling_block['color'],
                'is_recursive': False,
                'inner_block_scale': 0.0
            }
            # Check if inner block exceeds height limit
            if inner_block_rect.top < self.TOP_BOUNDARY:
                self.last_reward = -100.0
                self.game_over = True
                return
            
            # SFX: Secondary pop sound
            self.last_reward += 1.0
            self.stacked_blocks.append(inner_block)
            self._create_particles(inner_block_rect.midbottom, inner_block['color'])
        
        self.score = len([b for b in self.stacked_blocks if b['color_name'] == self.stack_base_color])

        # 4. Check for win condition
        if self.score >= self.WIN_SCORE:
            # SFX: Victory fanfare
            self.last_reward += 100.0
            self.game_over = True
            return
            
        self._spawn_block()

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame and numpy have swapped axes for width/height
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.PLAY_AREA_LEFT - 5, 0, 5, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.PLAY_AREA_RIGHT, 0, 5, self.SCREEN_HEIGHT))

        # Draw stacked blocks
        for block in self.stacked_blocks:
            self._draw_block(block)

        # Draw falling block
        if self.falling_block and not self.game_over:
            self._draw_block(self.falling_block, is_falling=True)
        
        # Draw particles
        self._update_and_draw_particles()

    def _draw_block(self, block, is_falling=False):
        rect = block['rect']
        color = block['color']
        
        # Glow effect for falling block
        if is_falling:
            glow_rect = rect.inflate(10, 10)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            glow_color = color + (50,)
            pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, glow_rect.topleft)

        # Main block
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        
        # Recursive outline
        if block['is_recursive']:
            inner_rect = rect.inflate(-8, -8)
            pygame.draw.rect(self.screen, self.RECURSIVE_OUTLINE_COLOR, inner_rect, width=2, border_radius=2)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        text_rect = score_text.get_rect(center=(self.SCREEN_WIDTH // 2, 25))
        self.screen.blit(score_text, text_rect)
        
        if self.game_over:
            status_text = "YOU WON!" if self.score >= self.WIN_SCORE else "GAME OVER"
            status_render = self.font.render(status_text, True, self.COLOR_TEXT)
            status_rect = status_render.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(status_render, status_rect)

    def _create_particles(self, pos, color):
        for _ in range(20):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-3, -1)],
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
            p['radius'] -= 0.05

            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
                color_with_alpha = p['color'] + (alpha,)
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                
                # Using gfxdraw for antialiasing
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color_with_alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color_with_alpha)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the environment directly and play manually.
    # This is not part of the required Gymnasium interface but is useful for testing.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use a visible driver
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display surface for manual play
    screen_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Recursive Stacker")

    while not done:
        # Action mapping for manual control
        # [movement, space, shift]
        # movement: 0-2 no-op, 3 left, 4 right
        action = [0, 0, 0] 
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_SPACE]:
            action[1] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS for smooth viewing

    env.close()