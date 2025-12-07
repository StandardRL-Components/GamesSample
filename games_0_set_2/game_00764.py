
# Generated: 2025-08-27T14:43:03.007637
# Source Brief: brief_00764.md
# Brief Index: 764

        
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
        "Controls: ←→ to move the block, and press space to drop it quickly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks to build a tower. Reach the target height to win, "
        "but don't let any blocks fall off the platform!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.BLOCK_SIZE = 20
        self.MOVE_SPEED = 10  # pixels per action
        self.MAX_STEPS = 2000

        self.TARGET_HEIGHT_Y = 40  # Win if top of stack reaches this y-coord
        
        self.INITIAL_FALL_SPEED = 1.0
        self.FAST_FALL_SPEED = 20.0
        self.DIFFICULTY_INTERVAL = 20 # Increase speed every 20 placements
        self.SPEED_INCREMENT = 0.2

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TARGET_LINE = (255, 255, 255)
        self.COLOR_FLOOR = (80, 80, 90)
        self.BLOCK_COLORS = [
            (231, 76, 60),   # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (241, 196, 15),  # Yellow
            (155, 89, 182),  # Purple
        ]

        # --- Block Shapes (offsets from origin in block units) ---
        self.BLOCK_SHAPES = {
            'I': [(0, -1), (0, 0), (0, 1), (0, 2)],
            'O': [(0, 0), (1, 0), (0, 1), (1, 1)],
            'T': [(-1, 0), (0, 0), (1, 0), (0, 1)],
            'L': [(-1, -1), (-1, 0), (-1, 1), (0, 1)],
            'S': [(-1, 1), (0, 1), (0, 0), (1, 0)],
        }
        self.SHAPE_KEYS = list(self.BLOCK_SHAPES.keys())

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Etc...        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.placements = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.stacked_blocks = []
        self.falling_block = None
        self.rng = None
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.placements = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        
        self.stacked_blocks = []
        self._create_floor()
        self._spawn_new_block()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _create_floor(self):
        floor_width_blocks = self.WIDTH // self.BLOCK_SIZE
        floor_shape = [(i - floor_width_blocks // 2, 0) for i in range(floor_width_blocks)]
        self.stacked_blocks.append({
            'x': self.WIDTH // 2,
            'y': self.HEIGHT - self.BLOCK_SIZE,
            'shape': floor_shape,
            'color': self.COLOR_FLOOR,
        })

    def _spawn_new_block(self):
        shape_key = self.rng.choice(self.SHAPE_KEYS)
        color = self.BLOCK_COLORS[self.rng.integers(0, len(self.BLOCK_COLORS))]
        self.falling_block = {
            'x': self.WIDTH // 2,
            'y': self.BLOCK_SIZE * 2,
            'shape_key': shape_key,
            'shape': self.BLOCK_SHAPES[shape_key],
            'color': color,
        }

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update game logic
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        is_fast_dropping = space_held

        move_x = 0
        if movement == 3:  # Left
            move_x = -self.MOVE_SPEED
        elif movement == 4:  # Right
            move_x = self.MOVE_SPEED

        self.falling_block['x'] += move_x
        self._clamp_falling_block_in_bounds()

        fall_distance = self.FAST_FALL_SPEED if is_fast_dropping else self.fall_speed
        self.falling_block['y'] += fall_distance
        
        reward += 0.01 # Small reward for staying in the game

        collision_surface_y = self._check_collision()

        if collision_surface_y is not None:
            placement_reward = self._place_block(collision_surface_y)
            reward += placement_reward

        terminated = self._check_termination()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self):
        return self.game_over or (self.steps >= self.MAX_STEPS)

    def _get_block_pixel_coords(self, block):
        coords = []
        for ox, oy in block['shape']:
            px = block['x'] + ox * self.BLOCK_SIZE
            py = block['y'] + oy * self.BLOCK_SIZE
            coords.append((px, py))
        return coords
    
    def _get_block_rects(self, block):
        rects = []
        for px, py in self._get_block_pixel_coords(block):
             rects.append(pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE))
        return rects

    def _clamp_falling_block_in_bounds(self):
        min_x, max_x = self.WIDTH, 0
        coords = self._get_block_pixel_coords(self.falling_block)
        if not coords: return
        
        for px, _ in coords:
            min_x = min(min_x, px)
            max_x = max(max_x, px)
        
        if min_x < 0:
            self.falling_block['x'] -= min_x
        if max_x + self.BLOCK_SIZE > self.WIDTH:
            self.falling_block['x'] -= (max_x + self.BLOCK_SIZE - self.WIDTH)

    def _check_collision(self):
        falling_rects = self._get_block_rects(self.falling_block)
        for stacked_block in self.stacked_blocks:
            stacked_rects = self._get_block_rects(stacked_block)
            for f_rect in falling_rects:
                collided_rect_idx = f_rect.collidelist(stacked_rects)
                if collided_rect_idx != -1:
                    return stacked_rects[collided_rect_idx].top
        return None

    def _place_block(self, landing_y):
        max_y_offset = max(oy for _, oy in self.falling_block['shape'])
        self.falling_block['y'] = landing_y - max_y_offset * self.BLOCK_SIZE - self.BLOCK_SIZE
        
        coords = self._get_block_pixel_coords(self.falling_block)
        min_x_abs = min(px for px, _ in coords)
        max_x_abs = max(px + self.BLOCK_SIZE for px, _ in coords)
        max_y_abs = max(py + self.BLOCK_SIZE for _, py in coords)

        if min_x_abs < 0 or max_x_abs > self.WIDTH or max_y_abs > self.HEIGHT:
            self.game_over = True
            # Sound: Negative buzzer
            return -100

        placement_reward = self._calculate_placement_reward()
        self.score += placement_reward
        
        self.stacked_blocks.append(self.falling_block)
        # Sound: Block place click
        
        self.placements += 1
        if self.placements > 0 and self.placements % self.DIFFICULTY_INTERVAL == 0:
            self.fall_speed = min(self.fall_speed + self.SPEED_INCREMENT, self.FAST_FALL_SPEED / 2)

        stack_top_y = min(r.top for b in self.stacked_blocks for r in self._get_block_rects(b))
        if stack_top_y <= self.TARGET_HEIGHT_Y:
            self.game_over = True
            # Sound: Victory fanfare
            self.score += 100
            return 100
            
        self._spawn_new_block()
        return placement_reward

    def _calculate_placement_reward(self):
        new_block_rects = self._get_block_rects(self.falling_block)
        
        support_rects = []
        for s_block in self.stacked_blocks:
            for s_rect in self._get_block_rects(s_block):
                for n_rect in new_block_rects:
                    if n_rect.left == s_rect.left and n_rect.top == s_rect.bottom:
                        support_rects.append(s_rect)
        
        if not support_rects:
            return 1.0

        new_min_x = min(r.left for r in new_block_rects)
        new_max_x = max(r.right for r in new_block_rects)
        support_min_x = min(r.left for r in support_rects)
        support_max_x = max(r.right for r in support_rects)

        if new_min_x < support_min_x or new_max_x > support_max_x:
            # Sound: Scrape sfx
            return -2.0
        else:
            # Sound: Positive chime
            return 5.0

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
        for x in range(0, self.WIDTH, self.BLOCK_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.BLOCK_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, self.TARGET_HEIGHT_Y), (self.WIDTH, self.TARGET_HEIGHT_Y), 2)

        for block in self.stacked_blocks:
            for r in self._get_block_rects(block):
                pygame.draw.rect(self.screen, block['color'], r)
                pygame.draw.rect(self.screen, self.COLOR_BG, r, 1)

        if self.falling_block and not self.game_over:
            glow_color = tuple(min(255, c + 50) for c in self.falling_block['color'])
            for r in self._get_block_rects(self.falling_block):
                glow_surface = pygame.Surface((self.BLOCK_SIZE * 2, self.BLOCK_SIZE * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*glow_color, 30), (self.BLOCK_SIZE, self.BLOCK_SIZE), self.BLOCK_SIZE)
                pygame.draw.circle(glow_surface, (*glow_color, 20), (self.BLOCK_SIZE, self.BLOCK_SIZE), self.BLOCK_SIZE * 0.8)
                self.screen.blit(glow_surface, (r.centerx - self.BLOCK_SIZE, r.centery - self.BLOCK_SIZE), special_flags=pygame.BLEND_RGBA_ADD)
                
                pygame.draw.rect(self.screen, self.falling_block['color'], r)
                pygame.draw.rect(self.screen, tuple(max(0, c - 40) for c in self.falling_block['color']), r, 2)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if len(self.stacked_blocks) > 1:
            stack_top_y = min(r.top for b in self.stacked_blocks[1:] for r in self._get_block_rects(b))
            height_blocks = (self.HEIGHT - stack_top_y) / self.BLOCK_SIZE
            height_text = self.font_small.render(f"Height: {height_blocks:.1f}", True, self.COLOR_TEXT)
            self.screen.blit(height_text, (self.WIDTH - height_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            win = any(r.top <= self.TARGET_HEIGHT_Y for b in self.stacked_blocks for r in self._get_block_rects(b))
            end_text_str = "YOU WIN!" if win else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "placements": self.placements,
        }
        
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv()
    
    obs, info = env.reset()
    
    running = True
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Stacker")

    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(30)

    env.close()