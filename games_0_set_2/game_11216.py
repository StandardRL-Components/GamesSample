import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:38:04.216316
# Source Brief: brief_01216.md
# Brief Index: 1216
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Drop colored blocks to form groups of three or more. Manage tall stacks, which split and rearrange, to clear the board and score points before time runs out."
    )
    user_guide = (
        "Controls: ←→ to move the falling block. Press space to drop the block instantly."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BLOCK_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // BLOCK_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // BLOCK_SIZE

    # Gameplay Constants
    MAX_TIME_STEPS = 120 * 60  # 120 seconds at 60 FPS
    WIN_SCORE = 1000
    STACK_SPLIT_HEIGHT = 5
    MATCH_THRESHOLD = 3
    INITIAL_FALL_SPEED = 2.0
    FALL_SPEED_INCREASE_INTERVAL = 600 # Every 10 seconds
    FALL_SPEED_INCREASE_AMOUNT = 0.5

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_RED = (255, 50, 50)
    COLOR_GREEN = (50, 255, 50)
    COLOR_BLUE = (50, 100, 255)
    BLOCK_COLORS = [COLOR_RED, COLOR_GREEN, COLOR_BLUE]
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (40, 50, 70, 180) # RGBA for transparency

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # State variables are initialized in reset()
        self.grid = []
        self.falling_block = None
        self.next_block_color = None
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.fall_speed = 0
        self.game_over = False
        self.space_was_held = False
        self.particles = []
        self.flash_effects = []
        
        # Using numpy's default_rng for modern random number generation
        self.np_random = np.random.default_rng()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        
        self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME_STEPS
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.space_was_held = False
        self.particles = []
        self.flash_effects = []
        self.game_over_message = ""

        self.next_block_color = self.BLOCK_COLORS[self.np_random.integers(len(self.BLOCK_COLORS))]
        self._create_new_falling_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.FALL_SPEED_INCREASE_INTERVAL == 0:
            self.fall_speed += self.FALL_SPEED_INCREASE_AMOUNT

        # Handle input
        self._handle_input(movement, space_held)
        
        # Update falling block
        self.falling_block['y'] += self.fall_speed
        
        # Check for collision and placement
        reward = 0
        if self._check_collision():
            placed_pos = self._place_block()
            if placed_pos[1] < 0: # Placed block out of bounds (top of screen)
                self.game_over = True
                self.game_over_message = "GAME OVER"
                reward -= 100
            else:
                reward += 0.1 # Reward for placing a block
                
                # --- Main Game Rule Loop ---
                # Repeatedly apply rules until the board is stable
                is_stable = False
                while not is_stable:
                    gravity_moved, _ = self._apply_gravity()
                    split_occurred, _ = self._check_stack_splits()
                    match_reward, match_found = self._check_matches()
                    
                    reward += match_reward
                    is_stable = not (gravity_moved or split_occurred or match_found)

                self._create_new_falling_block()
            
        self.space_was_held = space_held

        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over_message = "VICTORY!"
        elif self.time_remaining <= 0:
            reward -= 100
            terminated = True
            self.game_over_message = "TIME UP"
        
        if self.game_over:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement: 3=left, 4=right
        if movement == 3:
            self.falling_block['x'] -= self.BLOCK_SIZE
        elif movement == 4:
            self.falling_block['x'] += self.BLOCK_SIZE
        
        # Wrap around screen
        if self.falling_block['x'] < 0:
            self.falling_block['x'] = self.SCREEN_WIDTH - self.BLOCK_SIZE
        if self.falling_block['x'] >= self.SCREEN_WIDTH:
            self.falling_block['x'] = 0

        # Drop block on space press (rising edge)
        if space_held and not self.space_was_held:
            while not self._check_collision():
                self.falling_block['y'] += 1

    def _check_collision(self):
        gx = int(self.falling_block['x'] / self.BLOCK_SIZE)
        gy = int((self.falling_block['y'] + self.BLOCK_SIZE) / self.BLOCK_SIZE)

        if gy >= self.GRID_HEIGHT:
            return True
        if gy >= 0 and self.grid[gy][gx] is not None:
            return True
        return False

    def _place_block(self):
        gx = int(self.falling_block['x'] / self.BLOCK_SIZE)
        gy = int(self.falling_block['y'] / self.BLOCK_SIZE)
        
        gy = min(gy, self.GRID_HEIGHT - 1)
        
        if gy < 0: # Trying to place above screen
            return gx, gy
        
        if self.grid[gy][gx] is not None:
             # Column is full, find first empty spot from top
            for i in range(self.GRID_HEIGHT):
                if self.grid[i][gx] is None:
                    gy = i - 1
                    break
            else: # No empty spot found, stack is full
                gy = -1 # Signal game over
        
        if gy >= 0:
            self.grid[gy][gx] = self.falling_block['color']
            self._create_particles(gx * self.BLOCK_SIZE + self.BLOCK_SIZE/2, gy * self.BLOCK_SIZE + self.BLOCK_SIZE/2, self.falling_block['color'], 5)
        
        return gx, gy

    def _create_new_falling_block(self):
        start_x = self.np_random.integers(self.GRID_WIDTH)
        self.falling_block = {
            'x': start_x * self.BLOCK_SIZE,
            'y': 0,
            'color': self.next_block_color
        }
        self.next_block_color = self.BLOCK_COLORS[self.np_random.integers(len(self.BLOCK_COLORS))]

    def _apply_gravity(self):
        moved = False
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2, -1, -1):
                if self.grid[y][x] is not None and self.grid[y+1][x] is None:
                    fall_dist = 0
                    while y + 1 + fall_dist < self.GRID_HEIGHT and self.grid[y + 1 + fall_dist][x] is None:
                        fall_dist += 1
                    
                    if fall_dist > 0:
                        self.grid[y + fall_dist][x] = self.grid[y][x]
                        self.grid[y][x] = None
                        moved = True
        return moved, 0

    def _check_matches(self):
        to_clear = set()
        visited = set()

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] is not None and (x, y) not in visited:
                    color = self.grid[y][x]
                    component = set()
                    q = deque([(x, y)])
                    
                    while q:
                        cx, cy = q.popleft()
                        if (cx, cy) in visited or not (0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT):
                            continue
                        
                        if self.grid[cy][cx] == color:
                            visited.add((cx, cy))
                            component.add((cx, cy))
                            q.append((cx + 1, cy))
                            q.append((cx - 1, cy))
                            q.append((cx, cy + 1))
                            q.append((cx, cy - 1))

                    if len(component) >= self.MATCH_THRESHOLD:
                        to_clear.update(component)

        if not to_clear:
            return 0, False

        reward = 0
        for x, y in to_clear:
            color = self.grid[y][x]
            self.grid[y][x] = None
            self.score += 10
            reward += 1
            self._create_particles(x * self.BLOCK_SIZE + self.BLOCK_SIZE/2, y * self.BLOCK_SIZE + self.BLOCK_SIZE/2, color, 15)
            self.flash_effects.append({'pos': (x, y), 'color': (255, 255, 255), 'duration': 10})

        return reward, True

    def _check_stack_splits(self):
        split_occurred = False
        for x in range(self.GRID_WIDTH):
            stack_top_y = -1
            for y in range(self.GRID_HEIGHT):
                if self.grid[y][x] is not None:
                    stack_top_y = y
                    break
            
            if stack_top_y == -1: continue

            stack_bottom_y = self.GRID_HEIGHT - 1
            while stack_bottom_y > stack_top_y and self.grid[stack_bottom_y][x] is None:
                stack_bottom_y -= 1
            
            height = stack_bottom_y - stack_top_y + 1
            
            if height > self.STACK_SPLIT_HEIGHT:
                split_occurred = True
                stack_colors = [self.grid[y][x] for y in range(stack_top_y, stack_bottom_y + 1)]
                
                for y in range(stack_top_y, stack_bottom_y + 1):
                    self.grid[y][x] = None
                    self._create_particles(x * self.BLOCK_SIZE + self.BLOCK_SIZE/2, y * self.BLOCK_SIZE + self.BLOCK_SIZE/2, (200,200,200), 3)

                left_colors = stack_colors[:len(stack_colors)//2]
                right_colors = stack_colors[len(stack_colors)//2:]
                
                for i, col_data in enumerate([(-1, left_colors), (1, right_colors)]):
                    offset, colors = col_data
                    target_x = (x + offset + self.GRID_WIDTH) % self.GRID_WIDTH
                    
                    target_y = self.GRID_HEIGHT - 1
                    while target_y >= 0 and self.grid[target_y][target_x] is not None:
                        target_y -= 1
                    
                    for j in range(len(colors) - 1, -1, -1):
                        if target_y - (len(colors)-1-j) >= 0:
                            self.grid[target_y - (len(colors)-1-j)][target_x] = colors[j]
        
        return split_occurred, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, self.BLOCK_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.BLOCK_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] is not None:
                    self._draw_block(self.screen, x, y, self.grid[y][x])

        if not self.game_over and self.falling_block:
            gx = int(self.falling_block['x'] / self.BLOCK_SIZE)
            gy = int(self.falling_block['y'] / self.BLOCK_SIZE)
            color = self.falling_block['color']
            
            glow_center = (int(self.falling_block['x'] + self.BLOCK_SIZE/2), int(self.falling_block['y'] + self.BLOCK_SIZE/2))
            for i in range(10, 0, -2):
                pygame.gfxdraw.aacircle(self.screen, glow_center[0], glow_center[1], int(self.BLOCK_SIZE/2 + i), (color[0], color[1], color[2], 50-i*5))
            
            self._draw_block(self.screen, gx, gy, color, pos_override=(self.falling_block['x'], self.falling_block['y']))

        self._update_and_draw_particles()
        self._update_and_draw_flashes()

    def _draw_block(self, surface, gx, gy, color, pos_override=None):
        px, py = (gx * self.BLOCK_SIZE, gy * self.BLOCK_SIZE) if pos_override is None else pos_override
        
        outer_rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(surface, color, outer_rect)
        
        inner_color = tuple(min(255, c + 40) for c in color)
        inner_rect = pygame.Rect(px + 2, py + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4)
        pygame.draw.rect(surface, inner_color, inner_rect, border_radius=2)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
                s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
                s.fill((p['color'][0], p['color'][1], p['color'][2], alpha))
                self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

    def _update_and_draw_flashes(self):
        for f in self.flash_effects[:]:
            f['duration'] -= 1
            if f['duration'] <= 0:
                self.flash_effects.remove(f)
            else:
                alpha = max(0, min(255, int(255 * (f['duration'] / 10))))
                s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
                s.fill((f['color'][0], f['color'][1], f['color'][2], alpha))
                self.screen.blit(s, (f['pos'][0] * self.BLOCK_SIZE, f['pos'][1] * self.BLOCK_SIZE))

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(20, 41)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(2, 6)
            })

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))

        time_sec = self.time_remaining // 60
        timer_text = self.font_main.render(f"TIME: {time_sec}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 5))
        
        preview_surf = pygame.Surface((100, 80), pygame.SRCALPHA)
        preview_surf.fill(self.COLOR_UI_BG)
        pygame.draw.rect(preview_surf, self.COLOR_UI_TEXT, preview_surf.get_rect(), 2, border_radius=5)
        
        next_text = self.font_main.render("NEXT", True, self.COLOR_UI_TEXT)
        preview_surf.blit(next_text, (50 - next_text.get_width()//2, 5))

        if self.next_block_color:
            self._draw_block(preview_surf, 0, 0, self.next_block_color, pos_override=(50-self.BLOCK_SIZE//2, 35))
        
        self.screen.blit(preview_surf, (self.SCREEN_WIDTH//2 - 50, self.SCREEN_HEIGHT - 90))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message_text = self.font_large.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            self.screen.blit(message_text, (self.SCREEN_WIDTH//2 - message_text.get_width()//2, self.SCREEN_HEIGHT//2 - message_text.get_height()//2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Code ---
    # Set the video driver to a non-dummy one for display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Initialize pygame display for manual play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
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

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            if terminated:
                print(f"Game Over! Final Score: {info['score']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS

    env.close()