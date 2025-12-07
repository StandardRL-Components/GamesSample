import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select a block, "
        "then move to an adjacent block of the same color and press space again to connect. "
        "Press shift to deselect."
    )

    game_description = (
        "Connect adjacent blocks of the same color in a grid to clear them and achieve a "
        "high score before the 30-second timer runs out. Longer chains give more points!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 10
        self.NUM_COLORS = 3  # Red, Green, Blue
        self.MAX_TIME_SECONDS = 30
        self.FPS = 30
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS

        # Observation and action spaces
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
        self.font_small = pygame.font.Font(None, 36)

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (45, 55, 65)
        self.COLOR_EMPTY = (35, 45, 55)
        self.BLOCK_COLORS = {
            1: (231, 76, 60),   # Red
            2: (46, 204, 113),  # Green
            3: (52, 152, 219),  # Blue
        }
        self.COLOR_CURSOR = (241, 196, 15)
        self.COLOR_SELECTED = (255, 255, 255)
        self.COLOR_TEXT = (236, 240, 241)

        # Grid layout calculation
        self.top_margin = 60
        self.bottom_margin = 20
        self.side_margin = 20
        grid_h = self.HEIGHT - self.top_margin - self.bottom_margin
        grid_w = self.WIDTH - self.side_margin * 2
        self.block_size = min(grid_h // self.GRID_ROWS, grid_w // self.GRID_COLS)
        self.block_padding = self.block_size // 10
        self.grid_pixel_w = self.GRID_COLS * self.block_size
        self.grid_pixel_h = self.GRID_ROWS * self.block_size
        self.grid_start_x = (self.WIDTH - self.grid_pixel_w) // 2
        self.grid_start_y = self.top_margin

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.particles = None
        self.last_action_was_press = {'space': False, 'shift': False}
        self.connection_anim = None
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_pos = None
        self.particles = []
        self.connection_anim = None
        self.last_action_was_press = {'space': False, 'shift': False}

        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = 0

        # Unpack factorized action
        movement, space_action, shift_action = action[0], action[1], action[2]
        
        # Detect press (rising edge)
        space_pressed = space_action == 1 and not self.last_action_was_press['space']
        shift_pressed = shift_action == 1 and not self.last_action_was_press['shift']
        self.last_action_was_press['space'] = (space_action == 1)
        self.last_action_was_press['shift'] = (shift_action == 1)

        self._handle_movement(movement)
        reward += self._handle_actions(space_pressed, shift_pressed)

        self._update_animations()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self._is_board_clear():
                reward += 50 # Final board clear bonus
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            bool(terminated),
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement != 0:
            dy, dx = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dy) % self.GRID_ROWS
            self.cursor_pos[1] = (self.cursor_pos[1] + dx) % self.GRID_COLS

    def _handle_actions(self, space_pressed, shift_pressed):
        if shift_pressed and self.selected_pos:
            self.selected_pos = None
            return 0

        if not space_pressed:
            return 0

        cursor_r, cursor_c = self.cursor_pos
        
        if not self.selected_pos:
            if self.grid[cursor_r, cursor_c] != 0:
                self.selected_pos = [cursor_r, cursor_c]
            return 0
        
        is_adjacent = abs(self.selected_pos[0] - cursor_r) + abs(self.selected_pos[1] - cursor_c) == 1
        is_same_color = self.grid[self.selected_pos[0], self.selected_pos[1]] == self.grid[cursor_r, cursor_c]
        
        if is_adjacent and is_same_color:
            component, color_id = self._find_connected_component(self.selected_pos)
            
            if len(component) < 2:
                self.selected_pos = None
                return 0
                
            self.connection_anim = {'path': list(component), 'timer': 5, 'color': self.BLOCK_COLORS[color_id]}

            num_before = np.sum(self.grid == color_id)

            for r, c in component:
                self.grid[r, c] = 0
                self._create_particles(r, c, self.BLOCK_COLORS[color_id])

            self._apply_gravity()
            
            self.selected_pos = None

            reward = len(component)
            
            num_after = np.sum(self.grid == color_id)
            if num_after == 0 and num_before == len(component):
                reward += 10

            return reward
        else:
            self.selected_pos = None
            return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        pygame.draw.rect(
            self.screen, self.COLOR_GRID,
            (self.grid_start_x, self.grid_start_y, self.grid_pixel_w, self.grid_pixel_h),
            border_radius=5
        )

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_id = self.grid[r, c]
                rect = self._get_block_rect(r, c)
                
                block_color = self.BLOCK_COLORS.get(color_id, self.COLOR_EMPTY)
                pygame.draw.rect(self.screen, block_color, rect, border_radius=4)
        
        self._render_cursor()
        if self.selected_pos:
            self._render_selection()
        
        if self.connection_anim and self.connection_anim['timer'] > 0:
            points = [self._get_block_center(r, c) for r, c in self.connection_anim['path']]
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.connection_anim['color'], False, points, width=8)

        self._render_particles()

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        time_left = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_TEXT if time_left > 5 else self.BLOCK_COLORS[1]
        timer_text = self.font_large.render(f"{time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "BOARD CLEARED!" if self._is_board_clear() else "TIME'S UP!"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)


    def _render_cursor(self):
        r, c = self.cursor_pos
        rect = self._get_block_rect(r, c)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, width=4, border_radius=5)

    def _render_selection(self):
        r, c = self.selected_pos
        rect = self._get_block_rect(r, c)
        pulse = abs(math.sin(self.steps * 0.2))
        width = int(2 + pulse * 3)
        pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, width=width, border_radius=5)

    def _update_animations(self):
        if self.connection_anim:
            self.connection_anim['timer'] -= 1

        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                rect = pygame.Rect(p['pos'][0] - size//2, p['pos'][1] - size//2, size, size)
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), border_radius=2)
                self.screen.blit(shape_surf, rect)

    def _generate_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            if self._has_valid_moves():
                break

    def _has_valid_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color = self.grid[r, c]
                if c + 1 < self.GRID_COLS and self.grid[r, c + 1] == color:
                    return True
                if r + 1 < self.GRID_ROWS and self.grid[r + 1, c] == color:
                    return True
        return False
        
    def _find_connected_component(self, start_pos):
        r_start, c_start = start_pos
        color_to_match = self.grid[r_start, c_start]
        if color_to_match == 0:
            return [], 0
        
        q = deque([start_pos])
        visited = {tuple(start_pos)}
        component = []

        while q:
            r, c = q.popleft()
            component.append((r, c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                   tuple((nr, nc)) not in visited and self.grid[nr, nc] == color_to_match:
                    visited.add((nr, nc))
                    q.append([nr, nc])
        return component, color_to_match

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
    
    def _create_particles(self, r, c, color):
        center_x, center_y = self._get_block_center(r, c)
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 1],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(4, 8)
            })

    def _get_block_rect(self, r, c):
        x = self.grid_start_x + c * self.block_size + self.block_padding
        y = self.grid_start_y + r * self.block_size + self.block_padding
        size = self.block_size - 2 * self.block_padding
        return pygame.Rect(x, y, size, size)

    def _get_block_center(self, r, c):
        rect = self._get_block_rect(r, c)
        return rect.centerx, rect.centery

    def _check_termination(self):
        return self.timer <= 0 or self._is_board_clear()

    def _is_board_clear(self):
        return np.all(self.grid == 0)

if __name__ == "__main__":
    # This part allows a human to play the game.
    # It requires a display.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Connect-the-Blocks")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    movement = 0
    space_held = 0
    shift_held = 0
    
    running = True
    while running:
        # Action polling for continuous control
        action_changed = False
        old_movement = movement
        old_space = space_held
        old_shift = shift_held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        
        current_movement = 0
        if keys[pygame.K_UP]:
            current_movement = 1
        elif keys[pygame.K_DOWN]:
            current_movement = 2
        elif keys[pygame.K_LEFT]:
            current_movement = 3
        elif keys[pygame.K_RIGHT]:
            current_movement = 4
        
        current_space = 1 if keys[pygame.K_SPACE] else 0
        current_shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        if (current_movement != movement or 
            current_space != space_held or 
            current_shift != shift_held):
            
            movement = current_movement
            space_held = current_space
            shift_held = current_shift
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward > 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward}")
                
            if terminated or truncated:
                print("Game Over!")
                print(f"Final Score: {info['score']}")
                
                # Render final state
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                pygame.time.wait(2000)
                obs, info = env.reset()
                continue
        else:
            # If no key state change, we can just render the last observation
            pass

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    pygame.quit()