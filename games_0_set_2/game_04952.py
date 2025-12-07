
# Generated: 2025-08-28T03:31:39.872635
# Source Brief: brief_04952.md
# Brief Index: 4952

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for visual effects
class Particle:
    """A single particle for explosion effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = np_random.integers(15, 30)
        self.size = np_random.integers(3, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.vy += 0.1  # A bit of gravity

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 20))))
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.size, self.size), self.size)
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

class ScorePopup:
    """A floating text effect for score feedback."""
    def __init__(self, x, y, text, font, color):
        self.x = x
        self.y = y
        self.text = text
        self.font = font
        self.color = color
        self.lifespan = 45  # frames
        self.vy = -1.5

    def update(self):
        self.y += self.vy
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 45))))
            text_surface = self.font.render(self.text, True, self.color)
            text_surface.set_alpha(alpha)
            surface.blit(text_surface, (int(self.x - text_surface.get_width() / 2), int(self.y)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to select a group of 3 or more matching fruits."
    )

    game_description = (
        "Match cascading fruits in a frantic race against time! Create large combos to maximize your score and reach 2000 points before the 60-second timer runs out."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 8
    NUM_FRUIT_TYPES = 5
    MIN_MATCH_SIZE = 3
    WIN_SCORE = 2000
    GAME_DURATION_SECONDS = 60
    FPS = 30

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 45, 60)
    COLOR_GRID_LINES = (50, 70, 90)
    COLOR_WHITE = (230, 230, 230)
    COLOR_GOLD = (255, 215, 0)
    COLOR_CURSOR = (255, 255, 255)
    
    FRUIT_COLORS = [
        (220, 50, 50),   # Red
        (50, 220, 50),   # Green
        (80, 80, 255),   # Blue
        (255, 165, 0),   # Orange
        (160, 32, 240),  # Purple
    ]
    
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
        
        self.ui_font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.ui_font_small = pygame.font.SysFont("Arial", 24)
        self.popup_font = pygame.font.SysFont("Arial", 20, bold=True)

        self.grid_rect = pygame.Rect(0, 0, 360, 288)
        self.grid_rect.center = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20)
        self.cell_size = self.grid_rect.width // self.GRID_WIDTH
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.prev_space_held = False
        
        self.grid = self._generate_initial_grid()
        self.fruit_props = [[{'y': self.grid_rect.top + y * self.cell_size} for _ in range(self.GRID_WIDTH)] for y in range(self.GRID_HEIGHT)]
        
        self.processing_board = False
        self.process_timer = 0
        self.combo_count = 0
        
        self.particles = []
        self.score_popups = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        self._update_animations()

        if self.processing_board:
            self.process_timer -= 1
            if self.process_timer <= 0:
                cascade_reward, cascade_score = self._resolve_board_state()
                reward += cascade_reward
                self.score += cascade_score
        else:
            self._handle_input(movement)
            if space_pressed:
                match_reward, match_score = self._initiate_match()
                if match_reward > 0:
                    reward += match_reward
                    self.score += match_score

        terminated = self.time_remaining <= 0 or self.score >= self.WIN_SCORE
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                self.win = True
                reward += 100
            else:
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1

    def _initiate_match(self):
        x, y = self.cursor_pos
        if self.grid[y][x] is None:
            return 0, 0
            
        group = self._find_connected_group(x, y)
        
        if len(group) >= self.MIN_MATCH_SIZE:
            # sfx: Match success
            self.combo_count = 1
            match_score = int(len(group) * 10 * (1 + 0.1 * (len(group) - self.MIN_MATCH_SIZE)))
            self._remove_fruits(group, match_score)
            self.processing_board = True
            self.process_timer = 10 # Animation delay before fall
            return len(group), match_score
        else:
            # sfx: Match fail
            return 0, 0

    def _resolve_board_state(self):
        total_reward = 0
        total_score = 0
        
        self._apply_gravity()
        self._fill_new_fruits()
        self.process_timer = 20 # Fall animation time

        cascades = self._find_all_matches()
        if cascades:
            # sfx: Combo
            self.combo_count += 1
            reward_for_step = 5 * (self.combo_count - 1)
            for group in cascades:
                combo_bonus = 1.0 + self.combo_count * 0.5
                cascade_score = int(len(group) * 10 * combo_bonus)
                reward_for_step += len(group)
                total_score += cascade_score
                self._remove_fruits(group, cascade_score)
            total_reward += reward_for_step
            self.process_timer += 10 # Add more time for combo animation
        else:
            self.processing_board = False
            self.combo_count = 0
            
        return total_reward, total_score

    def _remove_fruits(self, group, score_val):
        avg_x, avg_y = 0, 0
        for x, y in group:
            if self.grid[y][x] is not None:
                self._create_particles(x, y, self.grid[y][x])
                self.grid[y][x] = None
                avg_x += x
                avg_y += y
        
        if group:
            avg_x /= len(group)
            avg_y /= len(group)
            popup_x = self.grid_rect.left + avg_x * self.cell_size + self.cell_size / 2
            popup_y = self.grid_rect.top + avg_y * self.cell_size
            self.score_popups.append(ScorePopup(popup_x, popup_y, f"+{score_val}", self.popup_font, self.COLOR_GOLD))

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] is not None:
                    if y != empty_row:
                        self.grid[empty_row][x] = self.grid[y][x]
                        self.grid[y][x] = None
                        self.fruit_props[empty_row][x]['y'] = self.fruit_props[y][x]['y']
                    empty_row -= 1

    def _fill_new_fruits(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] is None:
                    self.grid[y][x] = self.np_random.integers(0, self.NUM_FRUIT_TYPES)
                    self.fruit_props[y][x]['y'] = self.grid_rect.top - self.cell_size * 2

    def _find_connected_group(self, start_x, start_y):
        fruit_type = self.grid[start_y][start_x]
        if fruit_type is None: return []
            
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        group = []

        while q:
            x, y = q.popleft()
            group.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[ny][nx] == fruit_type:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _find_all_matches(self):
        all_matches = []
        visited = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in visited and self.grid[y][x] is not None:
                    group = self._find_connected_group(x, y)
                    if len(group) >= self.MIN_MATCH_SIZE:
                        all_matches.append(group)
                        for pos in group: visited.add(pos)
        return all_matches

    def _generate_initial_grid(self):
        grid = [[-1] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                possible_fruits = list(range(self.NUM_FRUIT_TYPES))
                if x > 1 and grid[y][x-1] == grid[y][x-2]:
                    if grid[y][x-1] in possible_fruits: possible_fruits.remove(grid[y][x-1])
                if y > 1 and grid[y-1][x] == grid[y-2][x]:
                    if grid[y-1][x] in possible_fruits: possible_fruits.remove(grid[y-1][x])
                if not possible_fruits: possible_fruits = list(range(self.NUM_FRUIT_TYPES))
                grid[y][x] = self.np_random.choice(possible_fruits)
        return grid

    def _update_animations(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles: p.update()
        
        self.score_popups = [s for s in self.score_popups if s.lifespan > 0]
        for s in self.score_popups: s.update()

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                props = self.fruit_props[y][x]
                target_y = self.grid_rect.top + y * self.cell_size
                props['y'] += (target_y - props['y']) * 0.4 # Interpolate for smooth fall

    def _create_particles(self, grid_x, grid_y, fruit_type):
        center_x = self.grid_rect.left + grid_x * self.cell_size + self.cell_size / 2
        center_y = self.grid_rect.top + grid_y * self.cell_size + self.cell_size / 2
        color = self.FRUIT_COLORS[fruit_type]
        for _ in range(15): self.particles.append(Particle(center_x, center_y, color, self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, self.grid_rect, border_radius=10)

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                fruit_type = self.grid[y][x]
                if fruit_type is not None:
                    props = self.fruit_props[y][x]
                    px = self.grid_rect.left + x * self.cell_size
                    self._draw_fruit(self.screen, fruit_type, px, props['y'])

        for i in range(1, self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_rect.left + i * self.cell_size, self.grid_rect.top), (self.grid_rect.left + i * self.cell_size, self.grid_rect.bottom))
        for i in range(1, self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_rect.left, self.grid_rect.top + i * self.cell_size), (self.grid_rect.right, self.grid_rect.top + i * self.cell_size))
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, self.grid_rect, width=2, border_radius=10)

        if not self.processing_board and not self.game_over:
            cx, cy = self.cursor_pos
            rect = pygame.Rect(self.grid_rect.left + cx * self.cell_size, self.grid_rect.top + cy * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 4, border_radius=5)

        for p in self.particles: p.draw(self.screen)
        for s in self.score_popups: s.draw(self.screen)

    def _draw_fruit(self, surface, fruit_type, x, y):
        size = self.cell_size
        center_x, center_y = int(x + size // 2), int(y + size // 2)
        radius = int(size * 0.38)
        color = self.FRUIT_COLORS[fruit_type]
        darker_color = tuple(max(0, c-40) for c in color)

        if fruit_type == 0: # Red Circle
            pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, darker_color)
            pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, color)
        elif fruit_type == 1: # Green Square
            rect = pygame.Rect(x + size*0.15, y + size*0.15, size*0.7, size*0.7)
            pygame.draw.rect(surface, color, rect, border_radius=int(size*0.1))
            pygame.draw.rect(surface, darker_color, rect, width=3, border_radius=int(size*0.1))
        elif fruit_type == 2: # Blue Triangle
            points = [(center_x, y + size*0.15), (x + size*0.15, y + size*0.85), (x + size*0.85, y + size*0.85)]
            pygame.gfxdraw.aapolygon(surface, points, darker_color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif fruit_type == 3: # Orange Hexagon
            points = [(center_x + radius * math.cos(math.pi/3*i), center_y + radius * math.sin(math.pi/3*i)) for i in range(6)]
            pygame.gfxdraw.aapolygon(surface, points, darker_color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif fruit_type == 4: # Purple Star
            points = [(center_x + (radius if i%2==0 else radius*0.5) * math.cos(math.pi/5*i - math.pi/2), center_y + (radius if i%2==0 else radius*0.5) * math.sin(math.pi/5*i - math.pi/2)) for i in range(10)]
            pygame.gfxdraw.aapolygon(surface, points, darker_color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_ui(self):
        score_text = self.ui_font_small.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (20, 10))
        
        time_str = f"{max(0, self.time_remaining // self.FPS):02d}"
        time_color = self.COLOR_WHITE if self.time_remaining > 10 * self.FPS else (255, 80, 80)
        time_text = self.ui_font_small.render(f"Time: {time_str}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 20, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "TIME'S UP!"
            end_text = self.ui_font_large.render(end_text_str, True, self.COLOR_GOLD if self.win else self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To run and visualize the game with human controls
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Matcher")
    
    running = True
    total_reward = 0
    
    key_to_action = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}

    while running:
        movement_action, space_action = 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        for key, move in key_to_action.items():
            if keys[key]:
                movement_action = move
                break
        
        if keys[pygame.K_SPACE]:
            space_action = 1
            
        action = [movement_action, space_action, 0] # Movement, Space, Shift
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
        
        env.clock.tick(env.FPS)
        
    env.close()